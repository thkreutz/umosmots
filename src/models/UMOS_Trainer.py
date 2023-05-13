import torch
import numpy as np
from tqdm import tqdm
import time


class UMOS_Trainer():
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    def save_checkpoint(self, checkpoint, output_path_file=""):
        #print("=> Saving checkpoint")
        torch.save(checkpoint, output_path_file)

    def load_checkpoint(self, checkpoint, model):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])

    def training_step_sparse(self, model, dataset, batch_size, optimizer, criterion, device):
        
        # we "batch" each batch... because too many samples per batch.
        permutation_dataset = torch.randperm(len(dataset))

        pbar = tqdm(range(len(dataset)))
        k = 0
        for i in pbar:
            loss_ = []

            X = dataset[permutation_dataset[i]] # loads random "batch" -> all mts of voxels of a scene crop
            if X.size()[0] < 128:
                # continue with next if not enough data. 
                k += 1
                continue

            # batch everything in the "batch"
            permutation_batch = torch.randperm(X.size()[0])
            X = X.to(device=device)

            for j in range(0, X.size()[0], batch_size):
                indices = permutation_batch[j: j + batch_size]
                batch_x = X[indices]

                # Forward pass
                #print(batch_x.shape)
                output = model(batch_x)
                loss = criterion(output, batch_x)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update pbar loss
                pbar.set_postfix(loss=loss.data, steps=k, n_examples=X.size()[0])
                loss_.append(loss.data.cpu().numpy())
                #del loss, output
                # k are our actual steps
                k += 1

            with open("%s_history.txt"%self.checkpoint_path.split(".")[0], "a") as f:
                f.write("%s %s\r\n" % ( int(round(time.time() * 1000)), np.mean(loss_)) )
            #print(k)
            # every 1000 batches make a checkpoint
            # Checkpoint
            if k > 1000:
                checkpoint = {
                    "state_dict" : model.state_dict(),
                    "optimizer" : optimizer.state_dict()
                }
                self.save_checkpoint(checkpoint, self.checkpoint_path)
                k = 0

    def train_sparse(self, epochs, batch_size, model, dataset, optimizer, criterion, device, LOAD_MODEL):
        # Load from checkpoint if needed
        if LOAD_MODEL:
            self.load_checkpoint(torch.load(self.checkpoint_path), model)

        print("=> Begin Training")

        # Train for #epochs
        for epoch in range(epochs):

            # Training step
            self.training_step_sparse(model, dataset, batch_size, optimizer, criterion, device)

            # Checkpoint
            checkpoint = {
                "state_dict" : model.state_dict(),
                "optimizer" : optimizer.state_dict()
            }
            # model checkpoint after epoch finished
            self.save_checkpoint(checkpoint, "%s_%s.pth.tar" % (self.checkpoint_path.split(".")[0], epoch)) 

    def predict(self, model, X):
        Z = model.encoder(X)
        return Z.detach().cpu().numpy()
