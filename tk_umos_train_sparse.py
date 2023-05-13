from src.data.MOTS_Datasets import MOTS_File_Dataset
from src.models import umos_ae
from src.models import UMOS_Trainer
import torch
import time
import os
import argparse
import json

def get_id():
    return round(time.time() * 1000)

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def main(args):
    ### TODO: Argument Parser [X] done
    # args: output_path, load_weights_path, radius, window_size, dataset, epochs, batch_size
    # Save args
    
    # args.output_path is on default /workspace/UMOS/models/
    load_weights_path = args.load_weights_path

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    continue_training = not load_weights_path == None
    if not continue_training:
        mid = get_id()
        output_path = os.path.join(args.output_path, "%s"%mid)
        os.mkdir(output_path) # make path
        # save args
        with open(os.path.join(output_path, "args.txt"), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        # checkpoint path
        output_path = os.path.join(output_path, "checkpoint.pth.tar")
        print("Checkpoint Path=", output_path)
    else:
        output_path = load_weights_path
        output_path = os.path.join(output_path, "checkpoint.pth.tar")
        print("Loading from Checkpoint Path=", output_path)
        
    device = get_device()
    torch.backends.cudnn.enabled = False

    # Create dataset   
    dataset = MOTS_File_Dataset(args.dataset, radius=args.radius, window_size=args.window_size)
    
    print("Dataset=%s, Window Size=%s, Radius=%s => Neighborhood shape=%s" % (args.dataset, args.window_size, args.radius, dataset.d.shape))

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    # Init Trainer
    trainer = UMOS_Trainer.UMOS_Trainer(checkpoint_path=output_path)
    
    # TODO: [X] Store args
    # TODO: [X] Speed up MTS neighborhood computation 
    # TODO: [X] Sequence length/window size up to 64 bit

    # Init model, loss function, and optimizer
    model, criterion, optimizer = umos_ae.create_UMOSAE(input_channels=dataset.d.shape[0], window_size=args.window_size, embedding_dim=args.embedding_dim)
    
    # Start training
    trainer.train_sparse(EPOCHS, BATCH_SIZE, model, dataset, optimizer, criterion, device, LOAD_MODEL=continue_training)


if __name__ == "__main__":
    # parse arguments or change default values
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', '-o',  type=str, default='models', help="path to the output of the saved weights")
    parser.add_argument('--load_weights_path', '-l',  type=str, default=None, help="path to some stored weights to continue training")
    parser.add_argument('--radius', '-r',  type=int, default=1, help="radius of the mts occupancy neighborhood")
    parser.add_argument('--window_size', '-w',  type=int, default=10, help="sequence length/time window size to compute features")
    parser.add_argument('--dataset', '-d',  type=str, default="data/SlidingWindowsPreprocessed/kitti_moving-10-01", help="path to the sparse tensor dataset, requires preprocessing -> tk_umos_preprocess_script.py")
    parser.add_argument( '--epochs', '-e', type=int, default=5, help="number of epochs, low number is enough because we process large scenes with high number of frames")
    parser.add_argument( '--batch_size', '-b', type=int, default=512, help="batch size for training, can try different ones, they will be used on each crop of the sequence")
    parser.add_argument( '--embedding_dim', '-ed', type=int, default=16, help="embedding dimension")

    args = parser.parse_args()

    main(args)
