import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from matplotlib import pyplot as plt
import getpy as gp
import torch
import MinkowskiEngine as ME
from sklearn.mixture import GaussianMixture
import json
from joblib import dump
import sys
import time
import argparse

from src.data import tk_data
from src.models.umos_ae import UMOSAE
from src.utils.tk_utils import log
from src.data.MOTS_Datasets import MOTS_Dataset

def find_overlap(p, gt_frame):
    #p=mots_predictions[0]
    vsall_preds = {}
    for i in set(p):
        vsall = [1 if i == x else 0 for x in p]
        vsall_preds[i] = vsall

    cluster_overlaps = []
    cluster_list = []
    for k, v in vsall_preds.items():
        overlap = jaccard_score(gt_frame, v)
        #print(k, overlap)
        cluster_overlaps.append(overlap)
        cluster_list.append(k)

    cluster_overlaps = np.array(cluster_overlaps)
    # top 3 overlaps
    top_k = cluster_overlaps.argsort()[-5:][::-1]
    top_k, cluster_overlaps[top_k]
    
    return np.array(cluster_list)[top_k], cluster_overlaps[top_k]

def map_predictions(mapping, mots_predictions):
    mapped_predictions = []
    for preds in tqdm(mots_predictions):
        temp = [1 if l in mapping else 0 for l in preds]
        mapped_predictions.append(temp)
    return mapped_predictions

def evaluate_miou(sorted_labels, predictions):
    miou_scores = []
    accuracy_scores = []
    for i, gold in enumerate(sorted_labels):
        miou_scores.append(jaccard_score(gold, predictions[i]))
        accuracy_scores.append(accuracy_score(gold, predictions[i]))
        
    print("mIoU: ", np.mean(miou_scores))
    print("mAcc: ", np.mean(accuracy_scores))
    #fig, ax = plt.subplots(1,2, figsize=(10,5))
    #ax[0].hist(miou_scores)
    #ax[0].set_title("IoU")
    #ax[1].hist(accuracy_scores)
    #ax[1].set_title("Acc")
    return miou_scores, accuracy_scores

def sort_points_labels(temp, l):
    sortidxs = np.arange(len(temp))
    sortidxs_a = temp[:,2].argsort()
    temp = temp[sortidxs_a]
    sortidxs_b = temp[:,1].argsort(kind='mergesort')
    temp = temp[sortidxs_b]
    sortidxs_c = temp[:,0].argsort(kind='mergesort')
    temp = temp[sortidxs_c]

    lbls_temp = np.array(l)
    permut_labels = lbls_temp[sortidxs_a][sortidxs_b][sortidxs_c]
    return temp, permut_labels

def allign_gt_voxel_labels(vs, labels, window_size=10):
    # Disclaimer: We did not map back to voxel predictions to points.
    # We still argue that evaluating on the voxel level is not a threat to the validity of the evaluation.
    # Since this is how the evaluation was done in the paper, we kept the implementation as it is here now.
    # If you want, you can do that on your own, or ask us for some guidance how to implement mapping voxel predictions back to points :)
    
    # Voxelization of points AND point labels
    # Q has the voxelized pointcloud frames
    # L holds the "voxelized" labels -> one label for each voxel
    Q = vs
    L = labels

    print("Normalizing frames to positive voxel coordinates")
    normalized_frames = []
    #window_size = 10
    for j in tqdm(range(len(Q)-(window_size-1))):
        X = ME.utils.batched_coordinates(Q[j:j+window_size])
        X[:,1:] += (-1 * torch.min(X[:,1:], dim=0)[0])

        normalized_frames.append(X[X[:,0] == window_size-1][:,1:])
        #normalized_frames.append(X[X[:,0] == int( (window_size/2))-1][:,1:])
    
    print("Sorting voxels and labels to allign with MOTS predictions")
    # Sort voxel and their labels in each frame (start at window_size - 1 with the labels because normalized frames made sliding windows.)
    sorted_frames = []
    sorted_labels = []
    for frame, lbls in zip(normalized_frames, L[window_size-1:]):
    #for frame, lbls in zip(normalized_frames, L[int((window_size/2))-1:]):
        tempo, permut_labels = sort_points_labels(frame.numpy(), lbls)
        sorted_frames.append(tempo)
        sorted_labels.append(permut_labels)
        
    return sorted_frames, sorted_labels


def clustering_predictions(embs, gm):
    predictions = []
    for emb in tqdm(embs):
        preds = gm.predict(emb.reshape(emb.shape[0], emb.shape[1]))
        predictions.append(list(preds))
    return predictions

def save_predictions(predictions, clustering_outpath, window_size, mapped=False):
    # path for each clustering model and sequences
    if not os.path.exists(clustering_outpath):
        os.makedirs(clustering_outpath)
    
    for i, preds in enumerate(predictions):
        fname = "%s.npy" % (i + (window_size - 1))
        fname = fname.zfill(10)
        if not mapped:
            np.save(os.path.join(clustering_outpath, "raw_%s" % fname), preds)
        else:
            np.save(os.path.join(clustering_outpath, "mapped_%s" % fname), preds)

def load_data(scene_path, window_size=10, voxel_size=0.1, crop_range=50, start_frame=0, end_frame=-1):
    ### Basically Constants -> Voxel_size is other hyperparameter to evaluate
    #voxel_size=0.1
    #crop_range = 50  -> increase or decrease the crop range to evaluate on larger or smaller range of the scene.

    # load point cloud and labels
    calib_path = os.path.join(scene_path, "calib.txt") 
    poses_path = os.path.join(scene_path, "poses.txt")
    labels_path = os.path.join(scene_path, "labels")
    scene_path = os.path.join(scene_path, "velodyne")

    scans, _, labels = tk_data.read_semantic_kitti_frames(scene_path, label_path=labels_path, calib_path=calib_path, 
                                poses_path=poses_path, with_labels=True, start_frame=start_frame, end_frame=end_frame, cutoff=-1,
                                crop_range=crop_range, remove_ground=True, crop=True, with_pcd=False, verbose=True)

    # voxelization
    vs_voxelize = [ME.utils.sparse_quantize(torch.Tensor(c).contiguous(), labels=torch.Tensor(l).type(torch.int32), quantization_size=voxel_size) for c,l in zip(scans,labels)]
    vs = [vs[0].numpy() for vs in vs_voxelize] # voxels
    vs_labels = [vs[1].numpy() for vs in vs_voxelize] # labels
    sorted_vs, sorted_vs_labels = allign_gt_voxel_labels(vs, vs_labels, window_size) # just sort them to allign with the predictions that we make... 

    # cant understand where -100 comes from, its there once or twice, we just go and ignore.
    temp = []
    for x in sorted_vs_labels:
        temp.append([i if i != -100 else 0 for i in x])
    sorted_vs_labels = temp

    # for voxel-to-point remapping if required, but we didnt do that here.
    #vs_prime = [ME.utils.sparse_quantize(torch.Tensor(s).contiguous(), quantization_size=voxel_size, return_index=True, return_inverse=True) for s in scans]

    print("Create sliding windows...")
    frames = []
    for j in range(len(scans) - (window_size - 1)):
        X = ME.utils.batched_coordinates(vs[j:j + window_size])
        frames.append(X)
    
    return frames, sorted_vs, sorted_vs_labels


def main(data_root_path, scene_path, eval_scene, model_root_path, model_id, test_clusters, start_frame=0, end_frame=-1):

    #data_root_path = "/workspace/approaches/WACV_UMOS/data" ## TO_MODIFY: Modify this to data root path
    #model_root_path = "/workspace/approaches/WACV_UMOS/models"  ## TO_MODIFY: Modify this to your model root path

    start = time.time()
    log("****** STARTING EXPERIMENT *******")
    log("Eval scene path: %s"%scene_path)


    # Model
    model_path = os.path.join(model_root_path, model_id)
    checkpoint_path = os.path.join(model_path, "checkpoint_1.pth.tar") # can change which checkpoint to use here
    log("Path to model checkpoint: %s" % checkpoint_path)

    #log("Model path=%s"%model_path)

    # General parameters  
    with open(os.path.join(model_path, "args.txt"), 'r') as f:
        args = json.load(f)

    # nn parameters
    radius = args["radius"]
    embedding_dim = args["embedding_dim"]
    window_size = args["window_size"]
    # experiment parameters
    test_clusters = [10, 15, 20]
    n_runs = 1 # how often to repeat the evaluation.
    n_uniform_samples = 200000
    log("Model+Experiment args: r=%s, e=%s, w=%s, n_clusters=%s, n_runs=%s, n_uniform_samples=%s" % (radius, embedding_dim, window_size, 
                                                                                                test_clusters, n_runs, n_uniform_samples))

    log("Reading and preparing data....")
    # reads sliding window frames and voxelizes them
    frames, vs, labels = load_data(scene_path, window_size=window_size, start_frame=start_frame, end_frame=end_frame)
    dataset = MOTS_Dataset(frames, r=radius, window_size=window_size)
    #a, b = dataset[0]
    #print(a.shape, b.shape)

    log("Loading model...")
    model = UMOSAE(input_channels=dataset.mots_utils.d.shape[0], window_size=window_size, embedding_dim=embedding_dim).cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    
    log("Encoding MOTS...")
    # encode MOTS for each frame sequentially
    embs = []
    for i in tqdm(range(len(frames))):
        with torch.no_grad():
            features, _ = dataset[i] # one batch == all MOTS of a single frame -> shape (n_voxels, n_channels, window_size)
            # voxels in here correspond to the voxels in vs, they are not ordered the same though.
            embs.append(model.projection_head(torch.flatten(model.encoder(features.cuda()), start_dim=1)).cpu().detach().numpy())
    print(embs[0].shape)
    # only consider first 10 frames for clustering -> use this model top predict the rest of the scene
    all_embs = np.vstack(embs[:10])
    all_embs = all_embs.reshape(all_embs.shape[0], all_embs.shape[1])


    log("Clustering...")
    for i in range(n_runs):
        log("*********** Run %s/%s ***********"%(i+1, n_runs))
        for n_clusters in tqdm(test_clusters):
            clustering_name = "k%s-%s" % (n_clusters, i)
            clustering_outpath = os.path.join(model_path, "predictions", eval_scene, clustering_name)
            log("Clustering... Outpath clustering predictions: %s"%clustering_outpath)
            log("Num embeddings:%s"%len(all_embs) )
            # uniform sampling if enough samples in all_embs, otherwise cluster on all first 10 frames
            if len(all_embs) >= n_uniform_samples:
                gm = GaussianMixture(n_components=n_clusters, random_state=0).fit(all_embs[np.random.choice(len(all_embs), n_uniform_samples, replace=False)])
            else:
                gm = GaussianMixture(n_components=n_clusters, random_state=0).fit(all_embs)

            ## Predictions from the trained gmm
            log("Clustering done. Predicting...")
            predictions = clustering_predictions(embs, gm)
            log("Saving predictions...")
            save_predictions(predictions, clustering_outpath, window_size, mapped=False)
            log("Saving clustering model...")
            dump(gm, os.path.join(clustering_outpath, "gmm.joblib"))

            # miou eval

            # show contingency matrix for general overlap between clusters.
            print("Contingency matrix: shows gt->cluster assignment in %")
            cm = metrics.cluster.contingency_matrix(labels[0], predictions[0]) / len(labels[0])
            cm = cm.round(5)
            cm_data = [
                [" "] +  list(np.arange(n_clusters)),
                ["Static"] + list(cm[0]),
                ["Moving"] + list(cm[1])
            ]
            # format string
            formstr = "{: >10}" * len(cm_data[0])
            for row in cm_data:
                print(formstr.format(*row))

            top_k, cluster_overlaps = find_overlap(predictions[0], labels[0])

            print("Cluster-wise IoU for the first frame...")
            iou_overlap_table = [
                [" "] + list(top_k),
                ["IoU"] + list(cluster_overlaps.round(5))
            ]
            formstr = "{: >10}" * len(iou_overlap_table[0])
            for row in iou_overlap_table:
                print(formstr.format(*row))

            mapping = top_k[cluster_overlaps > 0.15]
            mapped_predictions = map_predictions(mapping, predictions)
            miou_scores, accuracy_scores = evaluate_miou(labels, mapped_predictions)

            # also save the mapped predictions
            log("Saving mapped predictions...")
            save_predictions(mapped_predictions, clustering_outpath, window_size, mapped=True)

            # frame_wise mious
            log("Saving frame-wise iou scores...")
            np.save(os.path.join(clustering_outpath, "frame_mious.npy"), miou_scores)



    log("****** ENDING EXPERIMENT *******")
    end = time.time()
    print("Elapsed time (in minutes):", ((end - start)/60) )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d',  type=str, default='data/ExampleRawKITTI', help="path where to find the SemanticKITTI/sequences folder")
    parser.add_argument('--scene', '-sc',  type=str, default='City1', help="scene name for sequence")
    parser.add_argument('--sequence', '-seq',  type=str, default='-22', help="sequence id")
    parser.add_argument('--model_root', '-mp',  type=str, default="models/", help="root path where the models are stored")
    parser.add_argument('--model', '-mid',  type=str, default="1661045943760", help="the model id to evaluate")
    parser.add_argument('--start', '-st',  type=int, default=0, help="start frame number (stay within bounds, there is no check implemented)")
    parser.add_argument('--end', '-end',  type=int, default=200, help="end frame number (stay within bounds, there is no check implemented)")
    args = parser.parse_args()

    ### DATA args
    # for scenes we have a dictionary that maps City1, City2, and Campus1 to a respective ID in SemanticKITTI sequences folder
    # you can use that or directly reference by sequence ids 
    # -> using -seq allows to evaluate on arbitrary SemanticKITTI sequences (if there exist labels) if you are curious about the performance of this approach with a mobile sensor.
    data_root_path = args.data
    scenes = {"Campus1" : "22",
          "City1" : "23",
           "City2" : "24"
    }


    if int(args.sequence) != -1:
        # using direct query
        # must have sequence directory in the same folder as the data_path, i.e., -> path/to/data_path/<...,args.seq,22,23,24,, ...>
        scene_path = os.path.join(data_root_path, args.sequence)  # name in output directory
        eval_scene = args.sequence # name in output directory
    else:
        scene_path = os.path.join(data_root_path, scenes[args.scene])
        eval_scene = scenes[args.scene] # name in output directory

    ### MODEL args
    model_root_path = args.model_root
    model_id = args.model

    test_clusters = [10, 15, 20] # hardcoded here.
    
    main(data_root_path, scene_path, eval_scene, model_root_path, model_id, test_clusters, start_frame=args.start, end_frame=args.end)
    #scene = sys.argv[1]
    #mid = sys.argv[2]
    #print(scene, mid)
