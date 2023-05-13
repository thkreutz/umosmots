# Loading data
import glob
import sys
import numpy as np
import open3d as o3d
import os
from src.utils.tk_utils import log
from src.visualization.tk_visualization import vis_sequence
import pandas as pd
from tqdm import tqdm
import json


def read_polyterasse_frames(path="data/Polyterasse/tannenstrasse/lidar_points", n_frames=-1, with_pcd=False, crop=False, crop_range=50):
    # path="data/Polyterasse/tannenstrasse/lidar_points"
    if path == "":
        print("no path to a scene given..")
        return None
    
    files = sorted(glob.glob(os.path.join(path, "*")))
    if n_frames == -1:
        all_frames = len(files)
    else:
        all_frames = n_frames
    print("n_files=%s, reading n_frames=%s"%(len(files), all_frames))
    
    scans = []
    for file in tqdm(files[:all_frames]):
        scan = pd.read_csv(file, delimiter=" ").values[:, 0:3]
    
        if crop:
            xlim = [-crop_range,crop_range]
            ylim = [-crop_range,crop_range]
            zlim = [-crop_range,crop_range] # check semantickitti method for varying z dim

            crop_mask = (scan[:, 0]  <= xlim[1] ) & (scan[:, 0] >= xlim[0]) & (scan[:, 1] <= ylim[1]) & (scan[:, 1] >= ylim[0]) & (scan[:, 2] <= zlim[1]) & (scan[:, 2] >= zlim[0])        
            scan = scan[crop_mask]
        scans.append(scan)

    pcd = []
    if with_pcd:
        pcd = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)) for points in scans] 

        
    return scans, pcd


def read_vlp16_frames(path="", start_frame=0, end_frame=2000, crop=True, crop_range=50):
    #path="/path/to/DataVLP16/data_train/2019-01-18-11-56-16_velodyne-vlp-16-data-towardslibrary2/data_pcl/*"
    if path == "":
        print("no path to a scene given..")
        return None
    files = sorted(glob.glob(os.path.join(path, "*")))
    #print(files)
    sort_idxs = np.argsort([int(f.split("/")[-1].split("_")[0]) for f in files])
    files = [files[i] for i in sort_idxs]

    print("n_files=%s" % len(files))
    # end_frame capped at 2000 right now
    pcd = [o3d.io.read_point_cloud(f) for f in tqdm(files[start_frame:end_frame])]

    scans = []
    # Crop pointcloud
    for pc in pcd:
        scan = np.asarray(pc.points)
        if crop:
            xlim = [-crop_range,crop_range]
            ylim = [-crop_range,crop_range]
            zlim = [-crop_range,crop_range] # check semantickitti method for varying z dim

            crop_mask = (scan[:, 0]  <= xlim[1] ) & (scan[:, 0] >= xlim[0]) & (scan[:, 1] <= ylim[1]) & (scan[:, 1] >= ylim[0]) & (scan[:, 2] <= zlim[1]) & (scan[:, 2] >= zlim[0])        
            scan = scan[crop_mask]
        scans.append(scan)
    
    return scans, pcd


def read_semantic_kitti_frames(scan_path, label_path, calib_path, poses_path, start_frame=0,
                             end_frame=200, crop_range=50, remove_ground=True, crop=True, 
                             verbose=True, with_pcd=False, align_scans=True, cutoff=-1, with_labels=True):

    if scan_path == "":
        print("no path to a scene given..")
        return None

    calib = parse_calibration(calib_path)
    poses = parse_poses(poses_path, calib)
    scan_files = sorted(glob.glob(os.path.join(scan_path, "*")))
    if with_labels:
        label_files = sorted(glob.glob(os.path.join(label_path, "*")))

    #log("n_files=%s" % len(scan_files))
    if(end_frame == len(scan_files)):
        return -1

    if end_frame == -1:
        end_frame = len(scan_files)

    # get first pose, we align based on initial pose
    pose_anchor = poses[start_frame]
    aligned_scans = []
    scans = []
    scans_labels = []
    # Read and crop to x(20m) limit first
    if verbose:
        print("1. Read scans")
    for i, scan_filename in tqdm(enumerate(scan_files[start_frame:end_frame+1]), disable= (not verbose)):
        # read current frame
        #print("reading", scan_filename)
        scan = np.fromfile(scan_filename, dtype=np.float32).reshape((-1,4))
        #scan = np.vstack ( (scan, [0,0,0,1]) ) # add "center" point at the end
        
        ## read labels
        if with_labels:
            gt_labels = np.fromfile(label_files[start_frame+i], dtype=np.int32) & 0xFFFF
        # Map all labels to 0 and 1 for moving or stationary
        #pred_point_labels = [ [0 if l == 9 else 1 for l in lbls] for lbls in tqdm(pred_pt_labels)]
        
        # for the scan we remove everything further than 20m already here
        #o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan[:, 0:3]))
        if crop:
            xlim = [-crop_range,crop_range]
            ylim = [-crop_range,crop_range]
            if remove_ground:
                zlim = [cutoff, crop_range] 
            else:
                zlim = [-crop_range,crop_range] 

            crop_mask = (scan[:, 0]  <= xlim[1] ) & (scan[:, 0] >= xlim[0]) & (scan[:, 1] <= ylim[1]) & (scan[:, 1] >= ylim[0]) & (scan[:, 2] <= zlim[1]) & (scan[:, 2] >= zlim[0])        
            scan = scan[crop_mask]
            if with_labels:
                gt_labels = gt_labels[crop_mask]
            
        scans.append(scan)
        
        if with_labels:
            # map to mos  (# >=251 from semantickitti labelling scheme, check semantic-kitti-all.yaml in official dataset)
            mos_idxs = gt_labels >= 251 
            static_idxs = gt_labels < 251
            gt_labels[mos_idxs] = 1  # just map them to 1 or 0
            gt_labels[static_idxs] = 0
        
            scans_labels.append(gt_labels)
        
    # ego-motion compensation
    if align_scans:
        if verbose:
            print("2. Ego-motion compensation")
        for i, scan in tqdm(enumerate(scans), disable=(not verbose)):
            # convert points to homogenous coordinates (x, y, z, 1)
            #scan = np.asarray(o3dpcd.points)
            points = np.ones((scan.shape[0], 4))
            points[:, 0:3] = scan[:, 0:3]
            #remissions = scan[:, 3]

            # pose alignment
            diff = np.matmul(np.linalg.inv(pose_anchor), poses[start_frame+i])
            tpoints = np.matmul(diff, points.T).T
            #tpoints[:, 3] = remissions

            aligned_scans.append(tpoints[:, :3])
    else:
        aligned_scans = [sc[:,:3] for sc in scans]
    
    
    ### would directly turn the scans into o3d pcs, optional.
    geos=[]
    if with_pcd:
        if verbose:
            print("Making o3d pcds.")
        geos = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(f)) for f in aligned_scans]
        
    return aligned_scans, geos, scans_labels

def parse_calibration(filename):
    """ read calibration file with given filename
      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename
      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    file.close()

    return poses


def get_crop_regions(pc, n_tiles):

    # splits a pointcloud into tile-based regions

    M, N, _ = (pc.get_axis_aligned_bounding_box().get_extent()).astype(int) 
    M = M // n_tiles
    N = N // n_tiles
    N = M # to same offset in both directions
    minbounds, maxbounds = pc.get_min_bound().astype(int), pc.get_max_bound().astype(int) # we do not care about border
    # get the region offsets
    tiles = [[(x,y,minbounds[2]), (x+M, y+N, maxbounds[2])] for x in range(minbounds[0], maxbounds[0], M) for y in range(minbounds[1], maxbounds[1], N)]
    # bounding boxes
    #bbs = [o3d.geometry.AxisAlignedBoundingBox(tile[0], tile[1]) for tile in tiles]
    #for bb in bbs:
    #    bb.color = [0,0,0]
    
    ## Improvement: Depending on radius, need overlap of radius+1, then discard left/right/top/bottom/front/depth-most radius 
    # voxels for perfect sliding crop overlaps

    ## Improvement:
    # need to remove voxels on the outside, they would reduce the miss rate in the hash table
    
    return tiles # list of [ (minbound, maxbound), ... ]


# just calls visualization from src.visualozation.tk_visualization
def visualize_sequence(pcs):
    """ Visualizes a sequence of pointcloud frames in open3d

    Args:
        pcs (list): Pointcloud sequence to visualize
    """    
    vis_sequence(pcs)

