from src.data import tk_data
from src.utils.tk_utils import log
import glob
import MinkowskiEngine as ME
from tqdm import tqdm
import numpy as np
import os
import torch

class DataPreprocessor():

    def __init__(self, dataset="semantic_kitti", input_path="", 
           output_path="", window_size=10, voxel_size=0.1):
        '''
        possible datasets: semantic_kitti (seqs specified), kitti (all seqs in folder), polyterasse, vlp16
        '''
        
        #input_path="/path/to/kitti/seq/*/velodyne_points/data", 
        #output_path="/your/output/path"
        self.input_path = input_path
        self.output_path = output_path
        self.window_size = window_size
        self.voxel_size = 0.1
        self.dataset = dataset

    def run(self):

        if self.dataset == "semantic_kitti":
            # filter scenes for training set here, we know semantickitti is {00-10} \ {8}
            scenes = [os.path.join(self.input_path, str(i).zfill(2), "velodyne/") for i in range(11) if i != 8]
        else:
            # get all scenes, we assume here that the given path contains using glob  
            # to path of the parent folder of the point cloud files
            scenes = sorted(glob.glob(self.input_path))


        print(self.input_path)
        print("Scenes: ", scenes)
        self.preprocess_data(scenes, self.output_path, window_size=self.window_size, voxel_size=self.voxel_size)

    #global_lens = []
    def preprocess_sparse_tensor(self, pcs, window_size=10, voxel_size=0.1, sceneid=""):

        ws = {}
        #nps = #[np.asarray(c.points) for c in pcs]
        vs = [ME.utils.sparse_quantize(torch.Tensor(c).contiguous(), quantization_size=voxel_size).numpy() for c in pcs]
        # sliding window for each frame
        for j in range(len(pcs)-(window_size - 1)):
            X = ME.utils.batched_coordinates(vs[j:j + window_size])
            name = "%s-%s" % (sceneid, j)
            if X.shape[0] > 100 and len(set(X[:,0].numpy())) == window_size: 
                ws[name] = X
                #global_lens.append(X.shape[0])
        return ws

    def save_preprocessed(self, ws, outpath="", sceneid=""):
        outdir = os.path.join(outpath, sceneid)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print("Saving to %s" % outdir)
        for key, item in ws.items():
            torch.save(item, os.path.join(outdir, "%s.pt"%key))
            #continue
        
    def preprocess_data(self, scenes, outdir, window_size=10, voxel_size=0.1):
        # scenes are the scenes that we want to preprocess
        # scene will get id 0 to len(scenes)
        
        for i, scene in enumerate(scenes):
            ## here we just loop through the scenes, we do not care about their actual ID when we store them.
            ## If semantickitti is preprocessed, then there will be a folder with ID 08 
            ## -> The ID that we save under has nothing to do with the validation sequence, 
            ## if you check the logs then you will notice that 08 is skipped. =)
            scene_id = str(i) 
            log("Preprocessing %s/%s = %s" % (i+1, len(scenes), scene))

            # read data
            if self.dataset == "polyterasse":
                pcs, _ = tk_data.read_polyterasse_frames(path=scene)
            elif self.dataset == "kitti" or self.dataset == "semantic_kitti":
                # need calib and poses for kitti
                calib_path = os.path.join("/".join(scene.split("/")[:-2]), "calib.txt") 
                poses_path = os.path.join("/".join(scene.split("/")[:-2]), "poses.txt")
                # you can also train on all frames, however, we trained on the first 200 frames only. thats why the limit here is set to 200
                pcs, _, _ = tk_data.read_semantic_kitti_frames(scene, label_path="", calib_path=calib_path, 
                                           poses_path=poses_path, with_labels=False, start_frame=0, end_frame=200, cutoff=-1,
                                           crop_range=50, remove_ground=True, crop=True, with_pcd=False, verbose=True)
            elif self.dataset == "vlp16":
                pcs, _ = tk_data.read_vlp16_frames(path=scene, end_frame=200)[5:] ## start a few frames delayed because its sometimes a bit occluded in the beginning
            
            # get windows
            print("Sliding windows + voxelization...")
            ws = self.preprocess_sparse_tensor(pcs, window_size=window_size, voxel_size=voxel_size, sceneid=scene_id)
            # savewindows
            #print("Saving to %s"%outdir)
            self.save_preprocessed(ws, outdir, scene_id)

