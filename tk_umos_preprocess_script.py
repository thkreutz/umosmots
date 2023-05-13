from src.data.preprocessing import DataPreprocessor
import sys
import argparse
import os

if __name__ == "__main__":
    # You can also call the code below from a jupyter notebook.
    # You only have to preprocess the training set for your model.
    # Evaluation does not need preprocessing as its done here.

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', '-o',  type=str, default='data/SlidingWindowsPreprocessed', help="output path")
    parser.add_argument('--dataset', '-d', type=str, default="semantic_kitti", help="name of the dataset")
    parser.add_argument('--dataset_path', '-dp',  type=str, default="data/SemanticKITTI/sequences/", help="output path")
    parser.add_argument('--window_size', '-w',  type=int, default=10, help="sequence length")
    parser.add_argument('--voxel_size', '-v',  type=float, default=0.1, help="voxel size")

    args = parser.parse_args()

    # input path format
    # your/path/to/sequences/*/lidar_point_frames
    # Outputs -> .../your/path/to/your_name/windowed_and_voxelized_frames

    ### semantic_kitti -> KITTI sequences with given pose and calibration files
    data_preprocessor = DataPreprocessor(dataset=args.dataset, 
                input_path=args.dataset_path,
                output_path=os.path.join(args.output_path, "%s-%s-%s"%(args.dataset, args.window_size, 
                                                                    str(args.voxel_size).replace(".", ""))), 
                window_size=args.window_size, voxel_size=args.voxel_size)

    data_preprocessor.run()

    ### Polyterasse/Tannenstrasse

    #data_preprocessor = DataPreprocessor(dataset="polyterasse", 
    #            input_path="/path/to/datasets/Polyterasse/*/lidar_points", 
    #            output_path="/your/output/path", 
    #            window_size=20, voxel_size=0.1)


    ### VLP16 Data
    #data_preprocessor = DataPreprocessor(dataset="vlp16", 
    #            input_path="/path/to/DataVLP16/data_train/*/data_pcl", 
    #            output_path="/your/output/path", 
    #            window_size=20, voxel_size=0.1)

