# umosmots
Public repository for our WACV2023 Accepted Paper: "Unsupervised 4D LiDAR Moving Object Segmentation in Stationary Settings
with Multivariate Occupancy Time Series" by Thomas Kreutz, Max Mühlhäuser, and Alejandro Sanchez Guinea [arxiv](https://arxiv.org/abs/2212.14750)

# Abstract
In this work, we address the problem of unsupervised moving object segmentation (MOS) in 4D LiDAR data recorded from a stationary sensor, where no ground truth annotations are involved. Deep learning-based state-of-the-art methods for LiDAR MOS strongly depend on annotated ground truth data, which is expensive to obtain and scarce in existence.
To close this gap in the stationary setting, we propose a novel 4D LiDAR representation based on multivariate time series that relaxes the problem of unsupervised MOS to a time series clustering problem. More specifically, we propose modeling the change in occupancy of a voxel by a multivariate occupancy time series (MOTS), which captures spatio-temporal occupancy changes on the voxel level and its surrounding neighborhood. To perform unsupervised MOS, we train a neural network in a self-supervised manner to encode MOTS into voxel-level feature representations, which can be partitioned by a clustering algorithm into moving or stationary. Experiments on stationary scenes from the Raw KITTI dataset show that our fully unsupervised approach achieves performance that is comparable to that of supervised state-of-the-art approaches.

![alt text](https://github.com/thkreutz/umosmots/blob/main/demo.gif)

# Dependencies and Installation
Coming soon...

# Code and usage example
Coming soon...

# Pre-trained model
Coming soon...

# Data
Coming soon...

# Citation
Coming soon...
