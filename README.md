# umosmots
Public repository for our WACV2023 Accepted Paper: "Unsupervised 4D LiDAR Moving Object Segmentation in Stationary Settings
with Multivariate Occupancy Time Series" by Thomas Kreutz, Max Mühlhäuser, and Alejandro Sanchez Guinea

# Introduction
In this work, we address the problem of unsupervised
moving object segmentation (MOS) in 4D LiDAR data
recorded from a stationary sensor, where no ground truth
annotations are involved. Deep learning-based state-of-
the-art methods for LiDAR MOS strongly depend on anno-
tated ground truth data, which is expensive to obtain and
sparse in existence. To close this gap in the stationary set-
ting, we propose a novel 4D LiDAR representation based
on multivariate time series that relaxes the problem of un-
supervised MOS to a time series clustering problem. More
specifically, we propose modeling the change in occupancy
of a voxel by a multivariate occupancy time series (MOTS),
which captures spatio-temporal occupancy changes on the
voxel level and its surrounding neighborhood. To perform
unsupervised MOS, we train a neural network in a self-
supervised manner to encode MOTS into voxel-level feature
representations, which can be partitioned by a clustering al-
gorithm into moving or stationary. Experiments on station-
ary scenes from the Raw KITTI dataset show that our fully
unsupervised approach achieves performance that is com-
parable to that of supervised state-of-the-art approaches.

![alt text](https://github.com/thkreutz/umosmots/blob/main/demo.gif)

# Dependencies and Installation
#### Coming soon...

# Code and usage example
#### Coming soon...

# Pre-trained model
#### Coming soon...

# Data
#### Coming soon...

# Citation
Coming soon...
