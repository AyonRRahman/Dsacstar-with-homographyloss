# Homography-based loss project
- [Introduction](#introduction)
- [Installation](#installation)
- [Datasets Setup](#datasets-setup)
- [Training](#training)
- [Testing and Results ](#testing-and-results)

## Introduction
This repository is about replementation of DSAC\* [(found in this github repo)](https://github.com/vislearn/dsacstar). DSAC\* is a learning-based visual re-localization method to get the camera 6-D pose (the camera rotation and translation) from single new image from the specific scene. This can be done by training DSAC\* on the specific scene images.
DSAC\* is a combination of [Scene Coordinate Regression](https://ieeexplore.ieee.org/document/6619221) with CNNs and [Differentiable RANSAC (DSAC)](https://arxiv.org/abs/1611.05705) for end-to-end training. 

![](overview.png)
The DSAC is implemented in C++ and this code changes the loss function for DSAC and replaces it with homography-based loss function presented in this [paper](https://arxiv.org/abs/2205.01937).The idea behind this change is to enhance the training of this pipline to decrease the estimated pose error. 

## Installation
The repository contains an environment.yml that you can easily install the conda enviroment:
```bash
conda env create -f environment.yml
conda activate dsacstar_with_homography
```
You compile and install the C++ extension for DSAC\* by executing:
```bash
cd dsacstar
python setup.py install
```
## Datasets Setup 

Kindly refer to this [(this github repo)](https://github.com/clementinboittiaux/homography-loss-function) for datasets structure and how to setup the datasets for training and testing.

Here you can find instructions to setup datasets for use with this code.
```bash
cd datasets
```
## Cambridge and 7-Scenes

We provide the script [datasetup.py](datasets/datasetup.py) for setting up Cambridge and 7-Scenes datasets. The script can be
called with either the name of the dataset to setup, *e.g.*, `7-Scenes`, or the name of a specific scene, *e.g.*,
`KingsCollege`. For example, if you want to setup the whole Cambridge dataset:
```shell
python datasetup.py Cambridge
```
Or if you want to only setup the *chess* scene of 7-Scenes dataset:
```shell
python datasetup.py chess
```
All possibilities can be accessed by running:
```shell
python datasetup.py -h
```


## Custom dataset

We also support custom datasets in **COLMAP** model format.  
⚠️ Please note that only **RADIAL** camera models are supported for now.

The custom dataset folder must contain:
- The COLMAP model: `cameras`, `images` and `points3D` files in `.bin` or `.txt` format.
- A folder named `images` containing all images in the model.
- A file named `list_db.txt` with the name of all the images used for training, one image name per line.
- A file named `list_query.txt` with the name of all the images used for testing, one image name per line.

The final outline of the folder should look like this:
> - mydataset
>   - images
>     - frame001.jpg
>     - frame002.jpg
>     - frame003.jpg
>     - ...
>   - cameras.bin
>   - images.bin
>   - points3D.bin
>   - list_db.txt
>   - list_query.txt

An example of `list_db.txt` or `list_query.txt`:
```text
frame001.jpg
frame002.jpg
frame003.jpg
...
```

## Training
For training, run the following command
```
python dsacstar_training_with_homography.py <Dataset-Name> <Scene-Name>  --iterations 10000 --network_in <Path to file for initializing the Network>
```
## Testing and Results 
```
python Dsacstar_test.py <Dataset-Name> <Scene-Name> --path_to_checkpoints <Path to the folder of Checkpoints> --run <Tensorboard Folder Name>
```
## Monitor Training & Testing Result
```
tensorboard --logdir <Tensorboard Folder Name>
```
