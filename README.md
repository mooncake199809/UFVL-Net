# UFVL-Net
This is the PyTorch implementation of our paper "UFVL-Net: A Unified Framework for Visual Localization across Multiple Indoor Scenes".

## Highlights
- Once-for-multiple-scenes
UFVL-Net optimizes visual localization tasks of various scenes collectively using a multi-task learning manner, which challenges the conventional wisdom that SCoRe typically trains a separate model for each scene. 

- Competive performance
UFVL-Net delivers extraordinary performances on two benchmarks and complex real scenes. We demonstrate that once the training for UFVL-Net is done, UFVL-Net can generalize to new scenes with much fewer parameters by freezing the task-shared parameters.

## Environment Setup
To set up the enviroment you can easily run the following command:
- Create environment
```buildoutcfg
conda create -n ufvlnet python=3.7
conda activate ufvlnet
```
- Install torch, we verify UFVL-Net with pytorch 1.10.1 and cuda 11.3.
```buildoutcfg
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
- Subsequently, we need to build the cython module to install the PnP solver:
```buildoutcfg
cd ./pnpransac
rm -rf build
python setup.py build_ext --inplace
```
- Install openmmlab packages.
```buildoutcfg
pip install mmcv-full==1.5.0
```
