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
- Compile the ufvl-net as a package.
```buildoutcfg
cd ufvl_net
pip install -e .
cd .. 
export PYTHONPATH=./ufvl_net/
```
## Data Preparation
We utilize two standard datasets (i.e, 7-Scenes and 12-Scenes) to evaluate our method.
- 7-Scenes: The 7-Scenes dataset can be downloaded from [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).
- 12-Scenes: The 12-Scenes dataset can be downloaded from [12-Scenes](https://graphics.stanford.edu/projects/reloc/).
## Model Zoo 
For evaluation, we provide the checkpoints of 7-Scenes dataset in [Google Drive](https://drive.google.com/drive/folders/1M4Knz3V-uGTSHUoxigZnlAiy3qMClJuE?usp=sharing). 
For evaluation, we also provide the checkpoints of 12-Scenes dataset in [Google Drive](https://drive.google.com/drive/folders/1M4Knz3V-uGTSHUoxigZnlAiy3qMClJuE?usp=sharing). 
- Note: We integrate these models into a single one. You can do the evaluation following the description in *Quick Start - Test*).


