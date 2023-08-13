# Realizing visual localization across multiple scenes
This is the PyTorch implementation of our paper  
(1) "OFVL-MS: Once for Visual Localization across Multiple Indoor Scenes"  (ICCV2023 accept)
![overall](https://github.com/mooncake199809/UFVL-Net/blob/main/assets/OFVL_overall.png)
(2) "UFVL-Net: A Unified Framework for Visual Localization across Multiple Indoor Scenes".
![overall](https://github.com/mooncake199809/UFVL-Net/blob/main/assets/overall.png)

## Highlights
- Once-for-multiple-scenes
Both OFVL-MS and UFVL-Net optimize visual localization tasks of various scenes collectively using a multi-task learning manner,  which challenges the conventional wisdom that SCoRe typically trains a separate model for each scene. OFVL-MS realizes layer-wise parameters sharing, while UFVL-Net realizes channel-wise and kernel-wise sharing polices.

- Competive performance
Both OFVL-MS and UFVL-Net deliver extraordinary performances on two benchmarks and complex real scenes. We demonstrate that once the training for our methods are done, our methods can generalize to new scenes with much fewer parameters by freezing the task-shared parameters.

# OFVL-MS
The code is coming soon.

# UFVL-Net
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
- Real-world Scenes: The real-world dataset can be downloaded from [RealWorld-Scenes](https://drive.google.com/drive/folders/1rHILFijnb8wfQiT-5gWLvDJWbOqHMZwx).
## Model Zoo 
For evaluation, we provide the checkpoints of 7-Scenes dataset in [Google Drive](https://drive.google.com/drive/folders/1l4vWMz7mo49R1gMBxl932-DdavfhxiBO). 
For evaluation, we also provide the checkpoints of 12-Scenes dataset in [Google Drive](https://drive.google.com/drive/folders/1Yw-DskJD7hCPo-WIXfPvHI5mP5UgRgJ9). 
- Note: We integrate these models into a single one. You can do the evaluation following the description in *Quick Start - Test*).

## Quick Start

We provide *Test* code of ViT-MVT as follows: 

### Test
To test our trained models, you need to put the downloaded model in `./weights`.
To test a specific model in a specific scene, you need to modify the config file in ./config/7scenes/7scenes.py or ./config/12scenes/12scenes.py
The structure of the config file is described as follow:
```buildoutcfg
dataset_type: 'ufvl_net.SevenScenes' or 'ufvl_net.TWESCENES'
root: the root path of the dataset
scene: the scene name that you want to test
share_type: the type of weight sharing ("channel" or "kernel")
data: the config of the dataset
model: the config of the model
```
- If you want to test UFVL-Net-M with channel-wise sharing policy on the chess scene of 7-Scenes dataset, you need to modify the lines 8, 10, and 39 as "chess", "channel", and depth=34. Then, you could use the following command:
```buildoutcfg
python tools/test.py ./configs/7scenes/7scenes.py ./weights/34_channel_7scenes.pth --metrics accuracy
```


