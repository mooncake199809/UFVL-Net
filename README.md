# Visual localization with SCoRE methods
*** This is a collection of our visual localization frameworks. 

> [**OFVL-MS**](./configs/ofvl_ms) (```@ICCV'23```): **OFVL-MS: Once for Visual Localization across Multiple Indoor Scenes**

> [**UFVL-Net**](./configs/ufvl_net) (```TIM23```): **UFVL-Net: A Unified Framework for Visual Localization across Multiple Indoor Scenesss**

## Highlights
- Once-for-multiple-scenes.
Both OFVL-MS and UFVL-Net optimize visual localization tasks of various scenes collectively using a multi-task learning manner,  which challenges the conventional wisdom that SCoRe typically trains a separate model for each scene. OFVL-MS realizes layer-wise parameters sharing, while UFVL-Net realizes channel-wise and kernel-wise sharing polices.

- Competive performance.
Both OFVL-MS and UFVL-Net deliver extraordinary performances on two benchmarks and complex real scenes. We demonstrate that once the training for our methods are done, our methods can generalize to new scenes with much fewer parameters by freezing the task-shared parameters.

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
- LIVL: The real-world  LIVL dataset can be downloaded from [RealWorld-Scenes](https://drive.google.com/drive/folders/1rHILFijnb8wfQiT-5gWLvDJWbOqHMZwx).

## LIVL Dataset
LIVL dataset collection equipment contains a mobile chassis, a RealSense D435 camera, and a VLP-16 laser radar. LIVL dataset records RGB-D images and corresponding camera poses of four different indoor environments. 
The dataset is available at [here](https://drive.google.com/drive/folders/1rHILFijnb8wfQiT-5gWLvDJWbOqHMZwx).
Specifically, we utilize the ROS system to record RGB images and aligned depth images with corresponding timestamp $T_{1}$, Furthermore, we obtain point clouds with timestamp $T_{2}$ provided by VLP-16 laser radar. Then, we generate final RGB-D images and corresponding point clouds through aligning $T_{1}$ and $T_{2}$. Ultimately, We utilize the LiDAR-based SLAM system A-LOAM to compute the ground truth pose.  
For each scene, four sequences are recorded, in which three sequences are used for training and one sequence for testing. 

![overall](https://github.com/mooncake199809/UFVL-Net/tree/main/assets/Room.png)
Scene (i): a room spanning about $12 \times 9 m^{2}$ with $3109$ images for training and $1112$ images for testing. 

![](https://github.com/mooncake199809/UFVL-Net/tree/main/assets/Hall.png) 
Scene (ii): a hall spanning about $12 \times 5 m^{2}$ with $2694$ images for training and $869$ images for testing. 

![](https://github.com/mooncake199809/UFVL-Net/tree/main/assets/Parking_lot1.png)
Scene (iii): a parking lot spanning about $8 \times 6 m^{2}$ with $2294$ images for training and $661$ images for testing. 

![](https://github.com/mooncake199809/UFVL-Net/tree/main/assets/Parking_lot2.png)
Scene (iv): a parking lot spanning about $8 \times 8 m^{2}$ with $2415$ images for training and $875$ images for testing.
