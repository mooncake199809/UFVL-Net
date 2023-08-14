# UFVL-Net: A Unified Framework for Visual Localization across Multiple Indoor Scenes

![overall](https://github.com/mooncake199809/UFVL-Net/blob/main/assets/overall.png)

## Model Zoo 
For evaluation, we provide the checkpoints of 7-Scenes dataset in [Google Drive](https://drive.google.com/drive/folders/1l4vWMz7mo49R1gMBxl932-DdavfhxiBO). 
For evaluation, we also provide the checkpoints of 12-Scenes dataset in [Google Drive](https://drive.google.com/drive/folders/1Yw-DskJD7hCPo-WIXfPvHI5mP5UgRgJ9). 
- Note: We integrate these models into a single one. You can do the evaluation following the description in *Quick Start - Test*).

## Quick Start

We provide *Test* code of UFVL-Net as follows: 

### Test
To test our trained models, you need to put the downloaded model in `./weights`.
To test a specific model in a specific scene, you need to modify the config file in ./configs/ufvl_net/7scenes.py or ./configs/ufvl_net/12scenes.py
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
python tools/test.py ./configs/ufvl_net/7scenes.py ./weights/34_channel_7scenes.pth --metrics accuracy
```