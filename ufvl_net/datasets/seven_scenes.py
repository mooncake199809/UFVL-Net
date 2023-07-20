import os
import random

import cv2
import imgaug
import numpy as np
import torch
from imgaug import augmenters as iaa
from torch.utils.data import Dataset

from .builder import DATASETS


@DATASETS.register_module()
class SevenScenes(Dataset):
    # root就是--data_path
    CLASSES = [
        'chess'
    ]
    def __init__(self, root, dataset='7S', scene='heads', split='train', 
                    model='fdanet', aug='False', **kwargs):
        self.intrinsics_color = np.array([[525.0, 0.0,     320.0],
                       [0.0,     525.0, 240.0],
                       [0.0,     0.0,  1.0]])

        self.intrinsics_depth = np.array([[585.0, 0.0,     320.0],
                       [0.0,     585.0, 240.0],
                       [0.0,     0.0,  1.0]])

        self.intrinsics_depth_inv = np.linalg.inv(self.intrinsics_depth)            
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)            
        self.model = model
        self.dataset = dataset
        self.aug = aug                                                              
        self.root = os.path.join(root, '7Scenes')                                   
        self.calibration_extrinsics = np.loadtxt(os.path.join(self.root, 
                        'sensorTrans.txt'))                                      
        self.scene = scene
        self.split = split                          # 模式选择
        self.obj_suffixes = ['.color.png','.pose.txt', '.depth.png',
                '.label.png']                       # 后缀
        self.obj_keys = ['color','pose', 'depth','label']
        # 这里设定了训练/测试的图片
        with open(os.path.join(self.root, '{}{}'.format(self.split,         # ./data/7Scenes/tarin或test.txt
                '.txt')), 'r') as f:
            self.frames = f.readlines()
            if self.dataset == '7S' or self.split == 'test':
                # 列表['chess seq-03 frame-000000\n', 'chess seq-03 frame-000001\n', .............]
                self.frames = [frame for frame in self.frames \
                if self.scene in frame]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index].rstrip('\n')         
        scene, seq_id, frame_id = frame.split(' ')      # chess seq-03 frame-000000
        objs = {}
        objs['color'] = '/mnt/share/sda-2T/xietao/'+ self.scene + '/' + seq_id + '/' + frame_id + '.color.png'
        objs['pose'] = '/mnt/share/sda-2T/xietao/' + self.scene + '/' + seq_id + '/' + frame_id + '.pose.txt'        # Twc
        objs['depth'] = '/mnt/share/sda-2T/xietao/' + self.scene + '/' + seq_id + '/' + frame_id + '.depth.png'
        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose = np.loadtxt(objs['pose'])                 # 位姿文件(np)
        if self.split == 'test':
            img, pose = to_tensor_query(img, pose)
            ret = dict(img=img, gt_lables=pose)
            return ret

        depth = cv2.imread(objs['depth'],-1)
        
        pose[0:3,3] = pose[0:3,3] * 1000
     
        depth[depth==65535] = 0
        depth = depth * 1.0
        depth = get_depth(depth, self.calibration_extrinsics, self.intrinsics_color, self.intrinsics_depth_inv)
       
        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv, self.dataset)
        img, coord, mask = data_aug(img, coord, mask, self.aug)   # 进行数据增强

        coord = coord[4::8,4::8,:]  # [60, 80, 3]
        mask = mask[4::8,4::8].astype(np.float16)   #[60 80]
        img, coord, mask = to_tensor(img, coord, mask)     
        
        ret = dict(img=img, gt_lables=coord, mask=mask)

        return ret

    def evaluate(self, results, *args, **kwargs):
        transl_err_list = list()
        rot_err_list = list()
        for i in range(len(results)):
            transl_err_list.append(results[i]['trans_error_med'])
            rot_err_list.append(results[i]['rot_err_med'])
        res_ = np.array([transl_err_list, rot_err_list]).T

        return dict(median_trans_error=np.median(res_[:, 0]), 
                    median_rot_error=np.median(res_[:, 1]), 
                    accuracy=np.sum((res_[:, 0] <= 0.050) * (res_[:, 1] <= 5)) * 1. / len(res_)
                    )

# depth：深度图[480 640]
# 返回深度图对齐到RGB图后，RGB相应的深度信息
def get_depth(depth, calibration_extrinsics, intrinsics_color,
              intrinsics_depth_inv):
    """Return the calibrated depth image (7-Scenes).
    Calibration parameters from DSAC (https://github.com/cvlab-dresden/DSAC)
    are used.
    """
    '''
    利用深度摄像头内参矩阵把深度平面坐标（深度图坐标）转换到深度摄像头空间坐标，
    再利用外参计算旋转矩阵和平移矩阵，把深度摄像头空间坐标转换到RGB摄像头空间坐标，
    最后利用RGB摄像头内参矩阵把RGB摄像头空间坐标转换到RGB平面坐标（RGB图坐标）。
    这里只记录一下最终测试程序的思路：
    '''
    img_height, img_width = depth.shape[0], depth.shape[1]
    depth_ = np.zeros_like(depth)  # [480 640]
    x = np.linspace(0, img_width - 1, img_width)  # 640
    y = np.linspace(0, img_height - 1, img_height)  # 480

    xx, yy = np.meshgrid(x, y)  # 坐标网格化[img_width img_height]
    xx = np.reshape(xx, (1, -1))  # [1, img_width*img_height]
    yy = np.reshape(yy, (1, -1))  # [1, img_width*img_height]
    ones = np.ones_like(xx)  # [1, img_width*img_height]

    pcoord_depth = np.concatenate((xx, yy, ones), axis=0)  # [3, img_width*img_height], 像素坐标
    depth = np.reshape(depth, (1, img_height * img_width))  # [1, img_width*img_height]

    ccoord_depth = np.dot(intrinsics_depth_inv, pcoord_depth) * depth  # 像素坐标-->归一化坐标-->相机坐标[3, img_width*img_height]

    ccoord_depth[1, :] = - ccoord_depth[1, :]
    ccoord_depth[2, :] = - ccoord_depth[2, :]

    ccoord_depth = np.concatenate((ccoord_depth, ones), axis=0)  # [4, img_width*img_height]
    ccoord_color = np.dot(calibration_extrinsics, ccoord_depth)  # [3, img_width*img_height],RGB相机坐标

    ccoord_color = ccoord_color[0:3, :]
    ccoord_color[1, :] = - ccoord_color[1, :]
    ccoord_color[2, :] = depth

    pcoord_color = np.dot(intrinsics_color, ccoord_color)  # RGB像素坐标*Z
    pcoord_color = pcoord_color[:, pcoord_color[2, :] != 0]

    pcoord_color[0, :] = pcoord_color[0, :] / pcoord_color[2, :] + 0.5  # RGB像素坐标
    pcoord_color[0, :] = pcoord_color[0, :].astype(int)
    pcoord_color[1, :] = pcoord_color[1, :] / pcoord_color[2, :] + 0.5
    pcoord_color[1, :] = pcoord_color[1, :].astype(int)
    pcoord_color = pcoord_color[:, pcoord_color[0, :] >= 0]
    pcoord_color = pcoord_color[:, pcoord_color[1, :] >= 0]

    pcoord_color = pcoord_color[:, pcoord_color[0, :] < img_width]
    pcoord_color = pcoord_color[:, pcoord_color[1, :] < img_height]

    depth_[pcoord_color[1, :].astype(int),
           pcoord_color[0, :].astype(int)] = pcoord_color[2, :]
    return depth_


def get_coord(depth, pose, intrinsics_color_inv, dataset):
    """Generate the ground truth scene coordinates from depth and pose.
    """
    img_height, img_width = depth.shape[0], depth.shape[1]  # 480 640
    mask = np.ones_like(depth)
    mask[depth == 0] = 0  # 深度为0处，数值为0，否则为1
    mask = np.reshape(mask, (img_height, img_width, 1))  # [480 640 1]
    x = np.linspace(0, img_width - 1, img_width)
    y = np.linspace(0, img_height - 1, img_height)

    xx, yy = np.meshgrid(x, y)

    xx = np.reshape(xx, (1, -1))  # [1, 640*480]
    yy = np.reshape(yy, (1, -1))  # [1, 640*480]
    ones = np.ones_like(xx)  # [1, 640*480]
    pcoord = np.concatenate((xx, yy, ones), axis=0)  # [3, 640*480],像素坐标

    depth = np.reshape(depth, (1, img_height * img_width))  # [1, 640*480]
    ccoord = np.dot(intrinsics_color_inv, pcoord) * depth  # 相机坐标 [3 640*480]
    ccoord = np.concatenate((ccoord, ones), axis=0)  # 相机坐标 [4 640*480]

    # if dataset == 'my':
    #     scoord  = np.dot(np.swapaxes(ccoord,0,1), pose)
    # else:
    scoord = np.dot(pose, ccoord)  # 世界坐标 [3 640*480]
    scoord = np.swapaxes(scoord, 0, 1)  # 世界坐标 [640*480 3]

    scoord = scoord[:, 0:3]
    scoord = np.reshape(scoord, (img_height, img_width, 3))  # 世界坐标 [480 640 3]
    scoord = scoord * mask
    mask = np.reshape(mask, (img_height, img_width))  # [480 640]

    return scoord, mask


def data_aug(img, coord, mask, aug=True, sp_coords=None):
    img_h, img_w = img.shape[0:2]
    if aug:
        trans_x = random.uniform(-0.2, 0.2)  # 平移
        trans_y = random.uniform(-0.2, 0.2)

        aug_add = iaa.Add(random.randint(-20, 20))

        scale = random.uniform(0.7, 1.5)  # 缩放
        rotate = random.uniform(-30, 30)  # 旋转
        shear = random.uniform(-10, 10)  # 裁剪

        aug_affine = iaa.Affine(scale=scale, rotate=rotate,
                                shear=shear, translate_percent={"x": trans_x, "y": trans_y})
        aug_affine_lbl = iaa.Affine(scale=scale,rotate=rotate,
                    shear=shear,translate_percent={"x": trans_x, "y": trans_y},
                    order=0,cval=1)
        img = aug_add.augment_image(img)
    else:
        trans_x = random.randint(-3, 4)
        trans_y = random.randint(-3, 4)

        aug_affine = iaa.Affine(translate_px={"x": trans_x, "y": trans_y})

    padding = torch.randint(0, 255, size=(img_h,img_w, 3)).data.numpy().astype(np.uint8)
    padding_mask = np.ones((img_h, img_w)).astype(np.uint8)

    img = aug_affine.augment_image(img)
    coord = aug_affine.augment_image(coord)
    mask = aug_affine.augment_image(mask)
    mask = np.round(mask)
    padding_mask = aug_affine.augment_image(padding_mask)
    img = img + (1 - np.expand_dims(padding_mask, axis=2)) * padding

    if isinstance(sp_coords, np.ndarray):
        ia_kpts = []
        out_kpts = []
        for i in range(sp_coords.shape[0]):
            ia_kpts.append(imgaug.Keypoint(x=sp_coords[i][0], y=sp_coords[i][1]))
        ia_kpts = imgaug.KeypointsOnImage(ia_kpts, shape=img.shape)
        ia_kpts = aug_affine_lbl.augment_keypoints(ia_kpts)
        for i in range(len(ia_kpts)):
            out_kpts.append(np.array((ia_kpts[i].x, ia_kpts[i].y)))
        out_kpts = np.stack(out_kpts, axis=0)
        return img, coord, mask, out_kpts
    else:
        return img, coord, mask


def to_tensor(img, coord_img, mask):
    img = img.transpose(2, 0, 1)
    coord_img = coord_img.transpose(2, 0, 1)
    img = img / 255.
    img = img * 2. - 1.
    coord_img = coord_img / 1000.
    img = torch.from_numpy(img).float()
    coord_img = torch.from_numpy(coord_img).float()
    mask = torch.from_numpy(mask).float()
    return img, coord_img, mask


def to_tensor_query(img, pose):
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = img * 2. - 1.
    img = torch.from_numpy(img).float()
    pose = torch.from_numpy(pose).float()
    return img, pose


def data_aug_label(img, coord, mask, lbl, aug=True):
    img_h, img_w = img.shape[0:2]
    if aug:
        trans_x = random.uniform(-0.2, 0.2)
        trans_y = random.uniform(-0.2, 0.2)

        aug_add = iaa.Add(random.randint(-20, 20))

        scale = random.uniform(0.7, 1.5)
        rotate = random.uniform(-30, 30)
        shear = random.uniform(-10, 10)

        aug_affine = iaa.Affine(scale=scale, rotate=rotate,
                                shear=shear, translate_percent={"x": trans_x, "y": trans_y})
        aug_affine_lbl = iaa.Affine(scale=scale, rotate=rotate,
                                    shear=shear, translate_percent={"x": trans_x, "y": trans_y},
                                    order=0, cval=1)
        img = aug_add.augment_image(img)
    else:
        trans_x = random.randint(-3, 4)
        trans_y = random.randint(-3, 4)

        aug_affine = iaa.Affine(translate_px={"x": trans_x, "y": trans_y})
        aug_affine_lbl = iaa.Affine(translate_px={"x": trans_x, "y": trans_y},
                                    order=0, cval=1)

    padding = torch.randint(0, 255, size=(img_h,
                                          img_w, 3)).data.numpy().astype(np.uint8)
    padding_mask = np.ones((img_h, img_w)).astype(np.uint8)

    img = aug_affine.augment_image(img)
    coord = aug_affine.augment_image(coord)
    mask = aug_affine.augment_image(mask)
    mask = np.round(mask)
    lbl = aug_affine_lbl.augment_image(lbl)
    padding_mask = aug_affine.augment_image(padding_mask)
    img = img + (1 - np.expand_dims(padding_mask, axis=2)) * padding

    return img, coord, mask, lbl


def one_hot(x, N=25):
    one_hot = torch.FloatTensor(N, x.size(0), x.size(1)).zero_()
    one_hot = one_hot.scatter_(0, x.unsqueeze(0), 1)
    return one_hot


def to_tensor_label(img, coord_img, mask, lbl, N1=25, N2=25):
    img = img.transpose(2, 0, 1)
    coord_img = coord_img.transpose(2, 0, 1)
    img = img / 255.
    img = img * 2. - 1.
    coord_img = coord_img / 1000.
    img = torch.from_numpy(img).float()
    coord_img = torch.from_numpy(coord_img).float()
    mask = torch.from_numpy(mask).float()
    lbl = torch.from_numpy(lbl/1.0).long()
    lbl_oh = one_hot(lbl, N=N1)
    return img, coord_img, mask, lbl, lbl_oh


def to_tensor_query_label(img, pose):
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = img * 2. - 1.
    img = torch.from_numpy(img).float()
    pose = torch.from_numpy(pose).float()

    return img, pose
