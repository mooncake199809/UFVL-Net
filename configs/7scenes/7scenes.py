_base_ = [
    '../_base_/default_runtime.py'
]

dataset_type = 'ufvl_net.SevenScenes'
# root='/home/dk/SCORE_Methods/EAAINet/data/'
root='/home/dk/SCORE_Methods/EAAINet/data/'
scene='chess'
all_scene = ['chess', 'fire', 'heads', 'pumpkin', 'redkitchen', 'office', 'stairs']
share_type = "channel"          # channel or kernel

test_pipeline = []
eval_pipeline = []
custom_imports = dict(imports=['ufvl_net.models'])

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        root=root,
        scene=scene,
        split='train'),
    val=dict(
        type=dataset_type,
        root=root,
        scene=scene,
        split='test'),
    test=dict(
        type=dataset_type,
        root=root,
        scene=scene,
        split='test'))

model = dict(
    type='ufvl_net.FDANET',
    backbone=dict(
        type='SEResNet',
        depth=34,
        stem_channels=16,
        expansion = 1,
        strides=(1, 1, 2, 2),
        use_maxpool=False,
        num_stages=4,
        # conv_cfg=dict(type='Conv2d_share'),
        out_indices=(3, ),
        style='pytorch',
        drop_path_rate=0.1,
        ),
    head=dict(
        type='RegHead',
        in_channel=512),
    dataset="7Scenes")
