_base_ = ['qdtrack_faster-rcnn_r50_fpn.py',
		  '../../_base_/default_runtime.py']
### MODEL
model = dict(
	detector=dict(
		backbone=dict(norm_cfg=dict(requires_grad=False), style='caffe'),
		rpn_head=dict(bbox_coder=dict(clip_border=False)),
		roi_head=dict(
			bbox_head=dict(bbox_coder=dict(clip_border=False), num_classes=1)),
		init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'work_dirs/faster_rcnn_r50_caffe_fpn_person/faster_rcnn_r50_caffe_fpn_person_ap551.pth'
        )
	),
	track_head=dict(train_cfg=dict(assigner=dict(neg_iou_thr=0.5)))
)
### DATASET
dataset_type = 'MTMDCDataset'
dataset_type_MOT = 'MOTChallengeDataset'

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], 
    std=[1.0, 1.0, 1.0], 
    to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        img_scale=(1088, 1088),
        # img_scale=(1080, 1920),
        bbox_clip_border=False,
        share_params=True,
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(
        type='SeqRandomCrop',
        share_params=False,
        crop_size=(1088, 1088),
        # crop_size=(1080, 1920)
        bbox_clip_border=False
        ),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
        ]),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1088, 1088),
        # img_scale=(1080, 1920),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data_root = '/data1/kaist_mtmdc/'
data_root_mot = '/data1/MOT17/'
data = dict(
    # samples_per_gpu=4,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/kaist_mtmdc_train.json',
        img_prefix=data_root,
        key_img_sampler=dict(interval=23),
        ref_img_sampler=dict(
        	num_ref_imgs=1,
        	frame_range=20,
        	filter_key_img=True,
        	method='uniform'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type_MOT,
        # ann_file=data_root + 'annotations/kaist_mtmdc_val.json',
        ann_file=data_root_mot + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root_mot + 'train',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type_MOT,
        # ann_file=data_root + 'annotations/kaist_mtmdc_val.json',
        ann_file=data_root_mot + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root_mot + 'train',
        ref_img_sampler=None,
        pipeline=test_pipeline),
)
### RUNTIME
# optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
	policy='step', 
	warmup='linear',
	warmup_iters=500,
	warmup_ratio=1.0 / 3,
	step=[3])
# resume_from = 'work_dirs/MTA/qdtrack-frcnn_r50/latest.pth'
load_from = None
# work_dir = 'work_dirs/KAIST_MTMDC/qdtrack-frcnn_r50_kaist_mtmdc_alls'
work_dir = 'work_dirs/KAIST_MTMDC/qdtrack-frcnn_r50_kaist_mot_style_lr_half_1088_1088_all'
total_epochs = 4
evaluation = dict(metric=['bbox', 'track'], 
				  start=4
                #   ,resfile_path='work_dirs/KAIST_MTMDC/qdtrack-frcnn_r50_kaist_mtmdc_test'
                  )
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
