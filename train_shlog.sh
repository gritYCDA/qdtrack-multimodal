#!/usr/bin/env bash

'''
Default setting
* model-> rpn_head, roi_head 둘다 clip_border=Flase
* train_pipeline에 bbox_clip_border=False
* samples_per_gpu=2
* workers_per_gpu=2
* image_size 1088,1088

'''




'''
Baseine: qdtrack-frcnn_r50
Train: KAIST_MTMDC
qdtrack-frcnn_r50_kaist_mot_style_lr_half_1088_1088_all

'''
# coco-pretrained
# PROT=29500 ./tools/dist_train.sh configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_kaist_mtmdc.py 8
# IDF1: 0.5520, IDP: 0.8000, IDR: 0.4210, Rcll: 0.4820, Prcn: 0.9150, GT: 1017, MT: 222, PT: 297, ML: 498, FP: 7275, FN: 83790, IDs: 408, FM: 2118, MOTA: 0.4340, MOTP: 0.1850, IDt: 282, IDa: 198, IDm: 117, mAP: 0.5520

# from-scratch - resnet50:imagenet
PROT=29501 ./tools/dist_train.sh configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_kaist_mtmdc_scratch.py 8


'''
Baseine: qdtrack-frcnn_r50
Train: MOT17-half
qdtrack-frcnn_r50_fpn_4e_mot17_reproduce
'''
# coco-pretrained
PROT=29502 ./tools/dist_train.sh configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_mot17.py 8

# from-scratch - resnet50:imagenet
PROT=29503 ./tools/dist_train.sh configs/mot/qdtrack/qdtrack_faster-r   