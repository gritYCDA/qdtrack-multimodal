_base_ = 'qdtrack_faster-rcnn_r50_fpn_mta.py'
data_root = 'data/MTA_ext_short_coco/'
data = dict(
    val=dict(
        ann_file=data_root + 'train/mta_train.json'),
    test=dict(
        ann_file=data_root + 'test/mta_test.json')
    )
evaluation = dict(metric=['bbox', 'track'], 
				  resfile_path='work_dirs/MTA/qdtrack-frcnn_r50_train',
				  eval_data_type='train')