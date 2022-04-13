_base_ = '../../_base_/models/faster_rcnn_r50_fpn.py'

model=dict(
	type='QDTrack',
	track_head=dict(
		type='QuasiDenseRoIHead',
		roi_extractor=dict(
			type='SingleRoIExtractor',
			roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
			out_channels=256,
			featmap_strides=[4, 8, 16, 32]),
		embed_head=dict(
			type='QuasiDenseEmbedHead',
			num_convs=4,
			num_fcs=1,
			embed_channels=256,
			norm_cfg=dict(type='GN', num_groups=32),
			loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
			loss_track_aux=dict(
				type='L2Loss',
				neg_pos_ub=3,
				pos_margin=0,
				neg_margin=0.1,
				hard_mining=True,
				loss_weight=1.0),
			),
		train_cfg=dict(
			assigner=dict(
				type='MaxIoUAssigner',
				pos_iou_thr=0.7,
				neg_iou_thr=0.3,
				min_pos_iou=0.5,
				match_low_quality=False,
				ignore_iof_thr=-1),
			sampler=dict(
				type='CombinedSampler',
				num=256,
				pos_fraction=0.5,
				neg_pos_ub=3,
				add_gt_as_proposals=True,
				pos_sampler=dict(type='InstanceBalancedPosSampler'),
				neg_sampler=dict(type='RandomSampler')))
		),
	tracker=dict(
		type='QuasiDenseEmbedTracker',
		init_score_thr=0.9,
		obj_score_thr=0.5,
		match_score_thr=0.5,
		memo_tracklet_frames=30,
		memo_backdrop_frames=1,
		memo_momentum=0.8,
		nms_conf_thr=0.5,
		nms_backdrop_iou_thr=0.3,
		with_cats=True,
		match_metric='bisoftmax'),
)
