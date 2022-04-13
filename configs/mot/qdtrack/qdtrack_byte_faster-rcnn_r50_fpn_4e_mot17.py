_base_ = 'qdtrack_faster-rcnn_r50_fpn_4e_mot17.py'

model = dict(
	tracker=dict(
		_delete_=True,
		type='BYTETracker',
	)
)