from mmdet.models import build_detector, build_head

from mmtrack.core import outs2results
from .base import BaseMultiObjectTracker
from ..builder import MODELS, build_tracker

from mmtrack.utils import plot_img, plot_det_bboxes

@MODELS.register_module()
class QDTrack(BaseMultiObjectTracker):

	def __init__(self,
				 detector=None,
				 track_head=None,
				 tracker=None,
				 init_cfg=None):
		super().__init__(init_cfg)
		self.tracker_cfg = tracker
		if detector is not None:
			self.detector = build_detector(detector)
		if track_head is not None:
			self.track_head = build_head(track_head)
		if tracker is not None:
			self.tracker = build_tracker(tracker)

	def init_tracker(self):
		self.tracker = build_tracker(self.tracker_cfg)

	def forward_train(self,
					  img,
					  img_metas,
					  gt_bboxes,
					  gt_labels,
					  gt_match_indices,
					  ref_img,
					  ref_img_metas,
					  ref_gt_bboxes,
					  ref_gt_labels,
					  ref_gt_match_indices,
					  gt_bboxes_ignore=None,
					  gt_masks=None,
					  ref_gt_bboxes_ignore=None,
					  ref_gt_masks=None,
					  **kwargs):

		x = self.detector.extract_feat(img)
		ref_x = self.detector.extract_feat(ref_img)

		losses = dict()

		# RPN forward and loss
		if self.detector.with_rpn:
			proposal_cfg = self.detector.train_cfg.get(
				'rpn_proposal', self.detector.test_cfg.rpn)
			losses_rpn, proposal_list = self.detector.rpn_head.forward_train(
				x,
				img_metas,
				gt_bboxes,
				gt_labels=None,
				gt_bboxes_ignore=gt_bboxes_ignore,
				proposal_cfg=proposal_cfg)
			losses.update(losses_rpn)
		else:
			proposal_list = proposals

		losses_detect = self.detector.roi_head.forward_train(
			x, img_metas, proposal_list, gt_bboxes, gt_labels,
			gt_bboxes_ignore, gt_masks, **kwargs)
		losses.update(losses_detect)

		ref_proposals = self.detector.rpn_head.simple_test_rpn(ref_x, ref_img_metas)

		losses_track = self.track_head.forward_train(
			x, img_metas, proposal_list, gt_bboxes, gt_labels,
			gt_match_indices, ref_x, ref_img_metas, ref_proposals,
			ref_gt_bboxes, ref_gt_labels, gt_bboxes_ignore, gt_masks,
			ref_gt_bboxes_ignore, **kwargs)
		losses.update(losses_track)

		return losses

	def simple_test(self, img, img_metas, rescale=False):
		frame_id = img_metas[0].get('frame_id', -1)
		if frame_id == 0:
			self.init_tracker()

		x = self.detector.extract_feat(img)
		proposal_list = self.detector.rpn_head.simple_test_rpn(x, img_metas)
		det_bboxes, det_labels = self.detector.roi_head.simple_test_bboxes(
			x, img_metas, proposal_list, self.detector.roi_head.test_cfg, rescale)
		
		det_bboxes = det_bboxes[0]
		det_labels = det_labels[0]
		num_classes = self.detector.roi_head.bbox_head.num_classes

		det_results = outs2results(
			bboxes=det_bboxes, 
			labels=det_labels, 
			num_classes=num_classes)

		track_feats = self.track_head.simple_test(x, det_bboxes, img_metas)
		if track_feats is not None:
			track_bboxes, track_labels, track_ids, track_embeds = self.tracker.match(
				det_bboxes,
				det_labels,
				track_feats,
				frame_id,
				return_embed=True)

			track_results = outs2results(
				bboxes=track_bboxes,
				labels=track_labels,
				ids=track_ids,
				features=track_embeds,
				num_classes=num_classes)
			return dict(
				det_bboxes=det_results['bbox_results'],
				track_bboxes=track_results['bbox_results'])
		else:
			import numpy as np
			dummy_results = [
				np.zeros((0, 6+256), dtype=np.float32)
				for i in range(self.detector.roi_head.bbox_head.num_classes)
			]
			return dict(
				det_bboxes=det_results['bbox_results'],
				track_bboxes=dummy_results)

		


    	



