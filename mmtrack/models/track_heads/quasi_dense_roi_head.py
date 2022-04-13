import torch

from abc import ABCMeta

from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models import HEADS, build_head, build_roi_extractor
from mmcv.runner import BaseModule


@HEADS.register_module()
class QuasiDenseRoIHead(BaseModule, metaclass=ABCMeta):

    def __init__(self,
                 roi_extractor=None,
                 embed_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if embed_head is not None:
        	self.init_embed_head(roi_extractor, embed_head)

        self.init_assigner_sampler()

    def init_embed_head(self, roi_extractor, embed_head):
    	self.roi_extractor = build_roi_extractor(roi_extractor)
    	self.embed_head = build_head(embed_head)

    def init_assigner_sampler(self):
    	"""Initialize assigner and sampler."""
    	self.bbox_assigner = None
    	self.bbox_sampler = None
    	if self.train_cfg:
    		self.bbox_assigner = build_assigner(self.train_cfg.assigner)
    		self.bbox_sampler = build_sampler(
    			self.train_cfg.sampler, context=self)

    @property
    def with_track(self):
        """bool: whether the RoI head contains a `embed_head`"""
        return hasattr(self, 'embed_head') and self.embed_head is not None

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_x,
                      ref_img_metas,
                      ref_proposals,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      *args,
                      **kwargs):
        losses = dict()

        if self.with_track:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            if ref_gt_bboxes_ignore is None:
                ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            key_sampling_results, ref_sampling_results = [], []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                key_sampling_results.append(sampling_result)

                ref_assign_result = self.bbox_assigner.assign(
                    ref_proposals[i], ref_gt_bboxes[i], ref_gt_bboxes_ignore[i],
                    ref_gt_labels[i])
                ref_sampling_result = self.bbox_sampler.sample(
                    ref_assign_result,
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in ref_x])
                ref_sampling_results.append(ref_sampling_result)

            key_bboxes = [res.pos_bboxes for res in key_sampling_results]
            key_feats = self._track_forward(x, key_bboxes)
            ref_bboxes = [res.bboxes for res in ref_sampling_results]
            ref_feats = self._track_forward(ref_x, ref_bboxes)

            match_feats = self.embed_head.match(key_feats, ref_feats,
                                                key_sampling_results,
                                                ref_sampling_results)
            asso_targets = self.embed_head.get_track_targets(
                gt_match_indices, key_sampling_results, ref_sampling_results)
            loss_track = self.embed_head.loss(*match_feats, *asso_targets)

            losses.update(loss_track)

        return losses

    def _track_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        track_feats = self.roi_extractor(
            x[:self.roi_extractor.num_inputs], rois)
        track_feats = self.embed_head(track_feats)
        return track_feats

    def simple_test(self, x, det_bboxes, img_metas):
        if det_bboxes.size(0) == 0:
            return None

        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]['scale_factor']).to(det_bboxes.device)
        track_feats = self._track_forward(x, [track_bboxes])

        return track_feats