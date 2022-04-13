import os
import os.path as osp
import tempfile

import mmcv
import motmetrics as mm
import numpy as np
from mmcv.utils import print_log
from mmdet.core import eval_map
from mmdet.datasets import DATASETS
from .coco_video_dataset import CocoVideoDataset

from mmtrack.core import results2outs
import pickle

@DATASETS.register_module()
class MTADataset(CocoVideoDataset):

	CLASSES = ('person', )

	def __init__(self,
				 *args,
				 **kwargs):
		super().__init__(*args, **kwargs)

		self.mta_feature_pkl_root = 'work_dirs/clustering/config_runs'
		
	def format_results(self, 
					   results, 
					   resfile_path=None, 
					   metrics=['track'], 
					   eval_data_type='test'):
		assert isinstance(results, dict), 'results must be a dict.'
		if resfile_path is None:
			tmp_dir = tempfile.TemporaryDirectory()
			resfile_path = tmp_dir.name
		else:
			tmp_dir = None
			if osp.exists(resfile_path):
				print_log('remove previous results.', self.logger)
				import shutil
				shutil.rmtree(resfile_path)

		resfiles = dict()
		for metric in metrics:
			resfiles[metric] = osp.join(resfile_path, metric)
			os.makedirs(resfiles[metric], exist_ok=True)

		inds = [i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0]
		num_vids = len(inds)
		assert num_vids == len(self.vid_ids)
		inds.append(len(self.data_infos))
		vid_infos = self.coco.load_vids(self.vid_ids)
		names = [_['name'] for _ in vid_infos]
		names = [name.split('/')[-1] for name in names]

		for i in range(num_vids):
			for metric in metrics:
				formatter = getattr(self, f'format_{metric}_results')
				formatter(results[f'{metric}_bboxes'][inds[i]:inds[i+1]],
						  self.data_infos[inds[i]:inds[i+1]],
						  f'{resfiles[metric]}/{names[i]}.txt')

		for i in range(num_vids):
			self.mta_format_track_results(
				cam_id=i,
				results=results['track_bboxes'][inds[i]:inds[i+1]],
				infos=self.data_infos[inds[i]:inds[i+1]],
				resfile=f'{resfiles[metric]}/track_results_{i}.txt',
				eval_data_type=eval_data_type)
		return resfiles, names, tmp_dir

	def mta_format_track_results(self, 
								 cam_id, 
								 results, 
								 infos, 
								 resfile, 
								 outdir='mta_qdtrack_base/pickled_appearance_features',
								 eval_data_type='test'):
		with open(resfile, 'wt') as f:
			f.writelines(
					"frame_no_cam,cam_id,person_id,xtl,ytl,xbr,ybr\n")
			feats = []
			for res, info in zip(results, infos):
				frame = info['frame_id']
				outs_track = results2outs(bbox_results=res)
				pid_to_feat = dict()
				for bbox, label, id, feat in zip(outs_track['bboxes'],
											     outs_track['labels'],
											     outs_track['ids'],
											     outs_track['features']):
					pid_to_feat[id] = feat
					x1, y1, x2, y2, conf = bbox
					x1 = int(x1)
					y1 = int(y1)
					x2 = int(x2)
					y2 = int(y2)
					f.writelines(
						f'{frame},{cam_id},{id},{x1:d},{y1:d},{x2:d},{y2:d}\n')

				dirname = osp.join(self.mta_feature_pkl_root, outdir, eval_data_type)
				os.makedirs(dirname, exist_ok=True)
				filename = 'frameno_{}_camid_{}.pkl'.format(frame, cam_id)
				feature_pickle_path = osp.join(dirname, filename)
				with open(feature_pickle_path, 'wb') as handle:
					pickle.dump(pid_to_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def format_track_results(self, results, infos, resfile):
		with open(resfile, 'wt') as f:
			for res, info in zip(results, infos):
				frame = info['frame_id'] + 1
				outs_track = results2outs(bbox_results=res)
				for bbox, label, id in zip(outs_track['bboxes'],
										   outs_track['labels'],
										   outs_track['ids']):
					x1, y1, x2, y2, conf = bbox
					f.writelines(
						f'{frame},{id},{x1:.3f},{y1:.3f},{(x2-x1):.3f},' +
						f'{(y2-y1):.3f},{conf:.3f},-1,-1,-1\n')

	def format_bbox_results(self, results, infos, resfile):
		with open(resfile, 'wt') as f:
			for res, info in zip(results, infos):
				frame = info['frame_id'] + 1
				outs_det = results2outs(bbox_results=res)
				for bbox, label in zip(outs_det['bboxes'], outs_det['labels']):
					x1, y1, x2, y2, conf = bbox
					f.writelines(
						f'{frame},-1,{x1:.3f},{y1:.3f},{(x2-x1):.3f},' + 
						f'{(y2-y1):.3f},{conf:.3f}\n')

	def evaluate(self,
				 results,
				 metric='track',
				 logger=None,
				 resfile_path=None,
				 bbox_iou_thr=0.5,
				 track_iou_thr=0.5,
				 eval_data_type='test'):

		eval_results = dict()
		if isinstance(metric, list):
			metrics = metric
		elif isinstance(metric, str):
			metrics = [metric]
		else:
			raise TypeError('metric must be a list or a str.')
		allowed_metrics = ['bbox', 'track']
		for metric in metrics:
			if metric not in allowed_metrics:
				raise keyError(f'metric {metric} is not supported')

		if 'track' in metrics:
			resfiles, names, tmp_dir = self.format_results(
				results, resfile_path, metrics, eval_data_type)
			print_log('Evaluate CLEAR MOT results.', logger=logger)
			distth = 1 - track_iou_thr

			accs = []
			for name in names:
				gt_file = osp.join(self.img_prefix, 'test', name, 'gt.txt')
				res_file = osp.join(resfiles['track'], f'{name}.txt')
				gt = mm.io.loadtxt(gt_file)
				res = mm.io.loadtxt(res_file)
				ini_file = osp.join(self.img_prefix, f'{name}/seqinfo.ini')
				if osp.exists(ini_file):
					acc, ana = mm.utils.CLEAR_MOT_M(
						gt, res, ini_file, distth=distth)
				else:
					acc = mm.utils.compare_to_groundtruth(
						gt, res, distth=distth)
				accs.append(acc)

			mh = mm.metrics.create()
			summary = mh.compute_many(
				accs,
				names=names,
				metrics=mm.metrics.motchallenge_metrics,
				generate_overall=True)
			str_summary = mm.io.render_summary(
				summary,
				formatters=mh.formatters,
				namemap=mm.io.motchallenge_metric_names)
			print(str_summary)

			eval_results.update({
				mm.io.motchallenge_metric_names[k]: v['OVERALL']
				for k, v in summary.to_dict().items()
			})

			if tmp_dir is not None:
				tmp_dir.cleanup()

		if 'bbox' in metrics:
			if isinstance(results, dict):
				bbox_results = results['det_bboxes']
			elif isinstance(results, list):
				bbox_results = results
			else:
				raise TypeError('results must be a dict or a list.')
			annotations = [self.get_ann_info(info) for info in self.data_infos]
			mean_ap, _ = eval_map(
				bbox_results,
				annotations,
				iou_thr=bbox_iou_thr,
				dataset=self.CLASSES,
				logger=logger)
			eval_results['mAP'] = mean_ap

		for k, v in eval_results.items():
			if isinstance(v, float):
				eval_results[k] = float(f'{(v):.3f}')

		return eval_results