import os.path as osp
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import json
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--type', type=str, default='train')
	parser.add_argument('--mta_dataset_folder', 
						type=str, default='data/MTA/')
	parser.add_argument('--coco_mta_output_folder', 
						type=str, default='data/MTA_coco/')
	parser.add_argument('--sampling_rate', type=int, default=41)
	parser.add_argument('--camera_ids', type=str, default="0,1,2,3,4,5")

	args = parser.parse_args()
	
	args.mta_dataset_folder = osp.join(args.mta_dataset_folder, args.type)
	args.coco_mta_output_folder = osp.join(args.coco_mta_output_folder, args.type)
	args.camera_ids = list(map(int, args.camera_ids.split(",")))
	return args

def xyxy2xywh(bbox):
	_bbox = bbox.tolist()
	return [
		_bbox[0],
		_bbox[1],
		_bbox[2] - _bbox[0] + 1,
		_bbox[3] - _bbox[1] + 1
	]

def constrain_bbox_to_img_dims(xyxy_bbox,
							   img_dims=(1920, 1080)):
	img_width, img_height = img_dims
	xtl, ytl, xbr, ybr = xyxy_bbox

	if xtl < 0: xtl = 0
	if xtl >= img_width: xtl = img_width

	if ytl < 0: ytl = 0
	if ytl >= img_height: ytl = img_height

	if xbr < 0: xbr = 0
	if xbr >= img_width: xbr = img_width

	if ybr < 0: ybr = 0
	if ybr >= img_height: ybr = img_height

	return [xtl, ytl, xbr, ybr]

def get_frame_annotation(cam_coords_frame: pd.DataFrame,
						 video_id: int,
						 image_id: int,
						 image_size: tuple,
						 annotation_id: int,
						 ins_maps: dict,
						 ins_id: int):
	annotations = []

	for idx, ped_row in cam_coords_frame.iterrows():

		pid = ped_row['person_id'] 
		if pid in ins_maps:
			instance_id = ins_maps[pid]
		else:
			instance_id = ins_id
			ins_maps[pid] = ins_id
			ins_id += 1

		bbox = [
			int(ped_row["x_top_left_BB"]),
			int(ped_row["y_top_left_BB"]),
			int(ped_row["x_bottom_right_BB"]),
			int(ped_row["y_bottom_right_BB"])
		]
		bbox = constrain_bbox_to_img_dims(bbox)
		bbox = xyxy2xywh(np.asarray(bbox))

		width, height = bbox[2], bbox[3]
		area = float(width * height)

		annotation = {
			"bbox": bbox,
			"area": area,
			"iscrowd": 0,
			"image_id": image_id,
			"category_id": 1,
			"video_id": video_id,
			"instance_id" : instance_id,
			"id": annotation_id,
		}
		annotations.append(annotation)
		annotation_id += 1
	return annotation_id, annotations, ins_maps, ins_id

def convert_annotations(mta_dataset_path,
						coco_mta_dataset_path,
						sampling_rate,
						camera_ids,
						img_dims=(1920, 1080),
						person_ids_name="person_id",
						type='train'):
	
	coco_dict = {
		'info' : {
			'description' : 'MTA',
			'url' : 'mta.',
			'version' : '1.0',
			'year' : '2021',
			'contributor': 'Sanghyun Woo',
			'data_created' : '2021/12/19',
		},
		'licences' : [{
			'url': 'http://creativecommons.org/licenses/by-nc/2.0',
            'id': 2,
            'name': 'Attribution-NonCommercial License'
		}],
		'videos' : [],
		'images' : [],
		'annotations' : [],
		'categories' : [
			{
				'supercategory':'person',
				'id': 1,
				'name':'person'
			},
			{
				'supercategory': 'background',
				'id': 2,
				'name': 'background'
			}
			]
	}

	ins_maps = dict() # mtmc-specific
	ins_id = 0
	current_annotation_id = 0
	current_video_id = 0
	current_image_id = 0
	for cam_id in camera_ids:
		print('processing cam_{}'.format(cam_id))

		coco_gta_dataset_video_path = osp.join(coco_mta_dataset_path, 
											   'cam_{}'.format(cam_id))
		os.makedirs(coco_gta_dataset_video_path, exist_ok=True)

		cam_path = osp.join(mta_dataset_path, 'cam_{}'.format(cam_id))
		csv_path = osp.join(cam_path, 'coords_fib_cam_{}.csv'.format(cam_id))
		video_path = osp.join(cam_path, 'cam_{}.mp4'.format(cam_id))

		cam_coords = pd.read_csv(csv_path)

		sourcefile = open(osp.join(coco_gta_dataset_video_path, 'gt.txt'), 'w')
		for idx, ped_row in cam_coords.iterrows():
			bbox = [
				ped_row['x_top_left_BB'],
				ped_row['y_top_left_BB'],
				ped_row['x_bottom_right_BB'],
				ped_row['y_bottom_right_BB']
			]
			bbox = constrain_bbox_to_img_dims(bbox)
			bbox = xyxy2xywh(np.asarray(bbox))
			gt = [
				ped_row['frame_no_cam']+1,
				ped_row['person_id'],
				bbox[0],
				bbox[1],
				bbox[2],
				bbox[3],
				1, # conf
				1, # class id
				1. # visibility
			]
			print(*gt, sep=",", file=sourcefile)
		sourcefile.close()
	
		video_capture = cv2.VideoCapture(video_path)

		frame_nos_cam = list(set(cam_coords["frame_no_cam"]))
		frame_nos_cam = frame_nos_cam[0::sampling_rate]

		pbar = tqdm(total=len(frame_nos_cam))

		def updateTqdm(*a):
			pbar.update()

		current_frame_id = 0
		for frame_no_cam in frame_nos_cam:
			updateTqdm()

			cam_coords_frame = cam_coords[cam_coords['frame_no_cam'] == frame_no_cam]

			current_annotation_id, frame_annotations, ins_maps, ins_id = \
				get_frame_annotation(
					cam_coords_frame=cam_coords_frame,
					video_id=current_video_id,
					image_id=current_image_id,
					image_size=img_dims,
					annotation_id=current_annotation_id,
					ins_maps=ins_maps,
					ins_id=ins_id)
			image_name = 'image_{}.jpg'.format(current_frame_id)
			image_path_gta_coco = osp.join(coco_gta_dataset_video_path, image_name)

			video_name = osp.join(type, 'cam_{}'.format(cam_id))
			image_name = osp.join(video_name, image_name)

			coco_dict['videos'].append({
				'id' : current_video_id,
				'width' : img_dims[0],
				'height' : img_dims[1],
				'name' : video_name, 
				'frame_range' : len(frame_nos_cam)
				})
			coco_dict['images'].append({
				'id': current_image_id,
				'video' : video_name, 
				'width': img_dims[0],
				'height': img_dims[1],
				'file_name' : image_name,
				'video_id' : current_video_id,
				'frame_id' : current_frame_id,
				})
			coco_dict['annotations'].extend(frame_annotations)

			video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no_cam)

			ret, frame =  video_capture.read()
			if ret:
				cv2.imwrite(filename=image_path_gta_coco, img=frame)

			current_frame_id += 1
			current_image_id += 1
		current_video_id += 1

	with open(osp.join(coco_mta_dataset_path, 'mta_{}.json'.format(type)), 'w') as fp:
		json.dump(coco_dict, fp=fp, sort_keys=True, indent=3)

	return coco_dict

def main():
	args = parse_args()
	convert_annotations(mta_dataset_path=args.mta_dataset_folder,
						coco_mta_dataset_path=args.coco_mta_output_folder,
						sampling_rate=args.sampling_rate,
						camera_ids=args.camera_ids,
						type=args.type)


if __name__ == '__main__':
	main()