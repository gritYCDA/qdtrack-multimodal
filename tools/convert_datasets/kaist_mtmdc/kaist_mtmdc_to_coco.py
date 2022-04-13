import os.path as osp
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import json
import argparse

from typing import Dict, List, Tuple

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root',
                        type=str,
                        default='data/kaist_mtmdc/')
    parser.add_argument('--sampling_rate',
                        type=int,
                        default=1)
    parser.add_argument('--n_camera', 
                        type=int, 
                        default=16)
    args = parser.parse_args()
    
    args.ann_root = osp.join(args.dataset_root, 'annotations')
    args.train_data_root = osp.join(args.dataset_root, 'train')
    args.test_data_root = osp.join(args.dataset_root, 'test')
    args.n_camera = args.n_camera
    return args

def constrain_bbox_to_img_dims(
    xywh_bbox : List[int],
    img_dims : Tuple[float, float] = (1920, 1080),
) -> List[int]:
    
    img_width, img_height = img_dims
    xtl, ytl, w, h = xywh_bbox

    assert w > 0 and h > 0

    if xtl < 0: xtl = 0
    if xtl >= img_width: xtl = img_width

    if ytl < 0: ytl = 0
    if ytl >= img_height: ytl = img_height

    if xtl + w >= img_width: w = img_width - xtl
    if ytl + h >= img_height : h = img_height - ytl
    
    return [xtl, ytl, w, h]

def get_inst_annotations(
    ann : pd.DataFrame,
    video_id : int,
    frame_id : int,
    image_id: int,
    image_size: Tuple[int, int],
    annotation_id: int,
    person_id_map: dict,
    new_person_id: int
) -> Tuple[List, int, Dict, int]:
    annotations = []
    for idx, _ann in ann.iterrows():
        person_id = _ann["Person#"]
        if person_id == 0:
            continue
        if person_id in person_id_map:
            person_id = person_id_map[person_id]
        else:
            person_id = new_person_id
            person_id_map[person_id] = new_person_id
            new_person_id += 1

        bbox = [_ann["x"], _ann["y"], _ann["width"], _ann["height"]]
        area = float(_ann["width"] * _ann["height"])
        annotation = {
            "bbox" : bbox,
            "area" : area,
            "iscrowd" : 0,
            "image_id" : image_id,
            "category_id" : 1,
            "video_id" : video_id,
            "frame_id" : frame_id,
            "instance_id" : person_id,
            "id" : annotation_id
        }
        annotation_id += 1
        annotations.append(annotation)
    return annotations, annotation_id, person_id_map, new_person_id

def save_mot_style_ann(
    annotation_root : str,
    scenario_to_ann_dir : Dict,
    scenarios: List,
    type: str = 'train',
    sampling_rate : int = 1,
    img_dims : Tuple[int, int] = (1920, 1090),
):

    for scenario in scenarios:
        print(f'save mot style ann {type}_{scenario}')
        root = osp.join(type, scenario)
        ann_path = osp.join(annotation_root, scenario_to_ann_dir[scenario])
        csvs = os.listdir(ann_path); csvs.sort()
        for cam_id, csv in enumerate(csvs):
            video_name = osp.join(root, f'c{cam_id+1:02}')
            ann = pd.read_csv(osp.join(ann_path, csv))
            output_root = '/'.join(annotation_root.split('/')[:-1])
            output_path = osp.join(output_root, type, scenario, f'c{cam_id+1:02}')
            os.makedirs(output_path, exist_ok=True)
            if sampling_rate > 1:
                output_file = open(osp.join(output_path, 'gt_short.txt'), 'w')
            else:                
                output_file = open(osp.join(output_path, 'gt.txt'), 'w')

            for idx, _ann in ann.iterrows():
                if _ann['Person#'] == 0:
                    continue
                if _ann['Frame#'] % sampling_rate != 0:
                    continue
                bbox = [
                    _ann['x'],
                    _ann['y'],
                    _ann['width'],
                    _ann['height']
                ]
                # bbox = constrain_bbox_to_img_dims(bbox)
                gt = [
                    int(_ann['Frame#'] // sampling_rate) + 1,
                    _ann['Person#'],
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    1,
                    1,
                    1.
                ]
                print(*gt, sep=",", file=output_file)
            output_file.close()


def get_coco_style_ann(
    data_root: str,
    annotation_root : str,
    scenario_to_ann_dir : Dict,
    scenarios : List, 
    type : str = 'train',
    sampling_rate : int = 1,
    img_dims : Tuple[int, int] = (1920, 1080),
) -> Dict:
    assert type in ['train', 'test']
    video_id = 0
    image_id = 0
    annotation_id = 0
    person_id_map = dict()
    new_person_id = 0

    coco_dict = {
        'info' : {
            'description' : 'kaist_mtmdc',
            'version' : '1.0',
            'year': '2022',
            'contributor': 'kaist_rcv',
        },
        'videos' : [],
        'images' : [],
        'annotations' : [],
        'categories' : [
            {
                'supercategory': 'person',
                'id' : 1,
                'name' : 'person'
            },
            {
                'supercategory': 'background',
                'id' : 2,
                'name' : 'background'
            }
        ]

    }
    for scenario_id, scenario in enumerate(scenarios): # per scenario (s01, s02, ...)
        
        ann_path = osp.join(annotation_root, scenario_to_ann_dir[scenario])
        csvs = os.listdir(ann_path); csvs.sort()
        if type == 'train':
            data_dir = osp.join(type, scenario)    
            for cam_id, csv in enumerate(csvs): # per video (cam 01, 02, 03, ... 16)
            
                video_name = osp.join(data_dir, f'c{cam_id+1:02}')
                ann = pd.read_csv(osp.join(ann_path, csv))
                frames = list(set(ann["Frame#"]))
                frames = frames[0::sampling_rate]

                coco_dict['videos'].append({
                    'id' : video_id,
                    'width' : img_dims[0],
                    'height' : img_dims[1],
                    'name' : video_name,
                    'camera_id' : cam_id,
                    'scenario_id' : scenario_id,
                    'frame_range' : len(frames),
                })
                frame_id = 0
                
                pbar = tqdm(total=len(frames))
                def update_tqdm(*a):
                    pbar.update()
                for frame in frames: # per frame (000001.jpg, 000002.jpg, ..., 007360.jpg)
                    update_tqdm()
                    
                    image_name = osp.join(video_name, 'rgb', f'{frame:06}.jpg')
                    ann_frame = ann[ann["Frame#"]==frame]
                    ann_ins, annotation_id, person_id_map, new_person_id =\
                        get_inst_annotations(
                            ann=ann_frame,
                            video_id=video_id,
                            frame_id=frame_id,
                            image_id=image_id,
                            image_size=img_dims,
                            annotation_id=annotation_id,
                            person_id_map=person_id_map,
                            new_person_id=new_person_id
                        )
                    coco_dict['images'].append({
                        'id': image_id,
                        'video_id' : video_id,
                        'frame_id' : frame_id,
                        'video' : video_name,
                        'width' : img_dims[0],
                        'height' : img_dims[1],
                        'file_name' : image_name,
                        'camera_id' : cam_id,
                        'scenario_id' : scenario_id,
                    })
                    coco_dict['annotations'].extend(ann_ins)
                    frame_id += 1
                    image_id += 1

                video_id += 1

        else:

            cams = os.listdir(osp.join(data_root, type, scenario)); cams.sort()
            for cam_id, cam in enumerate(cams):

                video_name = osp.join(type, scenario, cam)
                ann = pd.read_csv(osp.join(ann_path, csvs[cam_id]))
                ann_frames = list(set(ann["Frame#"]))
                
                frames = os.listdir(osp.join(data_root, video_name, 'rgb')); frames.sort()
                frames = frames[0::sampling_rate]
                coco_dict['videos'].append({
                    'id' : video_id,
                    'width' : img_dims[0],
                    'height' : img_dims[1],
                    'name' : video_name,
                    'camera_id' : cam_id,
                    'scenario_id' : scenario_id,
                    'frame_range' : len(frames)
                    })
                frame_id = 0
                pbar = tqdm(total=len(frames))
                def update_tqdm(*a):
                    pbar.update()
                # frame : "000000.jpg, int(frame[:-4]) : 0"
                for frame in frames:
                    update_tqdm()
                    if int(frame[:-4]) in ann_frames:
                        ann_frame = ann[ann["Frame#"]==int(frame[:-4])]
                        ann_ins, annotation_id, person_id_map, new_person_id =\
                        get_inst_annotations(
                            ann=ann_frame,
                            video_id=video_id,
                            frame_id=frame_id,
                            image_id=image_id,
                            image_size=img_dims,
                            annotation_id=annotation_id,
                            person_id_map=person_id_map,
                            new_person_id=new_person_id
                        )
                        coco_dict['annotations'].extend(ann_ins)
                    image_name = osp.join(video_name, 'rgb', frame)
                    coco_dict['images'].append({
                        'id': image_id,
                        'video_id': video_id,
                        'frame_id': frame_id,
                        'video' : video_name,
                        'width' : img_dims[0],
                        'height' : img_dims[1],
                        'file_name' : image_name,
                        'camera_id' : cam_id,
                        'scenario_id' : scenario_id
                        })
                    frame_id += 1
                    image_id += 1
                video_id += 1
    return coco_dict

def convert_annotations(
    annotation_root : str,
    train_data_root : str,
    test_data_root : str,
    sampling_rate : int,
):
    
    csv_ann_root = os.path.join(annotation_root, 'csv_gt')
    ann_dirs = os.listdir(csv_ann_root); ann_dirs.sort()
    scenario_to_ann_dir = {ann_dir.split('_')[0] : ann_dir for ann_dir in ann_dirs}
    
    train_scenarios = os.listdir(train_data_root); train_scenarios.sort()
    test_scenarios = os.listdir(test_data_root); test_scenarios.sort()
    
    # save_mot_style_ann(
    #     annotation_root=csv_ann_root, 
    #     scenario_to_ann_dir=scenario_to_ann_dir, 
    #     scenarios=train_scenarios, 
    #     type='train', 
    #     sampling_rate=sampling_rate
    # )

    save_mot_style_ann(
        annotation_root=csv_ann_root, 
        scenario_to_ann_dir=scenario_to_ann_dir, 
        scenarios=test_scenarios, 
        type='test', 
        sampling_rate=sampling_rate
    )

    # train_dict = get_coco_style_ann(
    #     data_root='data/kaist_mtmdc',
    #     annotation_root=csv_ann_root, 
    #     scenario_to_ann_dir=scenario_to_ann_dir, 
    #     scenarios=train_scenarios, 
    #     type='train', 
    #     sampling_rate=sampling_rate
    # )
    # with open(osp.join(annotation_root, 'kaist_mtmdc_train.json'), 'w') as fp:
    #     json.dump(train_dict, fp=fp, sort_keys=True, indent=3)
    
    test_dict = get_coco_style_ann(
        data_root='data/kaist_mtmdc',
        annotation_root=csv_ann_root, 
        scenario_to_ann_dir=scenario_to_ann_dir, 
        scenarios=test_scenarios, 
        type='test',
        sampling_rate=sampling_rate,
    )
    with open(osp.join(annotation_root, 'kaist_mtmdc_test_short.json'), 'w') as fp:
        json.dump(test_dict, fp=fp, sort_keys=True, indent=3)


def main():
    args = parse_args()
    convert_annotations(
        annotation_root=args.ann_root,
        train_data_root=args.train_data_root,
        test_data_root=args.test_data_root,
        sampling_rate=args.sampling_rate
    )


if __name__ == '__main__':
    main()