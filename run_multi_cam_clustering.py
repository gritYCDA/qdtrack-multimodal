import os
import argparse
import mmcv
import json
from clustering.multi_cam_clustering import (splitted_clustering_from_weights, 
                                            find_clustering_weights)

class Run_multi_cam_clustering:
    def __init__(self,args):

        self.cfg = mmcv.Config.fromfile(args.config).root
        self.cfg.config_basename = os.path.basename(args.config).replace(".py","")
        self.cfg.repository_root = os.path.abspath(os.path.dirname(__file__))
        self.cfg.config_run_path = os.path.join(self.cfg.repository_root
                                                , "work_dirs"
                                                , "clustering"
                                                , "config_runs"
                                                , self.cfg.config_basename)
        os.makedirs(self.cfg.config_run_path,exist_ok=True)


    def run(self):
        if self.cfg.find_weights.run:
            find_clustering_weights(
                test_track_results_folder=self.cfg.test_track_results_folder
                 , train_track_results_folder=self.cfg.train_track_results_folder
                 , work_dirs=self.cfg.work_dirs
                 , test_dataset_folder=self.cfg.test_dataset_folder
                 , train_dataset_folder=self.cfg.train_dataset_folder
                 , mc_cfg=self.cfg
                 , cam_count=self.cfg.cam_count
                 , take_frames_per_cam=self.cfg.find_weights.take_frames_per_cam
                 , weight_search_configs=self.cfg.find_weights.weight_search_configs
                 , dist_name_to_distance_weights=self.cfg.find_weights.dist_name_to_distance_weights
                 , config_basename=self.cfg.config_basename
                 , person_identifier="person_id"
            )


        if self.cfg.cluster_from_weights.run:
            splitted_clustering_from_weights(
                test_track_results_folder=self.cfg.test_track_results_folder
                 , train_track_results_folder=self.cfg.train_track_results_folder
                 , work_dirs=self.cfg.work_dirs
                 , test_dataset_folder=self.cfg.test_dataset_folder
                 , train_dataset_folder=self.cfg.train_dataset_folder
                 , mc_cfg=self.cfg
                 , cam_count=self.cfg.cam_count
                 , best_weights_path=self.cfg.cluster_from_weights.best_weights_path
                 , default_weights=self.cfg.cluster_from_weights.default_weights
                 , config_basename=self.cfg.config_basename
                 , person_identifier="person_id"
                 , n_split_parts=self.cfg.cluster_from_weights.split_count
            )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    run_clustering = Run_multi_cam_clustering(args)
    run_clustering.run()