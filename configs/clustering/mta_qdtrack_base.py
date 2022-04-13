root = {
        "work_dirs" : "work_dirs",
        "train_track_results_folder" : "work_dirs/MTA/qdtrack-frcnn_r50_train/track",
        "test_track_results_folder" : "work_dirs/MTA/qdtrack-frcnn_r50_test/track",
        "train_dataset_folder" : "data/MTA_ext_short_coco/train",
        "test_dataset_folder" : "data/MTA_ext_short_coco/test",
        "cam_count" : 6,

        "cluster_from_weights" : {
            "split_count": 1,
            # "best_weights_path" : "work_dirs/clustering/config_runs/mta_qdtrack_base/best_weights/best_weights.csv",
            "best_weights_path" : "no_path",
            "default_weights" : {
                "are_tracks_cam_id_frame_disjunct": 1,
                "are_tracks_frame_overlap_disjunct": 1,
                "overlapping_match_score": 1.8,
                "feature_mean_distance":2.5,
                "track_pred_pos_start_distance" : 0.325
                },
            "dataset_type" : "test",
            "run" : True
        },

        "find_weights" : {
                #For fast testing a number of frames can be provided. To take all frames None can be passed.
                "take_frames_per_cam" : None,
                #Default weights which will be partly overwritten during the search
                "dist_name_to_distance_weights" : {
                    "are_tracks_cam_id_frame_disjunct": 1,
                    "are_tracks_frame_overlap_disjunct": 1,
                    "overlapping_match_score": 0,
                    "feature_mean_distance": 0,
                    "track_pred_pos_start_distance" : 0
            },
            "weight_search_configs" : [
                { "dist_name" : "feature_mean_distance",
                                "start_value" : 4.25,
                                "stop_value" : 5,
                                "steps" : 3},

                {"dist_name": "overlapping_match_score",
                              "start_value": 1.8,
                              "stop_value": 2.3,
                              "steps": 3},

                {"dist_name": "track_pred_pos_start_distance",
                              "start_value": 0.1,
                              "stop_value": 1,
                              "steps": 5}
            ],
        "run" : False
        }

}