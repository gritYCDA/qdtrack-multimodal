
for cam in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
	python demo/demo_mot_vis.py \
		configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_kaist_mtmdc.py \
		--checkpoint work_dirs/KAIST_MTMDC/qdtrack-frcnn_r50_kaist_mtmdc/latest.pth \
		--input data/kaist_mtmdc/all_video/s17/rgb/NIA_MTMDC_s17_c${cam}_pm_sunny.avi \
		--output test${cam}.mp4 \
		--fps 23
done