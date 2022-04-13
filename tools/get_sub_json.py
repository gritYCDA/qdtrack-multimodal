import json
from collections import defaultdict

ori_json_path = 'data/kaist_mtmdc/annotations/kaist_mtmdc_test.json'
with open(ori_json_path, 'r') as f:
	ori = json.load(f)


small_dict = defaultdict(list)
small_dict['categories'] = ori['categories']
small_dict['info'] = ori['info']
for ori_ in ori['annotations']:
	if ori_['video_id'] == 0:
		small_dict['annotations'].append(ori_)
for ori_ in ori['images']:
	if ori_['video_id'] == 0:
		small_dict['images'].append(ori_)
for ori_ in ori['videos']:
	if ori_['id'] == 0:
		small_dict['videos'].append(ori_)
with open('data/kaist_mtmdc/annotations/kaist_mtmdc_debug.json', 'w') as f:
	json.dump(small_dict, f)
