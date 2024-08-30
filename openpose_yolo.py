'''
author: DeaYong Kim
date: 2024.08.30
last_modified: 2024.08.30

###################

openpose와 yolo의 통합을 통해서
사람 인덱스 재분류

#################
'''

from dragon import dragonV
from dragon import dragonY
from collections import defaultdict
import json

##file path && constant##
json_folder_path = '/media/yong/SAMSUNG1/json/'
video_path  = '../gcdn_n/run_net/video/pp085_fast_rear.mp4'
yolo_data_path = ''

margin = 50
id = ''
##openpose part##
all_pos_data_list = dragonV.from_jsonfolder_to_list(json_folder_path)

##yolo part##
if yolo_data_path == '':
    results = dragonY.get_results_tracking_data_from_video(video_path)
    results_list = dragonY.get_all_frame_data_list_from_yolo_results(results)

    roi_list = defaultdict(lambda: [])
    
    for frame_idx in range(len(results_list)):
        for element in results_list[frame_idx]:
            if element['id'] == -1: continue

            track_id = element['id']
            box = element['xywh']
            x, y, w, h = box
            track = roi_list[track_id]
            track.append([frame_idx, int(x), int(y), int(w), int(h)])
else:
    with open(yolo_data_path, 'r', encoding='utf-8') as file:
        roi_list = json.load(file)


##Integration part##

##Play video##
