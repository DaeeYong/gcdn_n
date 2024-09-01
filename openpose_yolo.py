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
import cv2

##file path && constant##
json_folder_path = '/Users/ivory/Documents/github/gcdn_n/json/'
video_path  = '../gcdn_n/run_net/video/pp085_fast_rear.mp4'
yolo_data_path = './data_test.json'

id = '4'
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
cap = cv2.VideoCapture(video_path)
id_data_list = roi_list[id]
id_data_len = len(roi_list[id])
now_frame_pointer = 0
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 동영상 프레임을 읽어서 화면에 표시
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Reached the end of the video.")
        break
    
    ###### Yolo rendering part########
    frame_number_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    
    if now_frame_pointer < id_data_len:
        now_data = id_data_list[now_frame_pointer]
        now_idx = now_data[0]
    else:
        now_frame_pointer = -1

    if(now_idx == frame_number_idx):
        x1, y1, x2, y2 = now_data[1:]

        pt1 = (x1, y1)    
        pt2 = (x2, y2)
        cv2.rectangle(frame, pt1 , pt2, color=(255, 0, 0) ,thickness=2)
        now_frame_pointer += 1

        roi = frame[y1:y2, x1:x2]
    cv2.imshow('Video Playback', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()