from dragon import dragonReadNet25 as d25
import cv2
import numpy as np
from collections import defaultdict
from dragon import dragonY
from dragon import dragonV
import json

proto = '/Users/ivory/Documents/github/gcdn_n/body_25/pose_deploy.prototxt'
weights = '/Users/ivory/Documents/github/gcdn_n/body_25/pose_iter_584000.caffemodel'
video_path = '/Users/ivory/Documents/github/gcdn_n/run_net/video/pp085_fast_rear.mp4'
#video_path = '/Users/ivory/Documents/github/gcdn_n/run_net/video/gait1_front.mp4'
img_path = '/Users/ivory/Documents/github/gcdn_n/media/iu.jpg'

#results = dragonY.get_results_tracking_data_from_video(video_path)
#results_list = dragonY.get_all_frame_data_list_from_yolo_results(results)
#history_list = dragonY.get_each_id_data_from_yolo_result(results_list)
#net = cv2.dnn.readNetFromCaffe(proto, weights)

with open('./data.json', 'r', encoding='utf-8') as file:
    history_list = json.load(file)

cap = cv2.VideoCapture(video_path)
id = '4'
margin = 50
id_data_list = history_list[id]
now_frame_pointer = 0
print(history_list.keys())

id_data_len = len(id_data_list)

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        ###### Yolo rendering part########
        frame_number_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        
        if now_frame_pointer < id_data_len:
            now_data = id_data_list[now_frame_pointer]
            now_idx = now_data[0]
        else:
            now_frame_pointer = -1

        if(now_idx == frame_number_idx):
            x, y, w, h = now_data[1:]

            w += margin
            h += margin

            pt1_x = round(x - w/2)
            pt1_y = round(y - h/2)

            pt2_x = round(x + w/2)
            pt2_y = round(y + h/2)
            
            pt1 = (pt1_x, pt1_y)    
            pt2 = (pt2_x, pt2_y)
            cv2.rectangle(frame, pt1 , pt2, color=(255, 0, 0) ,thickness=2)
            now_frame_pointer += 1

            roi = frame[pt1_y:pt2_y, pt1_x:pt2_x]
            cv2.imshow("Tracking", roi)
        #############################

        #####OpenPose################
        '''
            roi = frame[pt1_y:pt2_y, pt1_x:pt2_x]
            inblob = d25.preprocess_image_blob(roi)
            net.setInput(inblob)
            output = net.forward()
            joint_pos_list, confidence_list, _ = d25.get_position_from_netoutput(output, roi)
            d25.mark_on_image(roi, joint_pos_list)
        
        '''
        ####q#########################
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
