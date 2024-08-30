import cv2
import json

image_path = 'iu.jpg'
video_path = './run_net/video/pp085_fast_rear.mp4'

with open('./data.json', 'r', encoding='utf-8') as file:
    history_list = json.load(file)

id = '4'

######################
sorted_items = sorted(history_list.items(), key=lambda item: len(item[1]), reverse=True)
new_dict = dict(sorted_items[:2])

print(new_dict.keys())
id = (input("Enter the keys: "))
######################

cap = cv2.VideoCapture(video_path)

margin = 50
id_data_list = history_list[id]
now_frame_pointer = 0
id_data_len = len(id_data_list)


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
    cv2.imshow('Video Playback', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()