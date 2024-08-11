from dragon import dragonV
import numpy as np

#############################################
video_path = '/Users/ivory/Documents/github/gcdn_n/run_net/video/pp085_fast_rear.mp4'
joint_path_xlsx = '/Users/ivory/Documents/github/gcdn_n/run_net/input/fast_rear.xlsx'
label_path_npy = '/Users/ivory/Documents/github/gcdn_n/run_net/output/predicted.npy'
#label_path_npy = '/Users/ivory/Documents/github/gcdn_n/run_net/labels/velocity_based/pp085/fast.npy'
#############################################

label_list = (np.load(label_path_npy)).tolist()
frame_data_list = dragonV.xlsx2data(joint_path_xlsx)

paired_data_list = dragonV.make_dataAndGtPair(frame_data_list, label_list)
dragonV.render_result_on_video(video_path, paired_data_list, 'video')