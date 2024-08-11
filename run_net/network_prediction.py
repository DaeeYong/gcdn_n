from dragon import dragonV
import numpy as np
import torch
import torch
import torch.nn as nn

################################################################################################
input_path_name = '/Users/ivory/Documents/github/gcdn_n/run_net/input/fast_rear.xlsx'
output_path_name = '/Users/ivory/Documents/github/gcdn_n/run_net/output/predicted.npy'
seleted_openpose_joint_idx_list = [8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]
input_size = 26
output_size = 4
hidden_size = 128
################################################################################################

frame_list = dragonV.xlsx2data(input_path_name)
#normalization
norm_frame_list =  dragonV.nomalize_data(frame_list)
#select lower joint
selected_norm_frame_data_list = dragonV.get_selected_joint_pos_frame_list(norm_frame_list, seleted_openpose_joint_idx_list)

input_features = dragonV.sliding_window(selected_norm_frame_data_list, 9)
input_data_torch = torch.from_numpy((np.array(input_features)))
input_data_torch = input_data_torch.to(torch.float32)

has_nan = torch.isnan(input_data_torch).any().item()
print("NaN 값이 있는지 여부:", has_nan)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        out = self.sigmoid(out)
        return out
    
model = LSTM(input_size, hidden_size, output_size)
model_weights = torch.load('../models/100_epochs.pth', map_location=torch.device('cpu'))
model.load_state_dict(model_weights)
model.eval()

output = model(input_data_torch)
predicted = (output > 0.5).int()
predicted_np =  predicted.numpy()

margin_np = np.array([[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]])
m_predicted_np = np.append(margin_np, predicted_np, axis=0)
m_predicted_np = np.append(m_predicted_np, margin_np, axis=0)

np.save(output_path_name, m_predicted_np)