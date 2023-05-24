# 使用一个经过调参的lstm模型作为baseline

import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, h_size, num_layers, pred_len):
        super().__init__()
        self.hidden_layer_size = h_size
        self.n_layers = num_layers
        self.pred_len = pred_len
        self.linear = nn.Linear(h_size,pred_len)

        self.lstm = nn.LSTM(input_size, h_size, num_layers,batch_first=True)

    def forward(self,input_seq,init_base):
        lstm_out, self.hidden_cell = self.lstm(input_seq,init_base)
        predicts = self.linear(lstm_out)
        return predicts[:,-self.pred_len:,-1] # (batch_size, seq_len) 默认标签数据在最后一个特征上
