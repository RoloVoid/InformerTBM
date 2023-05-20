# 使用一个经过调参的lstm模型作为baseline

import torch
from torch import nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer=100,output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer

        self.lstm = nn.LSTM(input_size, hidden_layer)

    def forward(self,input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq)
        predicts = self.linear(lstm_out)
        return predicts[-1]
