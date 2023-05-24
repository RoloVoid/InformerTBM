# 测试lstm是否正常工作

from baseline.lstm import LSTM
import torch

input_size = 4
h_size = 8
num_layers = 2
pred_len = 2
batch_size = 16

class TestLSTM():
    def test_forward(self):
        # input (Batch_size,seq_len,feature_d)
        input = torch.randn((16,8,input_size))
        model = LSTM(input_size,h_size,num_layers,pred_len)
        h_0 = torch.randn((num_layers,batch_size,h_size)) # (D*num_layers, batch_size, h_size) D若为单向lstm则为1, h另有约定，这里是hidden_size
        c_0 = torch.randn((num_layers,batch_size,h_size)) # 上下文和隐藏态的初始化基本一致

        print(model(input,(h_0,c_0)))
        print(model(input,(h_0,c_0)).shape) # 应返回需要预测的维度，即(batch_size, seq_len)

        

