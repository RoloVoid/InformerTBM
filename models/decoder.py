import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self,self_attn,cross_attn,
                 d_model,d_ff=None,dropout=0.1,
                 activation="relu"):
        super(DecoderLayer,self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attn = self_attn
        self.cross_attn = cross_attn

        # 两层卷积进行处理
        self.conv1 = nn.Conv1d(in_channels=d_model,out_channels=d_ff,kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff,out_channels=d_model,kernel_size=1)

        # 标准化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout=nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self,x,cross,x_mask=None,cross_mask=None):
        x = x + self.dropout(self.self_attn(
            x,x,x,attn_mask = x_mask
        )[0]) # 自注意力部分

        x = self.norm1(x)

        x = x + self.dropout(self.cross_attn(
            x,cross,cross, attn_mask=cross_mask
        )[0]) # 交错注意力部分
        
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1)) # test
        return self.norm3(x+y)
    
class Decoder(nn.Module):
    def __init__(self,layers,norm_layer=None):
        super(Decoder,self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self,x,cross,x_mask=None,cross_mask=None):
        for layer in self.layers:
            x = layer(x,cross,x_mask=x_mask,cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x