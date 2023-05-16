# informer基于复杂的时序参数进行了适配
# 由于TBM滚刀数据集以min和轮次为单位收集，没有严格的分层时序特征，这里进行简化

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# transformer的基本位置编码, 来自：
# Attention Is All You Need
# pytorch的原生transformer实现
class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super(PositionalEmbedding,self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad=False

        pos = torch.arange(0,max_len).float().unsqueeze(1) # (max_len,1)
        # PE_(pos,2i) = sin((pos/10000)^{\frac{2i}{d_model}})
        # PE_(pos,2i+1) = cos((pos/10000)^{\frac{2i}{d_model}})
        div_term = (torch.arange(0,d_model,2).float()*-(math.log(10000.0)/d_model)).exp()
        pe[:,0::2] = torch.sin(pos*div_term)
        pe[:,1::2] = torch.cos(pos*div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe) # pe作为模型参数(属性)但不更新

    def forward(self,x):
        return self.pe[:, :x.size(1)]

# Informer源码的可学习TokenEncoding
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2

        # 一维卷积将输入的特征空间映射到d_model上
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,out_channels=d_model, # channels 在这里指序列的特征
            kernel_size=3, padding=padding,padding_mode='circular'
        )

        # 所有的卷积初始化 kaiming
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu') 

    def forward(self, x):
        # x: [N, seq_len, c_in]
        x = self.tokenConv(x.permute(0,2,1)).transpose(1,2)
        return x

# Informer源码的FixEmbedding
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding,self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.requires_grad = False

        pos = torch.arange(0,c_in).float().unsqueeze(1) # (c_in,1)
        div_term = (torch.arange(0,d_model,2).float()*-(math.log(10000.0)))

        w[:,0::2] = torch.sin(pos*div_term)
        w[:,1::2] = torch.cos(pos*div_term)

        # 给定编码层位置权重之后进行固定的编码，故为FixEmbedding
        self.emb = nn.Embedding(c_in,d_model)
        self.emb.weight = nn.Parameter(w,requires_grad=False)

    def forward(self,x):
        return self.emb(x).detach()

# 盾构机掘进以min为单位，temporal attention简化为单min
# 依然是4min为一组进行编码，另外加一个8min的编码，一般盾构机平均工作8min会停止
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed'):
        super(TemporalEmbedding, self).__init__()
        min1_size = 4
        min2_size = 8
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        self.min1_embed = Embed(min1_size,d_model)
        self.min2_embed = Embed(min2_size,d_model)

    def forward(self,x):
        x = x.long()

        min1_x = self.min1_embed(x[:,:,1])
        min2_x = self.min2_embed(x[:,:,0])

        return min1_x+min2_x

# 将所有的数据Embedding
class DataEmbedding(nn.Module):
    def __init__(self,c_in,d_model,embed_type='fixed',dropout=0.1):
        
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        x = self.value_embedding(x)+self.pos_embedding(x)

        return self.dropout(x)