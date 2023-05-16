import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import DataEmbedding
from models.encoder import Encoder,EncoderLayer,ConvLayer
from models.attn import ProbAttn,FullAttn,AttnLayer
from models.decoder import Decoder,DecoderLayer

# 基于Informer2020的源码进行了适配
class InformerTBM(nn.Module):
    def __init__(self,enc_in,dec_in,
                 c_out,seq_len,label_len,out_len,
                 factor, d_model, n_heads, # 模型维度和注意力头设置
                 e_layers,d_layers,d_ff, # d_ff是模型中的隐藏层层数
                 dropout,attn,embed,activation, # dropout，注意力、编码、激活函数选择
                 output_attn, device,scale,
                 distil,mix # 蒸馏和混合
                 ):
        
        super(InformerTBM,self).__init__()
        self.pred_len = out_len 
        self.attn = attn
        self.output_attention = output_attn

        # 参见embed部分的设计，在原informer的基础上做了修改
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout) # [B, L, E] -> [B, L, D]
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout) # [B, L, E] -> [B, L, D]

        # 参见attn部分，基本保留了原Informer的部分
        Attn = ProbAttn if attn=='prob' else FullAttn
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttnLayer(Attn(False,factor,scale,dropout,output_attn),
                              n_heads,d_model,mix=False),
                    d_model,d_ff, # d_ff是encoder层的中间层数
                    dropout=dropout,
                    activation=activation
                ) for i in range(e_layers)
            ], # 编码层
            [
                ConvLayer(d_model) for l in range(e_layers-1) 
            ] if distil else None, # 卷积层比编码层少一层
            norm_layer = torch.nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # self_attn需要mask
                    AttnLayer(Attn(True,factor,attn_dropout=dropout,output_attn=False),
                              d_model,n_heads,mix=mix),
                    # cross_attn不需要mask
                    AttnLayer(FullAttn(False,factor,attn_dropout=dropout,output_attn=False),
                              d_model,n_heads,mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for i in range (d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # 最后用一个映射回到原过程
        self.projection = nn.Linear(d_model,c_out,bias=True)

    def forward(self,x_enc,x_dec, enc_self_mask=None,dec_self_mask=None,dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc) # [B, L, E] -> [B, L, D]
        enc_out,attns = self.encoder(enc_out,attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out,enc_out,x_mask=dec_self_mask,cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:,-self.pred_len::], attns
        else: return dec_out[:,-self.pred_len:,:] # [B,L,D]



