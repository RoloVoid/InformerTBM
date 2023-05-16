import os
import numpy as np
import pandas as pd

import torch
from utils.tools import StandardScaler
from torch.utils.data import Dataset, DataLoader

type_map = {'train':1,'val':2,'test':0}

# 此处的盾构机数据处理任务默认是一个MS(Mutivariate predict univariate，即多变量预测单变量)任务
# informer的原设计中对上述的不同情况进行了区分，此处的简化不再区分
class TBMDataset(Dataset):
    def __init__(self,root_path='../data/shieldmachine/knife',flag='train',size=None,
                 split='刀号',data_path='temp1.xlsx',
                 target='修正磨损量',scale=False,cols=None):
        
        # 依据informer架构的设计:
        # size[seq_len, label_len, pred_len]
        if size==None:
            self.seq_len = 8*4*4 # 根据数据集调换，下同
            self.label_len = 8*4
            self.pred_len = 8*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # 初始化
        assert flag in ['train','test','val']
        self.set_type=type_map[flag]

        self.target = target
        self.scale = scale
        self.scaler = StandardScaler() # 使用实现的标准化方法
        self.split = split # 数据集的划分依据，这里用刀号

        self.root_path = root_path
        self.data_path = data_path
        
        # 取数据
        self.__read_data__()

    def __read_data__(self):

        # 数据集标准化方法
        df_raw = pd.read_excel(os.path.join(self.root_path,self.data_path)) # 默认数据格式是xslx

        # 根据刀号划分
        self.separ=df_raw[self.split].unique()

        # 1/2训练，1/4测试，1/4验证
        l = len(self.separ)
        border1s = [0,l//4,l//4*3]
        border2s = [l//4,l//4*3,l]

        b1= border1s[self.set_type]
        b2= border2s[self.set_type]

        # 通过给定的值数组来筛选行的pandas方法
        df_extract = df_raw[df_raw[self.split].isin(self.separ[b1:b2])]

        # [序号, ...(各种特征), 刀号], 掐头去尾
        cols_data=df_extract.columns[1:-1]
        df_data=df_extract[cols_data]

        # 标准化参数
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x=data
        self.data_y=data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y
        
    def __len__(self):
        return len(self.data_x)-self.seq_len - self.pred_len+1
        
class TBMDataset_Pred(Dataset):
    def __init__(self,root_path,flag='pred',
                 size=None, features='S', data_path='',
                 target='',scale=True,cols=None):
        if size == None:
            self.seq_len = 8*4*4
            self.label_len = 8*4
            self.pred_len = 8*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # init
        assert flag=='pred'

        self.features = features
        self.target = target
        self.scale = scale
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_path__()

    def __read_data__(self):
        self.scaler = StandardScaler() # 归一化工具
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 此处所在的数据集标注：
        # df_raw ['features'...,target feature]

        # 用于提取需要的特征
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)

        df_raw = df_raw[cols+[self.target]]

        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)

        df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else: data=df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __get_item__(self,index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end = self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x,seq_y
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1





        
        

