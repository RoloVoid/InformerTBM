import os
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from utils.visualization.base import VisualBase

class TbVisualizer(VisualBase):
    def __init__(self,path):
        super(TbVisualizer,self).__init__(path)

    # 在这个任务里，数据来源是npy, 文件夹下包括pred.npy real.npy metric.npy三个文件
    def get_data(self):
        data = os.listdir(self.path)
        if len(data)<1: 
            print("this dir is empty")
            return
        cur = os.getcwd()
        os.chdir(self.path)
        pred = np.load('pred.npy')
        real = np.load('real.npy')
        metrics = np.load('metrics.npy')
        print(metrics.shape)
        print(real.shape)
        print(pred.shape)
        print(metrics)

    def gen_Writer(self):
        writer = SummaryWriter(self.path)