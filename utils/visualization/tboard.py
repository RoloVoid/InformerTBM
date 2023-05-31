import os
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from utils.visualization.base import VisualBase

class TbVisualizer(VisualBase):
    def __init__(self,path):
        super(TbVisualizer,self).__init__(path)
        self.writer = self._gen_Writer() # self.writer.close()

    def write_data(self,loss,epoch):
        assert(self.writer is not None)
        self.writer.add_scalar("Loss/epoch",loss,epoch)
        self.writer.flush()

    def close(self):
        self.writer.close()
        
    def _gen_Writer(self):
        writer = SummaryWriter(self.path)
        return writer
    
