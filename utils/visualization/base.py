# 可视化基类

import os

class VisualBase:
    def __init__(self,path:str):
        assert(os.path.exists(path))
        self.path = path

    def get_data(self):
        pass