# 可视化基类

import os

class VisualBase:
    def __init__(self,path:str):
        # if not os.path.exists(path): os.makedirs(path,exist_ok=True)
        self.path = path

    def write_data(self):
        pass