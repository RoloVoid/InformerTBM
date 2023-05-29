# 测试TensorBoard可视化

from utils.visualization.tboard import TbVisualizer

test_path = './result/informerTBM'

class Test_tensorboard():
    def test_get_data(self):
        tsb = TbVisualizer(path=test_path)    
        tsb.get_data()