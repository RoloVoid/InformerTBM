# 检查环境是否正常

import torch

class Test_env():
    def test_torch_cuda(self):
        print('\n',torch.__version__)
        assert(torch.cuda.is_available())