import yaml
import torch
from proc.lstmproc import BaseLSTM


f = open('./lstm-param.yml')
# 读取参数, 输出一个参数字典
params = yaml.load(f.read(),Loader=yaml.FullLoader)
f.close()

p = BaseLSTM

test = p(params)
# 先简化，setting只是为了处理文件名
test.train(setting="")

torch.cuda.empty_cache()