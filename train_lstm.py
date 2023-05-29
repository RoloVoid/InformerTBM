import yaml
import torch
from proc.lstmproc import BaseLSTM


f = open('./params/lstm-param.yml')
# 读取参数, 输出一个参数字典
params = yaml.load(f.read(),Loader=yaml.FullLoader)
f.close()

p = BaseLSTM

t = p(params)
# 先简化，setting只是为了处理文件名
# t.train(setting="lstm")
t.test('LSTM')

# setting是训练好的模型参数的位置

torch.cuda.empty_cache()