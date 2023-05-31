import yaml
import torch
from proc.procedure import Informer_Procedure

f = open('./params/hyper-param.yml')
# 读取参数, 输出一个参数字典
params = yaml.load(f.read(),Loader=yaml.FullLoader)
f.close()

p = Informer_Procedure

test = p(params)
# setting 指向checkpoints中最优的参数组合
test.train(setting="informerTBM",load=True)
# test.train(setting="informerTBM")
test.test(setting="informerTBM")

torch.cuda.empty_cache()