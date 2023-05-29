import yaml
import torch
from proc.lstmproc import BaseLSTM
from proc.procedure import Informer_Procedure

if __name__ == '__main__':

    # 参数读取
    f1,f2 = open('./params/hyper-param.yml'),open('./params/lstm-param.yml')
    iftbm_Params = yaml.load(f1.read(),Loader=yaml.FullLoader)
    lstm_Params = yaml.load(f2.read(),Loader=yaml.FullLoader)
    f1.close()
    f2.close()

    m1,m2 = BaseLSTM(lstm_Params),Informer_Procedure(iftbm_Params)

    m1.train(setting='lstm')
    m2.train(setting='informerTBM')

    
    


