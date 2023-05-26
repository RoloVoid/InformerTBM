from data.data_loader import TBMDataset,TBMDataset_Pred
from models.model import InformerTBM

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

class Procedure():
    def __init__(self,args):
        self.args = args
        self.device = self._training_device()
        self.model = self._build_model().to(self.device)
    
    def _training_device(self):
        if self.args['use_gpu']:
            device = torch.device('cuda:{}'.format(self.args['gpu']))
            print('Use GPU: cuda:{}'.format(self.args['gpu']))
        else: 
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    # 只使用单Informer层，不堆叠
    def _build_model(self):
        model = InformerTBM(
            enc_in=self.args['enc_in'],
            dec_in=self.args['dec_in'],
            c_out=self.args['c_out'],
            seq_len=self.args['seq_len'],
            label_len=self.args['label_len'],
            out_len=self.args['pred_len'],
            factor=self.args['factor'],
            d_model=self.args['d_model'],
            n_heads=self.args['n_heads'],
            e_layers=self.args['e_layers'],
            d_layers=self.args['d_layers'],
            d_ff=self.args['d_ff'],
            dropout=self.args['dropout'],
            attn=self.args['attn'],
            embed=self.args['embed'],
            activation=self.args['activation'],
            output_attn=self.args['output_attn'],
            device=self.device,
            scale=self.args['scale'],
            distil=self.args['distil'],
            mix=self.args['mix']
        ).float()

        return model

    def _get_data(self, flag):
        Data = TBMDataset
        if flag == 'test': 
            shuffle_flag = False # 已经被刀号分割了，可以丢弃
            drop_last = True
            batch_size = self.args['batch_size']
        elif flag=='pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            Data = TBMDataset_Pred
        else:
            shuffle_flag = False
            drop_last = True
            batch_size = self.args['batch_size']

        dataset = Data(
            shuffle = self.args['shuffle'],
            root_path = self.args['root_path'],
            data_path=self.args['data_path'],
            flag=flag,
            size=[self.args['seq_len'], self.args['label_len'], self.args['pred_len']],
            # features==args['']
            target=self.args['target'],
            cols = self.args['cols']
        )

        print(flag,len(dataset))
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args['num_workers'],
            drop_last=drop_last
        )

        return dataset,data_loader
    
    def _select_optimizer(self):
        model_optim=optim.Adam(self.model.parameters(),lr=self.args['learning_rate'])
        return model_optim
    
    def _select_criterion(self):
        criterion=nn.MSELoss()
        return criterion
    
    # 验证
    def vali(self, vali_data, vali_loader,criterion):
        self.model.eval() # 预测时不启用dropout和batch_norm
        total_loss = []

        # 去掉了所有的time_mark
        for i,(batch_x,batch_y) in enumerate(vali_loader):
            pred, real = self._process_one_batch(
                vali_data, batch_x, batch_y
            )
            # 预测要限制梯度的反向传播
            loss = criterion(pred.detach().cpu(),real.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        # checkpoints，用于保存训练过程中的模型参数
        path = os.path.join(self.args['check_points'],setting)
        if not os.path.exists(path): os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = None
        if self.args['use_early_stopping']:
            early_stopping = EarlyStopping(patience=self.args['patience'],verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args['use_amp']: scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args['train_epochs']):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                pred,real = self._process_one_batch(
                    train_data,batch_x,batch_y
                )

                loss = criterion(pred,real)
                train_loss.append(loss.item())

                if (i+1)%100==0:
                    print(f'\titers: {0},epoch: {1} | loss: {2:.7f}'.format(i+1, epoch+1,loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args['train_epochs']-epoch)*train_steps-i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                # 此处见hyper-param部分
                if self.args['use_amp']:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data,vali_loader,criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # 早停
            if early_stopping:
                early_stopping(vali_loss,self.model,path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # 学习率动态调节
            adjust_learning_rate(model_optim, epoch+1, self.args)
        
        # 检查点是当前最优的模型
        best_model_path = path+'/'+'checkpoint.pth'

        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval() # 测试，取消dropout和batchnorm
        preds = []
        reals = []
        for i,(batch_x,batch_y) in enumerate(test_loader):
            pred,real = self._process_one_batch(
                test_data,batch_x,batch_y
            )

            preds.append(pred.detach().cpu().numpy())
            reals.append(real.detach().cpu().numpy())

            preds = np.array(preds)
            reals = np.array(reals)
            print('test shape:',preds.shape[-2], preds.shape[-1])
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            reals = reals.reshape(-1, reals.shape[-2],reals.shape[-1])
            print('test shape:', preds.shape, reals.shape)

            # result save
            folder_path = './result/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # 所有的误差
            mae, mse, rmse, mape, mspe = metric(preds, reals)
            print('mse:{}, mae:{}'.format(mse,mae))

            np.save(folder_path+'metrics.npy',np.array([mae,mse,rmse,mape,mspe]))
            np.save(folder_path+'pred.npy',preds)
            np.save(folder_path+'real.npy',reals)

            return
        
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args['checkpoints'],setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model_eval()

        preds = []
        for i,(batch_x,batch_y) in enumerate(pred_loader):
            pred,real = self._process_one_batch(
                pred_data,batch_x,batch_y
            )
            pred.append(pred.detach().cpu.numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1,preds.shape[-2],preds.shape[-1],dim=1).float().to(self.device)

        # 保存结果
        folder_path = './results/'+setting+'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path+'real_prediction.npy',preds)

        return
        
    def _process_one_batch(self,dataset_object, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        
        # 解码器 输入
        if self.args['padding']==0:
            dec_input = torch.zeros([batch_y.shape[0], self.args['pred_len'], batch_y.shape[-1]]).float()
        elif self.args['padding']==1:
            dec_input = torch.ones([batch_y.shape[0], self.args['pred_len'], batch_y.shape[-1]]).float()
        dec_input = torch.cat([batch_y[:,:self.args['label_len'],:],dec_input],dim=1).float().to(self.device)

        if self.args['output_attn']:
            outputs = self.model(batch_x, dec_input)[0]
        else:
            outputs = self.model(batch_x, dec_input)

        batch_y = batch_y[:,-self.args['pred_len']:,0:].to(self.device)

        return outputs[:,:,-1], batch_y[:,:,-1]





            




            
            

    
    

    