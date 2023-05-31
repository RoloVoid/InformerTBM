# LSTM基线训练规程

import torch
import numpy as np
import torch.nn as nn
from torch import optim
import os,time

from baseline.lstm import LSTM
from data.data_loader import TBMDataset,TBMDataset_Pred,DataLoader
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.visualization.tboard import TbVisualizer

class BaseLSTM():
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

    def _build_model(self):
        input_size=self.args['c_in']
        h_size = self.args['h_size']
        num_layers = self.args['num_layers']
        model = LSTM(
            input_size=input_size,
            h_size=h_size,
            num_layers=num_layers,
            pred_len = self.args['pred_len']
        ).to(self.device)

        batch_size = self.args['batch_size']
        self.h_i = torch.randn((num_layers,batch_size,h_size)).to(self.device)
        self.c_i = torch.randn((num_layers,batch_size,h_size)).to(self.device)

        return model

    def _get_data(self,flag):
        Data = TBMDataset
        shuffle_flag = False # 已经依据刀号分割了，可以弃置
        batch_size = self.args['batch_size']
        if flag == 'pred':
            drop_last = False
            batch_size = 1
            Data = TBMDataset_Pred
        else:
            drop_last = True

        dataset = Data (
            # shuffle = self.args['shuffle'],
            root_path = self.args['root_path'],
            data_path = self.args['data_path'],
            flag=flag,
            size = [self.args['seq_len'],self.args['label_len'],self.args['pred_len']],
            target = self.args['target'],
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

        return dataset, data_loader

    def _select_optimizer(self):
        m_optim = optim.Adam(self.model.parameters(),lr=self.args['learning_rate'])
        return m_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self,setting,load=False):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

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

        path = os.path.join(self.args['checkpoints'],setting)
        best_model_path = path+'/'+'checkpoint.pth'

        if load & os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))

        for epoch in range(self.args['train_epochs']):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                pred,real = self._process_one_batch(
                    batch_x,batch_y
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
                
                # 使用amp的优化方法，暂时用不上
                if self.args['use_amp']:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            if early_stopping:
                early_stopping(vali_loss,self.model,path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            adjust_learning_rate(model_optim, epoch+1, self.args)
        best_model_path = os.path.join(path,'checkpoint.pth')

        # 加载训练的最优值
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    

    def vali(self,vali_data, vali_loader,criterion):
        self.model.eval() 
        total_loss = []

        for i,(batch_x,batch_y) in enumerate(vali_loader):
            pred, real = self._process_one_batch(
                batch_x, batch_y
            )
            # 预测要限制梯度的反向传播
            loss = criterion(pred.detach().cpu(),real.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, load=True):
        test_data, test_loader = self._get_data(flag='test')

        if load:
            path = os.path.join(self.args['checkpoints'],setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval() # 测试，取消dropout和batchnorm
        preds = []
        reals = []
        tbr = TbVisualizer(os.path.join(self.args['predict_path'],'real'))
        tbp = TbVisualizer(os.path.join(self.args['predict_path'],'pred'))
        for i,(batch_x,batch_y) in enumerate(test_loader):
            pred,real = self._process_one_batch(
                batch_x,batch_y
            )

            tbr.write_data(real.mean(),i)
            tbp.write_data(pred.mean(),i)
                      
            preds.append(pred.detach().cpu().numpy())
            reals.append(real.detach().cpu().numpy())

        tbr.close()
        tbp.close()

        preds = np.array(preds)
        reals = np.array(reals)
        print('test shape:',preds.shape[-2], preds.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])[-1]
        reals = reals.reshape(-1, reals.shape[-2],reals.shape[-1])[-1]
        print('test shape:', preds.shape, reals.shape)

        # result save
        folder_path = os.path.join('./result',setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 所有的误差
        mae, mse, rmse, mape, mspe, r2 = metric(preds, reals)
        print(f'mse:{mse}, mae:{mae},rmse:{rmse},mape:{mape},mspe:{mspe},R2:{r2}')

        np.save(folder_path+'/metrics.npy',np.array([mae,mse,rmse,mape,mspe]))
        np.save(folder_path+'/pred.npy',preds)
        np.save(folder_path+'/real.npy',reals)

        return

    def predict(self,setting,load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args['checkpoints'],setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []
        for i,(batch_x,batch_y) in enumerate(pred_loader):
            pred,real = self._process_one_batch(
                batch_x,batch_y
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

    def _process_one_batch(self, batch_x, batch_y):
        # batch_size 

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()[:,-self.args['label_len']:,-1].to(self.device)

        outputs = self.model(batch_x,(self.h_i,self.c_i))

        return outputs, batch_y


    
