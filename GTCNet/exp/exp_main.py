from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import GTCNet
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import math

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.is_gtcnet = (self.args.model == 'GTCNet')

    def _build_model(self):
        model_dict = {
            'GTCNet': GTCNet,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        
        if self.is_gtcnet:
            from layers.gtc_layers import GTCNetLoss
            gtc_loss = GTCNetLoss(
                lambda_gate=getattr(self.args, 'lambda_gate', 1e-4),
                lambda_scale=getattr(self.args, 'lambda_scale', 1e-5),
                use_ms=bool(getattr(self.args, 'use_ms', 0)),
            )
            return gtc_loss
        else:
            mse_criterion = nn.MSELoss()
            mae_criterion = nn.L1Loss()
            return mse_criterion, mae_criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=True):
        
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input (kept for compatibility, not used by GTCNet/xPatch)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # ---- Forward ----
                if self.is_gtcnet:
                    outputs, aux = self.model(batch_x, return_aux=True)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # ---- Loss computation ----
                if self.is_gtcnet:
                    
                    mse_fn = nn.MSELoss()
                    loss = mse_fn(outputs, batch_y)
                else:
                    if not is_test:
                        
                        ratio = np.array([-1 * math.atan(i + 1) + math.pi / 4 + 1 for i in range(self.args.pred_len)])
                        ratio = torch.tensor(ratio).unsqueeze(-1).to(self.device)
                        pred = outputs * ratio
                        true = batch_y * ratio
                    else:
                        pred = outputs
                        true = batch_y
                    loss = criterion(pred, true)

                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        if self.is_gtcnet:
            
            gtc_criterion = self._select_criterion()
            
            vali_criterion = nn.MSELoss()
            test_criterion = nn.MSELoss()
        else:
            mse_criterion, mae_criterion = self._select_criterion()
            vali_criterion = mae_criterion
            test_criterion = mse_criterion

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

               
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                
                if self.is_gtcnet:
                    outputs, aux = self.model(batch_x, return_aux=True)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                
                if self.is_gtcnet:
                    gates = aux.get('gates', None)
                    scale_weights = aux.get('scale_weights', None)
                    loss, loss_dict = gtc_criterion(outputs, batch_y, gates=gates, scale_weights=scale_weights)
                else:
                    
                    ratio = np.array([-1 * math.atan(i + 1) + math.pi / 4 + 1 for i in range(self.args.pred_len)])
                    ratio = torch.tensor(ratio).unsqueeze(-1).to(self.device)
                    loss = mae_criterion(outputs * ratio, batch_y * ratio)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    if self.is_gtcnet:
                        print("\t  mse: {:.6f}  gate_reg: {:.6f}".format(
                            loss_dict.get('mse', 0), loss_dict.get('gate', 0)))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, vali_criterion, is_test=False)
            test_loss = self.vali(test_data, test_loader, test_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)  #drop last =False

        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        return
