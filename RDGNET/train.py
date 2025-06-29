import os

import h5py
import logging

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from model import UNet
from tqdm import tqdm
import pickle

class MSELoss(nn.Module):
    'ssd,reduction= mean'
    def __init__(self, reduction: str = 'mean') -> None:
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)

class Gtd_loss(nn.Module):
    'mse'
    def __init__(self, reduction: str = 'mean') -> None:
        super(Gtd_loss, self).__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input=input[:, [1, 0], :, :]
        return F.mse_loss(input, target, reduction=self.reduction)

class Laplacian(nn.Module):
    def __init__(self, device):
        super(Laplacian, self).__init__()
        kernel = np.array([[0., 1., 0.], [1., -4, 1.], [0., 1., 0.]])
        # kernel = np.array([[1., 0., 1.], [0., -4, 0.], [1., 0., 1.]])
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel.to(device=device), requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        # x1 = F.conv2d(self.rep(x1.unsqueeze(1)), self.weight, padding=0)
        # x2 = F.conv2d(self.rep(x2.unsqueeze(1)), self.weight, padding=0)
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=0)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=0)
        x = torch.cat([x1, x2], dim=1)
        return torch.mean(x ** 2)

class Gradient(nn.Module):
    def __init__(self, device):
        super(Gradient, self).__init__()
        kernel = np.array([[0., -1., 0.], [-1., 0, 1.], [0., 1., 0.]])
        # kernel = np.array([[1., 0., 1.], [0., -4, 0.], [1., 0., 1.]])
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel.to(device=device), requires_grad=False)


    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        # x1 = F.conv2d(self.rep(x1.unsqueeze(1)), self.weight, padding=0)
        # x2 = F.conv2d(self.rep(x2.unsqueeze(1)), self.weight, padding=0)
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=0)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=0)
        x = torch.cat([x1, x2], dim=1)
        return torch.mean(x ** 2)


class MultipleInput(Dataset):
    "DATA INPUT: T,R,PHI"

    def __init__(self, input1=None, input2=None, input3=None, transform=None):

        self.input1_data = input1
        self.input2_data = input2
        self.input3_data = input3
        self.transform = transform

    def __len__(self):

        return len(self.input1_data)

    def __getitem__(self, idx):
        sample1 = self.input1_data[idx]
        sample2 = self.input2_data[idx]
        sample3 = self.input3_data[idx]

        if self.transform:
            sample1, sample2, sample3 = self.transform(sample1, sample2, sample3)

        return torch.tensor(sample1), torch.tensor(sample2), torch.tensor(sample3)










if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    logging.info('Using device {}'.format(device))

    batch_size = 32
    epochs = 1000
    lr = 0.01
    lamda1 = 1
    lamda2 = 0.001  #0.001
    lamda3 = 1000  #1000
    lamda4 = 0     #1000
    patience = 20
    epochs_no_improve = 0
    filepath1 = ''
    filepath2 = ''
    savepath = './result/'
    model = UNet()
    model.to(device=device)


    with h5py.File(filepath1, 'r') as f:
        data1 = f['T'][:]
        data2 = f['D'][:]
        print('train:', data1.shape)

        data3 = f['transfield'][:]


    with h5py.File(filepath2, 'r') as ff:
        test2 = ff['T'][0:]
        test1 = ff['T'][:]
        print('test:', test1.shape)

    assert test1.shape == test2.shape, "数据大小不一致"
    
        
    dataset = MultipleInput(input1=data1, input2=data2, input3=data3)  # 利用 gettim 实现多输入
    testset = MultipleInput(input1=test1, input2=test2)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)

    test_loader = DataLoader(testset, batch_size=len(test1), shuffle=False, num_workers=0, pin_memory=True,
                             drop_last=True)
    n_train = len(dataset)
    logging.info('''Starting training:
        Epochs:          {}
        Batch size:      {}
        Learning rate:   {}
        Device:          {}
    '''.format(epochs, batch_size, lr, device.type))
    # loss
    SSD_LOSS = nn.MSELoss(reduction='sum')  # 指定求和
    GTD_LOSS = Gtd_loss(reduction='sum')
    Laplacian = Laplacian(device)
    Gradient = Gradient(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # initialization
    best_loss = float('inf')
    best_model_wts = model.state_dict()
    loss_test = []
    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0
        sample_cnt = 0
        
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch + 1, epochs), unit='img') as pbar:  # 进度条更新
            for batch in train_loader:  #

                T, D, h = batch

                T = T.to(device=device, dtype=torch.float32)
                D = D.to(device=device, dtype=torch.float32)
                h = h.to(device=device, dtype=torch.float32)

                hh, TT = model(T, D)
                # print(hh.shape)
                loss1 = lamda1 * SSD_LOSS(TT, D)  # ssd
                loss2 = lamda2 * GTD_LOSS(h, hh)  # gtd
                loss3 = lamda3 * Laplacian(hh)  #
                #loss4 = lamda4 * Gradient(hh)

                loss = loss1 + loss2 + loss3 #+ loss4
                epoch_loss += loss.item()
                sample_cnt += 1
                pbar.set_postfix(**{'loss (batch)': loss.item(), 'epoch avg loss:': epoch_loss / sample_cnt,
                                    'SSD:': loss1.item(), 'GTD': loss2.item(),
                                    'R1': loss3.item() })  #,'R2': loss4.item()

                # backward
                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                pbar.update(batch_size)

            # 验证模型
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                T, D, h = torch.tensor(test1), torch.tensor(test2), torch.tensor(test2)
                T = T.to(device=device, dtype=torch.float32)
                D = D.to(device=device, dtype=torch.float32)
                # h = h.to(device=device, dtype=torch.float32)
    
                hh, TT = model(T, D)  # 前向传播
                loss1 = lamda1 * SSD_LOSS(TT, D)  # ssd
                #loss2 = lamda2 * GTD_LOSS(h, hh)  # gtd
                loss3 = lamda3 * Laplacian(hh)  #
                #loss4 = lamda4 * Gradient(hh)
                loss = loss1 +  loss3 + #loss4     loss2 +            val_loss += loss.item()
    
            avg_val_loss = loss / len(test1)  # 计算平均损失
            loss_test.append(avg_val_loss)
            # check
            if avg_val_loss < best_loss:
                print(
                    f'Epoch {epoch + 1}, val_loss: {avg_val_loss:.4f} which is better than best {best_loss:.4f}, saving...')
                best_loss = avg_val_loss
                # save
                best_model_wts = model.state_dict()
    
                torch.save(best_model_wts, savepath+'Weight/'+ 'CP_epoch{}_{}.pth'.format(epoch + 1, sample_cnt))
                logging.info('Checkpoint {}_{} saved !'.format(epoch + 1, sample_cnt))
                epochs_no_improve = 0
    
            else:  
                epochs_no_improve += 1
    
            # earlystop
            if epochs_no_improve >= patience:  
                print(f'Early stopping at epoch {epoch + 1} due to no improvement in {patience} epochs.')  
                break  



