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
    '计算ssd,默认reduction= mean求平均'
    'MSELoss(reduction=’sum‘)指定求和'
    def __init__(self, reduction: str = 'mean') -> None:
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)

class Gtd_loss(nn.Module):
    'matlab的形变场，xy方向与python中的方向和正负不同，这里需要调换方向再计算mse'
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
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)        #kernel 是一个 NumPy 数组，它被转换为 PyTorch 的 FloatTensor，然后通过两次 unsqueeze 操作来增加两个额外的维度，使其形状变为 [1, 1, 3, 3]。这种操作是为了使 kernel 的形状与 PyTorch 卷积层期望的权重形状相匹配，即 [out_channels, in_channels, kernel_height, kernel_width]。
        self.weight = nn.Parameter(data=kernel.to(device=device), requires_grad=False)  #keneral参数不可训练
        # self.rep = nn.ReplicationPad2d(1)

    def forward(self, x):                                                          #forward进行传入参数或数据
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
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)        #kernel 是一个 NumPy 数组，它被转换为 PyTorch 的 FloatTensor，然后通过两次 unsqueeze 操作来增加两个额外的维度，使其形状变为 [1, 1, 3, 3]。这种操作是为了使 kernel 的形状与 PyTorch 卷积层期望的权重形状相匹配，即 [out_channels, in_channels, kernel_height, kernel_width]。
        self.weight = nn.Parameter(data=kernel.to(device=device), requires_grad=False)  #keneral参数不可训练
        # self.rep = nn.ReplicationPad2d(1)

    def forward(self, x):                                                          #forward进行传入参数或数据
        x1 = x[:, 0]
        x2 = x[:, 1]
        # x1 = F.conv2d(self.rep(x1.unsqueeze(1)), self.weight, padding=0)
        # x2 = F.conv2d(self.rep(x2.unsqueeze(1)), self.weight, padding=0)
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=0)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=0)
        x = torch.cat([x1, x2], dim=1)
        return torch.mean(x ** 2)


class MultipleInput(Dataset):
    "dataset实现多输入"

    def __init__(self, input1=None, input2=None, input3=None, transform=None):
        # 假设input1_data和input2_data是两个包含数据样本的列表或数组
        # transform是一个可选的数据转换函数或组合，用于在返回之前转换数据
        self.input1_data = input1
        self.input2_data = input2
        self.input3_data = input3
        self.transform = transform

    def __len__(self):
        # 返回数据集中的样本数
        # 假设两个输入数据集的长度是相同的
        return len(self.input1_data)

    def __getitem__(self, idx):
        # 根据索引idx获取样本
        sample1 = self.input1_data[idx]
        sample2 = self.input2_data[idx]
        sample3 = self.input3_data[idx]
        # print(sample1.shape)
        # 如果定义了转换函数，则应用转换
        if self.transform:
            sample1, sample2, sample3 = self.transform(sample1, sample2, sample3)

            # 返回包含两个输入的元组
        return torch.tensor(sample1), torch.tensor(sample2), torch.tensor(sample3)










if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    logging.info('Using device {}'.format(device))

    batch_size = 32
    learning_rate = 0.01
    epochs = 1000
    lr = 0.01
    lamda1 = 1
    lamda2 = 0.001  #0.001
    lamda3 = 1000  #1000
    lamda4 = 0     #1000
    patience = 30
    epochs_no_improve = 0
    filepath = './data/new.h5'
    savepath = './result/'
    model = UNet()
    model.to(device=device)
    filepath1 = './data/bg_10.h5'
    filepath2 = './data/acdc_test_50.h5'
    filepath3 = './data/bgtest_50.h5'
    filepath4 = './data/bg_150_1000.h5'
    filepath5 = './data/bg_1_958.h5'
    filepath6 = './data/innerpatient_slices.h5'
    with h5py.File(filepath4, 'r') as f:
        # data1 = f['T'][:] #[0:1000]  # 假设你要的数组在名为'dataset1'的数据集中
        # data2 = f['D'][:] # [0:1000] # 假设第二个数组在名为'dataset2'的数据集中
        # print('train:', data1.shape)

        # data3 = f['transfield'][:]#[0:1000]

        data1 = f['T'][0:1000] #[0:1000]  # 
        data2 = f['D'][0:1000] # [0:1000] # 
        print('train:', data1.shape)

        data3 = f['transfield'][0:1000]#[0:1000]

    # with h5py.File(filepath3, 'r') as ff:
    #     test2 = ff['T'][:]#[0:1000]  # 假设你要的数组在名为'dataset1'的数据集中
    #     test1 = ff['D'][:]# [0:1000] # 假设第二个数组在名为'dataset2'的数据集中
    #     print('test:', test1.shape)

    #     test3 = ff['transfield'][:] #[0:1000]
    with h5py.File(filepath6, 'r') as ff:
        test2 = ff['T'][0:2]#[0:1000]  # 假设你要的数组在名为'dataset1'的数据集中
        test1 = ff['T'][1:3]# [0:1000] # 假设第二个数组在名为'dataset2'的数据集中
        print('test:', test1.shape)

    assert test1.shape == test2.shape, "数据大小不一致"
    
        
    dataset = MultipleInput(input1=data1, input2=data2, input3=data3)  # 利用 gettim 实现多输入
    testset = MultipleInput(input1=test1, input2=test2)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    # shuffle打乱数据,可能会打乱对应关系?
    # pin_memory=True，数据加载器在 CUDA 环境中运行，返回的 tensors 将会预先被固定在 CUDA pin memory 中，这可以减少数据从CPU 到 GPU 的传输时间
    # drop_last如果数据集大小不能被 batch_size 整除，则设置为 True 时会丢弃最后一个不完整的批次。如果为 False（默认值），则会保留最后一个批次
    test_loader = DataLoader(testset, batch_size=len(test1), shuffle=False, num_workers=0, pin_memory=True,
                             drop_last=True)
    n_train = len(dataset)
    logging.info('''Starting training:
        Epochs:          {}
        Batch size:      {}
        Learning rate:   {}
        Device:          {}
    '''.format(epochs, batch_size, lr, device.type))
    # 指定loss函数
    SSD_LOSS = nn.MSELoss(reduction='sum')  # 指定求和
    GTD_LOSS = Gtd_loss(reduction='sum')
    Laplacian = Laplacian(device)
    Gradient = Gradient(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 初始化最佳损失和最佳模型权重
    best_loss = float('inf')  # 初始化为正无穷大
    best_model_wts = model.state_dict()  # 初始化为当前模型的权重
    loss_test = []
    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0
        sample_cnt = 0
        
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch + 1, epochs), unit='img') as pbar:  # 进度条更新
            for batch in train_loader:  #
                # 前向传播
                # print(epoch+1)
                T, D, h = batch
                # print(T.shape)
                # print(D.shape)
                T = T.to(device=device, dtype=torch.float32)  # 移动到由device变量指定的设备上(GPU)
                D = D.to(device=device, dtype=torch.float32)
                h = h.to(device=device, dtype=torch.float32)

                hh, TT = model(T, D)  # forward方法,此处等价于model.forward(data), 批次数据的并行计算（输入是(batch_size,数据),输出是(batch_size,输出)）
                # print(hh.shape)
                loss1 = lamda1 * SSD_LOSS(TT, D)  # ssd
                loss2 = lamda2 * GTD_LOSS(h, hh)  # gtd
                loss3 = lamda3 * Laplacian(hh)  #
                loss4 = lamda4 * Gradient(hh)
                # print(loss1, loss2.shape, loss3.shape)
                # detDloss = 5 * criterion4(map_pred)
                # sum_detD += detDloss.item()
                loss = loss1 + loss2 + loss3 + loss4  # + detDloss
                epoch_loss += loss.item()
                sample_cnt += 1
                # writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item(), 'epoch avg loss:': epoch_loss / sample_cnt,
                                    'SSD:': loss1.item(), 'GTD': loss2.item(),
                                    'R1': loss3.item(), 'R2': loss4.item()})  # 显示日志（进度条，损失）

                # 反向传播和优化1we    t
                optimizer.zero_grad()  # 清除旧的梯度
                loss.backward()  # 计算梯度

                nn.utils.clip_grad_value_(model.parameters(), 0.1)  # 梯度裁剪
                optimizer.step()  # 更新参数
                pbar.update(batch_size)

            # 验证模型
            model.eval()  # 设置模型为评估模式
            val_loss = 0.0
            with torch.no_grad():  # 关闭梯度计算
                T, D, h = torch.tensor(test1), torch.tensor(test2), torch.tensor(test2)
                T = T.to(device=device, dtype=torch.float32)  # 移动到由device变量指定的设备上(GPU)
                D = D.to(device=device, dtype=torch.float32)
                # h = h.to(device=device, dtype=torch.float32)
    
                hh, TT = model(T, D)  # 前向传播
                loss1 = lamda1 * SSD_LOSS(TT, D)  # ssd
                #loss2 = lamda2 * GTD_LOSS(h, hh)  # gtd
                loss3 = lamda3 * Laplacian(hh)  #
                loss4 = lamda4 * Gradient(hh)
                loss = loss1 +  loss3 + loss4  # + detDloss    loss2 +            val_loss += loss.item()  # 累加损失
    
            avg_val_loss = loss / len(test1)  # 计算平均损失
            loss_test.append(avg_val_loss)
            # 检查是否当前epoch的损失更小
            if avg_val_loss < best_loss:
                print(
                    f'Epoch {epoch + 1}, val_loss: {avg_val_loss:.4f} which is better than best {best_loss:.4f}, saving...')
                best_loss = avg_val_loss
                # 保存最佳模型权重
                best_model_wts = model.state_dict()
                # 可选：打印信息
    
                torch.save(best_model_wts, savepath+'Weight/'+ 'CP_epoch{}_{}.pth'.format(epoch + 1, sample_cnt))
                logging.info('Checkpoint {}_{} saved !'.format(epoch + 1, sample_cnt))
                epochs_no_improve = 0
    
            else:  
                epochs_no_improve += 1  # 增加计数器  
    
            # 检查是否达到了早停条件  
            if epochs_no_improve >= patience:  
                print(f'Early stopping at epoch {epoch + 1} due to no improvement in {patience} epochs.')  
                break  


            # 在训练结束后，加载最佳模型权重
            # model.load_state_dict(best_model_wts)
 
    # 保存模型权重

    # 测试模型
    # model.eval()  # 设置为评估模式
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for X, y in test_loader:
    #         pred = model(X)
    #         predicted = torch.argmax(pred, dim=1)
    #         total += y.size(0)
    #         correct += (predicted == y).sum().item()
    #
    #     print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')


