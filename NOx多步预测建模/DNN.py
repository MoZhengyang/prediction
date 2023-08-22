import pandas as pd
import numpy as np
import torch
import time
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def Split(X, Y, test_size=0.25):
    train_num = int(len(X)*(1-test_size))
    X_train = X[:train_num,]
    y_train = Y[:train_num,]
    X_test = X[train_num:,]
    y_test = Y[train_num:,]
    return X_train, X_test, y_train, y_test


def load_data():
    data = pd.read_excel('多步预测建模数据.xlsx')
    # 解析数据,采样周期40s
    data40 = np.array(data.iloc[0:-1:2, 1:])
    Load = data40[:, 0]
    O2 = data40[:, 1]
    FuelAF = data40[:, 2:8]
    SecA = data40[:, 8]
    SecB = data40[:, 9]
    SecC = data40[:, 10]
    SecD = data40[:, 11]
    SecE = data40[:, 12]
    SecF = data40[:, 13]
    Sofah = data40[:, 14]
    Sofaq = data40[:, 15]
    NOx = data40[:, 17]  # 滤波后的NOx
    Sofa = (Sofah + Sofaq) / 2  # sofa风合并
    SecCF = (SecC + SecF) / 2  # 二次风中层CF合并
    SecBE = (SecB + SecE) / 2  # 二次风下层BE合并
    # 对数据归一化到[0,1]
    NOx = (NOx - min(NOx)) / (max(NOx) - min(NOx))
    Load = (Load - min(Load)) / (max(Load) - min(Load))
    O2 = (O2 - min(O2)) / (max(O2) - min(O2))
    FuelAF = (FuelAF - np.min(FuelAF, 0)) / (np.max(FuelAF, 0) - np.min(FuelAF, 0))
    SecA = (SecA - min(SecA)) / (max(SecA) - min(SecA))
    SecD = (SecD - min(SecD)) / (max(SecD) - min(SecD))
    SecCF = (SecCF - min(SecCF)) / (max(SecCF) - min(SecCF))
    SecBE = (SecBE - min(SecBE)) / (max(SecBE) - min(SecBE))
    Sofa = (Sofa - min(Sofa)) / (max(Sofa) - min(Sofa))
    variables = np.zeros((len(NOx), 13))
    variables[:, 0] = O2
    variables[:, 1:7] = FuelAF
    variables[:, 7] = SecA
    variables[:, 8] = SecD
    variables[:, 9] = SecCF
    variables[:, 10] = SecBE
    variables[:, 11] = Sofa
    variables[:, 12] = NOx
    print(variables.shape)
    return variables, NOx


def embedding_data(variables, NOx):
    Len = 5000  # 建模用数据长度
    st = 100
    ed = st + Len
    Y = NOx[st:ed]
    delays = [1, 3, 3, 3, 2, 3, 5, 3, 3, 2, 4, 10, 1]  # 共计13个变量，每个变量的迟延时间
    orders = [3, 5, 4, 5, 6, 5, 5, 1, 5, 6, 6, 3, 4]  # 共计13个变量，每个变量的输入阶数
    X = np.zeros((Len, sum(orders)))
    count = 0
    for i in range(13):
        delay = delays[i]
        order = orders[i]
        for j in range(order):
            X[:, count] = variables[st - delay - j:ed - delay - j, i]
            count += 1
    print(f'X维度{X.shape},Y维度{Y.shape}')
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    #X = X.unsqueeze(axis=1)
    X_train, X_test, y_train, y_test = Split(X, Y, test_size=0.25)
    print(f'X_train.shape: {X_train.shape}, X_test.shape:{X_test.shape}')
    batch_size = 256
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle=True)
    return train_dl,test_dl


class DNN(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1))

    def forward(self, x):
        y_hat = self.fc(x)
        return y_hat


def mae(y_true, y_pred):
    metric =  torch.abs(y_true- y_pred)
    return metric


# 评价测试集
def evaluate_accuracy(data_iter, net, device=None):
    if device is None:
        device = list(net.parameters())[0].device
    acc_sum,n = 0.0, 0
    with torch.no_grad():
        for X,y in data_iter:
            net.eval()
            acc_sum += mae(net(X.to(device)),y.to(device)).sum().cpu().item()
            n += y.shape[0]
    return acc_sum/n


# 训练模型
def train_net(net, train_iter, test_iter, optimizer, loss, device, epochs):
    net = net.to(device)
    print('training on', device)
    batch_count = 0
    for epoch in range(1, epochs + 1):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()

        # 更新学习率
        if epoch % 5 == 0:  # 双阶段除以5
            for p in optimizer.param_groups:
                p['lr'] *= 0.85

        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            net.train()
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += torch.abs(y_hat - y).sum().cpu().item()
            n += y.numel()
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net, device)

        print('epoch %d, loss %.4f, trian_acc %.3f, test_acc %.3f, time %.2f sec'
               % (epoch, train_l_sum, train_acc_sum / n, test_acc, time.time() - start))


def main():
    variables, NOx = load_data()
    train_dl, test_dl = embedding_data(variables, NOx)
    net = DNN(58)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0)
    # 初始化weight
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.001)
            nn.init.constant_(m.bias, 0)
    epochs = 300   # 不反馈设置为3000, 反馈设置100
    train_net(net, train_dl, test_dl, optimizer, loss, device, epochs)


if __name__ == "__main__":
    main()
