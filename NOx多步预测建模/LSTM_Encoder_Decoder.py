import pandas as pd
import numpy as np
import torch
import time
from torch import nn
import random


def get_data():
    def mean(data):
        return data.mean()

    def std(data):
        return data.std()

    data = pd.read_excel('多步预测建模数据_min.xlsx')
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
    NOx_max = 650
    NOx_min = 200
    # NOx = (NOx - NOx_min)/(NOx_max - NOx_min)
    # NOx = NOx
    Load = (Load - mean(Load)) / std(Load)
    O2 = (O2 - mean(O2)) / std(O2)
    FuelAF = (FuelAF - np.mean(FuelAF, 0)) / (np.std(FuelAF, 0) + 1e-6)

    SecA = (SecA - mean(SecA)) / std(SecA)
    SecD = (SecD - mean(SecD)) / std(SecD)
    SecCF = (SecCF - mean(SecCF)) / std(SecCF)
    SecBE = (SecBE - mean(SecBE)) / std(SecBE)
    Sofa = (Sofa - mean(Sofa)) / std(Sofa)
    variables = np.zeros((len(NOx), 13))
    variables[:, 0] = O2
    variables[:, 1:7] = FuelAF
    variables[:, 7] = SecA
    variables[:, 8] = SecD
    variables[:, 9] = SecCF
    variables[:, 10] = SecBE
    variables[:, 11] = Sofa
    variables[:, 12] = NOx
    return variables, NOx


def Split(X, Y, test_size=0.25):
    train_num = int(len(X)*(1-test_size))
    X_train = X[:train_num,]
    y_train = Y[:train_num,]
    X_test = X[train_num:,]
    y_test = Y[train_num:,]
    return X_train, X_test, y_train, y_test


def random_iter(x_train, y_train, batch_size, enc_time_step, dec_time_step, stride):
    offset = random.randint(0, stride)  # 随机开始
    initial_indices = (np.arange(offset, len(x_train) - enc_time_step - dec_time_step, stride))  # 样本开始索引
    random.shuffle(initial_indices)  # 随机打断样本开始索引

    # 注意的是，y_train值已经前向推进了一步
    def data(pos, sign):
        if sign == 1:
            return x_train[pos:pos + enc_time_step]
        if sign == 2:
            x_dec = x_train[pos + enc_time_step:pos + enc_time_step + dec_time_step]
            # 第一维保持不变    # dec中氧量保持常数
            x_dec[1:, 0] = x_dec[0, 0]
            return x_dec
        if sign == 3:
            return y_train[pos + enc_time_step:pos + enc_time_step + dec_time_step]

    num_batches = len(initial_indices) // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        x_enc = torch.stack([data(j, 1) for j in initial_indices_per_batch])
        x_dec = torch.stack([data(j, 2) for j in initial_indices_per_batch])
        y = torch.stack([data(j, 3) for j in initial_indices_per_batch])
        # 对 NOx 归一化 仅batch
        nox_mean = x_enc[:, :, -1].mean(1)
        nox_std = x_enc[:, :, -1].std(1)

        x_enc[:, :, -1] = (x_enc[:, :, -1] - nox_mean.reshape(-1, 1)) / nox_std.reshape(-1, 1)
        x_dec[:, :, -1] = (x_dec[:, :, -1] - nox_mean.reshape(-1, 1)) / nox_std.reshape(-1, 1)
        y = (y - nox_mean.reshape(-1, 1, 1)) / nox_std.reshape(-1, 1, 1)
        yield x_enc, x_dec, y, (nox_mean, nox_std)


class RandomDataLoader:
    def __init__(self, x_train, y_train, batch_size, enc_time_step, dec_time_step, stride):
        self.x_train, self.y_train = x_train, y_train
        self.batch_size = batch_size
        self.enc_time_step, self.dec_time_step, self.stride = enc_time_step, dec_time_step, stride

    def __iter__(self):
        return random_iter(self.x_train, self.y_train, self.batch_size,
                           self.enc_time_step, self.dec_time_step, self.stride)


class RNNModel(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size=feature_size, hidden_size=32, num_layers=2, dropout=0.1)
        self.feature_size = feature_size
        self.num_hiddens = self.rnn.hidden_size
        self.num_directions = 1  # 单向

        self.val_loss_list = []
        self.train_loss_list = []
        self.min_loss = 1e6
        self.val_multi_step_loss_list = []  # 验证集每个epoch 多步loss  mean
        self.min_multi_step_loss = 1e6  # 再训练评价指标

        self.fc = nn.Sequential(
            nn.Linear(self.num_hiddens * self.num_directions, 16),
            nn.Linear(16, 1))

    def encode(self, x_enc):
        # LSTM state为元组形式，包括(h0,c0)
        # x为[时间步, batch_size, feature_szie]
        # RNN 返回为(L,N,D*H_out)
        _, state = self.rnn(x_enc)
        return state

    def decode(self, x_dec, state):
        H, _ = self.rnn(x_dec, state)
        return H

    def forward(self, x_enc, x_dec):
        # LSTM state为元组形式，包括(h0,c0)
        # x为[时间步, batch_size, feature_szie]
        # RNN 返回为(L,N,D*H_out)
        # encoder
        state = self.encode(x_enc)
        # decoder
        H = self.decode(x_dec, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,1)。
        y_hat = self.fc(H.reshape((-1, H.shape[-1])))
        return y_hat


# 评价测试集
def evaluate_mae(data_iter, net, state=None, device=None):
    net.eval()
    if device is None:
        device = list(net.parameters())[0].device
    mae_sum, n = 0.0, 0

    with torch.no_grad():
        for x_enc, x_dec, y, normlizer in data_iter:
            # x[batch_size,time_step,features]  y[batch_size,time_step,1]
            x_enc.transpose_(0, 1)
            x_dec.transpose_(0, 1)
            y.transpose_(0, 1)
            x_enc = x_enc.to(device)
            x_dec = x_dec.to(device)
            y = y.to(device)

            y_hat = net(x_enc, x_dec)  # 输出的y_hat(时间步数*批量大小,1) state(D∗num_layers,N,H)
            mae_sum += (torch.abs(y_hat.reshape(-1) - y.reshape(-1)) * normlizer[1].repeat(15)).sum().cpu().item()
            n += y.numel()
    return mae_sum / n


def call_save_model(net, val_loss):
    if val_loss < net.min_loss:
        net.min_loss = val_loss
        torch.save(net.state_dict(), "model/LSTM.pth")
        print(f'call:{val_loss}')


# 训练模型
def train_net(net, train_iter, test_iter, optimizer, loss, device, epochs):
    net = net.to(device)
    test_mae_list = net.val_loss_list
    test_multi_step_loss_list = net.val_multi_step_loss_list  # 验证集每个epoch多步loss, 进行迭代多步
    train_mae_list = net.train_loss_list

    for epoch in range(1, epochs + 1):

        train_l_sum, train_mae_sum, n, start = 0.0, 0.0, 0, time.time()

        for x_enc, x_dec, y, normlizer in train_iter:
            # x[batch_size,time_step,features]  y[batch_size,time_step,1]
            x_enc.transpose_(0, 1)
            x_dec.transpose_(0, 1)
            y.transpose_(0, 1)
            x_enc = x_enc.to(device)
            x_dec = x_dec.to(device)
            y = y.to(device)

            net.train()
            y_hat = net(x_enc, x_dec)  # 输出的y_hat(时间步数*批量大小,1) state(D∗num_layers,N,H)
            l = loss(y_hat.reshape(-1), y.reshape(-1))

            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                train_l_sum += l.cpu().item()
                train_mae_sum += (torch.abs(y_hat.reshape(-1) - y.reshape(-1)) * normlizer[1].repeat(15)).sum().cpu().item()
                n += y.numel()

        # y_pred, y_true = IterMutilStep(X_val, y_val, net, horizon=15, device=device) # 迭代多步
        test_mae = evaluate_mae(test_iter, net, device)  # 均值MAE
        test_mae_list.append(test_mae)
        train_mae_list.append(train_l_sum / n)

        # 每次决定是否保存模型
        call_save_model(net, test_mae)
        # 展示
        print('epoch %d, loss %.4f, trian_mae %.3f, test_mae %.3f, time %.2f sec'
              % (epoch, train_l_sum / n, train_mae_sum / n, test_mae, time.time() - start))


def main():
    variables, NOx = get_data()
    print(variables.shape)
    Len = 1100  # 建模用数据长度
    st = 100
    ed = st+Len
    Y = NOx[st:ed].reshape(-1, 1)
    delays = [1,1,1,1,1,1,1,1,1,1,1,1,1]  # 共计13个变量，每个变量的迟延时间
    orders = [1,1,1,1,1,1,1,1,1,1,1,1,1]  # 共计13个变量，每个变量的输入阶数
    X = np.zeros((Len, sum(orders)))
    count = 0
    for i in range(13):
        delay = delays[i]
        order = orders[i]
        for j in range(order):
            X[:,count] = variables[st-delay-j:ed-delay-j,i]
            count += 1
    print(f'X维度{X.shape},Y维度{Y.shape}')
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    X_train, X_test, y_train, y_test = Split(X, Y, test_size=0.22)
    print(f'X_train.shape: {X_train.shape}, X_test.shape:{X_test.shape}')

    batch_size = 64
    encoder_len = 20
    decoder_len = 15
    stride = 1
    train_dl = RandomDataLoader(X_train, y_train, batch_size, encoder_len, decoder_len, stride)
    test_dl = RandomDataLoader(X_test, y_test, batch_size, encoder_len, decoder_len, stride)
    # for x_enc, x_dec, y, normlizer in train_dl:
    #     print(x_enc.shape, x_dec.shape, y.shape)
    #     print(x_dec[0, :, -1])
    #     print(x_enc[0, :, -1])
    #     print(y[0, :, -1])
    #     break
    # 保存训练集epoch_mae和验证集loss
    # 保存验证集loss最小的模型参数，用于验证性能，以及再训练
    device = torch.device('cpu')
    print('training on', device)
    loss = nn.L1Loss(reduction='sum')

    # 模型
    input_size = 13
    net = RNNModel(input_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    epochs = 200
    train_net(net, train_dl, test_dl, optimizer, loss, device, epochs)


if __name__ ==  "__main__":
    main()