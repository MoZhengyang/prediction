import pandas as pd
import numpy as np
import random
import torch
import time
from torch import nn


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
    NOx = (NOx - 200) / 450
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
    return variables,NOx


def seq_iter(X, Y, batch_size, num_steps):
    # LSTM数据输入构造，X[batch,time,feature] Y[batch,time,output]
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    batch_num = (len(X) - offset) // batch_size  # batch数量
    num_samples = batch_num * batch_size  # 总样本数量
    Xs = X[offset:offset + num_samples, :]
    Ys = Y[offset:offset + num_samples, :]
    Xs, Ys = Xs.reshape(batch_size, batch_num, Xs.shape[1]), Ys.reshape(batch_size, batch_num, 1)

    for i in range(0, (batch_num//num_steps)*num_steps, num_steps):
        xl = Xs[:, i: i + num_steps]
        yl = Ys[:, i: i + num_steps]
        yield xl, yl


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, feature_size):
        super().__init__()
        self.rnn = rnn_layer  # 外部输入的rnn_layer,可以在外部定义为RNN, GRU, LSTM
        self.feature_size = feature_size
        self.num_hiddens = self.rnn.hidden_size

        # 如果RNN是双向的，num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
        else:
            self.num_directions = 2

        self.fc = nn.Sequential(
            nn.Linear(self.num_hiddens * self.num_directions, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Linear(4, 1))

    def forward(self, x, state):
        # LSTM state为元组形式，包括(h0,c0)
        # x为[时间步, batch_size, feature_szie]
        # RNN 返回为(L,N,D*H_out)
        H, state = self.rnn(x, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,1)。
        y_hat = self.fc(H.reshape((-1, H.shape[-1])))
        return y_hat, state

    # 获得初始状态参数
    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


def mae(y_true, y_pred):
    metric = torch.abs(y_true - y_pred)
    return metric


# 评价测试集
def evaluate_mae(data_iter, net, device=None):
    if device is None:
        device = list(net.parameters())[0].device
    ame_sum, n = 0.0, 0

    with torch.no_grad():
        for x, y in data_iter:
            x.transpose_(0, 1)
            y.transpose_(0, 1)
            x.to(device)
            y.to(device)

            state = None
            # 获得初始状态，随机抽样需要初始化data，否则就不初始化
            if state is None:
                state = net.begin_state(batch_size=x.shape[1], device=device)
            else:
                if isinstance(net, nn.Module) and not isinstance(state, tuple):
                    # state对于nn.GRU是个张量
                    state.detach_()
                else:
                    # state对于nn.LSTM是元组
                    for s in state:
                        s.detach_()
            net.eval()
            y_hat, state = net(x, state)
            ame_sum += mae(y_hat.reshape(-1), y.reshape(-1)).sum().cpu().item()
            n += len(y)
    return ame_sum / n


# 训练模型
def train_net(net, train_iter, test_iter, optimizer, loss, device, epochs):
    net = net.to(device)
    print('training on', device)
    batch_count = 0  # 记录总样本数量
    for epoch in range(1, epochs + 1):

        state = None
        train_l_sum, train_mae_sum, n, start = 0.0, 0.0, 0, time.time()

        # 更新虚学习率
        if epoch % 50 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.85

        for x, y in train_iter:
            # x[batch_size,time_step,features]  y[batch_size,time_step,1]
            x.transpose_(0, 1)
            y.transpose_(0, 1)
            x = x.to(device)
            y = y.to(device)

            # 获得初始状态，随机抽样需要初始化data，否则就不初始化
            if state is None:
                state = net.begin_state(batch_size=x.shape[1], device=device)
            else:
                if isinstance(net, nn.Module) and not isinstance(state, tuple):
                    # state对于nn.GRU是个张量
                    state.detach_()
                else:
                    # state对于nn.LSTM是元组
                    for s in state:
                        s.detach_()

            net.train()
            y_hat, state = net(x, state)  # 输出的y_hat(时间步数*批量大小,1) state(D∗num_layers,N,H)
            l = loss(y_hat.reshape(-1), y.reshape(-1))

            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_l_sum += l.cpu().item()
            train_mae_sum += torch.abs(y_hat.view(-1) - y).sum().cpu().item()
            n += y.numel()
            batch_count += 1
        test_mae = evaluate_mae(test_iter, net, device)

        print('epoch %d, loss %.4f, trian_mae %.3f, test_mae %.3f, time %.2f sec'
              % (epoch, train_l_sum, 450 * train_mae_sum / n, 450 * test_mae, time.time() - start))


class SeqDataLoader:
    def __init__(self, X_train, y_train, batch_size, num_steps, seq_iter):
        self.X_train, self.y_train = X_train, y_train
        self.batch_size, self.num_steps = batch_size, num_steps
        self.seq_iter = seq_iter

    def __iter__(self):
        return self.seq_iter(self.X_train, self.y_train, self.batch_size, self.num_steps)


def main():
    variables, NOx = load_data()
    Len = 4000
    st = 20
    ed = st + Len
    Y = NOx[st:ed].reshape(-1, 1)
    delays = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 共计13个变量，每个变量的迟延时间
    orders = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]    # 共计13个变量，每个变量的输入阶数
    # delays = [1, 3, 3, 3, 2, 3, 5, 3, 3, 2, 4, 10, 1]  # 共计13个变量，每个变量的迟延时间
    # orders = [3, 5, 4, 5, 6, 5, 5, 1, 5, 6, 6, 3, 4]  # 共计13个变量，每个变量的输入阶数
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
    # X = X.unsqueeze(axis=1)
    X_train, X_test, y_train, y_test = Split(X, Y, test_size=0.25)
    print(f'X_train.shape: {X_train.shape}, X_test.shape:{X_test.shape}')

    batch_size = 32
    num_steps = 16
    train_dl = SeqDataLoader(X_train, y_train, batch_size, num_steps, seq_iter)
    test_dl = SeqDataLoader(X_test, y_test, batch_size, num_steps, seq_iter)

    input_size = 13
    lstm_layer = nn.LSTM(input_size=input_size, hidden_size=16, num_layers=1)
    net = RNNModel(lstm_layer, input_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
    epochs = 10
    train_net(net, train_dl, test_dl, optimizer, loss, device, epochs)


if __name__ == "__main__":
    main()
