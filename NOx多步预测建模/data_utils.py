import pandas as pd
import numpy as np
import torch
import time
from torch import nn
import random


# 该函数面对不同数据需要重写
def get_nox_data():
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
    NOx_max = 650
    NOx_min = 200
    NOx = (NOx - NOx_min) / (NOx_max - NOx_min)
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
    print('(samples,features)', variables.shape)
    return variables, NOx


def split(X, Y, test_size=0.25):
    train_num = int(len(X)*(1-test_size))
    X_train = X[:train_num, ]
    y_train = Y[:train_num, ]
    X_test = X[train_num:, ]
    y_test = Y[train_num:, ]
    return X_train, X_test, y_train, y_test


def random_iter(x_train, y_train, batch_size, enc_time_step, dec_time_step, stride):
    offset = random.randint(0, stride)  # 随机开始便宜
    initial_indices = (np.arange(offset, len(x_train) - enc_time_step - dec_time_step, stride))  # 样本开始索引
    random.shuffle(initial_indices)  # 随机打断样本开始索引

    # 注意的是，y_train值已经前向推进了一步
    def data(pos, sign):
        if sign == 1:
            return x_train[pos:pos + enc_time_step]
        if sign == 2:
            # dec中氧量保持常数
            x_dec = x_train[pos + enc_time_step:pos + enc_time_step + dec_time_step]
            # 第一维保持不变
            x_dec[1:, 0] = x_dec[0, 0]
            # dec维度变为11里面只有控制量
            #x_dec = x_train[pos + enc_time_step:pos + enc_time_step + dec_time_step, 1:-1]
            return x_dec
        if sign == 3:
            return y_train[pos + enc_time_step:pos + enc_time_step + dec_time_step]

    num_batches = len(initial_indices) // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        x_enc = np.array([data(j, 1) for j in initial_indices_per_batch])
        x_dec = np.array([data(j, 2) for j in initial_indices_per_batch])
        y = np.array([data(j, 3) for j in initial_indices_per_batch])
        yield torch.tensor(x_enc, dtype=torch.float32), torch.tensor(x_dec, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class RandomDataLoader:
    def __init__(self, x_train, y_train, batch_size, enc_time_step, dec_time_step, stride):
        self.x_train, self.y_train = x_train, y_train
        self.batch_size = batch_size
        self.enc_time_step, self.dec_time_step, self.stride = enc_time_step, dec_time_step, stride

    def __iter__(self):
        return random_iter(self.x_train, self.y_train, self.batch_size,
                                self.enc_time_step, self.dec_time_step, self.stride)