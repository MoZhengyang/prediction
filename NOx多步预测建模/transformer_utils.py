import sys
import collections
from collections import defaultdict
from matplotlib import pyplot as plt
from IPython import display
import math
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
import numpy as np
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
import requests


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络
    ffn_num_input：模型特征维度，d_model
    ffn_num_hidden：中间隐藏层，大小随意
    ffn_num_outputs：需保证输入输出维度相同，因此设置为模型特征维度，d_model
    模型中可加入dropout，可参见transformer原码
    """
    def __init__(self, ffn_num_input, ffn_num_hidden, ffn_num_outputs,
                 **kwargs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hidden)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hidden, ffn_num_outputs)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化  residual and norm
    normalized_shape：输入LayerNorm的维度
    dropout：MA（多头attention）后的结果和FFN后的结果先dropout，再residual Add，再norm
    """
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


# def sequence_mask(b, valid_lens, value=-1e6):
#     """
#     :param b: 被拉成了（b*q_num,q_d）
#     :param valid_lens: (b*q_num,）
#     :param value: 默认很大的值
#     :return: sequence_masked scores
#     """  # 第一个batch的第一个头的query个元素
#     for i in range(len(valid_lens)):
#         b[i, valid_lens[i]:] = value
#     return b


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    #     :param b: 被拉成了（b*q_num,q_d）
    #     :param valid_lens: (b*q_num,）
    #     :param value: 默认很大的值
    #     :return: sequence_masked scores
    # 第一个batch的第一个头的query个元素
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X



def masked_softmax(scores, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作
    scores: 采用点积注意力计算出的得分 (batch_size, query_num, k_num) 对k_num维度先masked，再softmax
    valid_len：如dim==1，则valid_len仅代表batch_size的维度，需要对其进行repeat_interleave，变成batch_size*query_num
    长度，如dim==2，则为(batch_size, query_num)
    该masked函数仅执行sequence_mask
    """
    if valid_lens is None:
        return nn.functional.softmax(scores, dim=-1)
    else:
        shape = scores.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        scores = sequence_mask(scores.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(scores.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """缩放点积注意力
    dropout: 对注意力dropout
    """
    def __init__(self, dropout, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        """
        :param queries:
        :param keys:
        :param values:
        :param valid_lens:
        :return: (batch_size, query_num, d_v)
        """
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        # scores的维度:(batch_size, query_num, k_num)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """ 多头注意力机制
    key_size: key的特征维度，即最后一维的维度
    query_size, value_size与之类似，
    注意： query key value在原文中的初始特征维度都为d_model 512,是一个很高的维度。
    多头注意力机制中，将query key value都投影到一个较低纬度空间（d_model//head）
    Query@W_q(d_model, d/h)   key@W_k(d_model, d/h)   value@W_v(d_model, d/h)
    利用ScaledDotAttention计算出Attention后加权Values（batch， queries， d/h）记为head_i，这里的
    queries即是time_step
    最后concat(head_i) （batch， queries（T）， d/h*h）-> HEAD，  HEAD@Wo（d_model，d_model）得到最后输出
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """
        # queries，keys，values的形状: (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状: (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状: (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        """
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class EncoderBlock(nn.Module):
    """
    :param key_size:query_size:value_size: 其最后一维维度，特征维度，升维成d_model
    :param num_hiddens: d_model
    :param norm_shape: Layer_norm 最后两维（T序列长度，d_model）
    :param ffn_num_input: 位置前馈网络参数
    :param ffn_num_hiddens: 位置前馈网络参数
    :param num_heads:
    :param dropout:
    :param use_bias:
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout=0.1, use_bias=False, **kwargs):
        super().__init__()
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens=None):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class DecoderBlock(nn.Module):
    """解码器中第i个块,参数和encoder模块基本相同"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        """
        :param X: 训练时候，X是（batch, t, features）, 预测时候 X是（1, 1, features）
        :param state[encoder_output, enc_valid_lens, [decoder各层的key_values]]  state[2]预测时需使用
        :return:
        """
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None :   # 训练的时候state[2][self.i]一直为None
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)  # 预测时候，保存当前次的前一层的输出，用作key-value,attention1使用
        state[2][self.i] = key_values
        # 训练模式下的dec_valid_lens和预测模式下的dec_valid_lens
        # dec_valid_lens的维度为(batch_size,num_steps)，训练模式下对每个batch，为[1,2,...,num_steps]
        # 预测模式下，dec_valid_lens为None，不做masked，state保存了之前所有时间步的状态。
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # 训练模式下，key_values=X，dec_valid_lens为(batch_size,num_steps)，squence_masked self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力，enc_outputs的维度(batch_size, num_steps(scr的t), num_hiddens(d_model))
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X, horizon=-1):
        if horizon==-1:
            # X维度[b, t , d_model]  # P的维度[1, 1000, d_model]
            X = X + self.P[:, :X.shape[1], :].to(X.device)
        else:
            X = X + self.P[:, horizon, :].to(X.device)
        return self.dropout(X)


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        """
        :param vocab_size: 机器翻译中，源字典（包含"BOS","END"）的大小，也即是做词元 one_hot编码之后的维度
        预测时，vocab_size定义为特征维度
        :param key_size: Encoder中的key特征维度，统一转化为d_model(num_hiddens)
        :param query_size: Encoder中的query特征维度，统一转化为d_model(num_hiddens)
        :param value_size: Encoder中的value特征维度，统一转化为d_model(num_hiddens)
        :param num_hiddens: d_model
        :param norm_shape: Layer_norm一般为最后两维
        :param ffn_num_input: FFN输入和输出 d_model
        :param ffn_num_hiddens: FFN隐藏
        :param num_heads:
        :param num_layers: Encoder层数
        :param dropout:
        :param use_bias: 计算Attention时，是否bias
        :param kwargs:
        """
        super().__init__()
        self.num_hiddens = num_hiddens  # d_model
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # self.linear = nn.Linear(vocab_size, num_hiddens)  # 先线性变换到高维空间 再位置编码
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("encoder_block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，因此嵌入值乘以嵌入维度的平方根进行缩放，然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            # 保存每层block的attention_weights  masked_softmax(scores, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        """
        :param vocab_size:
        :param key_size:
        :param query_size:
        :param value_size:
        :param num_hiddens:
        :param norm_shape:
        :param ffn_num_input:
        :param ffn_num_hiddens:
        :param num_heads:
        :param num_layers:
        :param dropout:
        :param kwargs:
        """
        super().__init__()
        self._attention_weights = None
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("decoder_block" + str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        """
        :param X:
        :param state: state[encoder_output, enc_valid_lens, [decoder各层的key_values]]  state[2]预测时需使用
        state[2]训练阶段要保持为None
        :return: 预测结果和state，注意state预测阶段很重要，训练阶段无用
        """
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


# def xavier_init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.xavier_uniform_(m.weight)
#     if type(m) == nn.GRU:
#         for param in m._flat_weights_names:
#             if "weight" in param:
#                 nn.init.xavier_uniform_(m._parameters[param])


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super().forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated times."""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """Sum a list of numbers over time."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            # grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        print(f'epoch{epoch} loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
            f'tokens/sec on {str(device)}')


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """
    序列到序列模型的预测
    :param net:
    :param src_sentence: 待翻译的句子
    :param src_vocab: 源词库
    :param tgt_vocab: 目标词库
    :param num_steps: 源scr时间步数，预测步数
    :param device:
    :param save_attention_weights:
    :return:
    """
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]  # 获得scr词对应的index[9, 4, 3]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])   # padding后作为encoder输入
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)  # shape为(1,10) 1个batch
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(  # 构造decoder输入"bos"为开始 # shape为(1,1) 1个batch，1个time_step,预测下一个词元，且state保存在了state中
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)  # 预测时，每次保留dec_state,
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq



class TransformerEncoder_nox(nn.Module):
    """Transformer编码器"""
    def __init__(self, feature_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        """
        :param feature_size: 每一个time_step的特征维度
        :param key_size: Encoder中的key特征维度，统一转化为d_model(num_hiddens)
        :param query_size: Encoder中的query特征维度，统一转化为d_model(num_hiddens)
        :param value_size: Encoder中的value特征维度，统一转化为d_model(num_hiddens)
        :param num_hiddens: d_model
        :param norm_shape: Layer_norm一般为最后两维
        :param ffn_num_input: FFN输入和输出 d_model
        :param ffn_num_hiddens: FFN隐藏
        :param num_heads:
        :param num_layers: Encoder层数
        :param dropout:
        :param use_bias: 计算Attention时，是否bias
        :param kwargs:
        """
        super().__init__()
        self.num_hiddens = num_hiddens  # d_model
        self.linear = nn.Linear(feature_size, num_hiddens)  # 先线性变换到高维空间 再位置编码
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("encoder_block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，因此嵌入值乘以嵌入维度的平方根进行缩放，然后再与位置编码相加。
        X = self.pos_encoding(self.linear(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            # 保存每层block的attention_weights  masked_softmax(scores, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class TransformerDecoder_nox(nn.Module):
    """Transformer解码器"""
    def __init__(self, feature_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        """
        :param feature_size:
        :param key_size:
        :param query_size:
        :param value_size:
        :param num_hiddens:
        :param norm_shape:
        :param ffn_num_input:
        :param ffn_num_hiddens:
        :param num_heads:
        :param num_layers:
        :param dropout:
        :param kwargs:
        """
        super().__init__()
        self._attention_weights = None
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.linear = nn.Linear(feature_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("decoder_block" + str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, i))
        self.dense1 = nn.Linear(num_hiddens, num_hiddens//2)
        self.dense2 = nn.Linear(num_hiddens//2, 1)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state, horizon=-1):
        """
        :param X:
        :param state: state[encoder_output, enc_valid_lens, [decoder各层的key_values]]  state[2]预测时需使用
        state[2]训练阶段要保持为None
        horizon: 第多少步预测，用于决定位置编码
        :return: 预测结果和state，注意state预测阶段很重要，训练阶段无用
        """
        X = self.pos_encoding(self.linear(X) * math.sqrt(self.num_hiddens), horizon)
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        X = self.dense1(X)
        X = self.dense2(X)
        return X, state

    @property
    def attention_weights(self):
        return self._attention_weights


def train_nox(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """
    NOx时间序列多步预测使用的训练函数
    """
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            # grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        print(f'epoch{epoch} loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
            f'tokens/sec on {str(device)}')


