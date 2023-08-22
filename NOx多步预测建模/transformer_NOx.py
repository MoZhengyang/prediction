import numpy as np
from torch import nn
import torch
import transformer_utils as tu
import data_utils as du


def main():
    data, NOx = du.get_nox_data()
    # np.save('NOx.npy', data)
    # data = np.load('NOx.npy')
    Len = 4000  # 建模用数据长度
    st = 1000  # 开始位置
    ed = st + Len
    Y = NOx[st:ed].reshape(-1, 1)
    delays = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 共计13个变量，每个变量的迟延时间
    orders = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 共计13个变量，每个变量的输入阶数
    # delays = [1,3,3,3,2,3,5,3,3,2,4,10,1]  # 共计13个变量，每个变量的迟延时间
    # orders = [3,5,4,5,6,5,5,1,5,6,6,3,4]    # 共计13个变量，每个变量的输入阶数
    X = np.zeros((Len, sum(orders)))
    count = 0
    for i in range(13):
        delay = delays[i]
        order = orders[i]
        for j in range(order):
            X[:, count] = data[st - delay - j:ed - delay - j, i]
            count += 1
    print(f'X维度{X.shape},Y维度{Y.shape}')

    x_train, x_test, y_train, y_test = du.split(X, Y, test_size=0.25)
    print(f'x_train.shape: {x_train.shape}, x_test.shape:{x_test.shape}')

    # 注：设置enc_time_step为15时，最后一个将送入dec用于预测接下来来的输出batch，enc_time_step-1，features
    # 相当于将enc_time_step设置为14
    batch_size = 32
    enc_time_step = 14
    dec_time_step = 20
    stride = 5  # 滑动窗口的步数
    train_dl = du.RandomDataLoader(x_train, y_train, batch_size, enc_time_step, dec_time_step, stride)
    test_dl = du.RandomDataLoader(x_test, y_test, batch_size, enc_time_step, dec_time_step, stride)
    # for x, _, _ in train_dl:
    #     print(x.shape)

    # 开始训练模型
    num_hiddens, num_layers, dropout = 16, 2, 0.1
    lr, num_epochs = 0.005, 10
    device = 'cpu'
    ffn_num_input, ffn_num_hiddens = 16, 16
    num_heads = 4
    feature_size, key_size, query_size, value_size = x_train.shape[1], 16, 16, 16
    norm_shape = [16]
    # 修改
    encoder = tu.TransformerEncoder_nox(
        feature_size, key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)

    decoder = tu.TransformerDecoder_nox(
        feature_size, key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)

    valid_lens = None
    net = tu.EncoderDecoder(encoder, decoder)
    net.apply(tu.xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.MSELoss(reduction='sum')
    net.train()
    timer = tu.Timer()
    for epoch in range(num_epochs):
        timer.start()
        metric = tu.Accumulator(2)  # 训练损失总和，词元数量
        for batch in train_dl:
            x_enc, x_dec, y = [x.to(device) for x in batch]
            y_hat, _ = net(x_enc, x_dec, valid_lens)

            l = loss(y_hat.reshape(-1), y.reshape(-1))
            l.backward()      # 损失函数的标量进行“反向传播”
            # grad_clipping(net, 1)
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                metric.add(l.item(), y.numel())
        print(f'epoch{epoch} loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
            f'tokens/sec on {str(device)}')


if __name__ == "__main__":
    main()