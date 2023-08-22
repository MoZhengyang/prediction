from torch import nn
import torch
import transformer_utils as tf
from d2l.d2l_torch import load_data_nmt, train_seq2seq


def main():
    '''
    # 测试位置编码
    max_len = 1000
    num_hiddens = 512
    P = torch.zeros((1, max_len, num_hiddens))
    X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
        torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
    P[:, :, 0::2] = torch.sin(X)
    P[:, :, 1::2] = torch.cos(X)

    # 测试基于位置的FNN网络
    ffn = tf.PositionWiseFFN(4, 4, 8)
    ffn.eval()
    print('FNN', ffn(torch.ones((2, 3, 4))).shape)

    # LayerNorm测试
    tensor = torch.FloatTensor([[1, 2, 4, 1],
                                [6, 3, 2, 4],
                                [2, 4, 6, 1]])
    layer_norm = nn.LayerNorm(4)
    layer_out = layer_norm(tensor)

    # 测试缩放点积注意力
    queries, keys = torch.normal(0, 1, (2, 1, 2)), torch.ones((2, 10, 2))
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    valid_lens = torch.tensor([2, 1])
    attention = tf.DotProductAttention(dropout=0)
    attention.eval()
    s = attention(queries, keys, values, valid_lens)

    # 测试多头注意力
    num_hiddens, num_heads = 100, 5
    attention = tf.MultiHeadAttention(key_size=num_hiddens, query_size=num_hiddens,
                                      value_size=num_hiddens, num_hiddens=num_hiddens,
                                      num_heads=num_heads, dropout=0.5)
    print(attention)
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([1, 2, 3, 4]).repeat(batch_size, 1)
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(X, Y, Y, valid_lens).shape)

    # 测试encoderBlock
    x = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = tf.EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()
    print(encoder_blk(x, valid_lens).shape)

    # 测试decoderBlock
    decoder_blk = tf.DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
    decoder_blk.eval()
    X = torch.ones((2, 100, 24), dtype=torch.float32)
    # 注意state的定义
    state = [encoder_blk(X), None, [None]]
    print(decoder_blk(X, state)[0].shape)
    '''
    # 多个Block形成transformer TransformerEncoder TransformerDecoder
    # TransformerEncoder需要改造成预测使用的形式
    # 测试TransformerEncoder
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 10, tf.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)

    # encoder = tf.TransformerEncoder(
    #     len(src_vocab), key_size, query_size, value_size, num_hiddens,
    #     norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    #     num_layers, dropout)
    # for batch in train_iter:
    #     X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
    #     enc_out = encoder(X, X_valid_len)
    #     print(enc_out.shape)
    # TransformerDecoder需要改造成多步预测形式
    encoder = tf.TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = tf.TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)

    # for batch in train_iter:
    #     X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
    #     bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
    #     dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
    #
    #     enc_out = encoder(X, X_valid_len)
    #     # decoder初始化state[2][i]=None
    #     dec_state = decoder.init_state(enc_out, X_valid_len)
    #     dec_out, state = decoder(dec_input, dec_state)
    #     print(dec_out.shape)

    # 构造Transformer
    net = tf.EncoderDecoder(encoder, decoder)
    # 构造训练函数并用语言数据集验证
    tf.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 预测文本
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, _ = tf.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ',
              f'bleu {tf.bleu(translation, fra, k=2):.3f}')


if __name__ == "__main__":
    main()