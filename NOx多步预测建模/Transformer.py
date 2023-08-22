import torch
from torch import nn
from d2l import d2l_torch as d2l
from transformer_utils import TransformerEncoder as tfenc
from transformer_utils import TransformerDecoder as tfdec

def main():
    ffn = d2l.PositionWiseFFN(4, 4, 8)
    ffn.eval()
    print(ffn(torch.ones((2, 3, 4)))[0])

    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = d2l.EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()
    print(encoder_blk(X, valid_lens).shape)

    encoder = d2l.TransformerEncoder(
        200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
    encoder.eval()
    print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)

    decoder_blk = d2l.DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
    decoder_blk.eval()
    X = torch.ones((2, 100, 24), dtype=torch.float32)
    state = [encoder_blk(X, valid_lens), valid_lens, [None]]
    print(decoder_blk(X, state)[0].shape)



    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, 'cpu'
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

    encoder = d2l.TransformerEncoder(    # d2l.TransformerEncoder
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = d2l.TransformerDecoder(    # d2l.TransformerDecoder
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = d2l.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ',
              f'bleu {d2l.bleu(translation, fra, k=2):.3f}')

if __name__ == "__main__":
    main()