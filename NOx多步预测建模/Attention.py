import torch
from torch import nn
from d2l import d2l_torch as d2l


def main():
    encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                 num_layers=2)
    encoder.eval()
    decoder = d2l.Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                      num_layers=2)
    decoder.eval()
    X = torch.zeros((4, 7), dtype=torch.float32)  # (batch_size,num_steps)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
    print(output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape)
    


if __name__ == "__main__":
    main()