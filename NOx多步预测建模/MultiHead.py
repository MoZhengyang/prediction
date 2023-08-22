import math
import torch
from torch import nn
from d2l import d2l_torch as d2l




def main():
    num_hiddens, num_heads = 100, 5
    attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
    attention.eval()

    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens =  6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(X, Y, Y, valid_lens).shape)

    multihead_attn = nn.MultiheadAttention(embed_dim=100, num_heads=5, batch_first=True)
    attn_output, attn_output_weights = multihead_attn(X, Y, Y)
    print(attn_output.shape)


if __name__ == "__main__":
    main()