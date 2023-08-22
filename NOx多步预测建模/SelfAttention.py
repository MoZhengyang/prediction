import math
import torch
from torch import nn
from d2l import d2l_torch as d2l




def main():
    num_hiddens, num_heads = 100, 5
    # key_size, query_size, value_size, num_hiddens,
    attention = d2l.MultiHeadAttention(key_size=num_hiddens, query_size=num_hiddens, value_size=num_hiddens,
                                       num_hiddens=num_hiddens, num_heads=num_heads, dropout=0.5)
    print(attention.eval())

    batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    print(attention(X, X, X, valid_lens).shape)

if __name__ == "__main__":
    main()