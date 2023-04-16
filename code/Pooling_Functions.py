"""
This file contains all used pooling functions

created by Stephanie Henkel
"""

import torch
import numpy as np

import torch.nn as nn
from torch.nn.functional import softmax


def choose_pooling_function(function, input_dim):
    if function == "mean":
        return MeanPooling()
    elif function == "gated_attention":
        return GatedAttention(input_dim)
    elif function == "hopfield_pooling":
        return HopfieldPooling(input_dim)
    else:
        raise ValueError


class PoolingFunction(nn.Module):
    """
    This class implements the application of all pooling functions
    """
    def __init__(self):
        super(PoolingFunction, self).__init__()
        self.pooling_hidden_dim = 32

    def aggregate(self, x, bag_lengths):
        device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')
        x_pooled = torch.cat(list(map(self.forward, torch.split(x, bag_lengths))), dim=0).to(device=device)

        return x_pooled

    def forward(self, x):
        raise NotImplementedError


class MeanPooling(PoolingFunction):
    """
    This function implements the mean pooling function
    """
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=0, keepdim=True)


class GatedAttention(PoolingFunction):
    """
    This class implements the Gated Attention pooling function
    """
    def __init__(self, num_in):
        super(GatedAttention, self).__init__()

        self.attention_V = nn.Sequential(
            nn.Linear(num_in, self.pooling_hidden_dim),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(num_in, self.pooling_hidden_dim),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.pooling_hidden_dim, 1)

    def forward(self, x):
        a_v = self.attention_V(x)  # NxD
        a_u = self.attention_U(x)  # NxD
        a = self.attention_weights(a_v * a_u)  # element wise multiplication Nx1
        a = torch.transpose(a, 1, 0)  # 1xN
        a = softmax(a, dim=1)  # softmax over N

        m = torch.mm(a, x)  # 1 x num_features = num_in
        return m


class HopfieldPooling(PoolingFunction):
    """
    This class implements the Hopfield pooling function
    """
    def __init__(self, num_in):
        super(HopfieldPooling, self).__init__()
        self.d_model = self.pooling_hidden_dim
        self.num_heads = 1
        self.d_k = int(self.pooling_hidden_dim/self.num_heads)

        self.q = nn.Parameter(torch.Tensor(self.num_heads, 1, self.d_k))
        self.W_k = nn.Linear(num_in, self.d_model, bias=False)
        self.W_v = nn.Linear(num_in, num_in, bias=False)

        torch.nn.init.kaiming_normal_(self.q, mode='fan_in', nonlinearity='linear')
        torch.nn.init.kaiming_normal_(self.W_k.weight, mode='fan_in', nonlinearity='linear')
        torch.nn.init.kaiming_normal_(self.W_v.weight, mode='fan_in', nonlinearity='linear')

        self.beta = 1/np.sqrt(self.pooling_hidden_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        k = self.W_k(x).view(self.num_heads, -1, self.d_k)  # heads, num_state, d_k
        q = self.beta * self.q

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # (heads, 1, num_instances), multiplies for each head seperately
        attn_output_weights = softmax(attn_output_weights, dim=-1)

        # average attention weights over heads
        attn_output_weights = attn_output_weights.mean(dim=0)

        m = attn_output_weights @ self.W_v(x)

        return m


