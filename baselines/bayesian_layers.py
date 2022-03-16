"""
https://github.com/anndvision/quince/blob/main/quince/library/modules/variational.py
"""
import torch
from torch import distributions, nn


class Normal(nn.Module):
    def __init__(self,in_features, out_features,):
        super(Normal, self).__init__()
        self.mu = nn.Linear(in_features=in_features, out_features=out_features,bias=True,)
        sigma = nn.Linear( in_features=in_features,out_features=out_features,bias=True)
        self.sigma = nn.Sequential(sigma, nn.Softplus())

    def forward(self, inputs):
        return distributions.Normal(loc=self.mu(inputs), scale=self.sigma(inputs) + 1e-7)


class Categorical(nn.Module):
    def __init__(self,in_features, out_features,):
        super(Categorical, self).__init__()
        self.logits = nn.Linear(in_features=in_features, out_features=out_features,bias=True,)
        self.distribution = (distributions.Bernoulli if dim_output == 1 else distributions.Categorical)

    def forward(self, inputs):
        return self.distribution(logits=self.logits(inputs))
