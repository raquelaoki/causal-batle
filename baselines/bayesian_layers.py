"""
https://github.com/anndvision/quince/blob/main/quince/library/modules/variational.py
"""
import numpy as np
import torch

from torch import distributions, nn
from sklearn.metrics import mean_squared_error


class Normal(nn.Module):
    def __init__(self, in_features, out_features, ):
        super(Normal, self).__init__()
        self.mu = nn.Linear(in_features=in_features, out_features=out_features, bias=True, )
        self.sigma = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        self.softplus = nn.Softplus()

    def forward(self, inputs):
        return distributions.Normal(loc=self.mu(inputs),
                                    scale=self.softplus(self.sigma(inputs) + 1e-7))


class Categorical(nn.Module):
    def __init__(self, in_features, out_features, ):
        super(Categorical, self).__init__()
        self.logits = nn.Linear(in_features=in_features, out_features=out_features, bias=True, )
        self.distribution = distributions.Bernoulli
        self.in_features = in_features
        # self.distribution = (distributions.Bernoulli if out_features == 1 else distributions.Categorical)

    def forward(self, inputs):
        return self.distribution(logits=self.logits(inputs))


def criterion_function_dragonnet_t(batch, predictions, device='cpu'):
    t_predictions = predictions['t']
    t_obs = batch[2].to(device)
    return -t_predictions.log_prob(t_obs).mean()


def criterion_function_dragonnet_y(batch, predictions, device='cpu'):
    y0_predictions, y1_predictions = predictions['y0'], predictions['y1']
    y_obs = batch[1].to(device)
    t_obs = batch[2].to(device)

    loss_y0_babch = -y0_predictions.log_prob(y_obs[t_obs == 0]).mean()
    loss_y1_babch = -y1_predictions.log_prob(y_obs[t_obs == 1]).mean()

    return +loss_y0_babch + loss_y1_babch


def metric_function_dragonnet_t(batch, predictions):
    dif = batch[2] - predictions['t'].mean.cpu().detach().numpy()
    return np.abs(dif).mean()


def metric_function_dragonnet_y(batch, predictions):
    pred0 = predictions['y0'].sample([1,1]).reshape(-1,1).cpu().detach().numpy()
    pred1 = predictions['y1'].sample([1,1]).reshape(-1,1).cpu().detach().numpy()
    t_obs = batch[2].cpu().detach().numpy()
    y_pred = pred0 * (1 - t_obs) + pred1 * t_obs
    return mean_squared_error(batch[1], y_pred)

