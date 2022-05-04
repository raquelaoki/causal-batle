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
        #self.threshold = nn.Threshold(0.000001, 0.000001)
        self.softplus = nn.Softplus()

    def forward(self, inputs):
        return distributions.Normal(loc=self.mu(inputs),
                                    scale=self.softplus(self.sigma(inputs))+0.0000001)


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
    pred0 = predictions['y0'].sample([1, 1]).reshape(-1, 1).cpu().detach().numpy()
    pred1 = predictions['y1'].sample([1, 1]).reshape(-1, 1).cpu().detach().numpy()
    t_obs = batch[2].cpu().detach().numpy()
    y_pred = pred0 * (1 - t_obs) + pred1 * t_obs
    return mean_squared_error(batch[1], y_pred)


def criterion_function_dragonnet_targeted(batch, predictions, device='cpu'):
    y_obs = batch[1].to(device)
    t_obs = batch[2].to(device)
    t_predictions = predictions['t'].sample([1, 1]).reshape(-1, 1)  # .cpu().detach().numpy()
    y0_predictions = predictions['y0'].sample([1, 1]).reshape(-1, 1)  # .cpu().detach().numpy()
    y1_predictions = predictions['y1'].sample([1, 1]).reshape(-1, 1)  # .cpu().detach().numpy()
    epsilon = predictions['epsilon']
    y_pred = y0_predictions * (1 - t_obs) + y1_predictions * t_obs
    criterion = TargetedLoss()
    return criterion(y_obs=y_obs, y_pred=y_pred, t_obs=t_obs, t_pred=t_predictions, epsilon=epsilon)


class TargetedLoss(nn.Module):
    """
    Reference: (https://arxiv.org/pdf/1906.02120.pdf)
    """

    def __init__(self):
        super(TargetedLoss, self).__init__()

    def forward(self, y_obs, y_pred, t_obs, t_pred, epsilon):
        t_pred = (t_pred + 0.01) / 1.02
        t1 = torch.div(t_obs, t_pred)
        t0 = torch.div(torch.sub(1, t_obs), torch.sub(1, t_pred))
        epsilon = epsilon.reshape(-1, 1)
        t = torch.mul(torch.sub(t1, t0), epsilon)
        # epislon == 0 -> t is not used and result should be equal to rmse error
        pred = torch.add(y_pred, t)
        loss = torch.sub(y_obs, pred)
        loss = torch.pow(loss, 2)
        return torch.mean(loss)
