import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Local Imports
import baselines.bayesian_layers as bl
import baselines.dragonnet as dragonnet
import helper_tensorboard as ht
import utils

logger = logging.getLogger(__name__)


class causal_batle(nn.Module):
    """Causal Batle.
    To adopt alternative backbone models, update here.
    Input: basic architecture, n_covariates (int).
    Return: nn.Module model.
    """

    def __init__(self, n_covariates, dropout_p=0.5, units1=200, units2=100, units3=1, is_Image=False):
        super().__init__()
        self.backbone = dragonnet.dragonnet(n_covariates=n_covariates, type_original=False,
                                            units1=units1, units2=units2, units3=units3,
                                            use_dropout=True, dropout_p=dropout_p,
                                            is_Image=is_Image)

        # Update the head for our causal-batle proposed methodlogy
        self.backbone.dragonnet_head = causal_batle_head(units1=units1, units2=units2,
                                                         units3=units3, dropout_p=dropout_p,
                                                         n_covariates=n_covariates,
                                                         is_Image=is_Image)

    def forward(self, inputs):
        return self.backbone(inputs)


class causal_batle_head(nn.Module):
    """ Causal Batle Head.
    Used inside causal_batle, it attaches to the backbone architecture.
    """

    def __init__(self, n_covariates, units1=200, units2=100, units3=1,
                 dropout_p=0.5, is_Image=False):
        super(causal_batle_head, self).__init__()
        self.units1 = units1
        self.units2 = units2
        self.units3 = units3
        self.dropout_p = dropout_p
        self.n_covariates = n_covariates
        self.head_layer2_1_0 = nn.Linear(in_features=self.units1, out_features=self.units2)
        self.head_layer2_2_0 = nn.Linear(in_features=self.units2, out_features=self.units2)

        self.head_layer2_1_1 = nn.Linear(in_features=self.units1, out_features=self.units2)
        self.head_layer2_2_1 = nn.Linear(in_features=self.units2, out_features=self.units2)

        # Activation functions.
        self.elu = nn.ELU(alpha=0.25)
        self.tahn = nn.Tanh()
        self.dropout = nn.Dropout(p=self.dropout_p)

        # Returns outcome density
        self.outcome_layer_0 = bl.Normal(in_features=self.units2, out_features=self.units3)
        self.outcome_layer_1 = bl.Normal(in_features=self.units2, out_features=self.units3)
        self.t_predictions = bl.Categorical(in_features=self.units1, out_features=self.units3)

        # Discriminator
        self.sigmoid = nn.Sigmoid()
        self.head_layer_d = nn.Linear(in_features=self.units1, out_features=self.units2)
        self.d_predictions = nn.Linear(in_features=self.units2, out_features=1)

        # Reconstruction
        self.decoder_layer = decoder(n_covariates=self.n_covariates,
                                     units1=self.units1,
                                     is_Image=is_Image)

        # epsilon
        self.epsilon_weight = nn.Parameter(torch.rand([1]), requires_grad=True)
        self.t_weight = nn.Parameter(torch.tensor([2.0]), requires_grad=True)

    def forward(self, inputs):
        # Inputs = backbone output
        # Treatment specific - first layer.
        y0_hidden = self.elu(self.head_layer2_1_0(inputs))
        y1_hidden = self.elu(self.head_layer2_1_1(inputs))

        # Treatment specific - second layer.
        y0_hidden = self.elu(self.head_layer2_2_0(self.dropout(y0_hidden)))
        y1_hidden = self.elu(self.head_layer2_2_1(self.dropout(y1_hidden)))

        # Treatment specific - third layer.
        y0_predictions = self.outcome_layer_0(self.dropout(y0_hidden))
        y1_predictions = self.outcome_layer_1(self.dropout(y1_hidden))

        t_predictions = self.t_predictions(self.dropout(inputs))

        d_hidden = self.head_layer_d(inputs)
        d_predictions = self.sigmoid(self.d_predictions(d_hidden))

        x_reconstruction = self.decoder_layer(inputs)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        epsilon = self.epsilon_weight * torch.ones(inputs.shape[0]).to(device)
        t_weight = self.t_weight * torch.ones(inputs.shape[0]).to(device)

        predictions = {'y0': y0_predictions,
                       'y1': y1_predictions,
                       't': t_predictions,
                       'd': d_predictions,
                       'xr': x_reconstruction,
                       'epsilon': epsilon,
                       't_weight': t_weight}
        return predictions


class decoder(nn.Module):
    """Simple Autoenconder"""

    def __init__(self, n_covariates, units1, is_Image):
        super(decoder, self).__init__()
        self.is_Image = is_Image
        self.decoder_layer1_1 = nn.Linear(in_features=units1, out_features=units1)
        self.decoder_layer1_2 = nn.Linear(in_features=units1, out_features=units1)
        if self.is_Image:
            self.decoder_layer1_3 = nn.Linear(in_features=units1, out_features=28 * 28)
        else:
            self.decoder_layer1_3 = nn.Linear(in_features=units1, out_features=n_covariates)
        self.elu = nn.ELU()

    def forward(self, inputs):
        # print('inputs shape',inputs.shape, ' expected', self.units2)
        reconstruction = self.elu(self.decoder_layer1_1(inputs))
        reconstruction = self.elu(self.decoder_layer1_2(reconstruction))
        reconstruction = self.decoder_layer1_3(reconstruction)
        if self.is_Image:
            reconstruction = reconstruction.reshape(-1, 1, 28, 28)
        return reconstruction


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time.
    Used for MC-dropout.
    Reference: https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def fit_causal_batle(epochs,
                     model,
                     loader_train,
                     loader_test,
                     optimizer,
                     criterion,
                     metric_functions,
                     use_validation,
                     use_tensorboard,
                     device,
                     loader_val=None,
                     alpha=[],
                     path_logger='',
                     config_name='',
                     home_dir='',
                     use_validation_best=False):
    """
        Fit implementation: Contain epochs and batch iterator, optimization steps, and eval.
    :param home_dir:
    :param config_name:
    :param path_logger:
    :param use_tensorboard:
    :param loader_test:
    :param epochs: integer
    :param model: nn.Module
    :param loader_train: DataLoader
    :param optimizer: torch.optim
    :param criterion: List of criterions
    :param metric_functions:
    :param use_validation: Bool
    :param device: torch.device
    :param loader_val: DataLoader (Optional)
    :param alpha: alphas to balance losses, torch.Tensor

    :return: model: nn.Module after model.train() over all epochs
    :return: loss: dictionary with all losses calculated
    :return: metrics: disctionary with all metrics calcualted
    """

    logger.debug('...starting')

    if use_tensorboard:
        writer_tensorboard = ht.TensorboardWriter(path_logger=path_logger,
                                                  config_name=config_name,
                                                  home_dir=home_dir)

    if len(alpha) == 0:
        alpha = torch.ones(len(criterion))
    elif not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha)

    assert len(alpha) == len(criterion), 'Loss weights do not match number of losses'

    loss_train_t, metric_train_t = np.zeros(epochs), np.zeros(epochs)  # _t: propensity score loss
    loss_train_y, metric_train_y = np.zeros(epochs), np.zeros(epochs)  # _y: outcome loss
    loss_train_d, metric_train_d = np.zeros(epochs), np.zeros(epochs)  # _d: discriminator loss
    loss_train_r, metric_train_r = np.zeros(epochs), np.zeros(epochs)  # _r: reconstruction loss
    loss_train_a, loss_train_tg = np.zeros(epochs), np.zeros(epochs)  # _a: adversarial loss

    loss_val_t, metric_val_t = np.zeros(epochs), np.zeros(epochs)
    loss_val_y, metric_val_y = np.zeros(epochs), np.zeros(epochs)
    loss_val_d, metric_val_d = np.zeros(epochs), np.zeros(epochs)
    loss_val_r, metric_val_r = np.zeros(epochs), np.zeros(epochs)
    loss_val_a, loss_val_tg = np.zeros(epochs), np.zeros(epochs)

    second_update = ['backbone.dragonnet_head.head_layer2_1_0',
                     'backbone.dragonnet_head.head_layer2_2_0',
                     'backbone.dragonnet_head.head_layer2_1_1',
                     'backbone.dragonnet_head.head_layer2_2_1',
                     'backbone.dragonnet_head.outcome_layer_0.mu',
                     'backbone.dragonnet_head.outcome_layer_0.sigma',
                     'backbone.dragonnet_head.outcome_layer_1.mu',
                     'backbone.dragonnet_head.outcome_layer_1.sigma',
                     'backbone.dragonnet_head.t_predictions.logits',
                     'backbone.dragonnet_head.head_layer_d',
                     'backbone.dragonnet_head.d_predictions',
                     'backbone.dragonnet_head.decoder_layer.decoder_layer1_1',
                     'backbone.dragonnet_head.decoder_layer.decoder_layer1_2',
                     'backbone.dragonnet_head.decoder_layer.decoder_layer1_3',
                     ]

    if use_validation_best:
        best_loss = 999
        best_epoch = -1
        best_model = None

    for e in range(epochs):

        torch.cuda.empty_cache()
        _metrics_t, _metrics_y, _metrics_d, _metrics_r = [], [], [], []
        _loss_t, _loss_y, _loss_tg, _loss_d, _loss_r, _loss_a = [], [], [], [], [], []

        for i, batch in enumerate(loader_train):
            # set model to train mode
            model.train()

            # Calculate overall loss.
            optimizer.zero_grad()
            predictions = model(batch[0].to(device))
            lb_t, lb_y, lb_tg, lb_d, lb_r, _ = _calculate_criterion_causalbatle(batch=batch,
                                                                                criterion_function=criterion,
                                                                                predictions=predictions,
                                                                                device=device)
            loss_batch = alpha[0] * lb_t + alpha[1] * lb_y + alpha[2] * lb_tg + alpha[3] * lb_d + alpha[4] * lb_r
            loss_batch.backward()
            optimizer.step()

            # Calculate adversarial loss.
            optimizer.zero_grad()
            predictions = model(batch[0].to(device))
            #print('second pass')
            _, _, _, _, _, lb_a = _calculate_criterion_causalbatle(batch=batch,
                                                                   criterion_function=criterion,
                                                                   predictions=predictions,
                                                                   device=device)
            loss_adv = alpha[5] * lb_a
            loss_adv.backward()
            for name, layer in model.named_modules():
                try:
                    if name in second_update:
                        layer.weight.grad.data.zero_()
                except:
                    print('Except - ',name)
            optimizer.step()

            _loss_t.append(lb_t.cpu().detach().numpy())
            _loss_y.append(lb_y.cpu().detach().numpy())
            _loss_tg.append(lb_tg.cpu().detach().numpy())
            _loss_d.append(lb_d.cpu().detach().numpy())
            _loss_r.append(lb_r.cpu().detach().numpy())
            _loss_a.append(lb_a.cpu().detach().numpy())
            mb_t, mb_y, mb_d, mb_r = _calculate_metric_causalbatle(metric_functions=metric_functions,
                                                                   batch=batch,
                                                                   predictions=predictions)
            _metrics_t.append(mb_t)
            _metrics_y.append(mb_y)
            _metrics_d.append(mb_d)
            _metrics_r.append(mb_r)

        loss_train_t[e] = np.nanmean(_loss_t)
        loss_train_y[e] = np.nanmean(_loss_y)
        loss_train_tg[e] = np.nanmean(_loss_tg)
        loss_train_d[e] = np.nanmean(_loss_d)
        loss_train_r[e] = np.nanmean(_loss_r)
        loss_train_a[e] = np.nanmean(_loss_a)

        metric_train_t[e] = np.nanmean(_metrics_t)
        metric_train_y[e] = np.nanmean(_metrics_y)
        metric_train_d[e] = np.nanmean(_metrics_d)
        metric_train_r[e] = np.nanmean(_metrics_r)
        if use_validation:
            lm_val = _calculate_loss_metric_noopti(model=model, loader=loader_val, device=device,
                                                   criterion=criterion, metric_functions=metric_functions)
            loss_val_t[e], loss_val_y[e] = lm_val['loss_t'], lm_val['loss_y']
            loss_val_d[e], loss_val_r[e] = lm_val['loss_d'], lm_val['loss_r']
            loss_val_a[e], loss_val_tg[e] = lm_val['loss_a'], lm_val['loss_tg']
            metric_val_t[e], metric_val_y[e] = lm_val['metric_t'], lm_val['metric_y']
            metric_val_d[e], metric_val_r[e] = lm_val['metric_d'], lm_val['metric_r']

            if use_validation_best:
                current_loss = alpha[0] * lm_val['loss_t'] + alpha[1] * lm_val['loss_y'] + alpha[2] * lm_val['loss_tg']
                if current_loss < best_loss:
                    best_epoch = e
                    best_loss = current_loss
                    best_model = model.state_dict()

        else:
            loss_val_t[e], loss_val_y[e], loss_val_d[e], loss_val_r[e], loss_val_a[e] = None, None, None, None, None
            metric_val_t[e], metric_val_y[e], metric_val_d[e], metric_val_r[e] = None, None, None, None

        if use_tensorboard:
            values = {'loss_train_t': loss_train_t[e], 'loss_train_y': loss_train_y[e],
                      'loss_train_d': loss_train_d[e], 'loss_train_r': loss_train_r[e],
                      'loss_train_a': loss_train_a[e], 'loss_train_tg': loss_train_tg[e],
                      'metric_train_t': metric_train_t[e], 'metric_train_y': metric_train_y[e],
                      'metric_train_d': metric_train_d[e], 'metric_train_r': metric_train_r[e],
                      }
            writer_tensorboard = ht.update_tensorboar(writer_tensorboard, values, e)
            values = {'loss_val_t': loss_val_t[e], 'loss_val_y': loss_val_y[e],
                      'loss_val_d': loss_val_d[e], 'loss_val_r': loss_val_r[e],
                      'loss_val_a': loss_val_a[e], 'loss_val_tg': loss_val_tg[e],
                      'metric_val_t': metric_val_t[e], 'metric_val_y': metric_val_y[e],
                      'metric_val_d': metric_val_d[e], 'metric_val_r': metric_val_r[e]}
            writer_tensorboard = ht.update_tensorboar(writer_tensorboard, values, e)

    if use_validation_best:
        if best_epoch > 0:
            model.load_state_dict(best_model)

    # Calculating metrics on testing set - no dropout used here.
    lm_test = _calculate_loss_metric_noopti(model=model, loader=loader_test,
                                            device=device, criterion=criterion, metric_functions=metric_functions)
    metric_test_t, metric_test_y = lm_test['metric_t'], lm_test['metric_y']
    metric_test_d, metric_test_r = lm_test['metric_d'], lm_test['metric_r']

    loss = {'loss_train_t': loss_train_t, 'loss_val_t': loss_val_t,
            'loss_train_y': loss_train_y, 'loss_val_y': loss_val_y,
            'loss_train_d': loss_train_d, 'loss_val_d': loss_val_d,
            'loss_train_r': loss_train_r, 'loss_val_r': loss_val_r,
            'loss_train_a': loss_train_a, 'loss_val_a': loss_val_a,
            }
    metrics = {'metric_train_t': metric_train_t, 'metric_val_t': metric_val_t, 'metric_test_t': metric_test_t,
               'metric_train_y': metric_train_y, 'metric_val_y': metric_val_y, 'metric_test_y': metric_test_y,
               'metric_train_d': metric_train_d, 'metric_val_d': metric_val_d, 'metric_test_d': metric_test_d,
               'metric_train_r': metric_train_r, 'metric_val_r': metric_val_r, 'metric_test_r': metric_test_r
               }

    return model, loss, metrics


def _calculate_loss_metric_noopti(model, loader, device, criterion, metric_functions):
    """ Calculate loss and metric per epoch.
    Ideal for validation and test set.
    :param model:
    :param loader:
    :param device:
    :return: dictionary with losses and metrics.
    """
    _metrics_t, _metrics_y, _metrics_d, _metrics_r = [], [], [], []
    _loss_t, _loss_y, _loss_d, _loss_r, _loss_a, _loss_tg = [], [], [], [], [], []
    for i, batch in enumerate(loader):
        model.eval()
        predictions = model(batch[0].to(device))
        lb_t, lb_y, lb_tg, lb_d, lb_r, lb_a = _calculate_criterion_causalbatle(batch=batch,
                                                                               criterion_function=criterion,
                                                                               predictions=predictions,
                                                                               device=device)
        _loss_t.append(lb_t.cpu().detach().numpy())
        _loss_y.append(lb_y.cpu().detach().numpy())
        _loss_tg.append(lb_tg.cpu().detach().numpy())
        _loss_d.append(lb_d.cpu().detach().numpy())
        _loss_r.append(lb_r.cpu().detach().numpy())
        _loss_a.append(lb_a.cpu().detach().numpy())

        mb_t, mb_y, mb_d, mb_r = _calculate_metric_causalbatle(metric_functions=metric_functions,
                                                               batch=batch,
                                                               predictions=predictions)
        _metrics_t.append(mb_t)
        _metrics_y.append(mb_y)
        _metrics_d.append(mb_d)
        _metrics_r.append(mb_r)

    output = {
        'loss_t': np.nanmean(_loss_t),
        'loss_y': np.nanmean(_loss_y),
        'loss_d': np.nanmean(_loss_d),
        'loss_r': np.nanmean(_loss_r),
        'loss_a': np.nanmean(_loss_a),
        'loss_tg': np.nanmean(_loss_tg),
        'metric_t': np.nanmean(_metrics_t),
        'metric_y': np.nanmean(_metrics_y),
        'metric_d': np.nanmean(_metrics_d),
        'metric_r': np.nanmean(_metrics_r)
    }

    return output


def _calculate_criterion_causalbatle(criterion_function, batch, predictions, device='cpu'):
    loss_t = criterion_function[0](batch=batch, predictions=predictions, device=device)
    loss_y = criterion_function[1](batch=batch, predictions=predictions, device=device)
    loss_tg = criterion_function[2](batch=batch, predictions=predictions, device=device)
    loss_d = criterion_function[3](batch=batch, predictions=predictions, device=device)
    loss_r = criterion_function[4](batch=batch, predictions=predictions, device=device)
    loss_a = criterion_function[5](batch=batch, predictions=predictions, device=device)
    return loss_t, loss_y, loss_tg, loss_d, loss_r, loss_a


def _calculate_metric_causalbatle(metric_functions, batch, predictions):
    metrics_t = metric_functions[0](batch=batch, predictions=predictions)
    metrics_y = metric_functions[1](batch=batch, predictions=predictions)
    metrics_d = metric_functions[2](batch=batch, predictions=predictions)
    metrics_r = metric_functions[3](batch=batch, predictions=predictions)
    return metrics_t, metrics_y, metrics_d, metrics_r


def criterion_function_t(batch, predictions, device='cpu'):
    t_predictions = predictions['t']
    weights = predictions['t_weight']

    d_obs = batch[3].to(device)
    t_obs = batch[2].to(device)  # [d_obs == 1]
    # Weights
    loss = -t_predictions.log_prob(t_obs)
    loss = loss[d_obs == 1]
    weights = weights.reshape(-1, 1)[d_obs == 1]
    loss = torch.mul(loss, weights)
    return loss.mean()


def criterion_function_y(batch, predictions, device='cpu'):
    y0_predictions, y1_predictions = predictions['y0'], predictions['y1']
    y_obs = batch[1].to(device)
    # t_obs = batch[2].to(device)
    # d_obs = batch[3].to(device)
    mask0 = np.logical_and(batch[2] == 0, batch[3] == 1).to(torch.bool).to(device)
    mask1 = np.logical_and(batch[2] == 1, batch[3] == 1).to(torch.bool).to(device)
    loss_y0_babch = -y0_predictions.log_prob(y_obs[mask0]).mean()
    loss_y1_babch = -y1_predictions.log_prob(y_obs[mask1]).mean()
    if mask1.sum() == 0:
        return loss_y0_babch
    elif mask0.sum() == 0:
        return loss_y1_babch
    else:
        return loss_y0_babch + loss_y1_babch


def criterion_function_dragonnet_targeted(batch, predictions, device='cpu'):
    mask = np.equal(batch[3], 1).to(torch.bool).to(device)

    y_obs = batch[1][mask].to(device)
    t_obs = batch[2][mask].to(device)

    t_predictions = predictions['t'].sample([1, 1]).reshape(-1, 1)[mask]  # .cpu().detach().numpy()
    y0_predictions = predictions['y0'].sample([1, 1]).reshape(-1, 1)[mask]  # .cpu().detach().numpy()
    y1_predictions = predictions['y1'].sample([1, 1]).reshape(-1, 1)[mask]  # .cpu().detach().numpy()
    epsilon = predictions['epsilon']

    y_pred = y0_predictions * (1 - t_obs) + y1_predictions * t_obs
    criterion = TargetedLoss()
    # print('ep',epsilon.shape, y1_predictions.shape)
    epsilon = epsilon.reshape(-1, 1)[mask]
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


def criterion_function_discriminator(batch, predictions, device='cpu'):
    """ BCE loss. While T and Y are bayesian, D is kept non-bayesian due to discriminator/adversarial components.
    Reference (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
    :param batch:
    :param predictions:
    :param device:
    :return:
    """
    d_predictions = predictions['d']
    d_obs = batch[3].to(device)
    # -d_predictions.log_prob(d_obs).mean()
    criterion = nn.BCELoss()
    return criterion(d_predictions, d_obs)


def criterion_function_reconstruction(batch, predictions, device='cpu'):
    reconstruction = predictions['xr']
    x = batch[0].to(device)
    criterion = nn.MSELoss()
    return criterion(reconstruction.ravel(), x.ravel())


def criterion_function_adversarial(batch, predictions, device='cpu'):
    """
    Reference (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
    :param batch:
    :param predictions:
    :param device:
    :return:
    """

    d_predictions = predictions['d']
    d_obs = 1 - batch[3]  # Set source domain as 1.
    criterion = nn.BCELoss()
    # Calculate adversarial loss only over the source domain.
    return criterion(d_predictions[d_obs == 1], d_obs[d_obs == 1].to(device))


def metric_function_discriminator(batch, predictions):
    return roc_auc_score(batch[3], predictions['d'].cpu().detach().numpy())


def metric_function_reconstruction(batch, predictions):
    reconstruction = predictions['xr'].cpu().detach().numpy().reshape(-1, 1)
    return mean_squared_error(batch[0].reshape(-1, 1), reconstruction)
