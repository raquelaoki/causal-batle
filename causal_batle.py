import numpy as np
import pandas as pd
import keras.backend as K
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score

import logging

# Local Imports
import utils
import dragonnet
import helper_tensorboard as ht
import bayesian_layers as bl

logger = logging.getLogger(__name__)


class causal_batle(nn.Module):
    def __init__(self, n_covariates, dropout_p=0.5, units1=200, units2=100, units3=1):
        super().__init__()

        self.backbone = dragonnet.dragonnet(n_covariates=n_covariates, type_original=False,
                                            units1=units1, units2=units2, units3=units3,
                                            use_dropout=True, dropout_p=dropout_p)

        # Update the head for our causal-batle proposed methodlogy
        self.backbone.dragonnet_head = causal_batle_head(units1=units1, units2=units2,
                                                         units3=units3, dropout_p=dropout_p,
                                                         n_covariates=n_covariates)

    def forward(self, inputs):
        return self.backbone(inputs)


class causal_batle_head(nn.Module):
    """ Causal Batle Head.
    """

    def __init__(self, n_covariates, units1=200, units2=100, units3=1, dropout_p=0.5):
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
        self.decoder_layer = decoder(n_covariates=self.n_covariates, units1=self.units1)

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

        predictions = {'y0': y0_predictions,
                       'y1': y1_predictions,
                       't': t_predictions,
                       'd': d_predictions,
                       'xr': x_reconstruction}

        return predictions


class decoder(nn.Module):
    def __init__(self, n_covariates, units1):
        super(decoder, self).__init__()
        self.decoder_layer1_1 = nn.Linear(in_features=units1, out_features=units1)
        self.decoder_layer1_2 = nn.Linear(in_features=units1, out_features=units1)
        self.decoder_layer1_3 = nn.Linear(in_features=units1, out_features=n_covariates)
        self.elu = nn.ELU()

    def forward(self, inputs):
        # print('inputs shape',inputs.shape, ' expected', self.units2)
        reconstruction = self.elu(self.decoder_layer1_1(inputs))
        reconstruction = self.elu(self.decoder_layer1_2(reconstruction))
        reconstruction = self.decoder_layer1_3(reconstruction)
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
                     episilon=None):
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

    # use prefetch_generator and tqdm for iterating through data
    # pbar = tqdm(enumerate(BackgroundGenerator(train_data_loader, ...)),
    #            total=len(train_data_loader))
    # start_time = time.time()

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
    loss_train_a = np.zeros(epochs)  # _a: adversarial loss

    loss_val_t, metric_val_t = np.zeros(epochs), np.zeros(epochs)
    loss_val_y, metric_val_y = np.zeros(epochs), np.zeros(epochs)
    loss_val_d, metric_val_d = np.zeros(epochs), np.zeros(epochs)
    loss_val_r, metric_val_r = np.zeros(epochs), np.zeros(epochs)
    loss_val_a = np.zeros(epochs)

    for e in range(epochs):

        torch.cuda.empty_cache()
        _metrics_t, _metrics_y, _metrics_d, _metrics_r = [], [], [], []
        _loss_t, _loss_y, _loss_d, _loss_r, _loss_a = [], [], [], [], []

        for i, batch in enumerate(loader_train):
            # set model to train mode
            model.train()

            optimizer.zero_grad()
            predictions = model(batch[0].to(device))
            lb_t, lb_y, lb_d, lb_r, lb_a = _calculate_criterion_causalbatle(batch=batch,
                                                                            criterion_function=criterion,
                                                                            predictions=predictions,
                                                                            device=device)

            # Calculate tarnet loss
            loss_batch = alpha[0] * lb_t + alpha[1] * lb_y + alpha[2] * lb_d + alpha[3] * lb_r + alpha[4] * lb_a
            loss_batch.backward()
            optimizer.step()
            _loss_t.append(lb_t.cpu().detach().numpy())
            _loss_y.append(lb_y.cpu().detach().numpy())
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
        loss_train_d[e] = np.nanmean(_loss_d)
        loss_train_r[e] = np.nanmean(_loss_r)
        loss_train_a[e] = np.nanmean(_loss_a)

        metric_train_t[e] = np.nanmean(_metrics_t)
        metric_train_y[e] = np.nanmean(_metrics_y)
        metric_train_d[e] = np.nanmean(_metrics_d)
        metric_train_r[e] = np.nanmean(_metrics_r)

        # print('epoch', e, loss_train_t[e], loss_train_y[e])
        if use_validation:
            lm_val = _calculate_loss_metric_noopti(model=model, loader=loader_val,  device=device,
                                                   criterion=criterion, metric_functions=metric_functions)
            loss_val_t[e], loss_val_y[e] = lm_val['loss_t'], lm_val['loss_y']
            loss_val_d[e], loss_val_r[e] = lm_val['loss_d'], lm_val['loss_r']
            loss_val_a[e] = lm_val['loss_a']
            metric_val_t[e], metric_val_y[e] = lm_val['metric_t'], lm_val['metric_y']
            metric_val_d[e], metric_val_r[e] = lm_val['metric_d'], lm_val['metric_r']

        else:
            loss_val_t[e], loss_val_y[e], loss_val_d[e], loss_val_r[e], loss_val_a[e] = None, None, None, None, None
            metric_val_t[e], metric_val_y[e], metric_val_d[e], metric_val_r[e] = None, None, None, None

        if use_tensorboard:
            values = {'loss_train_t': loss_train_t[e], 'loss_train_y': loss_train_y[e],
                      'loss_train_d': loss_train_d[e], 'loss_train_r': loss_train_r[e], 'loss_train_a': loss_train_a[e],
                      'metric_train_t': metric_train_t[e], 'metric_train_y': metric_train_y[e],
                      'metric_train_d': metric_train_d[e], 'metric_train_r': metric_train_r[e],
                      }
            writer_tensorboard = ht.update_tensorboar(writer_tensorboard, values, e, set='train')
            values = {'loss_val_t': loss_val_t[e], 'loss_val_y': loss_val_y[e],
                      'loss_val_d': loss_val_d[e], 'loss_val_r': loss_val_r[e], 'loss_val_a': loss_val_a[e],
                      'metric_val_t': metric_val_t[e], 'metric_val_y': metric_val_y[e],
                      'metric_val_d': metric_val_d[e], 'metric_val_r': metric_val_r[e]}
            writer_tensorboard = ht.update_tensorboar(writer_tensorboard, values, e, set='val')


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
    :return:
    """
    _metrics_t, _metrics_y, _metrics_d, _metrics_r = [], [], [], []
    _loss_t, _loss_y, _loss_d, _loss_r, _loss_a = [], [], [], [], []
    for i, batch in enumerate(loader):
        model.eval()
        predictions = model(batch[0].to(device))
        lb_t, lb_y, lb_d, lb_r, lb_a = _calculate_criterion_causalbatle(batch=batch,
                                                                        criterion_function=criterion,
                                                                        predictions=predictions,
                                                                        device=device)
        _loss_t.append(lb_t.cpu().detach().numpy())
        _loss_y.append(lb_y.cpu().detach().numpy())
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
        'metric_t': np.nanmean(_metrics_t),
        'metric_y': np.nanmean(_metrics_y),
        'metric_d': np.nanmean(_metrics_d),
        'metric_r': np.nanmean(_metrics_r)
    }

    return output


def _calculate_criterion_causalbatle(criterion_function, batch, predictions, device='cpu'):
    loss_t = criterion_function[0](batch=batch, predictions=predictions, device=device)
    loss_y = criterion_function[1](batch=batch, predictions=predictions, device=device)
    loss_d = criterion_function[2](batch=batch, predictions=predictions, device=device)
    loss_r = criterion_function[3](batch=batch, predictions=predictions, device=device)
    loss_a = criterion_function[4](batch=batch, predictions=predictions, device=device)
    return loss_t, loss_y, loss_d, loss_r, loss_a


def _calculate_metric_causalbatle(metric_functions, batch, predictions):
    metrics_t = metric_functions[0](batch=batch, predictions=predictions)
    metrics_y = metric_functions[1](batch=batch, predictions=predictions)
    metrics_d = metric_functions[2](batch=batch, predictions=predictions)
    metrics_r = metric_functions[3](batch=batch, predictions=predictions)
    return metrics_t, metrics_y, metrics_d, metrics_r


def criterion_function_t(batch, predictions, device='cpu'):
    t_predictions = predictions['t']
    t_obs = batch[2].to(device)
    d_obs = batch[2].to(device)
    return -t_predictions.log_prob(t_obs[d_obs == 1]).mean()


def criterion_function_y(batch, predictions, device='cpu'):
    y0_predictions, y1_predictions = predictions['y0'], predictions['y1']
    y_obs = batch[1].to(device)
    t_obs = batch[2].to(device)
    d_obs = batch[3].to(device)
    mask0 = np.logical_and(t_obs == 0, d_obs == 1).to(dtype=torch.bool)
    mask1 = np.logical_and(t_obs == 1, d_obs == 1).to(dtype=torch.bool)
    loss_y0_babch = -y0_predictions.log_prob(y_obs[mask0]).mean()
    loss_y1_babch = -y1_predictions.log_prob(y_obs[mask1]).mean()
    if mask1.sum() == 0:
        return loss_y0_babch
    elif mask0.sum() == 0:
        return loss_y1_babch
    else:
        return loss_y0_babch + loss_y1_babch


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
    return criterion(reconstruction, x)


def criterion_function_adversarial(batch, predictions, device='cpu'):
    d_predictions = predictions['d']
    d_obs = 1 - batch[3]  # Set source domain as 1.
    criterion = nn.BCELoss()
    # Calculate adversarial loss only over the source domain.
    return criterion(d_predictions[d_obs == 1], d_obs[d_obs == 1].to(device))


def metric_function_discriminator(batch, predictions):
    return roc_auc_score(batch[3], predictions['d'].detach().numpy())


def metric_function_reconstruction(batch, predictions):
    reconstruction = predictions['xr'].detach().numpy().reshape(-1, 1)
    return mean_squared_error(batch[0].reshape(-1, 1), reconstruction)


