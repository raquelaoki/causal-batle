"""
References:
https://github.com/claudiashi57/dragonnet/blob/master/src/experiment/models.py
https://github.com/raquelaoki/M3E2/blob/main/resources/dragonnet.py

Alternative implementation of the Dragonnet model: A neural network to estimate treatment effects.
Adopting pytorch
"""
import numpy as np
import pandas as pd
import keras.backend as K

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score, roc_curve

import tensorflow as tf
import utils

import helper_tensorboard as ht


# from semi_parametric_estimation.ate import psi_naive, psi_tmle_cont_outcome
# Do I use this functions above? https://github.com/raquelaoki/M3E2/tree/main/resources/semi_parametric_estimation
#
# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, mean_squared_error
import logging

logger = logging.getLogger(__name__)


# logging.basicConfig(level=logging.DEBUG)


class dragonnet(nn.Module):
    def __init__(self, n_covariates, units1=200, units2=100,
                 units3=1, type_original=True, use_dropout=False,
                 dropout_p=0):
        super().__init__()
        self.units1 = units1
        self.units2 = units2
        self.units3 = units3
        self.representation_layer1_1 = nn.Linear(in_features=n_covariates, out_features=self.units1)
        self.representation_layer1_2 = nn.Linear(in_features=self.units1, out_features=self.units1)
        self.representation_layer1_3 = nn.Linear(in_features=self.units1, out_features=self.units1)

        self.type_original = type_original
        if self.type_original:
            self.dragonnet_head = dragonnet_original(self.units1, self.units2, self.units3)
        else:
            logging.debug("Gaussian Process not implemented yet.")
        # Activation functions.
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p
        self.batchnorm = nn.BatchNorm1d(self.units1)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_p)
        else:
            self.dropout = nn.Dropout(p=0)

    def forward(self, inputs):
        """
        Neural net predictive model. The dragon has three heads.
        Parameters:
            n_covariates: number of covariates
            u1: 3-layer representation component with u1 units
            u2: 2-layer outcome prediction +
            u3: final-layer of outcome prediction

        :return:
        """

        # Shared presentation.
        x = self.elu(self.representation_layer1_1(self.dropout(inputs)))
        x = self.elu(self.representation_layer1_2(self.batchnorm(x)))
        x = self.elu(self.representation_layer1_3(x))
        return self.dragonnet_head(self.batchnorm(self.dropout(x)))


class dragonnet_original(nn.Module):
    """ Dragonnet Original Head.
    """

    def __init__(self, units1=200, units2=100, units3=1, use_dropout=False, dropout_p=0):
        super(dragonnet_original, self).__init__()
        self.units1 = units1
        self.units2 = units2
        self.units3 = units3
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p

        self.head_layer2_1_0 = nn.Linear(in_features=self.units1, out_features=self.units2)
        self.head_layer2_2_0 = nn.Linear(in_features=self.units2, out_features=self.units2)
        self.outcome_layer_0 = nn.Linear(in_features=self.units2, out_features=self.units3)

        self.head_layer2_1_1 = nn.Linear(in_features=self.units1, out_features=self.units2)
        self.head_layer2_2_1 = nn.Linear(in_features=self.units2, out_features=self.units2)
        self.outcome_layer_1 = nn.Linear(in_features=self.units2, out_features=self.units3)

        self.t_predictions = nn.Linear(in_features=self.units1, out_features=1)

        # Activation functions.
        self.elu = nn.ELU(alpha=0.25)
        self.sigmoid = nn.Sigmoid()
        self.tahn = nn.Tanh()

        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_p)
        else:
            self.dropout = nn.Dropout(p=0)

    def forward(self, inputs):
        # Treatment specific - first layer.
        y0_hidden = self.elu(self.head_layer2_1_0(inputs))
        y1_hidden = self.elu(self.head_layer2_1_1(inputs))

        # Treatment specific - second layer.
        y0_hidden = self.elu(self.head_layer2_2_0(self.dropout(y0_hidden)))
        y1_hidden = self.elu(self.head_layer2_2_1(self.dropout(y1_hidden)))

        # Treatment specific - third layer.
        y0_predictions = self.outcome_layer_0(y0_hidden)
        y1_predictions = self.outcome_layer_1(y1_hidden)

        t_predictions = self.sigmoid(self.t_predictions(inputs))
        predictions = {'y0': y0_predictions,
                       'y1': y1_predictions,
                       't': t_predictions}

        return predictions


class TargetedLoss(nn.Module):

    def __init__(self):
        super(TargetedLoss, self).__init__()

    def forward(self, y_obs, y_pred, t_obs, t_pred, epislon=0.1):
        t1 = torch.div(t_obs,
                       torch.add(t_pred, 0.01))
        t0 = torch.div(torch.sub(t_obs, -1),
                       torch.add(torch.sub(t_pred, -1), 0.01))
        t = torch.mul(torch.sub(t1, t0), epislon)

        loss = torch.add(y_pred, t)
        loss = torch.sub(y_obs, loss)
        return torch.mean(loss)


def metric_function_dragonnet_t(batch, predictions):
    return roc_auc_score(batch[2], predictions['t'].detach().numpy())


def metric_function_dragonnet_y(batch, predictions):
    y_pred = predictions['y0'] * (1 - predictions['t']) + predictions['y1'] * predictions['t']
    return mean_squared_error(batch[1], y_pred.detach().numpy())


def criterion_function_dragonnet_targeted(batch, predictions, device='cpu'):
    y_obs = batch[1].to(device)
    t_obs = batch[2].to(device)
    t_predictions = predictions['t']
    y0_predictions, y1_predictions = predictions['y0'], predictions['y1']
    y_pred = y0_predictions * (1 - t_obs) + y1_predictions * t_obs
    criterion = TargetedLoss()
    return criterion(y_obs=y_obs, y_pred=y_pred, t_obs=t_obs, t_pred=t_predictions)


def criterion_function_dragonnet_t(batch, predictions, device='cpu'):
    t_predictions = predictions['t']
    t_obs = batch[2].to(device)
    criterion = nn.BCELoss()
    return criterion(t_predictions, t_obs)


def criterion_function_dragonnet_y(batch, predictions, device='cpu'):
    y0_predictions, y1_predictions = predictions['y0'], predictions['y1']
    y_obs = batch[1].to(device)
    t_obs = batch[2].to(device)

    criterion = nn.MSELoss()
    loss_y0_batch = criterion(y0_predictions[t_obs == 0], y_obs[t_obs == 0])
    loss_y1_batch = criterion(y1_predictions[t_obs == 1], y_obs[t_obs == 1])
    return loss_y0_batch + loss_y1_batch


def _calculate_criterion_dragonnet(criterion_function, batch, predictions, device='cpu'):
    loss_t = criterion_function[0](
        batch=batch,
        predictions=predictions,
        device=device)
    loss_y = criterion_function[1](
        batch=batch,
        predictions=predictions,
        device=device)
    loss_target = criterion_function[2](
        batch=batch,
        predictions=predictions,
        device=device
    )

    return loss_t, loss_y, loss_target


def _calculate_metric_dragonnet(metric_functions, batch, predictions):
    metrics_t = metric_functions[0](batch=batch,
                                    predictions=predictions)
    metrics_y = metric_functions[1](batch=batch,
                                    predictions=predictions)
    return metrics_t, metrics_y


def fit_dragonnet(epochs,
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
                  home_dir=''):
    """
        Fit implementation: Contain epochs and batch iterator, optimization steps, and eval.
    :param epochs: integer
    :param model: nn.Module
    :param loader_train: DataLoader
    :param optimizer: torch.optim
    :param criterion: List of criterions
    :param metrics_functions: List of metrics
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

    loss_train_t, metric_train_t = np.zeros(epochs), np.zeros(epochs)
    loss_train_y, metric_train_y = np.zeros(epochs), np.zeros(epochs)

    loss_val_t, metric_val_t = np.zeros(epochs), np.zeros(epochs)
    loss_val_y, metric_val_y = np.zeros(epochs), np.zeros(epochs)

    # Targeted losses.
    loss_train_ty, loss_val_ty = np.zeros(epochs), np.zeros(epochs)

    for e in range(epochs):
        # set model to train mode
        model.train()

        torch.cuda.empty_cache()
        _metrics_t, _metrics_y = [], []
        _loss_t, _loss_y, _loss_ty = [], [], []
        for i, batch in enumerate(loader_train):
            optimizer.zero_grad()
            predictions = model(batch[0].to(device))
            loss_batch_t, loss_batch_y, loss_batch_tar = _calculate_criterion_dragonnet(criterion_function=criterion,
                                                                                        batch=batch,
                                                                                        predictions=predictions,
                                                                                        device=device)

            metrics_batch_t, metrics_batch_y = _calculate_metric_dragonnet(metric_functions=metric_functions,
                                                                           batch=batch,
                                                                           predictions=predictions)

            loss_batch = alpha[0] * loss_batch_t + alpha[1] * loss_batch_y + alpha[2] * loss_batch_tar

            loss_batch.backward()
            optimizer.step()
            _loss_t.append(loss_batch_t.cpu().detach().numpy())
            _loss_y.append(loss_batch_y.cpu().detach().numpy())
            _loss_ty.append(loss_batch_tar.cpu().detach().numpy())
            _metrics_t.append(metrics_batch_t)
            _metrics_y.append(metrics_batch_y)

        loss_train_t[e] = np.mean(_loss_t)
        loss_train_y[e] = np.mean(_loss_y)
        loss_train_ty[e] = np.mean(_loss_ty)
        metric_train_t[e] = np.mean(_metrics_t)
        metric_train_y[e] = np.mean(_metrics_y)

        if use_validation:
            model.eval()
            batch = next(iter(loader_val))
            predictions = model(batch[0].to(device))

            loss_val_t[e], loss_val_y[e], loss_val_ty[e] = _calculate_criterion_dragonnet(criterion_function=criterion,
                                                                                          batch=batch,
                                                                                          predictions=predictions,
                                                                                          device=device)
            metric_val_t[e], metric_val_y[e] = _calculate_metric_dragonnet(metric_functions=metric_functions,
                                                                           batch=batch,
                                                                           predictions=predictions)
        else:
            loss_val_t[e], loss_val_y[e], loss_val_ty[e] = None, None, None
            metric_val_t[e], metric_val_y[e] = None, None

        if use_tensorboard:
            values = {'loss_train_t':loss_train_t[e], 'loss_train_y':loss_train_y[e],
                      'loss_train_ty':loss_train_ty[e], 'metric_train_t':metric_train_t[e],
                      'metric_train_y':metric_train_y[e]}
            writer_tensorboard = ht.update_tensorboar(writer_tensorboard, values, e, set='train')
            values = {'loss_val_t':loss_val_t[e], 'loss_val_y':loss_val_y[e], 'loss_val_ty':loss_val_ty[e],
                      'metric_val_t':metric_val_t[e], 'metric_val_y':metric_val_y[e]}
            writer_tensorboard = ht.update_tensorboar(writer_tensorboard, values, e, set='val')

    # Change model to eval mode
    model.eval()

    # Metrics on testins set
    batch = next(iter(loader_test))
    predictions = model(batch[0].to(device))
    metric_test_t, metric_test_y = _calculate_metric_dragonnet(metric_functions=metric_functions,
                                                               batch=batch,
                                                               predictions=predictions,
                                                               )

    loss = {'loss_train_t': loss_train_t,
            'loss_val_t': loss_val_t,
            'loss_train_y': loss_train_y,
            'loss_val_y': loss_val_y,
            'loss_val_ty': loss_val_ty,
            'loss_train_ty': loss_train_ty,
            }
    metrics = {'metric_train_t': metric_train_t,
               'metric_val_t': metric_val_t,
               'metric_train_y': metric_train_y,
               'metric_val_y': metric_val_y,
               'metric_test_t': metric_test_t,
               'metric_test_y': metric_test_y
               }

    # thhold = find_optimal_cutoff_wrapper_t(loader_train=loader_train, model=model, device=device)
    # metrics['thhold'] = thhold
    model.eval()

    return model, loss, metrics
