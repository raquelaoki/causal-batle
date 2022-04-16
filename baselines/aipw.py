import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import Tensor
#from torch.utils.data import Dataset, DataLoader, TensorDataset

# Local Imports
import utils
import helper_tensorboard as ht

logger = logging.getLogger(__name__)


class linearRegression(nn.Module):
    def __init__(self, input):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(input, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


class logisticRegression(nn.Module):
    def __init__(self, input):
        super(logisticRegression, self).__init__()
        self.linear = nn.Linear(input, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.linear(x))
        return out


class aipw(nn.Module):
    def __init__(self, n_covariates):
        super(aipw, self).__init__()
        self.n_covariates = n_covariates
        self.linearRegression_0 = linearRegression(self.n_covariates)
        self.linearRegression_1 = linearRegression(self.n_covariates)
        self.logisticRegression = logisticRegression(self.n_covariates)

    def forward(self, inputs):
        y0_predictions = self.linearRegression_0(inputs)
        y1_predictions = self.linearRegression_1(inputs)
        t_predictions = self.logisticRegression(inputs)

        predictions = {'y0': y0_predictions,
                       'y1': y1_predictions,
                       't': t_predictions}

        return predictions


def metric_function_aipw_y(batch, predictions, base):
    t_obs = batch[2]
    return mean_squared_error(batch[1][t_obs==base], predictions['y'+str(base)].cpu().detach().numpy()[t_obs==base])


def metric_function_aipw_t(batch, predictions):
    return roc_auc_score(batch[2], predictions['t'].cpu().detach().numpy())


def criterion_function_aipw_y(batch, predictions, device='cpu', base=0):
    y_obs = batch[1].to(device)
    t_obs = batch[2].to(device)
    criterion = nn.MSELoss()
    return criterion(predictions['y'+str(base)][t_obs==base], y_obs[t_obs==base])


def criterion_function_aipw_t(batch, predictions, device='cpu'):
    t_obs = batch[2].to(device)
    criterion = nn.BCELoss()
    return criterion(predictions['t'], t_obs)


def _calculate_criterion_aipw(criterion_function, batch, predictions, device='cpu'):
    loss_t = criterion_function[0](
        batch=batch,
        predictions=predictions,
        device=device)
    loss_y0 = criterion_function[1](
        batch=batch,
        predictions=predictions,
        device=device,
        base=0)
    loss_y1 = criterion_function[2](
        batch=batch,
        predictions=predictions,
        device=device,
        base=1
    )

    return loss_t, loss_y0, loss_y1


def _calculate_metric_aipw(metric_functions, batch, predictions):
    metrics_t = metric_functions[0](batch=batch,
                                    predictions=predictions)
    metrics_y0 = metric_functions[1](batch=batch,
                                     predictions=predictions,
                                     base=0)
    metrics_y1 = metric_functions[1](batch=batch,
                                     predictions=predictions,
                                     base=1)
    return metrics_t, metrics_y0, metrics_y1


def fit_aipw(epochs,
             model,
             loader_train,
             loader_test,
             optimizer,
             criterion,
             metric_functions,
             use_tensorboard,
             device,
             use_validation,
             loader_val=None,
             path_logger='',
             config_name='',
             home_dir='',
             alpha=[2,1,1],
             use_validation_best=False,
             ):
    logger.debug('...starting')

    # use prefetch_generator and tqdm for iterating through data
    # pbar = tqdm(enumerate(BackgroundGenerator(train_data_loader, ...)),
    #            total=len(train_data_loader))
    # start_time = time.time()

    if use_tensorboard:
        writer_tensorboard = ht.TensorboardWriter(path_logger=path_logger,
                                                  config_name=config_name,
                                                  home_dir=home_dir)

    loss_train_t, metric_train_t = np.zeros(epochs), np.zeros(epochs)
    loss_train_y0, metric_train_y0 = np.zeros(epochs), np.zeros(epochs)
    loss_train_y1, metric_train_y1 = np.zeros(epochs), np.zeros(epochs)

    loss_val_t, metric_val_t = np.zeros(epochs), np.zeros(epochs)
    loss_val_y0, metric_val_y0 = np.zeros(epochs), np.zeros(epochs)
    loss_val_y1, metric_val_y1 = np.zeros(epochs), np.zeros(epochs)

    if use_validation_best:
        best_loss = 999
        best_epoch = -1
        best_model = None

    for e in range(epochs):
        # set model to train mode
        model.train()

        torch.cuda.empty_cache()
        _metrics_t, _metrics_y0, _metrics_y1 = [], [], []
        _loss_t, _loss_y0, _loss_y1 = [], [], []
        for i, batch in enumerate(loader_train):
            optimizer.zero_grad()
            predictions = model(batch[0].to(device))
            loss_batch_t, loss_batch_y0, loss_batch_y1 = _calculate_criterion_aipw(criterion_function=criterion,
                                                                                   batch=batch,
                                                                                   predictions=predictions,
                                                                                   device=device)
            # print('loss batch',loss_batch_t, loss_batch_y0, loss_batch_y1 )
            metrics_batch_t, metrics_batch_y0, metrics_batch_y1 = _calculate_metric_aipw(
                metric_functions=metric_functions,
                batch=batch,
                predictions=predictions)

            loss_batch = loss_batch_t*alpha[0] + loss_batch_y0*alpha[1] + loss_batch_y1*alpha[2]
            loss_batch.backward()
            optimizer.step()

            _loss_t.append(loss_batch_t.cpu().detach().numpy())
            _loss_y0.append(loss_batch_y0.cpu().detach().numpy())
            _loss_y1.append(loss_batch_y1.cpu().detach().numpy())
            _metrics_t.append(metrics_batch_t)
            _metrics_y0.append(metrics_batch_y0)
            _metrics_y1.append(metrics_batch_y1)

        loss_train_t[e] = np.mean(_loss_t)
        loss_train_y0[e] = np.mean(_loss_y0)
        loss_train_y1[e] = np.mean(_loss_y1)
        metric_train_t[e] = np.mean(_metrics_t)
        metric_train_y0[e] = np.mean(_metrics_y0)
        metric_train_y1[e] = np.mean(_metrics_y1)

        if use_validation:
            model.eval()
            batch = next(iter(loader_val))
            predictions = model(batch[0].to(device))
            # print('val pred and batch', predictions['t'][0:10], batch[2][0:10])
            loss_val_t[e], loss_val_y0[e], loss_val_y1[e] = _calculate_criterion_aipw(criterion_function=criterion,
                                                                                      batch=batch,
                                                                                      predictions=predictions,
                                                                                      device=device)
            metric_val_t[e], metric_val_y0[e], metric_val_y1[e] = _calculate_metric_aipw(
                metric_functions=metric_functions,
                batch=batch,
                predictions=predictions)
            if use_validation_best:
                current_loss = metric_val_t[e]*alpha[0]+ metric_val_y0[e]*alpha[1]+ metric_val_y1[e] *alpha[2]
                if current_loss < best_loss:
                    best_epoch = e
                    best_loss = current_loss
                    best_model = model.state_dict()

        else:
            loss_val_t[e], loss_val_y0[e], loss_val_y1[e] = None, None, None
            metric_val_t[e], metric_val_y0[e], metric_val_y1[e] = None, None, None
        #print('checking metric t', metric_val_t[e], metric_train_t[e], e)
        if use_tensorboard:
            values = {'loss_train_t': loss_train_t[e], 'loss_train_y0': loss_train_y0[e],
                      'loss_train_y1': loss_train_y1[e], 'metric_train_t': metric_train_t[e],
                      'metric_train_y0': metric_train_y0[e], 'metric_train_y1': metric_train_y1[e]}
            writer_tensorboard = ht.update_tensorboar(writer_tensorboard, values, e)
            values = {'loss_val_t': loss_val_t[e], 'loss_val_y0': loss_val_y0[e],
                      'loss_val_y1': loss_val_y1[e], 'metric_val_t': metric_val_t[e],
                      'metric_val_y0': metric_val_y0[e], 'metric_val_y1': metric_val_y1[e]}
            writer_tensorboard = ht.update_tensorboar(writer_tensorboard, values, e)

    if use_validation_best:
        if best_epoch > 0:
            model.load_state_dict(best_model)
    model.eval()

    # Metrics on testins set
    batch = next(iter(loader_test))
    predictions = model(batch[0].to(device))
    metric_test_t, metric_test_y0, metric_test_y1 = _calculate_metric_aipw(
        metric_functions=metric_functions,
        batch=batch,
        predictions=predictions,
    )

    loss = {'loss_train_t': loss_train_t,
            'loss_val_t': loss_val_t,
            'loss_train_y0': loss_train_y0,
            'loss_val_y0': loss_val_y0,
            'loss_val_y1': loss_val_y1,
            'loss_train_y1': loss_train_y1
            }
    metrics = {'metric_train_t': metric_train_t,
               'metric_val_t': metric_val_t,
               'metric_train_y0': metric_train_y0,
               'metric_val_y0': metric_val_y0,
               'metric_train_y1': metric_train_y1,
               'metric_val_y1': metric_val_y1,
               'metric_test_t': metric_test_t,
               'metric_test_y0': metric_test_y0,
               'metric_test_y1': metric_test_y1
               }

    # thhold = find_optimal_cutoff_wrapper_t(loader_train=loader_train, model=model, device=device)
    # metrics['thhold'] = thhold
    model.eval()
    return model, loss, metrics


def fit_aipw_three_opt(epochs,
                       model,
                       loader_train,
                       loader_test,
                       optimizer,
                       criterion,
                       metric_functions,
                       use_tensorboard,
                       device,
                       use_validation,
                       loader_val=None,
                       path_logger='',
                       config_name='',
                       home_dir='',
                       alpha=[],
                       weight_1=1):
    logger.debug('...starting')

    # use prefetch_generator and tqdm for iterating through data
    # pbar = tqdm(enumerate(BackgroundGenerator(train_data_loader, ...)),
    #            total=len(train_data_loader))
    # start_time = time.time()

    if use_tensorboard:
        writer_tensorboard = ht.TensorboardWriter(path_logger=path_logger,
                                                  config_name=config_name,
                                                  home_dir=home_dir)

    alpha=[1,0,0]
    optimizer_t = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.9)
    model = _freeze_layers(model,alpha)

    model, loss_train_t, loss_val_t, metric_train_t, metric_val_t = epoch_opt(model=model,
                                                                              epochs=epochs,
                                                                              loader_train=loader_train,
                                                                              loader_val=loader_val,
                                                                              device=device,
                                                                              optimizer=optimizer_t,
                                                                              criterion=criterion,
                                                                              metric_functions=metric_functions,
                                                                              use_validation=use_validation,
                                                                              alpha=alpha)
    alpha=[0, 1,0]
    model = _freeze_layers(model,alpha)
    model, loss_train_y0, loss_val_y0, metric_train_y0, metric_val_y0 = epoch_opt(model=model,
                                                                                  epochs=epochs,
                                                                                  loader_train=loader_train,
                                                                                  loader_val=loader_val,
                                                                                  device=device,
                                                                                  optimizer=optimizer,
                                                                                  criterion=criterion,
                                                                                  metric_functions=metric_functions,
                                                                                  use_validation=use_validation,
                                                                                  alpha=alpha)
    alpha = [0, 0, 1]
    model = _freeze_layers(model, alpha)
    model, loss_train_y1, loss_val_y1, metric_train_y1, metric_val_y1 = epoch_opt(model=model,
                                                                                  epochs=epochs,
                                                                                  loader_train=loader_train,
                                                                                  loader_val=loader_val,
                                                                                  device=device,
                                                                                  optimizer=optimizer,
                                                                                  criterion=criterion,
                                                                                  metric_functions=metric_functions,
                                                                                  use_validation=use_validation,
                                                                                  alpha=alpha)

    if use_tensorboard:
        for e in range(epochs):
            values = {'loss_train_t': loss_train_t[e], 'loss_train_y0': loss_train_y0[e],
                      'loss_train_y1': loss_train_y1[e], 'metric_train_t': metric_train_t[e],
                      'metric_train_y0': metric_train_y0[e], 'metric_train_y1': metric_train_y1[e]}
            writer_tensorboard = ht.update_tensorboar(writer_tensorboard, values, e, set='train')
            values = {'loss_val_t': loss_val_t[e], 'loss_val_y0': loss_val_y0[e],
                      'loss_val_y1': loss_val_y1[e], 'metric_val_t': metric_val_t[e],
                      'metric_val_y0': metric_val_y0[e], 'metric_val_y1': metric_val_y1[e]}
            writer_tensorboard = ht.update_tensorboar(writer_tensorboard, values, e, set='val')

    model.eval()
    # Metrics on testins set
    batch = next(iter(loader_test))
    predictions = model(batch[0].to(device))
    metric_test_t, metric_test_y0, metric_test_y1 = _calculate_metric_aipw(
        metric_functions=metric_functions,
        batch=batch,
        predictions=predictions,
    )

    loss = {'loss_train_t': loss_train_t,
            'loss_val_t': loss_val_t,
            'loss_train_y0': loss_train_y0,
            'loss_val_y0': loss_val_y0,
            'loss_val_y1': loss_val_y1,
            'loss_train_y1': loss_train_y1
            }
    metrics = {'metric_train_t': metric_train_t,
               'metric_val_t': metric_val_t,
               'metric_train_y0': metric_train_y0,
               'metric_val_y0': metric_val_y0,
               'metric_train_y1': metric_train_y1,
               'metric_val_y1': metric_val_y1,
               'metric_test_t': metric_test_t,
               'metric_test_y0': metric_test_y0,
               'metric_test_y1': metric_test_y1
               }

    model.eval()
    return model, loss, metrics


def epoch_opt(model, epochs, loader_train, loader_val, device, optimizer, criterion, metric_functions, use_validation,
              alpha=[1, 0, 0]):
    loss_train, metric_train = np.zeros(epochs), np.zeros(epochs)
    loss_val, metric_val = np.zeros(epochs), np.zeros(epochs)
    model.train()

    for e in range(epochs):
        # set model to train mode
        model.train()

        torch.cuda.empty_cache()
        _metrics = []
        _loss = []
        for i, batch in enumerate(loader_train):
            optimizer.zero_grad()
            predictions = model(batch[0].to(device))
            loss_batch_t, loss_batch_y0, loss_batch_y1 = _calculate_criterion_aipw(criterion_function=criterion,
                                                                                   batch=batch,
                                                                                   predictions=predictions,
                                                                                   device=device)
            metrics_batch_t, metrics_batch_y0, metrics_batch_y1 = _calculate_metric_aipw(
                metric_functions=metric_functions,
                batch=batch,
                predictions=predictions)

            loss_batch = loss_batch_t * alpha[0] + loss_batch_y0 * alpha[1] + loss_batch_y1 * alpha[2]
            loss_batch.backward()
            optimizer.step()

            _loss.append(loss_batch.cpu().detach().numpy())
            _metrics.append(metrics_batch_t * alpha[0] + metrics_batch_y0 * alpha[1] + metrics_batch_y1 * alpha[2])

        loss_train[e] = np.mean(_loss)
        metric_train[e] = np.mean(_metrics)

        if use_validation:
            model.eval()
            batch = next(iter(loader_val))
            predictions = model(batch[0].to(device))
            loss_val_t, loss_val_y0, loss_val_y1 = _calculate_criterion_aipw(criterion_function=criterion,
                                                                             batch=batch,
                                                                             predictions=predictions,
                                                                             device=device)
            metric_val_t, metric_val_y0, metric_val_y1 = _calculate_metric_aipw(
                metric_functions=metric_functions,
                batch=batch,
                predictions=predictions)
            loss_val[e] = loss_val_t * alpha[0] + loss_val_y0 * alpha[1] + loss_val_y1 * alpha[2]
            metric_val[e] = metric_val_t * alpha[0] + metric_val_y0 * alpha[1] + metric_val_y1 * alpha[2]
        else:
            loss_val[e], metric_val[e] = None, None

    return model, loss_train, loss_val, metric_train, metric_val


def _freeze_layers(model, alpha):
    layers = ['logisticRegression', 'linearRegression_0', 'linearRegression_1']
    to_freeze = []
    for i, item in enumerate(alpha):
        if item == 1:
            to_unfreeze = [layers[i]+'.linear.weight', layers[i]+'.linear.bias']
        else:
            to_freeze.append(layers[i]+'.linear.weight')
            to_freeze.append(layers[i]+'.linear.bias')

    print('alpha and unfreeze and freeze', alpha, to_unfreeze, to_freeze)

    for name, layer in model.named_parameters():
        print('checking named layer', name)
        if name in to_freeze:
            layer.require_grad = False
            print('Frooze')
        else:
            layer.require_grad = True
            print( 'Not frooze')
    return model