import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score, roc_curve
import logging
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import helper_ate as ate
from dragonnet import dragonnet

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def _find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------
    list type, with optimal cutoff value
    https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    Adapted
    """

    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc.drop(0, axis=0, inplace=True)
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


def find_optimal_cutoff_wrapper_t(loader_train, model, device):
    target, predicted = [], []
    for i, batch in enumerate(loader_train):
        predictions = model(batch[0].to(device))
        target = np.concatenate((target, batch[2].numpy().reshape(-1)), 0)
        predicted = np.concatenate((predicted, predictions['t'].detach().numpy().reshape(-1)), 0)

    target = target.reshape(-1)
    predicted = predicted.reshape(-1)
    return _find_optimal_cutoff(target, predicted)[0]


def make_model(params):
    logging.debug('Model - %s', params['model_name'])
    if params['model_name'] == 'dragonnet':
        model = dragonnet(n_covariates=params['n_covariates'], units1=params['units1'], units2=params['units2'],
                          units3=params['units3'],
                          type_original=params['type_original'])
        criterion = [_criterion_function_dragonnet_t, _criterion_function_dragonnet_y]
        metric_functions = [_metric_function_dragonnet_t, _metric_function_dragonnet_y]
        fit = fit_dragonnet
    else:
        logging.warning('%s not implemented', params['model_name'])
    logger.debug('...model constructed')
    return model, criterion, metric_functions, fit


def _metric_function_dragonnet_t(batch, predictions):
    return roc_auc_score(batch[2], predictions['t'].detach().numpy())


def _metric_function_dragonnet_y(batch, predictions):
    y_pred = predictions['y0'] * (1 - predictions['t']) + predictions['y1'] * predictions['t']
    return mean_squared_error(batch[1], y_pred.detach().numpy())


def _criterion_function_dragonnet_t(batch,
                                    predictions,
                                    device='cpu', set='train'):
    t_predictions = predictions['t']
    t_obs = batch[2].to(device)
    criterion = nn.BCELoss()
    return criterion(t_predictions, t_obs)


def _criterion_function_dragonnet_y(batch,
                                    predictions,
                                    device='cpu', set='train'):
    y0_predictions, y1_predictions = predictions['y0'], predictions['y1']
    y_obs = batch[1].to(device)
    t_obs = batch[2].to(device)

    criterion = nn.MSELoss()
    loss_y0_batch = criterion(y0_predictions[t_obs == 0], y_obs[t_obs == 0])
    loss_y1_batch = criterion(y1_predictions[t_obs == 1], y_obs[t_obs == 1])
    return loss_y0_batch + loss_y1_batch


def _calculate_criterion_dragonnet(criterion_function, batch, predictions, device='cpu', set='train'):
    loss_t = _calculate_criterion(criterion_function=criterion_function[0],
                                  batch=batch,
                                  predictions=predictions,
                                  device=device,
                                  set=set)
    loss_y = _calculate_criterion(criterion_function=criterion_function[1],
                                  batch=batch,
                                  predictions=predictions,
                                  device=device,
                                  set=set)
    return loss_t, loss_y


def _calculate_criterion(criterion_function, batch, predictions, device='cpu', set='train'):
    return criterion_function(batch, predictions, device=device, set=set)


def _calculate_metric(metric_function, batch, predictions):
    return metric_function(batch, predictions)


def _calculate_metric_dragonnet(metrics_functions, batch, predictions):
    metrics_t = _calculate_metric(metric_function=metrics_functions[0],
                                  batch=batch,
                                  predictions=predictions)
    metrics_y = _calculate_metric(metric_function=metrics_functions[0],
                                  batch=batch,
                                  predictions=predictions)
    return metrics_t, metrics_y


class TensorboardWriter:
    def __init__(self, path_logger, name_config):
        date = self.get_date()
        full_path = (
                self.get_home_dir() + "/" + path_logger + date + "/" + name_config + "/"
        )
        print("Tensorboard folder path - {}".format(full_path))
        self.writer = SummaryWriter(log_dir=full_path)

    # Add day, month and year to path
    def get_date(self):
        now = datetime.now()  # Current date and time (Hour, minute)
        date = now.strftime("%Y_%m_%d_%H_%M")
        return date

    def get_home_dir(self):
        return os.getenv("HOME")

    def add_scalar(self, name_metric, value_metric, epoch):
        self.writer.add_scalar(name_metric, value_metric, epoch)

    def end_writer(self):
        # Make sure all pending events have been written to disk
        self.writer.flush()


def update_tensorboar(writer_tensorboard, values, e, set='train'):
    names = ['loss_t_' + set, 'loss_y_' + set, 'f1_t_' + set, 'mse_y_' + set]
    assert len(values) == len(names)
    for i in range(len(names)):
        writer_tensorboard.add_scalar(names[i], values[i], e)
    writer_tensorboard.end_writer()
    return writer_tensorboard


def fit_dragonnet(epochs,
                  model,
                  loader_train,
                  optimizer,
                  criterion,
                  metrics_functions,
                  use_validation,
                  device,
                  loader_val=None):
    logger.debug('...starting')

    loss_train_t, metric_train_t = np.zeros(epochs), np.zeros(epochs)
    loss_train_y, metric_train_y = np.zeros(epochs), np.zeros(epochs)

    if use_validation:
        loss_val_t, metric_val_t = np.zeros(epochs), np.zeros(epochs)
        loss_val_y, metric_val_y = np.zeros(epochs), np.zeros(epochs)

    for e in range(epochs):
        torch.cuda.empty_cache()
        _metrics_t, _metrics_y = 0, 0
        _loss_t, _loss_y = 0, 0
        for i, batch in enumerate(loader_train):
            optimizer.zero_grad()
            predictions = model(batch[0].to(device))
            loss_batch_t, loss_batch_y = _calculate_criterion_dragonnet(criterion_function=criterion,
                                                                        batch=batch,
                                                                        predictions=predictions,
                                                                        device=device)

            metrics_batch_t, metrics_batch_y = _calculate_metric_dragonnet(metrics_functions=metrics_functions,
                                                                           batch=batch,
                                                                           predictions=predictions)
            loss_batch = loss_batch_t + loss_batch_y
            loss_batch.backward()
            optimizer.step()
            _loss_t += loss_batch_t.cpu().detach().numpy()
            _loss_y += loss_batch_y.cpu().detach().numpy()
            _metrics_t += metrics_batch_t
            _metrics_t += metrics_batch_y

        loss_train_t[e] = _loss_t / (i + 1)
        loss_train_y[e] = _loss_y / (i + 1)
        metric_train_t[e] = _metrics_t / (i + 1)
        metric_train_y[e] = _metrics_y / (i + 1)

        if use_validation:
            batch = next(iter(loader_val))
            predictions = model(batch[0].to(device))

            loss_val_t[e], loss_val_y[e] = _calculate_criterion_dragonnet(criterion_function=criterion,
                                                                          batch=batch,
                                                                          predictions=predictions,
                                                                          device=device)
            metric_val_t[e], metric_val_y[e] = _calculate_metric_dragonnet(metrics_functions=metrics_functions,
                                                                           batch=batch,
                                                                           predictions=predictions)
        else:
            loss_val_t[e], loss_val_y[e] = None, None
            metric_val_t[e], metric_val_y[e] = None, None

    loss = {'loss_train_t': loss_train_t,
            'loss_val_t': loss_val_t,
            'loss_train_y': loss_train_y,
            'loss_val_y': loss_val_y
            }
    metrics = {'metric_train_t': metric_train_t,
               'metric_val_t': metric_val_t,
               'metric_train_y': metric_train_y,
               'metric_val_y': metric_val_y
               }

    thhold = find_optimal_cutoff_wrapper_t(loader_train=loader_train, model=model, device=device)
    metrics['thhold'] = thhold

    return model, loss, metrics


def fit_wrapper(params,
                loader_train, loader_test, loader_all,
                loader_val=None,
                use_tensorboard=False,
                use_validation=False):
    logger.debug("...fitting %s", params['model_name'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("...using %s", device)
    np.random.seed(params['seed'])

    if use_validation:
        best_metric_y_val = 999
        best_epoch = 0

    if use_tensorboard:
        writer_tensorboard = TensorboardWriter(path_logger, config_name)

    model, criterion, metric_functions, fit = make_model(params)

    if torch.cuda.is_available():
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    model, loss, metrics = fit(epochs=params['max_epochs'],
                               model=model,
                               loader_train=loader_train,
                               optimizer=optimizer,
                               criterion=criterion,
                               metrics_functions=metric_functions,
                               use_validation=use_validation,
                               device=device,
                               loader_val=loader_val)

    if use_tensorboard:
        values = [loss_train_t[e], loss_train_y[e], metrics_train_t[e], metrics_train_y[e]]
        writer_tensorboard = update_tensorboar(writer_tensorboard, values, e, set='train')
        if use_validation:
            values = [loss_val_t[e], loss_val_y[e], metrics_val_t[e], metrics_val_y[e]]
            writer_tensorboard = update_tensorboar(writer_tensorboard, values, e, set='val')

    # Metrics on testins set
    batch = next(iter(loader_test))
    predictions = model(batch[0].to(device))
    metrics['metric_test_t'] = _calculate_metric(metric_function=metric_functions[0],
                                                 batch=batch,
                                                 predictions=predictions,
                                                 )
    metrics['metric_test_y'] = _calculate_metric(metric_function=metric_functions[1],
                                                 batch=batch,
                                                 predictions=predictions,
                                                 )

    logging.debug("...calculating ate")
    ate_naive_train, ate_aipw_train = ate.calculate_ate(loader_train, model, single_batch=False,
                                                        include_aipw=True, thhold=metrics['thhold'])
    ate_naive_test, ate_aipw_test = ate.calculate_ate(loader_test, model, single_batch=True,
                                                      include_aipw=True, thhold=metrics['thhold'])
    ate_naive_all, ate_aipw_all = ate.calculate_ate(loader_all, model, single_batch=False,
                                                    include_aipw=True, thhold=metrics['thhold'])

    logging.debug("...fitting done.")

    # Organize output.
    # metrics = {'metric_y_train': metric_y_train, 'metric_t_train': metric_t_train,
    #           'metric_y_test': metric_y_test, 'metric_t_test': metric_t_test}
    # losses = {'loss_y_train': loss_train_y, 'loss_t_train': loss_train_t}

    ate_estimated = {'ate_naive_train': ate_naive_train, 'ate_aipw_train': ate_aipw_train,
                     'ate_naive_test': ate_naive_test, 'ate_aipw_test': ate_aipw_test,
                     'ate_naive_all': ate_naive_all, 'ate_aipw_all': ate_aipw_all}

    return metrics, loss, ate_estimated
