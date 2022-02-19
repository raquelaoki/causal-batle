import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
import logging
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import helper_ate as ate
from dragonnet import dragonnet

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def make_model(params):
    logging.debug('Model - %s', params['model_name'])
    if params['model_name'] == 'dragonnet':
        model = dragonnet(n_covariates=params['n_covariates'], units1=params['units1'], units2=params['units2'],
                          units3=params['units3'],
                          type_original=params['type_original'])
        criterion = [criterion_function_dragonnet_opt1, criterion_function_dragonnet_opt2]
        metric_functions = [metric_function_dragonnet_opt1, metric_function_dragonnet_opt2]
        fit = fit_dragonnet
    else:
        logging.warning('%s not implemented', params['model_name'])
    logger.debug('...model constructed')
    return model, criterion, metric_functions, fit


def metric_function_dragonnet_opt1(batch, predictions):
    return roc_auc_score(batch[2], predictions['t'].detach().numpy())


def metric_function_dragonnet_opt2(batch, predictions):
    y_pred = predictions['y0'] * (1 - predictions['t']) + predictions['y1'] * predictions['t']
    return mean_squared_error(batch[1], y_pred.detach().numpy())


def criterion_function_dragonnet_opt1(batch,
                                      predictions,
                                      device='cpu', set='train'):
    t_predictions = predictions['t']
    t_obs = batch[2].to(device)
    criterion = nn.BCEWithLogitsLoss()
    return criterion(t_predictions, t_obs)


def criterion_function_dragonnet_opt2(batch,
                                      predictions,
                                      device='cpu', set='train'):
    y0_predictions, y1_predictions = predictions['y0'], predictions['y1']
    y_obs = batch[1].to(device)
    t_obs = batch[2].to(device)

    criterion = nn.MSELoss()
    loss_y0_batch = criterion(y0_predictions[t_obs == 0], y_obs[t_obs == 0])
    loss_y1_batch = criterion(y1_predictions[t_obs == 1], y_obs[t_obs == 1])
    return loss_y0_batch + loss_y1_batch


def calculate_criterion(criterion_function, batch, predictions, device='cpu', set='train'):
    return criterion_function(batch, predictions, device=device, set=set)


def calculate_metric(metric_function, batch, predictions):
    return metric_function(batch, predictions)


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
    # Train only to predict t
    metric_opt1 = metrics_functions[0]
    metric_opt2 = metrics_functions[1]
    criterion_opt1 = criterion[0]
    criterion_opt2 = criterion[1]

    logger.debug('...starting opt1')
    model, loss_train_t, loss_val_t, metric_train_t, metric_val_t = fit_optimization(
        model=model,
        epochs=epochs,
        loader_train=loader_train,
        optimizer=optimizer,
        criterion=criterion_opt1,
        metric_function=metric_opt1,
        use_validation=use_validation,
        device=device,
        loader_val=loader_val
    )

    freeze = ['representation_layer1_1.weight',
              'representation_layer1_1.bias',
              'representation_layer1_2.weight',
              'representation_layer1_2.bias',
              'representation_layer1_3.weight',
              'representation_layer1_3.bias',
              'dragonnet_head.t_predictions.weight',
              'dragonnet_head.t_predictions.bias'
              ]

    for name, param in model.named_parameters():
        if name in freeze:
            param.require_grad=False

    logger.debug('...starting opt2')
    model, loss_train_y, loss_val_y, metric_train_y, metric_val_y = fit_optimization(
        model=model,
        epochs=epochs,
        loader_train=loader_train,
        optimizer=optimizer,
        criterion=criterion_opt2,
        metric_function=metric_opt2,
        use_validation=use_validation,
        device=device,
        loader_val=loader_val
    )

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

    return model, loss, metrics


def fit_optimization(model,
                     epochs,
                     loader_train,
                     optimizer,
                     criterion,
                     metric_function,
                     use_validation,
                     device,
                     loader_val=None):
    loss_train, metric_train = np.zeros(epochs), np.zeros(epochs)
    if use_validation:
        loss_val, metric_val = np.zeros(epochs), np.zeros(epochs)
    for e in range(epochs):
        torch.cuda.empty_cache()
        _metrics = 0
        _loss = 0
        for i, batch in enumerate(loader_train):
            optimizer.zero_grad()
            predictions = model(batch[0].to(device))
            loss_batch = calculate_criterion(criterion_function=criterion,
                                             batch=batch,
                                             predictions=predictions,
                                             device=device)
            metrics_batch = calculate_metric(metric_function=metric_function,
                                             batch=batch,
                                             predictions=predictions)
            _loss += loss_batch.cpu().detach().numpy()
            _metrics += metrics_batch
            loss_batch.backward()
            optimizer.step()
        loss_train[e] = _loss / i
        metric_train[e] = _metrics / i

        if use_validation:
            batch = next(iter(loader_val))
            predictions = model(batch[0].to(device))
            loss_val[e] = calculate_criterion(criterion_function=criterion,
                                              batch=batch,
                                              predictions=predictions,
                                              device=device)
            metric_val[e] = calculate_metric(metric_function=metric_function,
                                             batch=batch,
                                             predictions=predictions)
        else:
            loss_val, metric_val = None, None
    return model, loss_train, loss_val, metric_train, metric_val


def fit_wrapper(params,
                loader_train, loader_test, loader_all,
                loader_val=None, use_tensorboard=False):
    """Fit models.
    :param criterion:
    :param metrics:
    :param model:
    :param loader_train:
    :param loader_val:
    :param loader_test:
    :param loader_all:
    :param params: keys: units1, units2, units3, binary_target, seed, batch_size, epochs_adam, epochs_sgd, reg_l2
    :return:
    """
    logger.debug("...fitting %s", params['model_name'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("...using %s", device)
    np.random.seed(params['seed'])

    if loader_val:
        use_validation = True
        best_metric_y_val = 999
        best_epoch = 0
    else:
        use_validation = False

    if use_tensorboard:
        writer_tensorboard = TensorboardWriter(path_logger, config_name)

    model, criterion, metric_functions, fit = make_model(params)

    if torch.cuda.is_available():
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    # TODO: update loss and metrics (make into a dict)
    model, loss, metrics = fit(epochs=params['max_epochs'],
                               model=model,
                               loader_train=loader_train,
                               optimizer=optimizer,
                               criterion=criterion,
                               metrics_functions=metric_functions,
                               use_validation=False,
                               device=device,
                               loader_val=None)

    if use_tensorboard:
        values = [loss_train_t[e], loss_train_y[e], metrics_train_t[e], metrics_train_y[e]]
        writer_tensorboard = update_tensorboar(writer_tensorboard, values, e, set='train')
        if use_validation:
            values = [loss_val_t[e], loss_val_y[e], metrics_val_t[e], metrics_val_y[e]]
            writer_tensorboard = update_tensorboar(writer_tensorboard, values, e, set='val')

    # Metrics on testins set
    batch = next(iter(loader_test))
    predictions = model(batch[0].to(device))
    metrics['metric_test_t'] = calculate_metric(metric_function=metric_functions[0],
                                                batch=batch,
                                                predictions=predictions,
                                                )
    metrics['metric_test_y'] = calculate_metric(metric_function=metric_functions[1],
                                                batch=batch,
                                                predictions=predictions,
                                                )

    logging.debug("...calculating ate")
    ate_naive_train = ate.calculate_ate(loader_train, model, single_batch=False, type_ate='naive')
    ate_aipw_train = ate.calculate_ate(loader_train, model, single_batch=False, type_ate='aipw')
    ate_naive_test = ate.calculate_ate(loader_test, model, single_batch=True, type_ate='naive')
    ate_aipw_test = ate.calculate_ate(loader_test, model, single_batch=True, type_ate='aipw')
    ate_naive_all = ate.calculate_ate(loader_all, model, single_batch=False, type_ate='naive')
    ate_aipw_all = ate.calculate_ate(loader_all, model, single_batch=False, type_ate='aipw')

    logging.debug("...fitting done.")

    # Organize output.
    # metrics = {'metric_y_train': metric_y_train, 'metric_t_train': metric_t_train,
    #           'metric_y_test': metric_y_test, 'metric_t_test': metric_t_test}
    # losses = {'loss_y_train': loss_train_y, 'loss_t_train': loss_train_t}

    ate_estimated = {'ate_naive_train': ate_naive_train, 'ate_aipw_train': ate_aipw_train,
                     'ate_naive_test': ate_naive_test, 'ate_aipw_test': ate_aipw_test,
                     'ate_naive_all': ate_naive_all, 'ate_aipw_all': ate_aipw_all}

    return metrics, loss, ate_estimated
