import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
import logging
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import helper_ate as ate

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def make_metrics(binary_target=False, include_treatment=True):
    if binary_target:
        logger.warning("Binary targers not implemented")
    else:
        if include_treatment:
            return [mean_squared_error, roc_auc_score]
        else:
            return [mean_squared_error]


def make_criterion_dragonnet_original(binary_target=False):
    """
    Returns array with criterion objects. Called once on model definition.
    :param binary_target:
    :return: array with criterion objects
    """
    if binary_target:
        return [nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss()]
    else:
        return [nn.BCEWithLogitsLoss(), nn.MSELoss()]


def calculate_criterion_dragonnet_original(criterion, batch, t_predictions,
                                           y0_predictions, y1_predictions,
                                           device='cpu', set='train'):
    loss_t_batch = criterion[0](t_predictions, batch[2].to(device))
    y_predictions = y0_predictions * (t_predictions == 0) + y1_predictions * (t_predictions == 1)
    loss_y_batch = criterion[1](y_predictions, batch[1].to(device))
    loss_batch = loss_t_batch + loss_y_batch
    if set=='train':
        return loss_batch, loss_t_batch, loss_y_batch
    else:
        return loss_batch.cpu().detach().numpy(), loss_t_batch.cpu().detach().numpy(), loss_y_batch.cpu().detach().numpy()
def calculate_criterion(model_name, criterion, batch,
                        t_predictions, y0_predictions, y1_predictions,
                        device='cpu', set='train'):
    if model_name == 'dragonnet':
        return calculate_criterion_dragonnet_original(criterion, batch, t_predictions,
                                                      y0_predictions, y1_predictions,
                                                      device, set)


def calculate_metric(metrics, batch, t_predictions,
                     y0_predictions, y1_predictions,
                     device='cpu', include_treatment=True):
    y_predictions = y0_predictions * (t_predictions == 0) + y1_predictions * (t_predictions == 1)
    metric_y = metrics[0](batch[1], y_predictions.cpu().detach().numpy())
    metric_t = 0
    if include_treatment:
        metric_t = metrics[1](batch[2].to(device), t_predictions.cpu().detach().numpy())
    return metric_y, metric_t


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
    names = ['loss_' + set, 'loss_t_' + set, 'loss_y_' + set, 'f1_t_' + set, 'mse_y_' + set]
    assert len(values) == len(names)
    for i in range(len(names)):
        writer_tensorboard.add_scalar(names[i], values[i], e)
    writer_tensorboard.end_writer()
    return writer_tensorboard


def fit(model, criterion, metrics, params,
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
    else:
        use_validation = False

    if use_tensorboard:
        writer_tensorboard = TensorboardWriter(path_logger, config_name)

    if torch.cuda.is_available():
        model.to(device)

    optimizer1 = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer1, gamma=params["gamma"])
    # optimizer2 = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    loss_train = np.zeros(params['max_epochs'])
    loss_train_t, loss_train_y = np.zeros(params['max_epochs']), np.zeros(params['max_epochs'])
    metric_y_train, metric_t_train = np.zeros(params['max_epochs']), np.zeros(params['max_epochs'])

    if use_validation:
        loss_val = np.zeros(params['max_epochs'])
        loss_val_t, loss_val_y = np.zeros(params['max_epochs']), np.zeros(params['max_epochs'])
        metric_y_val, metric_t_val = np.zeros(params['max_epochs']), np.zeros(params['max_epochs'])

        best_metric_y_val = 999
        best_epoch = 0

    for e in range(params['max_epochs']):
        torch.cuda.empty_cache()
        _metrics_y, _metrics_t = [], []
        _loss, _loss_t, _loss_y = 0, 0, 0
        for i, batch in enumerate(loader_train):
            optimizer1.zero_grad()

            t_predictions, y0_predictions, y1_predictions = model(batch[0].to(device))
            loss_batch, loss_t_batch, loss_y_batch = calculate_criterion(params['model_name'], criterion,
                                                                         batch, t_predictions,
                                                                         y0_predictions, y1_predictions, device)
            _metrics_y_batch, _metrics_t_batch = calculate_metric(metrics, batch, t_predictions,
                                                                  y0_predictions, y1_predictions,
                                                                  device, include_treatment=True)
            _metrics_y.append(_metrics_y_batch)
            _metrics_t.append(_metrics_t_batch)

            loss_batch.backward()
            _loss += loss_batch.cpu().detach().numpy()
            _loss_t += loss_t_batch.cpu().detach().numpy()
            _loss_y += loss_y_batch.cpu().detach().numpy()

            optimizer1.step()
        loss_train[e] = _loss / i
        loss_train_t[e] = _loss_t / i
        loss_train_y[e] = _loss_y / i

        metric_y_train[e] = np.mean(_metrics_y)
        metric_t_train[e] = np.mean(_metrics_t)

        # Validation
        if use_validation:
            batch = next(iter(loader_val))
            t_predictions, y0_predictions, y1_predictions = model(batch[0].to(device))
            loss_val[e], loss_val_t[e], loss_val_y[e] = calculate_criterion(params['model_name'], criterion,
                                                                            batch, t_predictions,
                                                                            y0_predictions, y1_predictions, device,
                                                                            set='val')
            metric_y_val[e], metric_t_val[e] = calculate_metric(metrics, batch, t_predictions,
                                                                y0_predictions, y1_predictions,
                                                                device, include_treatment=True)

        # Updating tensorboard.
        if use_tensorboard:
            values = [loss_train[e], loss_train_t[e], loss_train_y[e], metric_t_train[e], metric_y_train[e]]
            writer_tensorboard = update_tensorboar(writer_tensorboard, values, e, set='train')
            if use_validation:
                values = [loss_val[e], loss_val_t[e], loss_val_y[e], metric_t_val[e], metric_y_val[e]]
                writer_tensorboard = update_tensorboar(writer_tensorboard, values, e, set='val')

    # Metrics on testins set
    batch = next(iter(loader_test))
    t_predictions, y0_predictions, y1_predictions = model(batch[0].to(device))
    metric_y_test, metric_t_test = calculate_metric(metrics, batch, t_predictions,
                                                        y0_predictions, y1_predictions,
                                                        device, include_treatment=True)
    #TODO: save metric_y_test, metric_t_test
    #TODO: save estimated ate
    ate_naive_train = ate.calculate_ate(loader_train, model, single_batch=False,type_ate='naive')
    ate_aipw_train = ate.calculate_ate(loader_train, model, single_batch=False,type_ate='aipw')
    ate_naive_test = ate.calculate_ate(loader_test, model, single_batch=True,type_ate='naive')
    ate_aipw_test = ate.calculate_ate(loader_test, model, single_batch=True,type_ate='aipw')
    ate_naive_all = ate.calculate_ate(loader_all, model, single_batch=False,type_ate='naive')
    ate_aipw_all = ate.calculate_ate(loader_all, model, single_batch=False,type_ate='aipw')


    logging.debug("...fitting done.")
    return ate_naive_train, ate_aipw_train, ate_naive_test, ate_aipw_test, ate_naive_all, ate_aipw_all
