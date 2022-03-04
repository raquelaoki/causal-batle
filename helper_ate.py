import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

def calculate_ate(data_loader, model,
                  single_batch=False,
                  include_aipw=True,
                  title=''):
    """
    Calculate the Average Treatment Effect
    :param data_loader: if neural networks, needs to be a DataLoader objs
    :param model: object
    :param single_batch: False (all, train - contain several batches), True (val, test)
    :param include_aipw: if True, calculate naive and aipw. If False, only calculate naive.
    :return:
    """
    if not single_batch:
        y_obs, t_obs = np.array([]), np.array([])
        y0_pred, y1_pred, t_pred = np.array([]), np.array([]), np.array([])

        for i, batch in enumerate(data_loader):
            y_obs = np.concatenate([y_obs.reshape(-1), batch[1].reshape(-1)], 0)
            t_obs = np.concatenate([t_obs.reshape(-1), batch[2].reshape(-1)], 0)
            predictions = model(batch[0])
            t_predictions, y0_predictions, y1_predictions = predictions['t'], predictions['y0'], predictions['y1']
            y0_pred = np.concatenate([y0_pred.reshape(-1), y0_predictions.detach().numpy().reshape(-1)], 0)
            y1_pred = np.concatenate([y1_pred.reshape(-1), y1_predictions.detach().numpy().reshape(-1)], 0)
            t_pred = np.concatenate([t_pred.reshape(-1), t_predictions.detach().numpy().reshape(-1)], 0)

    else:
        batch = next(iter(data_loader))
        y_obs = batch[1].detach().numpy().reshape(-1)
        t_obs = batch[2].detach().numpy().reshape(-1)
        predictions = model(inputs=batch[0])
        t_pred, y0_pred, y1_pred = predictions['t'], predictions['y0'], predictions['y1']
        t_pred = t_pred.detach().numpy().reshape(-1)
        y0_pred = y0_pred.detach().numpy().reshape(-1)
        y1_pred = y1_pred.detach().numpy().reshape(-1)

    print('Sample of t_pred -',title,'  ', t_pred[0:5],t_obs[0:5])
    print('Sample of y_pred -',title,'  ', y0_pred[0:5],y1_pred[0:5],y_obs[0:5])

    if include_aipw:
        return _naive_ate(y0_pred, y1_pred, t_pred), _aipw_ate(t_obs, y_obs, y0_pred, y1_pred, t_pred)
    else:
        _naive_ate(y0_pred, y1_pred, t_pred), None


def _naive_ate(y0_pred, y1_pred, t_pred):
    ite = (y1_pred - y0_pred)
    return np.mean(_truncate_by_g(attribute=ite, g=t_pred, level=0.05))


def _aipw_ate(t_obs, y_obs, y0_pred, y1_pred, t_pred):
    y0_pred, y1_pred, t_pred, t_obs, y_obs = _truncate_all_by_g(y0_pred, y1_pred, t_pred, t_obs, y_obs)
    y_pred = y0_pred * (1 - t_obs) + y1_pred * t_obs
    p_ = t_obs * (1.0 / t_pred) - (1.0 - t_obs) / (1.0 - t_pred)
    ite = p_ * (y_obs - y_pred) + y1_pred - y0_pred
    return np.mean(ite)


def _truncate_by_g(attribute, g, level=0.05):
    """
    Remove rows with too low or too high g values. attribute and g must have same dimensions.
    :param attribute: column we want to keep after filted
    :param g: filter
    :param level: limites
    :return: filted attribute column
    """
    assert len(attribute) == len(g), 'Dimensions must be the same!'
    keep_these = np.logical_and(g >= level, g <= 1. - level)
    return attribute[keep_these]


def _truncate_all_by_g(y0_pred, y1_pred, t_pred, t_obs, y_obs, truncate_level=0.05):
    orig_t_pred = np.copy(t_pred)
    y0_pred = _truncate_by_g(np.copy(y0_pred), orig_t_pred, truncate_level)
    y1_pred = _truncate_by_g(np.copy(y1_pred), orig_t_pred, truncate_level)
    t_pred = _truncate_by_g(np.copy(t_pred), orig_t_pred, truncate_level)
    t_obs = _truncate_by_g(np.copy(t_obs), orig_t_pred, truncate_level)
    y_obs = _truncate_by_g(np.copy(y_obs), orig_t_pred, truncate_level)
    return y0_pred, y1_pred, t_pred, t_obs, y_obs


