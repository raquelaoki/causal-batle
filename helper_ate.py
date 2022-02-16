import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------
    list type, with optimal cutoff value
    https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    """

    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


def calculate_ate(data_loader, model, single_batch=False, type_ate='naive'):
    """
    Calculate the Average Treatment Effect
    :param data_loader: if neural networks, needs to be a DataLoader objs
    :param model: object
    :param single_batch: True (all, train - contain several batches), False (val, test)
    :param type_ate: naive, aipw
    :return:
    """
    if single_batch:
        y_obs, t_obs, y0_pred, y1_pred, t_pred = [], [], [], [], []
        for i, batch in enumerate(data_loader):
            y_obs = np.concatenate([y_obs, batch[1]], 1)
            t_obs = np.concatenate([t_obs, batch[2]], 1)
            t_predictions, y0_predictions, y1_predictions = model(batch[0])
            y0_pred = np.concatenate([y0_pred, y0_predictions], 1)
            y1_pred = np.concatenate([y1_pred, y1_predictions], 1)
            t_pred = np.concatenate([t_pred, t_predictions], 1)
    else:
        batch = next(iter(data_loader))
        y_obs = batch[1]
        t_obs = batch[2]
        t_pred, y0_pred, y1_pred = model(batch[0])

    if type_ate == 'naive':
        return _naive_ate(t_obs, y0_pred, y1_pred, t_pred)
    elif type_ate == 'aipw':
        return _aipw_ate(t_obs, y_obs, y0_pred, y1_pred, t_pred)
    else:
        logging.debug("Option not implemented")


def _naive_ate(t_obs, y0_pred, y1_pred, t_pred):
    ite = (y1_pred - y0_pred)
    print(ite[0:10], t_pred[0:10])
    return np.mean(truncate_by_g(ite, t_pred, level=0.05))


def _aipw_ate(t_obs, y_obs, y0_pred, y1_pred, t_pred):
    y0_pred, y1_pred, t_pred, t_obs, y_obs = truncate_all_by_g(y0_pred, y1_pred, t_pred, t_obs, y_obs)
    y_pred = y0_pred * (1 - t_obs) + y1_pred * t_obs
    p_ = t_obs * (1.0 / t_pred) - (1.0 - t_obs) / (1.0 - t_pred)
    ite = p_ * (y_obs - y_pred) + y1_pred - y0_pred
    return np.mean(ite)


def truncate_by_g(attribute, g, level=0.05):
    keep_these = np.logical_and(g >= level, g <= 1. - level)
    print('keeping', keep_these[0:10])
    return attribute[keep_these]


def truncate_all_by_g(y0_pred, y1_pred, t_pred, t_obs, y_obs, truncate_level=0.05):
    orig_t_pred = np.copy(t_pred)
    y0_pred = truncate_by_g(np.copy(y0_pred), orig_t_pred, truncate_level)
    y1_pred = truncate_by_g(np.copy(y1_pred), orig_t_pred, truncate_level)
    t_pred = truncate_by_g(np.copy(t_pred), orig_t_pred, truncate_level)
    t_obs = truncate_by_g(np.copy(t_obs), orig_t_pred, truncate_level)
    y_obs = truncate_by_g(np.copy(y_obs), orig_t_pred, truncate_level)
    return y0_pred, y1_pred, t_pred, t_obs, y_obs


