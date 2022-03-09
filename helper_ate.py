import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def calculate_ate(loader_train, loader_test, loader_all, model, ate_method_list=['naive', 'ipw', 'aipw']):
    ate_train = _per_set_ate(loader_train, model, single_batch=False,
                             methods_list=ate_method_list, loader_name='train')
    ate_test = _per_set_ate(loader_test, model, single_batch=True,
                            methods_list=ate_method_list, loader_name='test')
    ate_all = _per_set_ate(loader_all, model, single_batch=False,
                           methods_list=ate_method_list, loader_name='all')

    ate_estimated = {}
    ate_estimated.update(ate_all)
    ate_estimated.update(ate_train)
    ate_estimated.update(ate_test)

    return ate_estimated


def _per_set_ate(data_loader,
                 model,
                 single_batch=False,
                 methods_list=['naive'],
                 loader_name='DEFAULT'):
    """
    Calculate the Average Treatment Effect
    :param include_ipw:
    :param data_loader: if neural networks, needs to be a DataLoader objs
    :param model: object
    :param single_batch: False (all, train - contain several batches), True (val, test)
    :param include_aipw: if True, calculate naive and aipw. If False, only calculate naive.
    :return:ate_naive_train
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

    implemented_methods = {'naive': _naive_ate,
                           'ipw': _ipw_ate,
                           'aipw': _aipw_ate,
                           }
    estimated_ate = {}
    for method in implemented_methods.keys():
        _key = 'ate_' + method + '_' + loader_name
        if method in methods_list:
            estimated_ate[_key] = implemented_methods[method](t_obs, y_obs, y0_pred, y1_pred, t_pred)
        else:
            estimated_ate[_key] = None

    return estimated_ate


def _naive_ate(t_obs, y_obs, y0_pred, y1_pred, t_pred):
    ite = (y1_pred - y0_pred)
    return np.mean(_truncate_by_g(attribute=ite, g=t_pred, level=0.05))


def _ipw_ate(t_obs, y_obs, y0_pred, y1_pred, t_pred):
    y0_pred, y1_pred, t_pred, t_obs, y_obs = _truncate_all_by_g(y0_pred, y1_pred, t_pred, t_obs, y_obs)
    ipw1 = y1_pred / t_pred
    ipw0 = y0_pred / (1.0 - t_pred)

    #print('predi', y1_pred[t_obs == 1], y0_pred[t_obs == 0])
    #print('predi - t', np.mean(t_pred[t_obs == 1]), np.mean(t_pred[t_obs == 0]))
    #print('predi - ipw', ipw1[t_obs == 1], ipw0[t_obs == 0])

    ipw1 = np.mean(ipw1[t_obs==1])
    ipw0 = np.mean(ipw0[t_obs==0])
    #print('averages ',ipw1, ipw0)

    return  ipw1-ipw0


def _aipw_ate(t_obs, y_obs, y0_pred, y1_pred, t_pred):
    y0_pred, y1_pred, t_pred, t_obs, y_obs = _truncate_all_by_g(y0_pred, y1_pred, t_pred, t_obs, y_obs)
    ite_dif = (y1_pred - y0_pred)
    ite_prop1 = t_obs * (y_obs - y1_pred) / t_pred
    ite_prop0 = (1 - t_obs) * (y_obs - y0_pred) / (1 - t_pred)
    ite = ite_dif + ite_prop1 - ite_prop0
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
