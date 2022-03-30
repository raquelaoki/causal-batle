import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def calculate_ate(loader_train, loader_test, loader_all, model,
                  ate_method_list=['naive', 'aipw'], device='cpu',
                  forward_passes=None, filter_d=False):
    """
    :param loader_train: torch.utils.data.DataLoader obj.
    :param loader_test: torch.utils.data.DataLoader obj.
    :param loader_all:  torch.utils.data.DataLoader obj.
    :param model: torch.nn Module.
    :param ate_method_list: list with ATE methods' names.
    :param device: cpu or cuda.
    :param forward_passes: None, it is only used for the bayesian version.
    :param filter_d: None, it is only used for Causal-Batle version.
    :return: dictionary with estimated ate in each loader x method.
    """
    ate_train = _per_set_ate(loader_train, model, make_predictions=_make_predictions_regular,
                             methods_list=ate_method_list, loader_name='train', device=device)
    ate_test = _per_set_ate(loader_test, model, make_predictions=_make_predictions_regular,
                            methods_list=ate_method_list, loader_name='test', device=device)
    ate_all = _per_set_ate(loader_all, model, make_predictions=_make_predictions_regular,
                           methods_list=ate_method_list, loader_name='all', device=device)

    ate_estimated = {}
    ate_estimated.update(ate_all)
    ate_estimated.update(ate_train)
    ate_estimated.update(ate_test)

    return ate_estimated


def calculate_ate_bayesian(loader_train, loader_test, loader_all, model,
                           ate_method_list=['naive', 'aipw'], device='cpu',
                           forward_passes=10, filter_d=False):
    """
    :param loader_train: torch.utils.data.DataLoader obj.
    :param loader_test: torch.utils.data.DataLoader obj.
    :param loader_all:  torch.utils.data.DataLoader obj.
    :param model: torch.nn Module.
    :param ate_method_list: list with ATE methods' names.
    :param device: cpu or cuda.
    :param forward_passes: number of forward passes for MC-dropout.
    :param filter_d: None, it is only used for Causal-Batle version.
    :return: dictionary with estimated ate in each loader x method.
    """
    ate_train = _per_set_ate(loader_train, model, make_predictions=_make_predictions_dropout,
                             methods_list=ate_method_list, loader_name='train',
                             device=device, forward_passes=forward_passes, filter_d=filter_d)
    ate_test = _per_set_ate(loader_test, model, make_predictions=_make_predictions_dropout,
                            methods_list=ate_method_list, loader_name='test',
                            device=device, forward_passes=forward_passes, filter_d=filter_d)
    ate_all = _per_set_ate(loader_all, model, make_predictions=_make_predictions_dropout,
                           methods_list=ate_method_list, loader_name='all',
                           device=device, forward_passes=forward_passes, filter_d=filter_d)

    ate_estimated = {}
    ate_estimated.update(ate_all)
    ate_estimated.update(ate_train)
    ate_estimated.update(ate_test)

    return ate_estimated


def _make_predictions_regular(data_loader, model, device, place_holder, filter_d=False):
    """Calculate predictions.

    :param data_loader: torch.utils.data.DataLoader obj.
    :param model: torch.nn Module.
    :param device: cpu or cuda.
    :param place_holder: .
    :param filter_d: None, it is only used for Causal-Batle version.
    :return: arrays with predictions
    """
    y_obs, t_obs = np.array([]), np.array([])
    y0_pred, y1_pred, t_pred = np.array([]), np.array([]), np.array([])

    for i, batch in enumerate(data_loader):
        y_obs = np.concatenate([y_obs.reshape(-1), batch[1].reshape(-1)], 0)
        t_obs = np.concatenate([t_obs.reshape(-1), batch[2].reshape(-1)], 0)
        predictions = model(batch[0].to(device))
        t_predictions, y0_predictions, y1_predictions = predictions['t'], predictions['y0'], predictions['y1']
        y0_pred = np.concatenate([y0_pred.reshape(-1), y0_predictions.cpu().detach().numpy().reshape(-1)], 0)
        y1_pred = np.concatenate([y1_pred.reshape(-1), y1_predictions.cpu().detach().numpy().reshape(-1)], 0)
        t_pred = np.concatenate([t_pred.reshape(-1), t_predictions.cpu().detach().numpy().reshape(-1)], 0)
    return t_pred, y0_pred, y1_pred, t_obs, y_obs


def _make_predictions_dropout(data_loader, model, device, forward_passes, filter_d=True):
    """
    Reference: (https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch)

    :param data_loader:torch.utils.data.DataLoader obj.
    :param model:torch.nn Module.
    :param device: cpu or cuda.
    :param forward_passes: number of MC dropout forward passes.
    :return: array with predictions.
    """
    y_obs, t_obs = np.array([]), np.array([])
    y0_pred_mean, y1_pred_mean, t_pred_mean = np.array([]), np.array([]), np.array([])

    for i, batch in enumerate(data_loader):

        predictions = model(batch[0].to(device))
        t_predictions, y0_predictions, y1_predictions = predictions['t'], predictions['y0'], predictions['y1']
        y0_pred = y0_predictions.mean.cpu().detach().numpy().reshape(-1, 1)
        y1_pred = y1_predictions.mean.cpu().detach().numpy().reshape(-1, 1)
        t_pred = t_predictions.mean.cpu().detach().numpy().reshape(-1, 1)

        # Make more predictions in the same batch.
        for j in range(forward_passes - 1):
            predictions = model(batch[0].to(device))
            t_predictions, y0_predictions, y1_predictions = predictions['t'], predictions['y0'], predictions['y1']
            y0_pred = np.concatenate([y0_pred, y0_predictions.mean.cpu().detach().numpy().reshape(-1, 1)], 1)
            y1_pred = np.concatenate([y1_pred, y1_predictions.mean.cpu().detach().numpy().reshape(-1, 1)], 1)
            t_pred = np.concatenate([t_pred, t_predictions.mean.cpu().detach().numpy().reshape(-1, 1)], 1)

        # Mean of predictions.
        y0_pred = np.mean(y0_pred, axis=1)
        y1_pred = np.mean(y1_pred, axis=1)
        t_pred = np.mean(t_pred, axis=1)

        if filter_d:
            d = batch[3].reshape(-1)
            # Concatenate Means.
            y0_pred_mean = np.concatenate([y0_pred_mean.reshape(-1), y0_pred[d == 1].reshape(-1)], 0)
            y1_pred_mean = np.concatenate([y1_pred_mean.reshape(-1), y1_pred[d == 1].reshape(-1)], 0)
            t_pred_mean = np.concatenate([t_pred_mean.reshape(-1), t_pred[d == 1].reshape(-1)], 0)
            # Concatenate obs.
            y_obs = np.concatenate([y_obs.reshape(-1), batch[1][d == 1].reshape(-1)], 0)
            t_obs = np.concatenate([t_obs.reshape(-1), batch[2][d == 1].reshape(-1)], 0)
        else:
            # Concatenate Means.
            y0_pred_mean = np.concatenate([y0_pred_mean.reshape(-1), y0_pred.reshape(-1)], 0)
            y1_pred_mean = np.concatenate([y1_pred_mean.reshape(-1), y1_pred.reshape(-1)], 0)
            t_pred_mean = np.concatenate([t_pred_mean.reshape(-1), t_pred.reshape(-1)], 0)
            # Concatenate obs.
            y_obs = np.concatenate([y_obs.reshape(-1), batch[1].reshape(-1)], 0)
            t_obs = np.concatenate([t_obs.reshape(-1), batch[2].reshape(-1)], 0)

    return t_pred_mean, y0_pred_mean, y1_pred_mean, t_obs, y_obs


def _per_set_ate(data_loader,
                 model,
                 make_predictions,
                 methods_list=['naive'],
                 loader_name='DEFAULT',
                 device='cpu',
                 forward_passes=None,
                 filter_d=False):
    """
    Calculate the Average Treatment Effect
    :param data_loader:torch.utils.data.DataLoader obj.
    :param model:torch.nn Module.
    :param make_predictions: function.
    :param device: cpu or cuda.
    :param methods_list: list with ate methods names to use.
    :param loader_name: str, set name (train, all, test).
    :param forward_passes: int.
    :param filter_d: bool.
    """

    t_pred, y0_pred, y1_pred, t_obs, y_obs = make_predictions(data_loader, model, device, forward_passes,
                                                              filter_d=filter_d)

    implemented_methods = {'naive': _naive_ate,
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
    """Calculate ATE using naive formula."""
    ite = (y1_pred - y0_pred)
    return np.mean(_truncate_by_g(attribute=ite, g=t_pred, level=0.05))


def _ipw_ate(t_obs, y_obs, y0_pred, y1_pred, t_pred):
    """Calculate ATE using IPW formula."""
    y0_pred, y1_pred, t_pred, t_obs, y_obs = _truncate_all_by_g(y0_pred, y1_pred, t_pred, t_obs, y_obs)
    ipw1 = y1_pred / t_pred
    ipw0 = y0_pred / (1.0 - t_pred)
    ipw1 = np.mean(ipw1[t_obs == 1])
    ipw0 = np.mean(ipw0[t_obs == 0])
    return ipw1 - ipw0


def _aipw_ate(t_obs, y_obs, y0_pred, y1_pred, t_pred):
    """Calculate ATE using AIPW formula."""
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
    assert len(attribute) == len(g), 'Dimensions must be the same!' + str(len(attribute)) + ' and ' + str(len(g))
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
