import pandas as pd
import logging
import helper_fit as hfit
import data_preprocessing as dp
from dragonnet import dragonnet

# temp
import numpy as np
import sklearn as sk

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def make_data(params):
    if params['data_name'] == 'gwas':
        data_s, data_t, tau = dp.make_gwas(params)
        return data_s, data_t, tau


def run_model(params):
    data_s, data_t, tau = make_data(params)

    # Running liner model tau = m1() - m0()
    x_train, t_train, y_train = data_t.x_train, data_t.t_train.reshape(-1), data_t.y_train
    x_test, t_test, y_test = data_t.x_test, data_t.t_test.reshape(-1), data_t.y_test
    x_train0, x_train1 = x_train[t_train == 0, :], x_train[t_train == 1, :]
    y_train0, y_train1 = y_train[t_train == 0], y_train[t_train == 1]

    m0 = sk.linear_model.LinearRegression().fit(x_train0, y_train0)
    m1 = sk.linear_model.LinearRegression().fit(x_train1, y_train1)

    pred0 = m0.predict(x_test)
    pred1 = m1.predict(x_test)

    small_test = {'pred0': pred0, 'obs': y_test, 't': t_test, 'pred1': pred1}

    if params['use_transfer']:
        tloader_train, tloader_val, tloader_test, tloader_all = data_t.loader(batch=params['batch_size'],
                                                                              shuffle=params['shuffle'],
                                                                              seed=0)
        sloader_train, sloader_val, sloader_test, sloader_all = data_s.loader(batch=params['batch_size'],
                                                                              shuffle=params['shuffle'],
                                                                              seed=0)
    else:
        tloader_train, tloader_val, tloader_test, tloader_all = data_t.loader(batch=params['batch_size'],
                                                                              shuffle=params['shuffle'],
                                                                              seed=0
                                                                              )

    metrics, loss, ate = hfit.fit_wrapper(params=params,
                                          loader_train=tloader_train,
                                          loader_test=tloader_test,
                                          loader_all=tloader_all,
                                          loader_val=tloader_val,
                                          use_validation=params['use_validation'])

    return metrics, loss, ate, tau, small_test


def run_methdos_(X_train, X_test, y_train, y_test, params):
    # TODO : update run_methods
    """Run all params['methods'] methods.
    This function will run the methods and generate a pd.DataFrame with the results.
    Args:
        X_train: Features of training set.
        X_test: Features of testing set.
        y_train: Target of training set.
        y_test: Target of testing set.
        params: Dictionary with parameters.
    Returns:
        pd.Dataframe with the results.
    """
    output = pd.DataFrame(columns=['method', 'config',
                                   'metric_train', 'metric_test'])

    for method in params.get('methods', ['new_method']):
        if method == 'new_method':
            model = NewMethod()
        elif method == 'baseline':
            model = Baseline()
        else:
            raise ValueError(f"Method {method} not implemented.")

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        out = {
            'method': method,
            'config': params.get('config', 'None'),
            'metric_train': metric(y_train, y_train_pred),
            'metric_test': metric(y_test, y_test_pred),
        }
        output = output.append(out, ignore_index=True)

    return output
