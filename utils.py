import pandas as pd
import logging
import helper_fit as hfit
import data_preprocessing as dp
from dragonnet import dragonnet

#temp
import numpy as np
import sklearn as sk



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def make_data(params):
    if params['data_name'] == 'gwas':
        data_s, data_t, tau = dp.make_gwas(params)
        return data_s, data_t, tau


def run_model(params):
    data_s, data_t, tau = make_data(params)

    #Running liner model
    x_train = np.concatenate([data_t.x_train,data_t.t_train],1)
    x_test = np.concatenate([data_t.x_test,data_t.t_test],1)
    t_counter = [1-item for item in data_t.t_test]
    x_counter = np.concatenate([data_t.x_test,t_counter],1)
    m1 = sk.linear_model.LinearRegression().fit(x_train,data_t.y_train)

    pred = m1.predict(x_test)
    pred_counter = m1.predict(x_counter)
    small_test={'pred':pred, 'obs':data_t.y_test,'t':data_t.t_test,'counter':pred_counter}

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
                                          loader_val=tloader_val)


    return  metrics, loss, ate, tau, small_test


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
