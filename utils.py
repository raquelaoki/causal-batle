import pandas as pd
import logging
import helper_fit as hfit
import helper_data as hd
from dragonnet import dragonnet

# temp
import numpy as np
# import sklearn as sk
import os

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def make_data(params):
    if params['data_name'] == 'gwas':
        data_s, data_t, tau = hd.make_gwas(params)
        return data_s, data_t, tau
    elif params['data_name'] == 'ihdp':
        # data_s, data_t, tau = dp.make_ihdp(params)
        return hd.make_ihdp(params)


def run_model(params):
    data_s, data_t, tau = make_data(params)

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
    #
    metrics, loss, ate = hfit.fit_wrapper(params=params,
                                          loader_train=tloader_train,
                                          loader_test=tloader_test,
                                          loader_all=tloader_all,
                                          loader_val=tloader_val,
                                          use_validation=params['use_validation'],
                                          use_tensorboard=params['use_tensorboard'])

    return metrics, loss, ate, tau


def organize(params, ate, tau, table=pd.DataFrame()):
    columns = ['model_name', 'config', 'data_name', 'tau',
               'ate_naive_all', 'ate_naive_train', 'ate_naive_test',
               'ate_ipw_all', 'ate_ipw_train', 'ate_ipw_test',
               'ate_aipw_all', 'ate_aipw_train', 'ate_aipw_test']

    if table.empty:
        table = pd.DataFrame(columns=set(columns))

    out = {
        'model_name': params['model_name'],
        'data_name': params['data_name'],
        'config': params['config_name'],
        'tau': tau,
        'ate_naive_train': ate['ate_naive_train'],
        'ate_ipw_train': ate['ate_ipw_train'],
        'ate_aipw_train': ate['ate_aipw_train'],
        'ate_naive_all': ate['ate_naive_all'],
        'ate_ipw_all': ate['ate_ipw_all'],
        'ate_aipw_all': ate['ate_aipw_all'],
        'ate_naive_test': ate['ate_naive_test'],
        'ate_ipw_test': ate['ate_ipw_test'],
        'ate_aipw_test': ate['ate_aipw_test'],
    }
    table = table.append(out, ignore_index=True)
    return table[columns]


def read_config_names(path):
    config_files = os.listdir(path)
    config_files = [path + item for item in config_files]
    return config_files
