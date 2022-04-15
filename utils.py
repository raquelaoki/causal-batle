import os
import logging
import numpy as np
import pandas as pd

# Local Imports.
import helper_fit as hfit
import helper_data as hd
from baselines.dragonnet import dragonnet

logger = logging.getLogger(__name__)


def make_data(params):
    """ It calls function to make the dataset.
    :param params: dictionary with parameters. Required: 'data_name'.
    :return: data: DataTargetAndSource or DataTarget class.
    :return: tau: True treatment effect.
    """
    if params['data_name'] == 'gwas':
        data, tau = hd.make_gwas(params)
        return data, tau
    elif params['data_name'] == 'ihdp':
        return hd.make_ihdp(params)


def run_model(params, model_seed=0):
    """ Given a set of parameters, it run the model.
    1. Creates the dataset (use seed for stability/reproducibility across several models).
    2. Make data loaders.
    3. Call fit_wrapper (make model, fits, estimate ate)
    :param params: dictionary
    :param model_seed: int
    :return metrics, loss and ate dictionaries, tau (true treatment effect value)
    """

    success = False
    while not success:
        try:
            data, tau = make_data(params)
            tloader_train, tloader_val, tloader_test, tloader_all = data.loader(batch=params['batch_size'])
            metrics, loss, ate = hfit.fit_wrapper(params=params,
                                                  loader_train=tloader_train,
                                                  loader_test=tloader_test,
                                                  loader_all=tloader_all,
                                                  loader_val=tloader_val,
                                                  use_validation=params['use_validation'],
                                                  use_tensorboard=params['use_tensorboard'],
                                                  model_seed=model_seed,
                                                  binfeat=data.binfeat,
                                                  contfeat=data.contfeat
                                                  )
            success = True
        except ValueError:
            model_seed = model_seed + 1
            params['seed_add_on'] = params['seed_add_on']+1
            print('...value error')

    return metrics, loss, ate, tau


def organize(params, ate, tau, table=pd.DataFrame(), b=1):
    """ Organize the outputs in a pd.DataFrame().
    :param params: dictionary
    :param ate: estimated values
    :param tau: true treatment effect
    :param table: if it already exists
    :param b: repetition
    :return:
    """

    columns = ['model_name', 'config', 'data_name','config_rep' ,'tau', 'b', 'source_size_p',
               'ate_naive_all', 'ate_naive_train', 'ate_naive_test',
               'ate_aipw_all', 'ate_aipw_train', 'ate_aipw_test']

    if table.empty:
        table = pd.DataFrame(columns=set(columns))

    out = {
        'model_name': params['model_name'],
        'data_name': params['data_name'],
        'config': params['data_name']+'_'+params['model_name'],
        'config_rep': params['config_name_seeds'],
        'tau': tau,
        'b': b,
        'source_size_p': params['source_size_p'],
        'ate_naive_train': ate['ate_naive_train'],
        'ate_aipw_train': ate['ate_aipw_train'],
        'ate_naive_all': ate['ate_naive_all'],
        'ate_aipw_all': ate['ate_aipw_all'],
        'ate_naive_test': ate['ate_naive_test'],
        'ate_aipw_test': ate['ate_aipw_test'],
    }
    table = table.append(out, ignore_index=True)
    return table[columns]


def read_config_names(path):
    """ Make a list with the names of all config files in a folder.
    Important: it assumes the folder contains only config files
    :param path: folder with config.yaml files
    :return:list with path for each config file.
    """
    config_files = os.listdir(path)
    config_files = [path + item for item in config_files]
    return config_files


def repeat_experiment(params, table=pd.DataFrame(), use_range_source_p=False, source_size_p=None,
                      save=False, output_save=''):
    """ Repeat the experiment b times.
    This function perform b repetitions of (Dataset, Model, Ate) - set by the config/params file.
    :param output_save:
    :param save:
    :param params: dictinary
    :param table: pd.DataFrame() - if not given, a new dataframe will be created.
    :param use_range_source_p: Bool, if true, we explore a range of source_size_p values (valid only for GWAS and IHDP)
    :param source_size_p: list with proportions.
    :return: pd.Dataframe with results of the b repetitions.
    """
    print(params['model_name'])
    b = params['repetitions']

    n_seeds = params['seed']

    for seed in range(n_seeds):
        params['seed'] = seed
        print('seed ', seed)
        logger.debug('seed '+str(seed))
        config_name = params['config_name']
        for i in range(b):
            if use_range_source_p:
                table = range_source_p(params, table, source_size_p, b=i)
            else:
                params['config_name_seeds'] = config_name + '_' + 'seed' + str(
                    params['seed']) + '_' + 'b' + str(i)
                print(params['config_name_seeds'])
                metrics, loss, ate, tau = run_model(params, model_seed=i)
                table = organize(params, ate, tau, table, b=i)
        if save:
            table.to_csv(output_save + '.csv', sep=',')
    return table


def range_source_p(params, table, source_size_p=None, b=1):
    """ Creates a range of experiments with same set of parameters, but different source_size_p.
    source_size_p: proportion of input data splited between target and source domain.
    Note 1: that this only makes sense on the GWAS and IHDP datasets, where we are artificially spliting the dataset
    to create a source and target domain. If the experiment already have this split, this function should not be used.
    Note 2: For small values of p, we need to increase the batch size to avoid batches with only one class.
    :param source_size_p: list of proportions to be tested.
    :param params: dictionary.
    :param table: pd.DataFrame().
    :param b: repetitions (not make inside this function, used only for organize())
    :return: pd.DataFrame() with range of experiments.
    """
    if not source_size_p:
        source_size_p = [0.2, 0.4, 0.6, 0.8]
    else:
        assert max(source_size_p) < 1 and min(source_size_p) > 0, 'Values on array are outsise range(0,1)'
    condig_name = params['config_name']
    for p in source_size_p:
        params['config_name'] = condig_name + '_' + str(p) + '_' + 'seed' + str(params['seed']) + '_' + 'b' + str(b)
        params['source_size_p'] = p
        params['config_name_seeds'] = params['config_name']
        metrics, loss, ate, tau = run_model(params, model_seed=b)
        table = organize(params, ate, tau, table, b=b)

    params['config_name'] = condig_name
    return table
