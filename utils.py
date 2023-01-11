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
    elif params['data_name'] == 'hcmnist':
        return hd.make_hcmnist(params)
    else:
        raise NotImplementedError(params['data_name'])


def run_model(params, model_seed=0, good_runs=0, data_counter=0):
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
            params['data_seed'] = data_counter
            data, tau = make_data(params)
            tloader_train, tloader_val, tloader_test, tloader_all = data.loader(batch=params['batch_size'])

            params['full_size_n'] = data.full_size_n
            params['target_size_n'] = data.target_size_n
            params['source_size_n'] = data.source_size_n

            metrics, loss, ate = hfit.fit_wrapper(params=params,
                                                  loader_train=tloader_train,
                                                  loader_test=tloader_test,
                                                  loader_all=tloader_all,
                                                  loader_val=tloader_val,
                                                  use_validation=params['use_validation'],
                                                  use_tensorboard=params['use_tensorboard'],
                                                  model_seed=model_seed,
                                                  binfeat=data.binfeat,
                                                  contfeat=data.contfeat,
                                                  )
            success = True
            good_runs += 1
        except ValueError:
            model_seed = model_seed + 1
            params['seed_add_on'] = params['seed_add_on'] + 1
            print('...value error (good runs before - ', good_runs, ')')
            good_runs = 0

    return metrics, loss, ate, tau, good_runs, params, data_counter + 1


def organize(params, ate, tau, table=pd.DataFrame(), b=1):
    """ Organize the outputs in a pd.DataFrame().
    :param params: dictionary
    :param ate: estimated values
    :param tau: true treatment effect
    :param table: if it already exists
    :param b: repetition
    :return:
    """

    columns = ['model_name', 'config', 'data_name', 'config_rep', 'tau', 'b', 'seed', 'source_size_p',
               'ate_naive_all', 'ate_naive_train', 'ate_naive_test',
               'ate_aipw_all', 'ate_aipw_train', 'ate_aipw_test',
               'epochs', 'alpha', 'lr', 'wd', 'x_t_shape', 'x_s_shape',
               'batch', 'dropout', 'repetitions', 'data_rep', 'range_size',
               'full_size_n', 'target_size_n', 'source_size_n',
               'data_seed',
               ]

    if table.empty:
        table = pd.DataFrame(columns=set(columns))

    if params['model_name'] == 'batle':
        source_size = params.get('source_size', 0)
        if params['use_fix_digit']:
            use_fix_digit = '_fix'
        else:
            use_fix_digit = '_random'
    else:
        source_size = None
        use_fix_digit = ''

    out = {
        'model_name': params['model_name'],
        'data_name': params['data_name'],
        'config': params['data_name'] + '_' + params['model_name'] + use_fix_digit,
        'config_rep': params['config_name_seeds'],
        'tau': tau,
        'b': b,
        'seed': params['seed'],
        'source_size_p': params.get('source_size_p', 1),
        'ate_naive_train': ate['ate_naive_train'],
        'ate_aipw_train': ate['ate_aipw_train'],
        'ate_naive_all': ate['ate_naive_all'],
        'ate_aipw_all': ate['ate_aipw_all'],
        'ate_naive_test': ate['ate_naive_test'],
        'ate_aipw_test': ate['ate_aipw_test'],
        'alpha': params['alpha'],
        'epochs': params['max_epochs'],
        'wd': params['weight_decay'],
        'lr': params['lr'],
        'x_t_shape': params.get('target_size', 1),
        'x_s_shape': source_size,
        'batch': params['batch_size'],
        'dropout': params['dropout_p'],
        'repetitions': params['repetitions'],
        'data_rep': params['seed'],
        'range_size': params.get('target_size', 1) * params.get('source_size_p', 1),
        'full_size_n': params.get('full_size_n', -1),
        'target_size_n': params.get('target_size_n', -1),
        'source_size_n': params.get('source_size_n', -1),
        'data_seed': params.get('data_seed', -1),
        'informative_source': params.get('informative_source', True)
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
                      save=False, output_save='', target_size=None):
    """ Repeat the experiment b times.
    This function perform b repetitions of (Dataset, Model, Ate) - set by the config/params file.
    :param output_save:
    :param save:
    :param params: dictinary
    :param table: pd.DataFrame() - if not given, a new dataframe will be created.
    :param use_range_source_p: Bool, if true, we explore a range of source_size_p values (valid only for GWAS and IHDP)
    :param source_size_p: list with proportions (GWAS and IHDP).
    :param target_size: list with target sizes (MNIST).
    :return: pd.Dataframe with results of the b repetitions.
    """
    print(params['model_name'])
    b = params['repetitions']

    n_seeds = params['seed']
    data_counter = 0

    for seed in range(n_seeds):
        params['seed'] = seed
        print('seed ', seed)
        logger.debug('seed ' + str(seed))
        config_name = params['config_name']
        good_runs = 0
        for i in range(b):
            if use_range_source_p:
                table, good_runs, data_counter = range_source_p(params=params,
                                                                table=table,
                                                                b=i,
                                                                good_runs=good_runs,
                                                                source_size_p=source_size_p,
                                                                target_size=target_size,
                                                                data_counter=data_counter
                                                                )
            else:

                params['config_name_seeds'] = config_name + '_' + 'seed' + str(
                    params['seed']) + '_' + 'b' + str(i)
                print(params['config_name_seeds'])
                metrics, loss, ate, tau, good_runs, params, data_counter = run_model(params,
                                                                                     model_seed=i,
                                                                                     good_runs=good_runs,
                                                                                     data_counter=data_counter)
                params['data_seed'] = data_counter
                table = organize(params, ate, tau, table, b=i)
        table['data_rep'] = n_seeds
        if save:
            table.to_csv(output_save + '.csv', sep=',')

    table['data_rep'] = n_seeds
    return table


def range_source_p(params, table, source_size_p=None, b=1, good_runs=0, target_size=None, data_counter=0):
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
    if params['data_name'] == 'hcmnist':
        if not target_size:
            range_sizes = [250, 500, 750, 1000]
        else:
            range_sizes = target_size
    else:
        if not source_size_p:
            range_sizes = [0.2, 0.4, 0.6, 0.8]
        else:
            assert max(source_size_p) < 1 and min(source_size_p) > 0, 'Values on array are outsise range(0,1)'
            range_sizes = source_size_p

    config_name = params['config_name']

    for p in range_sizes:
        params['config_name'] = config_name + '_' + str(p) + '_' + 'seed' + str(params['seed']) + '_' + 'b' + str(b)
        if params['data_name'] == 'hcmnist':
            params['target_size'] = p
        else:
            params['source_size_p'] = p
        params['config_name_seeds'] = params['config_name']
        metrics, loss, ate, tau, good_runs, params, data_counter = run_model(params,
                                                                             model_seed=b,
                                                                             good_runs=good_runs,
                                                                             data_counter=data_counter)
        params['data_seed'] = data_counter
        table = organize(params, ate, tau, table, b=b)

    params['config_name'] = config_name
    return table, good_runs, data_counter
