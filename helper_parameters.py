"""Parameters Check

Check the consistency of the parameters and complete some missing fields based on conditions.

"""
import logging
import pandas as pd
import yaml
import os

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def _running_on_colab():
    try:
        __IPYTHON__
        _in_ipython_session = True
    except NameError:
        _in_ipython_session = False
    return _in_ipython_session


def create_if_not_available(parameters, default_keys):
    for key in default_keys:
        parameters[key] = parameters.get(key, default_keys[key])
    return parameters


def parameter_loader(config_path=""):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    params = config['parameters']
    params = _check_params_consistency(params)
    return params


def _check_params_consistency(params):
    valid_model_names = ['dragonnet', 'aipw','bdragonnet']
    valid_data_names = ['ihdp', 'gwas']

    assert 'data_name' in params, 'data_name missing!'
    assert 'model_name' in params, 'model_name missing!'
    assert 'config_name' in params, 'config_name missing!'
    assert 'seed' in params, 'seed missing!'

    # Adding default values
    params['repetitions'] = params.get('repetitions', 10)
    params['ate_method_list'] = params.get('ate_method_list', ['naive', 'ipw'])

    assert params['data_name'] in valid_data_names, 'data_name not implemented!'
    assert params['model_name'] in valid_model_names, 'model_name not implemented!'

    if params['data_name'] == 'gwas':
        params['n_sample'] = params.get('n_sample', 5000)
        params['n_covariates'] = params.get('n_covariates', 1000)
        params['binary_target'] = params.get('binary_target', False)
        params['n_treatments'] = params.get('n_treatments', 1)
        params['use_overlap_knob'] = params.get('use_overlap_knob', False)
        params['overlap_knob'] = params.get('overlap_knob', 1)
    elif params['data_name'] == 'ihdp':
        assert 1 <= params['seed'] <= 10, 'Seed out of range (1-10)'
        params['n_covariates'] = params.get('n_covariates', 25)
        params['n_sample'] = params.get('n_sample', 747)
    else:
        logger.debug('%s not implemented', params['data_name'])

    if params['model_name'] == 'dragonnet':
        params['max_epochs'] = params.get('max_epochs', 50)
        params['batch_size'] = params.get('batch_size', 50)
        params['units1'] = params.get('units1', 200)
        params['units2'] = params.get('units2', 100)
        params['units3'] = params.get('units3', 1)
        params['lr'] = params.get('lr', 0.01)
        params['weight_decay'] = params.get('weight_decay', 0.05)
        params['dropout_p'] = params.get('dropout_p', 0.5)
        params['alpha'] = params.get('alpha', [1, 1, 0])
        params['use_validation'] = params.get('use_validation', False)
        params['use_dropout'] = params.get('use_dropout', False)
        params['use_tensorboard'] = params.get('use_tensorboard', True)
        params['use_source'] = params.get('use_source', False)
        params['shuffle'] = params.get('shuffle', False)
        params['type_original'] = params.get('type_original', True)
    elif params['model_name'] == 'bdragonnet':
        params['max_epochs'] = params.get('max_epochs', 50)
        params['batch_size'] = params.get('batch_size', 50)
        params['units1'] = params.get('units1', 200)
        params['units2'] = params.get('units2', 100)
        params['units3'] = params.get('units3', 1)
        params['lr'] = params.get('lr', 0.01)
        params['weight_decay'] = params.get('weight_decay', 0.05)
        params['dropout_p'] = params.get('dropout_p', 0.5)
        params['alpha'] = params.get('alpha', [1, 1, 0])
        params['use_validation'] = params.get('use_validation', False)
        params['use_dropout'] = params.get('use_dropout', True)
        params['use_tensorboard'] = params.get('use_tensorboard', True)
        params['use_source'] = params.get('use_source', False)
        params['shuffle'] = params.get('shuffle', False)
        params['type_original'] = params.get('type_original', False)
    elif params['model_name'] == 'aipw':
        assert 'n_covariates' in params, 'n_covariates missing'
        params['max_epochs'] = params.get('max_epochs', 50)
        params['batch_size'] = params.get('batch_size', 50)
        params['lr'] = params.get('lr', 0.01)
        params['weight_decay'] = params.get('weight_decay', 0.05)
        params['use_validation'] = params.get('use_validation', False)
        params['use_tensorboard'] = params.get('use_tensorboard', True)
        params['use_source'] = params.get('use_source', False)
        params['shuffle'] = params.get('shuffle', False)
        params['alpha'] = params.get('alpha', [1, 1, 1])

    else:
        logger.debug('%s not implemented', params['model_name'])

    # Tensorflow root path:
    if _running_on_colab:
        params['home_dir'] = '/content'
    else:
        params['home_dir'] = os.getenv("HOME")

    return params


def parameter_debug(data_name='gwas', model_name='dragonnet', max_epochs=100,
                    batch_size=200, lr=0.01, weight_decay=0.01, units1=100, units2=50, units3=1,
                    n_sample=5000, n_covariates=1000, use_validation=False, use_tensorboard=False,
                    use_dropout=False, dropout_p=0, use_overlap_knob=False, overlap_knob=1,
                    seed=1, alpha=[1, 1, 1], config_name='configA', ate_method_list=['naive']):
    params = {'data_name': data_name, 'model_name': model_name,
              'seed': seed, 'config_name': config_name,
              'use_tensorboard': use_tensorboard,
              'ate_method_list':ate_method_list}

    if params['data_name'] == 'gwas':
        params = _make_parameters_data_gwas(params=params,
                                            n_sample=n_sample,
                                            n_covariates=n_covariates,
                                            use_overlap_knob=use_overlap_knob,
                                            overlap_knob=overlap_knob
                                            )
    elif params['data_name'] == 'ihdp':
        params = _make_paramters_data_ihdp(params=params,
                                           use_overlap_knob=use_overlap_knob,
                                           overlap_knob=overlap_knob
                                           )
    else:
        logger.debug('...data not implemented')

    # print('update here',params)
    if params['model_name'] == 'dragonnet':
        params = _make_parameters_model_dragonnet(params, max_epochs=max_epochs,
                                                  batch_size=batch_size, lr=lr,
                                                  weight_decay=weight_decay, units1=units1, units2=units2,
                                                  units3=units3, use_validation=use_validation,
                                                  use_dropout=use_dropout, dropout_p=dropout_p,
                                                  alpha=alpha)

    params = _check_params_consistency(params)
    return params


def _make_parameters_data_gwas(params,
                               n_sample=5000,
                               n_covariates=1000,
                               binary_target=False,
                               n_treatments=1,
                               use_overlap_knob=False,
                               overlap_knob=1):
    params['n_sample'] = n_sample
    params['n_covariates'] = n_covariates
    params['binary_target'] = binary_target
    params['n_treatments'] = n_treatments
    params['use_overlap_knob'] = use_overlap_knob
    params['overlap_knob'] = overlap_knob
    return params


def _make_paramters_data_ihdp(params,
                              n_sample=747,
                              n_covariates=25,
                              binary_target=False,
                              use_overlap_knob=False,
                              overlap_knob=1):
    params['binary_target'] = binary_target
    params['use_overlap_knob'] = use_overlap_knob
    params['overlap_knob'] = overlap_knob
    params['n_covariates'] = n_covariates
    params['n_sample'] = n_sample
    params['n_treatments'] = 1
    return params


def _make_parameters_model_dragonnet(params,
                                     batch_size=200,
                                     max_epochs=100,
                                     decay=20,
                                     gamma=0.7,
                                     lr=0.01,
                                     weight_decay=0.05,
                                     type_original=True,
                                     units1=200,
                                     units2=100,
                                     units3=1,
                                     use_validation=False,
                                     use_dropout=False,
                                     dropout_p=0,
                                     alpha=[1, 1, 1]):
    params['use_source'] = False
    params['shuffle'] = True
    params['batch_size'] = batch_size
    params['max_epochs'] = max_epochs
    params['decay'] = decay
    params['alpha'] = alpha  # weights
    params['gamma'] = gamma  # Adam decay
    params['lr'] = lr  # learning rate
    params['weight_decay'] = weight_decay  # weight_decay
    params['units1'] = units1
    params['units2'] = units2
    params['units3'] = units3
    params['type_original'] = type_original
    params['use_validation'] = use_validation
    params['use_dropout'] = use_dropout
    params['dropout_p'] = dropout_p
    return params
