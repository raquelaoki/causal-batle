import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Local
import baselines.aipw as aipw
import baselines.bayesian_layers as bl
import baselines.dragonnet as dragonnet
import baselines.cevae as cevae
import causal_batle as cb
import helper_ate as ha

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def make_model(params, binfeat=[], contfeat=[]):
    """
    Create the model, criterion, and metrics according to type of model selected.
    :param params: dictionaty. Required key: {model_name}. Optinal keys: architecture parameters;
    :param binfeat: array with column numbers of binary covariates (Used by some baselines);
    :param contfeat: array with column numbers of contfeat covariates (Used by some baselines);
    :return: model(nn.Module), criterion (list of criterions),
        metric_functions (list of metrics), fit (funtion to train model)
    """
    logger.debug('Model - %s', params['model_name'])
    if params['model_name'] == 'dragonnet':
        model = dragonnet.dragonnet(n_covariates=params['n_covariates'],
                                    units1=params['units1'],
                                    units2=params['units2'],
                                    units3=params['units3'],
                                    type_original=params['type_original'])
        criterion = [dragonnet.criterion_function_dragonnet_t,
                     dragonnet.criterion_function_dragonnet_y,
                     dragonnet.criterion_function_dragonnet_targeted]
        metric_functions = [dragonnet.metric_function_dragonnet_t,
                            dragonnet.metric_function_dragonnet_y]
        fit = dragonnet.fit_dragonnet
        ate = ha.calculate_ate
    elif params['model_name'] == 'bdragonnet':
        model = dragonnet.dragonnet(n_covariates=params['n_covariates'],
                                    units1=params['units1'],
                                    units2=params['units2'],
                                    units3=params['units3'],
                                    type_original=params['type_original'],
                                    use_dropout=params['use_dropout'],
                                    dropout_p=params['dropout_p']
                                    )
        criterion = [bl.criterion_function_dragonnet_t,
                     bl.criterion_function_dragonnet_y]
        metric_functions = [bl.metric_function_dragonnet_t,
                            bl.metric_function_dragonnet_y]
        fit = dragonnet.fit_dragonnet
        ate = ha.calculate_ate_bayesian

    elif params['model_name'] == 'aipw':
        model = aipw.aipw(n_covariates=params['n_covariates'])
        criterion = [aipw.criterion_function_aipw_t,
                     aipw.criterion_function_aipw_y,
                     aipw.criterion_function_aipw_y]
        metric_functions = [aipw.metric_function_aipw_t,
                            aipw.metric_function_aipw_y,
                            aipw.metric_function_aipw_t]
        fit = aipw.fit_aipw
        ate = ha.calculate_ate
    elif params['model_name'] == 'batle':
        model = cb.causal_batle(n_covariates=params['n_covariates'],
                                units1=params['units1'],
                                units2=params['units2'],
                                units3=params['units3'],
                                dropout_p=params['dropout_p'])
        criterion = [cb.criterion_function_t,
                     cb.criterion_function_y,
                     cb.criterion_function_discriminator,
                     cb.criterion_function_reconstruction,
                     cb.criterion_function_adversarial]  # Missing adversarial
        metric_functions = [bl.metric_function_dragonnet_t,
                            bl.metric_function_dragonnet_y,
                            cb.metric_function_discriminator,
                            cb.metric_function_reconstruction]
        fit = cb.fit_causal_batle
        ate = ha.calculate_ate_bayesian
        logger.debug('Implementation in progress')
    elif params['model_name'] == 'cevae':
        model = cevae.cevae(n_covariates=params['n_covariates'],
                            binfeat=binfeat,
                            contfeat=contfeat)
        criterion = [cevae.criterion_l1l2,
                     cevae.criterion_t,
                     cevae.criterion_y,
                     cevae.criterion_kl,
                     cevae.criterion_l6l7]
        metric_functions = [cevae.metric_function_cevae_t,
                            cevae.metric_function_cevae_y]
        fit = cevae.fit_cevae
        ate = ha.calculate_ate_cevae
    else:
        logger.warning('%s not implemented', params['model_name'])
    logger.debug('...model constructed')
    return model, criterion, metric_functions, fit, ate


def fit_wrapper(params,
                loader_train, loader_test, loader_all,
                loader_val=None,
                use_tensorboard=False,
                use_validation=False,
                model_seed=0,
                binfeat=[],
                contfeat=[]):
    """ Wrap all model trainign functions.
    1. Call make_model() to create model, criterion, metrics,fit function and ate estimators.
    2. Set optimizer.
    3. Call fit()
    4. Call ate()

    :param params: dictionary with model and data parameters
    :param loader_train: pytorch DataLoader obj
    :param loader_test: pytorch DataLoader obj
    :param loader_all: pytorch DataLoader obj
    :param loader_val: pytorch DataLoader obj (Optional)
    :param use_tensorboard: Bool
    :param use_validation: Bool
    :return metrics: dictionary with metrics
    :return loss: dictionary with losses
    :return ate_estimated: dictionary with estimated ate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)

    logger.debug("...fitting %s", params['model_name'])
    logger.debug("...using %s", device)

    # TODO: implement
    if use_validation:
        best_metric_y_val = 999
        best_epoch = 0

    if use_tensorboard:
        path_logger = params.get('path_tensorboard', 'logs/')
    else:
        path_logger = None

    model, criterion, metric_functions, fit, ate = make_model(params, binfeat=binfeat, contfeat=contfeat)

    if torch.cuda.is_available():
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=params['lr'],
                                 weight_decay=params['weight_decay']
                                 )
    alpha = params['alpha']

    model, loss, metrics = fit(epochs=params['max_epochs'],
                               model=model,
                               loader_train=loader_train,
                               loader_test=loader_test,
                               optimizer=optimizer,
                               criterion=criterion,
                               metric_functions=metric_functions,
                               use_validation=use_validation,
                               use_tensorboard=use_tensorboard,
                               device=device,
                               loader_val=loader_val,
                               alpha=alpha,
                               path_logger=path_logger,
                               config_name=params['config_name'],
                               home_dir=params['home_dir'],
                               episilon=params['episilon'],  # Only used by dragonnet
                               weight_1=params['weight_1']  # Only used by causal batle
                               )

    logger.debug("...calculating ate")
    ate_estimated = ate(loader_train=loader_train,
                        loader_test=loader_test,
                        loader_all=loader_all,
                        model=model,
                        ate_method_list=params['ate_method_list'],
                        device=device,
                        forward_passes=params['forward_passes'],
                        filter_d=params['filter_d'])
    logger.debug("...Model done!")
    return metrics, loss, ate_estimated
