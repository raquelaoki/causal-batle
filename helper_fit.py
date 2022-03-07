import pandas as pd
import numpy as np
#from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score, roc_curve
import logging
import torch.nn as nn
import torch
import helper_ate as ate
import dragonnet

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Do i use find_optimal?
def _find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------
    list type, with optimal cutoff value
    https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    Adapted
    """

    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc.drop(0, axis=0, inplace=True)
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


def find_optimal_cutoff_wrapper_t(loader_train, model, device):
    target, predicted = [], []
    for i, batch in enumerate(loader_train):
        predictions = model(batch[0].to(device))
        target = np.concatenate((target, batch[2].numpy().reshape(-1)), 0)
        predicted = np.concatenate((predicted, predictions['t'].detach().numpy().reshape(-1)), 0)

    target = target.reshape(-1)
    predicted = predicted.reshape(-1)
    return _find_optimal_cutoff(target, predicted)[0]


def make_model(params):
    """
    Create the model, criterion, and metrics according to type of model selected.
    :param params: dictionaty. Required key: {model_name}. Optinal keys: architecture parameters;
    :return: model(nn.Module), criterion (list of criterions),
        metric_functions (list of metrics), fit (funtion to train model)
    """
    logging.debug('Model - %s', params['model_name'])
    if params['model_name'] == 'dragonnet':
        model = dragonnet.dragonnet(n_covariates=params['n_covariates'], units1=params['units1'],
                                    units2=params['units2'],
                                    units3=params['units3'],
                                    type_original=params['type_original'])
        criterion = [dragonnet.criterion_function_dragonnet_t,
                     dragonnet.criterion_function_dragonnet_y,
                     dragonnet.criterion_function_dragonnet_targeted]
        metric_functions = [dragonnet.metric_function_dragonnet_t,
                            dragonnet.metric_function_dragonnet_y]
        fit = dragonnet.fit_dragonnet
    else:
        logging.warning('%s not implemented', params['model_name'])
    logger.debug('...model constructed')
    return model, criterion, metric_functions, fit


def _calculate_criterion(criterion_function, batch, predictions, device='cpu', set='train'):
    return criterion_function(batch, predictions, device=device, set=set)


def _calculate_metric(metric_function, batch, predictions):
    return metric_function(batch, predictions)


def fit_wrapper(params,
                loader_train, loader_test, loader_all,
                loader_val=None,
                use_tensorboard=False,
                use_validation=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(params['seed'])

    logger.debug("...fitting %s", params['model_name'])
    logger.debug("...using %s", device)

    if use_validation:
        best_metric_y_val = 999
        best_epoch = 0

    if use_tensorboard:
        path_logger = params.get('path_tensorboard', 'logs')

    model, criterion, metric_functions, fit = make_model(params)

    if torch.cuda.is_available():
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    alpha = params['alpha']

    model, loss, metrics = fit(epochs=params['max_epochs'],
                               model=model,
                               loader_train=loader_train,
                               optimizer=optimizer,
                               criterion=criterion,
                               metrics_functions=metric_functions,
                               use_validation=use_validation,
                               use_tensorboard=use_tensorboard,
                               device=device,
                               loader_val=loader_val,
                               alpha=alpha,
                               path_logger=path_logger,
                               config_name=params['config_name']
                               )
    # Change model to eval mode
    model.eval()

    # Metrics on testins set
    batch = next(iter(loader_test))
    predictions = model(batch[0].to(device))
    metrics['metric_test_t'] = _calculate_metric(metric_function=metric_functions[0],
                                                 batch=batch,
                                                 predictions=predictions,
                                                 )
    metrics['metric_test_y'] = _calculate_metric(metric_function=metric_functions[1],
                                                 batch=batch,
                                                 predictions=predictions,
                                                 )

    logging.debug("...calculating ate")
    ate_naive_train, ate_aipw_train = ate.calculate_ate(loader_train, model, single_batch=False,
                                                        include_aipw=True, title='Train')
    ate_naive_test, ate_aipw_test = ate.calculate_ate(loader_test, model, single_batch=True,
                                                      include_aipw=True, title='test')
    ate_naive_all, ate_aipw_all = ate.calculate_ate(loader_all, model, single_batch=False,
                                                    include_aipw=True, title='all')

    logging.debug("...fitting done.")

    ate_estimated = {'ate_naive_train': ate_naive_train, 'ate_aipw_train': ate_aipw_train,
                     'ate_naive_test': ate_naive_test, 'ate_aipw_test': ate_aipw_test,
                     'ate_naive_all': ate_naive_all, 'ate_aipw_all': ate_aipw_all}

    return metrics, loss, ate_estimated
