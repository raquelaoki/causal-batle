import pandas as pd
import logging
import helper_fit as hfit
import data_preprocessing as dp
from dragonnet import dragonnet

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def make_model(params):
    logging.debug('Model - %s', params['model_name'])
    if params['model_name'] == 'dragonnet':
        model = dragonnet(n_covariates=params['n_covariates'], units1=params['units1'], units2=params['units2'],
                          units3=params['units3'], binary_target=params['binary_target'],
                          type_original=params['type_original'])

        criterion = hfit.make_criterion_dragonnet_original(params['binary_target'])
        metrics = hfit.make_metrics(include_treatment=True)

    logger.debug('...model constructed')
    return model, criterion, metrics


def make_data(params):
    if params['data_name'] == 'gwas':
        data_s, data_t, tau = dp.make_gwas(params)
        return data_s, data_t, tau


def run_model(params):
    data_s, data_t, tau = make_data(params)
    model, criterion, metrics = make_model(params)
    if params['use_transfer']:
        tloader_train, tloader_val, tloader_test, tloader_all = data_t.loader()
        sloader_train, sloader_val, sloader_test, sloader_all = data_s.loader()
    else:
        tloader_train, tloader_val, tloader_test, tloader_all = data_t.loader()

    loss_train, loss_train_y, loss_train_t, metric_t_train, metric_y_train=\
        hfit.fit(model, criterion, metrics, params,
                 tloader_train, tloader_test, tloader_all, tloader_val)
    return ate_naive_train, ate_aipw_train, ate_naive_test, ate_aipw_test, ate_naive_all, ate_aipw_all, tau
    #logger.debug('The end')
    #return tloader_train


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
