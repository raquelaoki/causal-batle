"""Parameters Check

Check the consistency of the parameters and complete some missing fields based on conditions.

"""
import logging

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_if_not_available(parameters, default_keys):
    for key in default_keys:
        parameters[key] = parameters.get(key, default_keys[key])
    return parameters



def parameter_debug(data_name='gwas', model_name='dragonnet', max_epochs=100,
                    batch_size=200, lr=0.01, weight_decay=0.01, units1=100, units2=50, units3=1,
                    n_sample=5000, n_covariates=1000, use_validation=False,
                    use_dropout=False, dropout_p=0, use_overlap_knob=False, overlap_knob=1):
    params = {'data_name': data_name}
    if params['data_name'] == 'gwas':
        params = make_parameters_data_gwas(params=params,
                                           n_sample=n_sample,
                                           n_covariates=n_covariates,
                                           use_overlap_knob=use_overlap_knob,
                                           overlap_knob=overlap_knob
                                           )

    params['model_name'] = model_name
    if params['model_name'] == 'dragonnet':
        params = make_parameters_model_dragonnet(params, max_epochs=max_epochs,
                                                 batch_size=batch_size, lr=lr,
                                                 weight_decay=weight_decay, units1=units1, units2=units2,
                                                 units3=units3, use_validation=use_validation,
                                                 use_dropout=use_dropout, dropout_p=dropout_p)
    return params


def make_parameters_data_gwas(params,
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


def make_parameters_model_dragonnet(params,
                                    batch_size=200,
                                    max_epochs=100,
                                    decay=20,
                                    alpha=0.1,
                                    gamma=0.7,
                                    lr=0.01,
                                    weight_decay=0.05,
                                    type_original=True,
                                    units1=200,
                                    units2=100,
                                    units3=1,
                                    use_validation=False,
                                    use_dropout=False,
                                    dropout_p=0):
    params['use_transfer'] = False
    params['shuffle'] = True
    params['batch_size'] = batch_size
    params['max_epochs'] = max_epochs
    params['decay'] = decay
    params['alpha'] = alpha  # decay downgrad
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
