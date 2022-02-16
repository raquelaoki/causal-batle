"""Parameters Check

Check the consistency of the parameters and complete some missing fields based on conditions.

"""
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parameter_debug(data_name='gwas', model_name='dragonnet', max_epochs=100):
    params = {}
    params['data_name'] = data_name
    if params['data_name'] == 'gwas':
        params = make_parameters_data_gwas(params)

    params['model_name'] = model_name
    if params['model_name'] == 'dragonnet':
        params = make_parameters_model_dragonnet(params, max_epochs=max_epochs)
    return params

def make_parameters_data_gwas(params,
                              n_sample=5000,
                              n_covariates=1000,
                              binary_target=False,
                              n_treatments=1):
    params['n_sample'] = n_sample
    params['n_covariates'] = n_covariates
    params['binary_target'] = binary_target
    params['n_treatments'] = n_treatments
    return params


def make_parameters_model_dragonnet(params,
                                    batch_size=64,
                                    max_epochs=100,
                                    decay=20,
                                    alpha=0.1,
                                    gamma=0.7,
                                    lr=0.001,
                                    wd=0.05,
                                    type_original=True,
                                    units1=100,
                                    units2=200,
                                    units3=1):
    params['use_transfer'] = False
    params['suffle'] = True
    params['batch_size'] = batch_size
    params['max_epochs'] = max_epochs
    params['decay'] = decay
    params['alpha'] = alpha  # decay downgrad
    params['gamma'] = gamma  # Adam decay
    params['lr'] = lr  # learning rate
    params['wd'] = wd  # weight_decay
    params['units1']=units1
    params['units2']=units2
    params['units3']=units3
    params['type_original']=type_original
    return params
