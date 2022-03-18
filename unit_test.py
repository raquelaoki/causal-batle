import helper_data as dp
from helper_parameters import parameter_debug
import unittest
import logging
import math
from utils import run_model

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataPrep(unittest.TestCase):
    def test_make_dragonnet(self):
        logger.debug('Test 1: GWAS and Dragonnet creation')
        params = parameter_debug(max_epochs=2, n_covariates=100, n_sample=1000,
                                 lr=0.01, batch_size=20, weight_decay=0.01, use_validation=True,
                                 units1=10, units2=5, use_dropout=True, dropout_p=0.5,
                                 use_overlap_knob=True, overlap_knob=0.5,
                                 )
        metrics, loss, ate, tau = run_model(params)
        self.assertFalse(math.isnan(ate['ate_naive_train']), "GWAS+Dragonnet failed")

    def test_idhp_aipw(self):
        logger.debug('Test : IHDP and AIPW')
        params = parameter_debug(data_name='ihdp', model_name='aipw', use_validation=True,
                                 use_tensorboard=True, max_epochs=2, ate_method_list=['aipw'],
                                 config_name='unit_test')
        metrics, loss, ate, tau = run_model(params)
        self.assertFalse(math.isnan(ate['ate_aipw_train']), 'IHDP+AIPW failed.')

    def test_ihdp_bdragonnet(self):
        logger.debug('Test: IHDP and Bayesian Dragonnet')
        params = parameter_debug(
            data_name='ihdp', model_name='bdragonnet', use_validation=True,
            use_tensorboard=False, max_epochs=2, ate_method_list=['naive','ipw','aipw'],
            config_name='unit_test'
        )
        metrics, loss, ate, tau = run_model(params)
        self.assertFalse(math.isnan(ate['ate_aipw_train']), 'IHDP+AIPW failed.')


if __name__ == '__main__':
    unittest.main()
