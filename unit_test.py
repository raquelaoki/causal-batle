""" Unit tests.

1. Gwas + Dragonnet, check if final ate is Nan
2. IHDP + AIPW, check if final ate is Nan
3. IHDP and Bayesian Dragonnet, check if final ate is Nan
4. GWAS + Causal-Batle, check if final ate is Nan (also combine source and target domains).
"""

import unittest
import logging
import math
import pandas as pd
# Local Imports.
import helper_data as dp
from helper_parameters import parameter_debug, _check_params_consistency
from utils import run_model, repeat_experiment

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataPrep(unittest.TestCase):
    def test_make_dragonnet(self):
        logger.debug('Test 1: GWAS and Dragonnet creation')
        params = parameter_debug(max_epochs=2, n_covariates=100, n_sample=1000,
                                 lr=0.01, batch_size=20, weight_decay=0.01, use_validation=True,
                                 units1=10, units2=5, dropout_p=0.5, use_overlap_knob=True, overlap_knob=0.5,
                                 repetitions=1,
                                 )

        metrics, loss, ate, tau, _ = run_model(params)
        self.assertFalse(math.isnan(ate['ate_naive_train']), "GWAS+Dragonnet failed (1 repetitions)")

    def test_idhp_aipw(self):
        logger.debug('Test : IHDP and AIPW')
        params = parameter_debug(data_name='ihdp', model_name='aipw', use_validation=True,
                                 use_tensorboard=True, max_epochs=2, ate_method_list=['aipw'],
                                 config_name='unit_test')
        metrics, loss, ate, tau, _ = run_model(params)
        self.assertFalse(math.isnan(ate['ate_aipw_train']), 'IHDP+AIPW failed (default repetitions)')

    def test_ihdp_bdragonnet(self):
        logger.debug('Test: IHDP and Bayesian Dragonnet')
        params = parameter_debug(
            data_name='ihdp', model_name='bdragonnet', use_validation=True,
            use_tensorboard=False, max_epochs=2, ate_method_list=['naive', 'ipw', 'aipw'],
            config_name='unit_test', alpha=[1, 1, 0], repetitions=1,forward_passes=10
        )
        table = repeat_experiment(params)
        self.assertFalse(math.isnan(table['ate_aipw_train'].values[0]), 'IHDP+AIPW failed.')

    def test_gwas_batle(self):
        logger.debug('Test: GWAS anc Causal-Batle')
        params = parameter_debug(
            data_name='gwas', model_name='batle', max_epochs=2, batch_size=200,
            use_validation=True, ate_method_list=['naive', 'aipw'],
            config_name='unit_test', lr=0.001, weight_decay=0.05, alpha=[1, 1, 1, 1, 1, 1],
            use_source=True, repetitions=5, forward_passes=10
        )
        metrics, loss, ate, tau, _ = run_model(params)
        self.assertFalse(math.isnan(ate['ate_aipw_train']), 'GWAS anc Causal-Batle failed.')

    def test_ihdp_cevae(self):
        logger.debug('Test: IHDP anc Cevae')

        params = {'model_name': 'cevae',
                  'data_name': 'ihdp',
                  'config_name': 'ihdp_cevae',
                  'seed': 1,
                  'repetitions': 1,
                  'alpha': [1],
                  'lr': 0.001,
                  'weight_decay': 0.05,
                  'use_validation': False}

        params = _check_params_consistency(params)

        table = repeat_experiment(params, table=pd.DataFrame(), use_range_source_p=False,
                                  save=False, output_save='')
        self.assertFalse(math.isnan(table['ate_naive_train'].values[0]), 'IHDP+CEVAE failed.')


if __name__ == '__main__':
    unittest.main()
