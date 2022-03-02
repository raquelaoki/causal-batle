import data_preprocessing as dp
from helper_parameters import make_parameters_data_gwas, parameter_debug
import unittest
import logging
import math

logging.basicConfig(level=logging.DEBUG)


class DataPrep(unittest.TestCase):
    def test_gwas(self):
        try:
            params = {'data_name':'gwas'}
            params = make_parameters_data_gwas(params, n_sample=1000, n_covariates=100)
            data_s, data_t, tau = dp.make_gwas(params, unit_test=True)
            _, _, loader_test, _ = data_t.loader()
            test_batch = next(iter(loader_test))
            self.assertTrue(data_t.x_test.shape[0] ==test_batch[0].shape[0], "GWAS: Empty Dataset")
        except:
            self.assertRaises("GWAS: Dataset creation failed")


    def test_make_dragonnet(self):
        try:
            params = parameter_debug(max_epochs=2, n_covariates=100, n_sample=1000,
                                     lr=0.01, batch_size=20, weight_decay=0.01, use_validation=True,
                                     units1=10, units2=5,  use_dropout=True, dropout_p=0.5,
                                     use_overlap_knob=True, overlap_knob=0.5,
                                     )
            params['seed'] = 10
            metrics, loss, ate, tau, small_test = run_model(params)
            self.assertTrue(math.isnan(ate['ate_naive_train']), "ATE: NAN value")
        except:
            self.assertRaises("Dragonnet: creation and/or fit failed.")

# TODO: parameters

if __name__ == '__main__':
    unittest.main()