import data_preprocessing as dp
from helper_parameters import make_parameters_data_gwas
import unittest
import logging


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

# TODO: parameters

if __name__ == '__main__':
    unittest.main()
