import unittest
import data_preprocessing as dp


class DataPrep(unittest.TestCase):
    def test_gwas(self):
        try:
            data_s, data_t = dp.make_gwas()
            _, _, loader_test, _ = data_t.loader()
            test_batch = next(iter(loader_test))
            self.assertTrue(data_t.x_test.shape[0] ==test_batch[0].shape[0], "GWAS: Empty Dataset")
        except:
            self.assertRaises("GWAS: Dataset creation failed")


if __name__ == '__main__':
    unittest.main()
