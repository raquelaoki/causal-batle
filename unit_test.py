import unittest
import data_preprocessing as dp


class DataPrep(unittest.TestCase):
    def test_gwas(self):
        try:
            data_s, data_t = dp.make_gwas()
            self.assertTrue(data_s.x.shape[0] > 0 and data_t.x.shape[0] > 0, "GWAS: Empty Dataset")
        except:
            self.assertRaises("GWAS: Dataset creation failed")


if __name__ == '__main__':
    unittest.main()
