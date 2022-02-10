from typing import Any

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import CompBioAndSimulated_Datasets.simulated_data_multicause as bcdata  # local library / github repo
import logging

logger = logging.getLogger(__name__)


class DataSource():
    def __init__(self, x):
        self.x = x


class DataTarget():
    def __init__(self, x, t, y):
        self.x = x
        self.t = t
        self.y = y


def make_gwas(params={}, seed=0):
    """ Make the gwas dataset.

    This function adapts the gwas dataset simulated at
    We split the original dataset 80%/20% for source/target dataset respectivaly.

    :param
        params: dictionary with dataset parameters. keys: sample_size, covariates_size
    :return
        data_s: DataSource class, unlabeled covariates
        data_t: DataTarget class, labeled (with treatment assigment and outcome) covariates
    """

    params.get('sample_size', 10000)
    params.get('covariates_size', 1000)
    params.get('n_treatments', 1)

    prop = 1 / params['covariates_size']
    data_setting = bcdata.gwas_simulated_data(prop_tc=prop,  # proportion ot true causes
                                              pca_path='CompBioAndSimulated_Datasets/data/tgp_pca2.txt',
                                              seed=seed,
                                              n_units=params['sample_size'],
                                              n_causes=params["n_treatments"] + params['n_covariates'],
                                              true_causes=params["n_treatments"])
    data_x, data_y, _, treatement_columns, treatment_effects, _ = data_setting.generate_samples()
    s_x, _, _, t_x, t_y, t_t = train_test_split(data_x, data_y, data_t, seed=seed, test_size=0.2)
    data_s = DataSource(s_x)
    data_t = DataTarget(t_x, t_t, t_y)
    return data_s, data_t
