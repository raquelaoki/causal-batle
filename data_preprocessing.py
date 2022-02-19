from typing import Any

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset

import CompBioAndSimulated_Datasets.simulated_data_multicause as bcdata  # local library / github repo
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class DataSource:
    def __init__(self, x):
        self.x = x


class DataTarget:
    def __init__(self, x, t, y, test_size=0.33, seed=0, validation=False):
        super(DataTarget, self).__init__()
        self.validation = validation

        x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(x, y, t,
                                                                             test_size=0.33, random_state=seed)
        if validation:
            x_train, x_val, y_train, y_val, t_train, t_val = train_test_split(x_train, y_train, t_train,
                                                                              test_size=0.15, random_state=seed)
            self.x_val = x_val.values
            self.y_val = y_val.reshape(-1, 1)
            self.t_val = t_val.reshape(-1, 1)
        self.x_train = x_train.values
        self.y_train = y_train.reshape(-1, 1)
        self.t_train = t_train.reshape(-1, 1)
        self.x_test = x_test.values
        self.y_test = y_test.reshape(-1, 1)
        self.t_test = t_test.reshape(-1, 1)
        self.x = x.values
        self.t = t.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

    def loader(self, shuffle=True, batch=32, seed=0):
        # Creating TensorDataset to use in the DataLoader.
        dataset_train = TensorDataset(Tensor(self.x_train), Tensor(self.y_train), Tensor(self.t_train))
        dataset_test = TensorDataset(Tensor(self.x_test), Tensor(self.y_test), Tensor(self.t_test))
        dataset_all = TensorDataset(Tensor(self.x), Tensor(self.y), Tensor(self.t))

        # Required: Create DataLoader for training the models.
        loader_train = DataLoader(dataset_train, shuffle=shuffle, batch_size=batch)
        loader_test = DataLoader(dataset_test, shuffle=False, batch_size=self.x_test.shape[0])
        loader_all = DataLoader(dataset_all, shuffle=False, batch_size=batch)

        if self.validation:
            dataset_val = TensorDataset(Tensor(self.x_val), Tensor(self.y_val), Tensor(self.t_val))
            loader_val = DataLoader(dataset_val, shuffle=shuffle, batch_size=self.x_val.shape[0])
        else:
            loader_val = None

        return loader_train, loader_val, loader_test, loader_all


def make_gwas(params, seed=0):
    """ Make the gwas dataset.

    This function adapts the gwas dataset simulated at
    We split the original dataset 80%/20% for source/target dataset respectivaly.

    :param
        params: dictionary with dataset parameters. keys: sample_size, covariates_size
    :return
        data_s: DataSource class, unlabeled covariates
        data_t: DataTarget class, labeled (with treatment assigment and outcome) covariates
    """
    # Adding default values
    params["n_sample"] = params.get('n_sample', 10000)
    params['n_covariates'] = params.get('n_covariates', 1000)
    params['n_treatments'] = params.get('n_treatments', 1)
    prop = 1 / params['n_covariates']
    data_setting = bcdata.gwas_simulated_data(prop_tc=prop,  # proportion ot true causes
                                              pca_path='CompBioAndSimulated_Datasets/data/tgp_pca2.txt',
                                              seed=seed,
                                              n_units=params['n_sample'],
                                              n_causes=params["n_treatments"] + params['n_covariates'],
                                              true_causes=params["n_treatments"])
    data_x, data_y, _, treatment_columns, treatment_effects, _ = data_setting.generate_samples(prop=[0.4, 0.2, 0.35])
    data_t = data_x.iloc[:, treatment_columns[0]].values
    data_x.drop(data_x.columns[treatment_columns].values[0], axis=1, inplace=True)
    s_x, t_x, _, t_y, _, t_t = train_test_split(data_x, data_y, data_t, random_state=seed, test_size=0.2)
    data_s = DataSource(s_x)
    data_t = DataTarget(t_x, t_t, t_y)
    logging.debug('Dataset Prep Complete')
    return data_s, data_t, treatment_effects[treatment_columns]
