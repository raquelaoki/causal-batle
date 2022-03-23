from typing import Any
from numpy.random import binomial
from scipy.special import expit
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
import helper_parameters as hp

import CompBioAndSimulated_Datasets.simulated_data_binarycause as bcdata  # local library / github repo
import logging

logger = logging.getLogger(__name__)


# logging.basicConfig(level=logging.DEBUG)


class DataTargetAndSource:
    def __init__(self, x, t, y, d, test_size=0.33, seed=0, use_validation=False):
        self.use_validation = use_validation
        x_train, x_test, y_train, y_test, t_train, t_test, d_train, d_test = train_test_split(x, y, t, d,
                                                                                              test_size=test_size,
                                                                                              random_state=seed)
        if self.use_validation:
            x_train, x_val, y_train, y_val, t_train, t_val, d_train, d_val = train_test_split(x_train, y_train,
                                                                                              t_train, d_train,
                                                                                              test_size=test_size / 2,
                                                                                              random_state=seed)
            self.x_val = x_val
            self.y_val = y_val.reshape(-1, 1)
            self.t_val = t_val.reshape(-1, 1)
            self.d_val = d_val.reshape(-1, 1)

        self.x_train = x_train
        self.y_train = y_train.reshape(-1, 1)
        self.t_train = t_train.reshape(-1, 1)
        self.d_train = d_train.reshape(-1, 1)

        self.x_test = x_test
        self.y_test = y_test.reshape(-1, 1)
        self.t_test = t_test.reshape(-1, 1)
        self.d_test = d_test.reshape(-1, 1)

        self.x = x
        self.t = t.reshape(-1, 1)
        self.y = y.reshape(-1, 1)
        self.d = d.reshape(-1, 1)

    def loader(self, batch=32, seed=0):
        # Creating TensorDataset to use in the DataLoader.
        dataset_train = TensorDataset(Tensor(self.x_train), Tensor(self.y_train),
                                      Tensor(self.t_train), Tensor(self.d_train))
        dataset_test = TensorDataset(Tensor(self.x_test), Tensor(self.y_test),
                                     Tensor(self.t_test), Tensor(self.d_test))
        dataset_all = TensorDataset(Tensor(self.x), Tensor(self.y),
                                    Tensor(self.t), Tensor(self.d))

        # Required: Create DataLoader for training the models.
        loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch)
        loader_test = DataLoader(dataset_test, shuffle=False, batch_size=self.x_test.shape[0])
        loader_all = DataLoader(dataset_all, shuffle=False, batch_size=batch)

        if self.use_validation:
            dataset_val = TensorDataset(Tensor(self.x_val), Tensor(self.y_val),
                                        Tensor(self.t_val), Tensor(self.d_val))
            loader_val = DataLoader(dataset_val, shuffle=True, batch_size=self.x_val.shape[0])
        else:
            loader_val = None

        return loader_train, loader_val, loader_test, loader_all


class DataTarget:
    def __init__(self, x, t, y, test_size=0.33, seed=0, use_validation=False):
        super(DataTarget, self).__init__()
        self.use_validation = use_validation

        x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(x, y, t,
                                                                             test_size=test_size, random_state=seed)
        if self.use_validation:
            x_train, x_val, y_train, y_val, t_train, t_val = train_test_split(x_train, y_train, t_train,
                                                                              test_size=test_size / 2,
                                                                              random_state=seed)
            self.x_val = x_val
            self.y_val = y_val.reshape(-1, 1)
            self.t_val = t_val.reshape(-1, 1)
        self.x_train = x_train
        self.y_train = y_train.reshape(-1, 1)
        self.t_train = t_train.reshape(-1, 1)
        self.x_test = x_test
        self.y_test = y_test.reshape(-1, 1)
        self.t_test = t_test.reshape(-1, 1)
        self.x = x
        self.t = t.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

    def loader(self, batch=32, seed=0):
        # Creating TensorDataset to use in the DataLoader.
        dataset_train = TensorDataset(Tensor(self.x_train), Tensor(self.y_train), Tensor(self.t_train))
        dataset_test = TensorDataset(Tensor(self.x_test), Tensor(self.y_test), Tensor(self.t_test))
        dataset_all = TensorDataset(Tensor(self.x), Tensor(self.y), Tensor(self.t))

        # Required: Create DataLoader for training the models.
        loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch)
        loader_test = DataLoader(dataset_test, shuffle=False, batch_size=self.x_test.shape[0])
        loader_all = DataLoader(dataset_all, shuffle=False, batch_size=batch)

        if self.use_validation:
            dataset_val = TensorDataset(Tensor(self.x_val), Tensor(self.y_val), Tensor(self.t_val))
            loader_val = DataLoader(dataset_val, shuffle=True, batch_size=self.x_val.shape[0])
        else:
            loader_val = None

        return loader_train, loader_val, loader_test, loader_all


def make_Data(data_x, data_t, data_y, data_x_source=None,
              seed=1, source_size=0.2, test_size=0.33,
              use_validation=False, use_source=False):
    if use_source:
        logger.debug('... combining source and target domains data.')

        if data_x_source:
            s_x = data_x_source
            t_x, t_y, t_t = data_x, data_y, data_t
        else:
            s_x, t_x, _, t_y, _, t_t = train_test_split(data_x, data_y, data_t, random_state=seed * 10,
                                                        test_size=source_size)

        n_source = s_x.shape[0]
        n_target = t_x.shape[0]

        x = np.concatenate([s_x, t_x], axis=0)
        t = np.concatenate([np.zeros(n_source).reshape(-1,1), t_t.reshape(-1,1)], axis=0)
        y = np.concatenate([np.zeros(n_source).reshape(-1,1), t_y.reshape(-1,1)], axis=0)
        d = np.concatenate([np.zeros(n_source), np.ones(n_target)], axis=0)

        permutation = np.random.permutation(len(y))
        x = x[permutation]
        t = t[permutation]
        y = y[permutation]
        d = d[permutation]

        data = DataTargetAndSource(x=x,
                                   t=t,
                                   y=y,
                                   d=d,
                                   use_validation=use_validation,
                                   test_size=test_size)
    else:
        logger.debug('... using only target domain data.')

        s_x, t_x, _, t_y, _, t_t = train_test_split(data_x, data_y, data_t, random_state=seed * 10,
                                                    test_size=source_size)
        t_x = t_x.values
        data = DataTarget(x=t_x, t=t_t, y=t_y, use_validation=use_validation, test_size=test_size)
    return data


def make_gwas(params, unit_test=False):
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
    _key = {'n_sample': 10000, 'n_covariates': 1000, 'n_treatments': 1, 'use_validation': False,
            'use_overlap_knob': False, 'overlap_knob': 1, 'seed': 1}
    params = hp.create_if_not_available(params, _key)
    seed = params['seed']
    prop = 1 / params['n_covariates']
    data_setting = bcdata.gwas_simulated_data(prop_tc=prop,  # proportion ot true causes
                                              pca_path='CompBioAndSimulated_Datasets/data/tgp_pca2.txt',
                                              seed=seed,
                                              n_units=params['n_sample'],
                                              n_causes=params["n_treatments"] + params['n_covariates'],
                                              true_causes=params["n_treatments"],
                                              unit_test=unit_test)
    data_x, data_y, data_t, tau = data_setting.generate_samples(prop=[0.4, 0.2, 0.35])

    if params['use_overlap_knob']:
        logger.debug('...adding overlap')
        x_scale = StandardScaler()
        x_sum = x_scale.fit_transform(data_x.sum(axis=1).values.reshape(-1, 1))
        prob = expit(x_sum).reshape(-1)
        prob_knob = [1 * (1 - params['overlap_knob']) if item > 0.5 else 0 for item in prob]
        prob_knob = prob_knob + params['overlap_knob'] * prob
        data_t = [binomial(1, item) for item in prob_knob]
        data_t = np.array(data_t)

    data_t = data_t.reshape(-1)
    data = make_Data(data_x=data_x,
                     data_y=data_y,
                     data_t=data_t,
                     seed=seed,
                     source_size=0.2,
                     use_validation=params['use_validation'],
                     use_source=params['use_source'])
    return data, tau[0]  # treatment_effects[treatment_columns]


def make_ihdp(params):
    seed = params['seed']
    assert 1 <= seed <= 10, 'Seed/Id out of range (0-10)'
    data_setting = bcdata.ihdp_data(path='/content/data/ihdp/', id=seed)
    data_x, data_y, data_t, tau = data_setting.generate_samples()

    data = make_Data(data_x=data_x,
                     data_y=data_y,
                     data_t=data_t,
                     seed=seed,
                     source_size=0.2,
                     use_validation=params['use_validation'],
                     use_source=params['use_source'])

    return data, tau
