"""
References:
    # https://pytorch.org/vision/stable/datasets.html
    quince/library/datasets/
"""

import logging
import numpy as np
import pandas as pd
import torch

from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets

# Local Imports
import CompBioAndSimulated_Datasets.simulated_data_binarycause as bcdata  # local library / github repo
import helper_parameters as hp
import quince.library.datasets.utils as quince

logger = logging.getLogger(__name__)


class DataTargetAndSource:
    def __init__(self, x, t, y, d, test_size=0.33, seed=0, use_validation=False, binfeat=[], contfeat=[],
                 full_size_n=-1, target_size_n=-1, source_size_n=-1,
                 ):
        self.use_validation = use_validation
        self.size = x.shape[0]
        self.target = d.sum()
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

        self.binfeat = binfeat
        self.contfeat = contfeat

        self.full_size_n = full_size_n
        self.target_size_n = target_size_n
        self.source_size_n = source_size_n

    def loader(self, batch=32):
        # Creating TensorDataset to use in the DataLoader.
        dataset_train = TensorDataset(Tensor(self.x_train), Tensor(self.y_train),
                                      Tensor(self.t_train), Tensor(self.d_train))
        dataset_test = TensorDataset(Tensor(self.x_test), Tensor(self.y_test),
                                     Tensor(self.t_test), Tensor(self.d_test))
        dataset_all = TensorDataset(Tensor(self.x), Tensor(self.y),
                                    Tensor(self.t), Tensor(self.d))

        # Required: Create DataLoader for training the models.
        max_size = int(np.min([len(self.t_test), batch]))

        loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch, drop_last=True)
        loader_test = DataLoader(dataset_test, shuffle=False, batch_size=max_size, drop_last=True)
        loader_all = DataLoader(dataset_all, shuffle=False, batch_size=batch, drop_last=True)

        if self.use_validation:
            max_size = int(np.min([len(self.t_val), batch]))
            dataset_val = TensorDataset(Tensor(self.x_val), Tensor(self.y_val),
                                        Tensor(self.t_val), Tensor(self.d_val))
            loader_val = DataLoader(dataset_val, shuffle=True, batch_size=max_size, drop_last=True)
        else:
            loader_val = None

        return loader_train, loader_val, loader_test, loader_all


class DataTarget:
    def __init__(self, x, t, y, test_size=0.33, seed=0, use_validation=False, binfeat=[], contfeat=[],
                 full_size_n=-1, target_size_n=-1, source_size_n=-1,
                 ):
        super(DataTarget, self).__init__()
        self.use_validation = use_validation
        self.target = x.shape[0]

        self.full_size_n = full_size_n
        self.target_size_n = target_size_n
        self.source_size_n = source_size_n
        x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(x, y, t,
                                                                             test_size=test_size,
                                                                             random_state=seed + 100)
        if self.use_validation:
            x_train, x_val, y_train, y_val, t_train, t_val = train_test_split(x_train, y_train, t_train,
                                                                              test_size=test_size / 2,
                                                                              random_state=seed + 100)
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
        self.binfeat = binfeat
        self.contfeat = contfeat

    def loader(self, batch=32):
        # Creating TensorDataset to use in the DataLoader.
        dataset_train = TensorDataset(Tensor(self.x_train), Tensor(self.y_train), Tensor(self.t_train))
        dataset_test = TensorDataset(Tensor(self.x_test), Tensor(self.y_test), Tensor(self.t_test))
        dataset_all = TensorDataset(Tensor(self.x), Tensor(self.y), Tensor(self.t))

        # Required: Create DataLoader for training the models.
        max_size = int(np.min([len(self.t_test), batch * 4]))
        loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch, drop_last=True)
        loader_test = DataLoader(dataset_test, shuffle=False, batch_size=max_size, drop_last=True)
        loader_all = DataLoader(dataset_all, shuffle=False, batch_size=batch, drop_last=True)

        if self.use_validation:
            max_size = int(np.min([len(self.y_val), batch]))
            dataset_val = TensorDataset(Tensor(self.x_val), Tensor(self.y_val), Tensor(self.t_val))
            loader_val = DataLoader(dataset_val, shuffle=True, batch_size=max_size, drop_last=True)
        else:
            loader_val = None

        return loader_train, loader_val, loader_test, loader_all


def make_DataClass(data_x, data_t, data_y,
                   data_x_source=None, seed=1, source_size=0.2, test_size=0.33,
                   use_validation=False, use_source=False, binfeat=[], contfeat=[], use_data_x_source=False,
                   seed_add_on=0,
                   informative_source=True,
                   ):
    """ It creates the data classes (DataTarget or DataTargetAndSource) from input data data_x, data_t, data_y.
    VALID ONLY FOR GWAS AND IHDP.
    It will alwyas split the dataset using source_size, even it the method only uses the target data
    (decision based on the experiments setup - check paper for more info).
    it sses seed to ensure reprodubility(same seed will generate same dataset). Recommends: use b as seed.

    :param data_x: np.matrix
    :param data_t: np.array
    :param data_y: np.array
    :param use_data_x_source: bool
    :param data_x_source: np.matrix (optional)
    :param seed: integer
    :param source_size: proportion, in[0,1]
    :param test_size: proportion, in[0,1]
    :param use_validation: bool
    :param use_source:bool
    :param binfeat: array with numbers of binary covariates columns
    :param contfeat: array with numbers of continous covariates columns
    :return: DataTarget or DataTargetAndSource class
    """
    if use_source:
        logger.debug('... combining source and target domains data.')
        if use_data_x_source:
            s_x = data_x_source
            t_x, t_y, t_t = data_x, data_y, data_t
        else:
            s_x, t_x, _, t_y, _, t_t = train_test_split(data_x, data_y, data_t, random_state=seed + seed_add_on,
                                                        test_size=source_size)

        if not informative_source:
            original_shapes = s_x.shape
            pandas = False
            if type(s_x) == pd.DataFrame:
                pandas = True
                col = s_x.columns
                s_x = s_x.values

            full = s_x.reshape(-1, 1)
            permutation = np.random.permutation(len(full))
            s_x = full[permutation]
            s_x = s_x.reshape(original_shapes)
            if pandas:
                s_x = pd.DataFrame(s_x)
                s_x.columns = col

        n_source = s_x.shape[0]
        n_target = t_x.shape[0]

        x = np.concatenate([s_x, t_x], axis=0)
        t = np.concatenate([np.zeros(n_source).reshape(-1, 1), t_t.reshape(-1, 1)], axis=0)
        y = np.concatenate([np.zeros(n_source).reshape(-1, 1), t_y.reshape(-1, 1)], axis=0)
        d = np.concatenate([np.zeros(n_source), np.ones(n_target)], axis=0)

        np.random.seed(seed + seed_add_on)
        permutation = np.random.permutation(len(y))
        x = x[permutation]
        t = t[permutation]
        y = y[permutation]
        d = d[permutation]

        data = DataTargetAndSource(x=x, t=t, y=y, d=d, use_validation=use_validation,
                                   test_size=test_size, binfeat=binfeat, contfeat=contfeat,
                                   full_size_n=x.shape[0], target_size_n=n_target, source_size_n=n_source,
                                   )
    else:
        logger.debug('... using only target domain data.')
        if source_size < 1:
            #  Only a proportion of the data in source is used (GWAS and IHDP)
            s_x, target_x, _, target_y, _, target_t = train_test_split(data_x, data_y, data_t,
                                                        random_state=seed + seed_add_on,
                                                        test_size=source_size)
            target_x = target_x.values
            np.random.seed(seed + seed_add_on)
            permutation = np.random.permutation(len(target_y))
            t_x = target_x[permutation]
            t_t = target_t[permutation]
            t_y = target_y[permutation]
            data = DataTarget(x=t_x, t=t_t, y=t_y, use_validation=use_validation, test_size=test_size,
                              binfeat=binfeat, contfeat=contfeat,
                              full_size_n=data_x.shape[0], target_size_n=target_x.shape[0], source_size_n=-1,
                              )
        else:
            #  All the data in source is used (HCMNIST)
            data = DataTarget(x=data_x, t=data_t, y=data_y, use_validation=use_validation, test_size=test_size,
                              binfeat=binfeat, contfeat=contfeat,
                              full_size_n=data_x.shape[0], target_size_n=data_x.shape[0], source_size_n=-1,
                              )
    return data


def make_gwas(params, unit_test=False):
    """ Make the gwas dataset.

    This function adapts the gwas dataset simulated at
    We split the original dataset 80%/20% for source/target dataset respectivaly.

    :param
        params: dictionary with dataset parameters. keys: sample_size, covariates_size
    :return
        data: DataSource class
    """
    # Adding default values
    _key = {'n_sample': 10000, 'n_covariates': 1000, 'n_treatments': 1, 'use_validation': False,
            'use_overlap_knob': False, 'overlap_knob': 1, 'seed': 1}
    params = hp.create_if_not_available(params, _key)
    seed = params['data_seed']
    prop = 1 / params['n_covariates']
    data_setting = bcdata.gwas_simulated_data(prop_tc=prop,  # proportion ot true causes
                                              pca_path='CompBioAndSimulated_Datasets/data/tgp_pca2.txt',
                                              seed=seed,
                                              n_units=params['n_sample'],
                                              n_causes=params["n_treatments"] + params['n_covariates'],
                                              true_causes=params["n_treatments"],
                                              unit_test=unit_test)
    data_x, data_y, data_t, tau = data_setting.generate_samples(prop=[0.4, 0.2, 0.35])  # prop is internal from GWAS.

    if params['use_overlap_knob']:
        logger.debug('...adding overlap')
        x_scale = StandardScaler()
        x_sum = x_scale.fit_transform(data_x.sum(axis=1).values.reshape(-1, 1))
        prob = expit(x_sum).reshape(-1)
        prob_knob = [1 * (1 - params['overlap_knob']) if item > 0.5 else 0 for item in prob]
        prob_knob = prob_knob + params['overlap_knob'] * prob
        data_t = [np.random.binomial(1, item) for item in prob_knob]
        data_t = np.array(data_t)

    data_t = data_t.reshape(-1)
    data = make_DataClass(data_x=data_x,
                          data_y=data_y,
                          data_t=data_t,
                          seed=params['data_seed'],
                          source_size=params['source_size_p'],
                          use_validation=params['use_validation'],
                          use_source=params['use_source'],
                          binfeat=[],
                          contfeat=list(range(data_x.shape[1])),
                          seed_add_on=params['seed_add_on'],
                          informative_source=params['informative_source'],
                          )
    return data, tau[0]  # treatment_effects[treatment_columns]


def make_ihdp(params):
    """Call functions from CompBioAndSimulate Repository

    :param params:
    :return:
    """
    seed = params['seed']
    assert 0 <= seed <= 8, 'Seed/Id out of range (0-8) ---' + str(seed)
    data_setting = bcdata.ihdp_data(path='/content/data/ihdp/', id=seed)
    data_x, data_y, data_t, tau = data_setting.generate_samples()
    data = make_DataClass(data_x=data_x,
                          data_y=data_y,
                          data_t=data_t,
                          seed=params['data_seed'],
                          source_size=params['source_size_p'],
                          use_validation=params['use_validation'],
                          use_source=params['use_source'],
                          binfeat=list(range(6, 25)),
                          contfeat=list(range(6)),
                          seed_add_on=params['seed_add_on'],
                          informative_source=params['informative_source'],
                          )

    return data, tau


def make_hcmnist(params):
    data_setting = HCMNIST('', download=True, seed=params['data_seed'],
                           use_fix_digit=params['use_fix_digit'],
                           target_size=params['target_size'],
                           use_source=params['use_source'],
                           source_size=params['source_size'],
                           )
    data = make_DataClass(data_x=data_setting.x_t,
                          data_y=data_setting.y_t,
                          data_t=data_setting.t_t,
                          data_x_source=data_setting.x_s,
                          seed=params['data_seed'],
                          source_size=1,  # Use all x_t available
                          use_validation=params['use_validation'],
                          use_source=params['use_source'],
                          binfeat=[],
                          contfeat=list(range(data_setting.x_t.shape[1])),
                          seed_add_on=params['seed_add_on'],
                          use_data_x_source=params['use_data_x_source'],
                          informative_source=params['informative_source'],
                          )
    tau = data_setting.tau.mean()
    return data, tau


class HCMNIST(datasets.MNIST):
    def __init__(self,
                 root=None,
                 download=False,
                 use_fix_digit=True,
                 target_size=1000,
                 seed=0,
                 use_source=False,
                 source_size=1000
                 ):
        # https://pytorch.org/vision/stable/datasets.html
        self.t_t = None
        self.y_t = None
        self.tau = None
        self.__class__.__name__ = "MNIST"
        super(HCMNIST, self).__init__(root, train=True, download=download)
        self.seed = seed
        self.data = (self.data.float().div(255) - 0.1307).div(0.3081)

        #  Fix seed and randomizing digits
        np.random.seed(self.seed)
        digits = list(range(10))
        np.random.shuffle(digits)

        #  Pick two digits to be on target-domain.
        mask_target = np.in1d(self.targets.numpy(), digits[0:2])
        self.x_t = self.data[mask_target]
        self.target_t = self.targets[mask_target]

        #  Simulates t and y
        self.create_hcmnist()

        # Getting correct number of samples on target-domain.
        target_samples = list(range(self.x_t.shape[0]))
        np.random.shuffle(target_samples)
        target_samples_selection = target_samples[0:target_size]
        self.x_t = self.x_t[target_samples_selection]
        self.target_t = self.target_t[target_samples_selection]
        self.t_t = self.t_t[target_samples_selection]
        self.y_t = self.y_t[target_samples_selection]

        # Setting source-domain:
        if use_source:
            if use_fix_digit:
                mask_source = np.in1d(self.targets.numpy(), digits[-1])  # Last Digit as source-domain.
                self.x_s = self.data[mask_source]
            else:
                # In-domain source domain
                self.x_s = self.data[mask_target]
                # Remove samples used by target domain
                source_samples_selection = target_samples[target_size:-1] #target not used
                self.x_s = self.x_s[source_samples_selection]

            # Fixing sample size as source_size
            source_samples = list(range(self.x_s.shape[0]))
            np.random.shuffle(source_samples)
            self.x_s = self.x_s[source_samples[0:source_size]]
        else:
            self.x_s = None

    def _fit_phi_model(self, domain=2):
        # Reference: quince/library/datasets/utils.py
        edges = torch.arange(-domain, domain + 0.1, (2 * domain) / 10)
        data = self.x_t.view(self.x_t.shape[0], -1)
        model = {}
        digits = torch.unique(self.target_t)
        for i, digit in enumerate(digits):
            lo, hi = edges[i: i + 2]
            ind = self.target_t == digit
            data_ind = data[ind].view(ind.sum(), -1)
            means = data_ind.mean(dim=-1)
            mu = means.mean()
            sigma = means.std()
            model.update(
                {
                    digit.item(): {
                        "mu": mu.item(),
                        "sigma": sigma.item(),
                        "lo": lo.item(),
                        "hi": hi.item(),
                    }
                }
            )
        return model

    def _phi(self, domain=2):
        # Reference: quince/library/datasets/utils.py
        # x = ((data.astype("float32") / 255.0) - 0.1307) / 0.3081
        z = np.zeros_like(self.target_t.numpy().astype("float32"))
        phi_model = self._fit_phi_model(domain=domain)

        for k, v in phi_model.items():
            ind = self.target_t == k
            x_ind = self.x_t[ind].reshape(ind.sum(), -1)
            means = x_ind.mean(axis=-1)
            z[ind] = quince.linear_normalization(
                np.clip((means - v["mu"]) / v["sigma"], -1.4, 1.4), v["lo"], v["hi"]
            )
        return np.expand_dims(z, -1)

    def create_hcmnist(self, sigma_y=1, theta=4):
        # adapted
        # https://github.com/anndvision/quince/blob/main/quince/library/datasets/hcmnist.py

        # Creates phi=[-2,2]
        phi = self._phi()
        #  Propensity Score.
        rng = np.random.RandomState(seed=self.seed)
        e_propscore = expit(phi.ravel() * 0.75 + 0.5)

        #  Treat Assig.
        self.t_t = rng.binomial(1, e_propscore).astype("float32")
        u = rng.binomial(1, 0.5, size=len(e_propscore)).astype("float32").ravel()

        #  Random Noise.
        eps = (sigma_y * rng.normal(size=self.t_t.shape)).astype("float32")

        #  Outcomes.
        mu0 = (quince.f_mu(x=phi.ravel(), t=0.0, u=u, theta=theta).astype("float32").ravel())
        mu1 = (quince.f_mu(x=phi.ravel(), t=1.0, u=u, theta=theta).astype("float32").ravel())
        y0 = mu0 + eps
        y1 = mu1 + eps
        self.y_t = self.t_t * y1 + (1 - self.t_t) * y0
        self.tau = mu1 - mu0
