# Causal-BaTLe

Author: 

## Introduction
This repository contains software and data for "Causal inference from small high-dimensional datasets".
The paper introduces an approach to estimate causal effects in small high-dimensional datasets using transfer learning.


## Usage

All our models are implemented in Pytorch. It supports GPUs and CPUs. 

### Hyperparameters
The implementation available is very flexible, meaning that the user must define
a set of parameters. We divide the parameters in three groups: 
1) Data parameters: responsible for defining which dataset will be
loaded/simulated, their dimensions (if suitable). The main parameter
is 'data_name'. See [Datasets](###Datasets) for details on implemented datasets.
2) Model parameters: The user needs to define model's name, neural network architecture (units, dropout, regular/bayesian), 
optimization parameters (epochs, batch, learning rate, validation, loss weights).
3) Utils parameters: The user needs to define if wants tensorboard, the config_file name, seeds, 
list of ate methods to use, etc.

All these paramters have default values. However, we highlight the most important parameters as: 
1) data_name and model_name: depends on the list of implemented methods, and define which 
functions and data the model will load.
2) max_epochs, lr (learning_rate), batch_size: important paramters in any 
neural network model. These can affect a lot the quality of the final estimates.
3) alpha: represent the loss weights. It changes depending on the model_name
adopted. Default value is equal weight for all losses. 

### Parameter creation
The current implementation supports two approaches: 
1) Reading from .yaml file (See examples in config/):
```python
params = hp.parameter_loader(config_path)
```

2) Using a function that creates the dictionary:
```python
params=parameter_debug(data_name='ihdp',max_epochs=50, lr=0.1,batch_size=50, 
                          units1=200,units2=100, weight_decay=0.05, use_validation=False,
                          use_dropout=False,dropout_p=0.5, use_overlap_knob=False,
                          seed=1, alpha=[1,1,0], config_name='config_test1', 
                          use_tensorboard=True, model_name='dragonnet')
```

Note: we also have the function `_check_params_consistency()`, which helps to 
ensure all parameters are properly set.


###Implemented methods

We currently implemented two ATE estimators, and 4 outcome models. 

#### Outcome Models

TODO: add references
* AIPW (aipw): 
* Dragonnet (dragonnet): 
* Bayesian Dragonnet (bdragonnet): 
* Causal-Batle (batle):

#### ATE estimators

Uses the outcome of the previous methods to estimate ATE.
* Naive
* AIPW

###Datasets
There are currently two datasets implemented: 
1) IHDP: Collection of datasets with ~700 samples and 23 covariates. Binary treatment and continuous outcome.
Downloaded from CEVAE repository [[link](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP/csv) ] 
2) GWAS: Synthetic dataset [[link](https://github.com/raquelaoki/CompBioAndSimulated_Datasets)]. Default values are 1000 samples
and 5000 covariates. Binary treatment and continous outcome. 

Check the publication for the original references of these datasets, and more details
on data preprocessing, and adaptation for transfer learning. 

###Experiments 

Each given parameter setting (yaml file) can run #seeds x #repetitions 
independent models. Each #seed will generate an independent dataset, and
each #repetition will run an independent model. 
```python
import pandas as pd
import helper_parameters as hp
from utils import repeat_experiment, run_model

table = pd.DataFrame()
output_path = '/outputs'
params = hp.parameter_loader(config_path=config)
table = repeat_experiment(params, table, use_range_source_p=False, 
                          save=True, output_save=output_path)
```
The object table will contain one #seeds x #repetitions output per row.

To run a single model, you can use: 
```python
params = hp.parameter_loader(config_path=config)
metrics, loss, ate, tau = run_model(params)
```

## Unit Test
We have 4 unit tests implemented that check the main functions implemented.
```python
!python -m unit_test
```

## References
* Bayesian Layers: https://github.com/oatml/ucate (bayesian NN + uncertainty) and https://github.com/anndvision/quince (with bounds)
* Dragonnet https://github.com/claudiashi57/dragonnet/blob/master/src/experiment/models.py
* CEVAE: https://github.com/rik-helwegen/CEVAE_pytorch/
* HCMNIST: quince/library/datasets/
