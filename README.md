# Causal-BaTLe

Author: Raquel Aoki

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

To run the unit tests in Colab, we use: 
```shell
!git clone https://github.com/raquelaoki/CompBioAndSimulated_Datasets.git
!git clone -b in_progress https://github.com/raquelaoki/causal-batle.git
!mv  -v /content/causal-batle/* /content/
!python -m unit_test
```


## References
* baselines https://github.com/oatml/ucate (bayesian NN + uncertainty) and https://github.com/anndvision/quince (with bounds)
* drgonnet https://github.com/claudiashi57/dragonnet/blob/master/src/experiment/models.py
* add gwas
* add bayesian layers 
* IN CONSTRUCTION

## Compute Canada Instructions (not used)

* Source code and data: /home/raoki/projects/rrg-ester/raoki/batle
* Env and checkpoints: /home/raoki/scratch/
* https://www.notion.so/Compute-Canada-Wiki-Ester-Lab-869c7e3b51b54fb39970e0fbd7b8af3f 

#### Virtualenv 

```commandline
module load python/3.9

virtualenv --no-download env

# Activate virtualenv from scratch:
source env/bin/activate

# Activate virtualenv from rrg-ester
source ~/scratch/env/bin/activate

# Deactivate virtualenv 
deactivate

# Initial install (or new packages)
pip install --no-index -r requirements.txt
```

#### Interactive Session

Setting up jupyter notebook
```commandline
# For interactive jupyter notebook:
echo -e '#!/bin/bash\nunset XDG_RUNTIME_DIR\njupyter notebook --ip $(hostname -f) --no-browser' > $VIRTUAL_ENV/bin/notebook.sh
chmod u+x $VIRTUAL_ENV/bin/notebook.sh
jupyter nbextension install --py jupyterlmod --sys-prefix #dont work
jupyter nbextension enable --py jupyterlmod --sys-prefix
jupyter serverextension enable --py jupyterlmod --sys-prefix
```

Request Session - Compute Canada
```commandline
source ~/scratch/env/bin/activate

# CPU
salloc --time=1:0:0  --cpus-per-task=1 --mem=4000M --account=rrg-ester srun $VIRTUAL_ENV/bin/notebook.sh

# GPU
salloc --time=1:0:0 --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --mem=4000M --account=rrg-ester srun $VIRTUAL_ENV/bin/notebook.sh
```

Local Machine
```commandline
ssh -L 8888:cdr767.int.cedar.computecanada.ca:8888 USER@cedar.computecanada.ca
```

Tensorboard 
```commandline
# Load the TensorBoard notebook extension
%load_ext tensorboard
%tensorboard --logdir logs
```

### TODO:
1) Save model based on validation;
2) Add main on unit_test.py
3) Add cevae; 
4) Add dr-cfr;
5) Add X-learner;
6) Add new dataset (vision);
7) Add BCCH dataset;
8) sensitive analysis. 
