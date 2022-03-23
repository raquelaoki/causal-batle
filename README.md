# Causal-BaTLe


## Compute Canada Instructions

* Source code and data: /home/raoki/projects/rrg-ester/raoki/batle

* Env and checkpoints: /home/raoki/scratch/
* https://www.notion.so/Compute-Canada-Wiki-Ester-Lab-869c7e3b51b54fb39970e0fbd7b8af3f 
* baselines https://github.com/oatml/ucate (bayesian NN + uncertainty) and https://github.com/anndvision/quince (with bounds)
* transfer learning https://medium.com/georgian-impact-blog/transfer-learning-part-1-ed0c174ad6e7
* drgonnet https://github.com/claudiashi57/dragonnet/blob/master/src/experiment/models.py


## Setup

### Virtualenv 

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

### Interactive Session

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

## Usage

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



### Baselines Instructions

ucate: 

### Experiments 

* run dataset x method b times -> csv
* there are Data = [dataset1, ..., datasetn]
* there are Methods = [method1, ..., methodn]
* combine all the csvs

### workflow
1) causal batle 
   * dropout for predictions 
2) Fit dragonnet
   * double check losses from dragonnet - here
3) AIPW 
   * gwas version is not good
4) save best epoch model 
5) save output for data + model
6) https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
7) Wrapper to run dragonnet b times
8) Fit causa-battle 
9) Fit cevae 
10) fit dr-cfr 2020
11) Fit X-learner,
12) Add new dataset (vision)
13) Add another dataset

Note: My discriminator loss is right! 