# Causal-BaTLe


## Compute Canada Instructions

* Source code and data: /home/raoki/projects/rrg-ester/raoki/batle

* Env and checkpoints: /home/raoki/scratch/
* https://www.notion.so/Compute-Canada-Wiki-Ester-Lab-869c7e3b51b54fb39970e0fbd7b8af3f 
* baselines https://github.com/oatml/ucate (bayesian NN + uncertainty) and https://github.com/anndvision/quince (with bounds)
* transfer learning https://medium.com/georgian-impact-blog/transfer-learning-part-1-ed0c174ad6e7
* drgonnet https://github.com/claudiashi57/dragonnet/blob/master/src/experiment/models.py

## TODOS
* cyberduck
* Use checkpoints
* TensorBoard
* virtualenv - Add version of packages 
* datasets
* Adding logging
* bayesian model: trick is on the sampler functions
* can I use kaggle dataset for my own researach? 
https://www.kaggle.com/andrewmvd/retinal-disease-classification
  https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/139291
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2769884/

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


### Logging 

Reference: 
https://www.machinelearningplus.com/python/python-logging-guide/


### Baselines Instructions

ucate: 

### Experiments 

run dataset x method b times -> csv

there are Data = [dataset1, ..., datasetn]

there are Methods = [method1, ..., methodn]

combine all the csvs
