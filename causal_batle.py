import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
import logging

logger = logging.getLogger(__name__)

# https://github.com/raquelaoki/M3E2/blob/main/model_m3e2.py

class batle(nn.Module):
    """New Method.
    TODO.
    input:
        TODO
    returns:
        TODO
    """

    def __init__(self, dataset_name, architecture):
        super().__init__()

        self.dataset_name = dataset_name
        self.architecture = architecture

        # TODO Define body of the architecture.

        # TODO Define adversatial component

        # TODO Define mixture of normal

        # TODO Define head

        # Activation functions and others
        self.dropout = nn.Dropout(dropoutp)
        self.tahn = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        logger.info('... Model initialization done!')

    def forward(self, batch):

        n = batch.shape[0]

        '''Dropout'''
        batch = self.dropout(batch)

        ''' Body'''

        ''' Adversarial'''

        ''' Mixture'''

        '''Outcome output - Y'''
        outcome_y = batch

        return outcome_y