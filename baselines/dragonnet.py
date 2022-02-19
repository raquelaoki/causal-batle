"""
References:
https://github.com/claudiashi57/dragonnet/blob/master/src/experiment/models.py
https://github.com/raquelaoki/M3E2/blob/main/resources/dragonnet.py

Alternative implementation of the Dragonnet model: A neural network to estimate treatment effects.
Adopting pytorch
"""
import numpy as np
import pandas as pd
import keras.backend as K

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf
import utils

# from semi_parametric_estimation.ate import psi_naive, psi_tmle_cont_outcome
# Do I use this functions above? https://github.com/raquelaoki/M3E2/tree/main/resources/semi_parametric_estimation
#
# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, mean_squared_error
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class dragonnet(nn.Module):
    def __init__(self, n_covariates, units1=200, units2=100,
                   units3=1, type_original=True):
        super().__init__()
        self.units1 = units1
        self.units2 = units2
        self.units3 = units3
        self.representation_layer1 = nn.Linear(in_features=n_covariates, out_features=self.units1)
        self.representation_layer1_ = nn.Linear(in_features=self.units1, out_features=self.units1)
        self.type_original = type_original
        if self.type_original:
            self.dragonnet_head = dragonnet_original(self.units1, self.units2, self.units3)
        else:
            logging.debug("Gaussian Process not implemented yet.")
        # Activation functions.
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.thhold = []  # For future metrics

    def forward(self, inputs):
        """
        Neural net predictive model. The dragon has three heads.
        Parameters:
            n_covariates: number of covariates
            u1: 3-layer representation component with u1 units
            u2: 2-layer outcome prediction +
            u3: final-layer of outcome prediction

        :return:
        """

        # Shared presentation.
        x = self.elu(self.representation_layer1(inputs))
        x = self.elu(self.representation_layer1_(x))
        x = self.elu(self.representation_layer1_(x))
        return self.dragonnet_head(x)


class dragonnet_original(nn.Module):
    """ Dragonnet Original Head.
    """

    def __init__(self, units1=200,units2=100, units3=1):
        super(dragonnet_original, self).__init__()
        self.units1 = units1
        self.units2 = units2
        self.units3 = units3

        self.head_layer2 = nn.Linear(in_features=self.units1, out_features=self.units2)
        self.head_layer2_ = nn.Linear(in_features=self.units2, out_features=self.units2)
        self.outcome_layer = nn.Linear(in_features=self.units2, out_features=self.units3)
        self.t_predictions = nn.Linear(in_features=self.units1, out_features=1)

        # Activation functions.
        self.elu = nn.ELU(alpha=0.5)
        self.sigmoid = nn.Sigmoid()
        self.tahn = nn.Tanh()

    def forward(self, inputs):
        # Treatment specific - first layer.
        y0_hidden = self.elu(self.head_layer2(inputs))
        y1_hidden = self.elu(self.head_layer2(inputs))

        # Treatment specific - second layer.
        y0_hidden = self.tahn(self.head_layer2_(y0_hidden))
        y1_hidden = self.tahn(self.head_layer2_(y1_hidden))

        # Treatment specific - third layer.
        y0_predictions = self.outcome_layer(y0_hidden)
        y1_predictions = self.outcome_layer(y1_hidden)

        y0_predictions = self.tahn(y0_predictions)
        y1_predictions = self.tahn(y1_predictions)

        t_predictions = self.sigmoid(self.t_predictions(inputs))
        # y01_predictions = torch.cat((y0_predictions,y1_predictions),1)
        predictions = {'y0':y0_predictions,
                       'y1':y1_predictions,
                       't':t_predictions, }
        return predictions