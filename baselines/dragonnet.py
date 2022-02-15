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
from sklearn.preprocessing import StandardScaler
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
    def __init__(self, n_covariates, units1=200, units2=100, units3=1,
                 binary_target=False, type_original=True):
        super().__init__()
        self.units1 = units1
        self.units2 = units2
        self.units3 = units3
        self.binary_target = binary_target
        self.representation_layer1 = nn.Linear(in_features=n_covariates, out_features=self.units1)
        self.representation_layer1_ = nn.Linear(in_features=self.units1, out_features=self.units1)
        self.type_original=type_original
        if self.type_original:
            self.dragonnet_head = dragonnet_original(self.units1, self.units2, self.units3, self.binary_target)
        else:
            logging.debug("Gaussian Process not implemented yet.")
        # Activation functions.
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.thhold=[] #For future metrics

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

        t_predictions, y0_predictions, y1_predictions = dragonnet_head(x)

        return t_predictions, y0_predictions, y1_predictions

    def dragonnet_metrics(self, t_predictions, y0_predictions, y1_predictions, batch, is_train=False, title=""):
        """ Need to be called once for each set (train / test/ val / all)
        :param t_predictions:
        :param y0_predictions:
        :param y1_predictions:
        :return:
        """
        y_pred_array = y0_predictions* (t_predictions == 0) + y1_predictions * (t_predictions == 1)
        t_pred_01 = [0 if item < 0.5 else 1 for item in t_predictions]
        f1_batch_t = f1_score(batch[2],t_pred_01)

        if self.binary_target:
            if is_train:
                thhold = utils.Find_Optimal_Cutoff(batch[1], y_pred_array)
                self.thhold.append(thhold)
            y_pred_array01 = [0 if item < np.mean(self.thhold) else 1 for item in y_pred_array]
            f1_batch = f1_score(batch[1], y_pred_array01)
            if not is_train:
                logging.debug('...Evaluation - {} set - F1 score {} - confusion matrix {}',
                              title,
                              f1_train,
                              confusion_matrix(batch[1], y_pred_array01).ravel())

            return f1_batch, f1_batch_t

        else:
            mse_batch = mean_squared_error(batch[1], y_pred_array)
            if not is_train:
                logging.debug('...Evaluation - {} set - MSE {}', title, mse_test)
            return mse_batch, f1_batch_t





class dragonnet_original(nn.Module):
    """ Dragonnet Original Head.

    """
    def __init__(self, units1, units2, units3, binary_target):
        self.binary_target =binary_target
        self.t_predictions = nn.Linear(units1, 1)
        self.head_layer2 = nn.Linear(in_features=units1, out_feautures=units2)
        self.head_layer2_ = nn.Linear(in_features=units2, out_feautures=units2)
        self.outcome_layer = nn.Linear(in_features=units2, out_feautures=units3)
        # Activation functions.
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # Propensity Score - Treatment Prediction.
        t_predictions = self.sigmoid(self.t_predictions(inputs))

        # Treatment specific - first layer.
        y0_hidden = self.elu(self.head_layer2(inputs)) #Missing: regularizers.l2
        y1_hidden = self.elu(self.head_layer2(inputs))

        # Treatment specific - second layer.
        y0_hidden = self.elu(self.head_layer2(y0_hidden))
        y1_hidden = self.elu(self.head_layer2(y1_hidden))

        # Treatment specific - third layer.
        y0_predictions = self.outcome_layer(y0_hidden)
        y1_predictions = self.outcome_layer(y1_hidden)

        if self.binary_target:
            y0_predictions=self.sigmoid(y0_predictions)
            y1_predictions=self.sigmoid(y1_predictions)

        return t_predictions, y0_predictions, y1_predictions


def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    return losst


def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))

    return loss0 + loss1


def dragonnet_loss_binarycross(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def treatment_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return binary_accuracy(t_true, t_pred)




