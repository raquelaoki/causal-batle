"""
References:
https://github.com/claudiashi57/dragonnet/blob/master/src/experiment/models.py
https://github.com/raquelaoki/M3E2/blob/main/resources/dragonnet.py

Alternative implementation of the Dragonnet model: A neural network to estimate treatment effects.

"""
import numpy as np
import pandas as pd
import keras.backend as K
from keras import callbacks
from keras import layers
from keras import regularizers
from tensorflow.keras.layers import Layer
from keras.models import Model
from keras.metrics import binary_accuracy
from tensorflow.keras.optimizers import SGD, Adam
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

# from semi_parametric_estimation.ate import psi_naive, psi_tmle_cont_outcome
# Do I use this functions above? https://github.com/raquelaoki/M3E2/tree/main/resources/semi_parametric_estimation
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
# from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, mean_squared_error
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class dragonnet:
    """
    y_scaler: scale y based on train y
    """
    def __init__(self, X, y, treatment, seed=0, binary_target=False):
        super(dragonnet, self).__init__()
        self.X = X.values
        self.y = y.reshape(-1, 1)
        self.treatment = treatment
        self.y_scaler = None
        self.binary_target = binary_target
        self.seed = seed
        logging.debug('Model: Running dragonnet')

    def fit(self, val_split=0.2, batch_size=64, epochs_adam=100, epochs_sgd=300, u1=200, u2=100, u3=1, reg_l2=0.01):
        """Fit dragonnet with given parameters
            :param val_split: validation split for training
            :param batch_size:
            :param epochs_adam:
            :param epochs_sgd:
            :param u1: dragonnet 3-layers shared representation hidden units
            :param u2: dragonnet 2-layer head-specific representation hidden units
            :param u3: dragonnet final-layer hidden units
            :param reg_l2: regularizer
            :return: metrics_test, metrics_train, y_pred
        """

        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(self.X, self.y,
                                                                            self.treatment, test_size=0.33)

        y_test_pred, y_train_pred, y_all_pred = self.train_and_predict_dragons(t_train=t_train,
                                                                               y_train=y_train,
                                                                               x_train=X_train,
                                                                               t_test=t_test,
                                                                               y_test=y_test,
                                                                               x_test=X_test,
                                                                               val_split=val_split,
                                                                               batch_size=batch_size,
                                                                               epochs_adam=epochs_adam,
                                                                               epochs_sgd=epochs_sgd,
                                                                               u1=u1, u2=u2, u3=u3,
                                                                               reg_l2=reg_l2)

        y_train_pred_array = y_train_pred[:, 0] * (t_train == 0) + y_train_pred[:, 1] * (t_train == 1)
        y_test_pred_array = y_test_pred[:, 0] * (t_test == 0) + y_test_pred[:, 1] * (t_test == 1)
        y_train_pred_array = self.y_scaler.inverse_transform(y_train_pred_array)
        y_test_pred_array = self.y_scaler.inverse_transform(y_test_pred_array)

        if self.binary_target:
            thhold = self.Find_Optimal_Cutoff(self.y_train, y_train_pred)
            y_train_pred_array01 = [0 if item < thhold else 1 for item in y_train_pred_array]
            y_test_pred_array01 = [0 if item < thhold else 1 for item in y_test_pred_array]
            f1_train = f1_score(y_train, y_train_pred_array01)
            f1_test = f1_score(y_test, y_test_pred_array01)
            logging.debug('...Evaluation - training set - F1 score {} - confusion matrix {}',
                          f1_train,
                          confusion_matrix(y_train, y_train_pred_array01).ravel())
            logging.debug('...Evaluation - test set - F1 score {} - confusion matrix {}',
                          f1_test,
                          confusion_matrix(y_test, y_test_pred_array01).ravel())
            return f1_test, f1_train, y_all_pred

        else:
            mse_train = mean_squared_error(y_train, y_train_pred_array)
            mse_test = mean_squared_error(y_test, y_test_pred_array)
            logging.debug('...Evaluation - training set - MSE {}', mse_train)
            logging.debug('...Evaluation - testing set - MSE {}', mse_test)
            return mse_test, mse_train, y_all_pred


    def train_and_predict_dragons(self,
                                  x_train,
                                  y_train,
                                  t_train,
                                  x_test,
                                  y_test,
                                  t_test,
                                  val_split=0.2,
                                  batch_size=64,
                                  epochs_adam=100,
                                  epochs_sgd=300,
                                  u1=200, u2=100, u3=1,
                                  reg_l2=0.01):
        """Makes Dragonnet Model.
        1) Defines the architecture with make_dragonnet()
        2) Define compiler and metrics
        3) Fit the model
        Returns:
            yt_test_pred, yt_train_pred, all_pred
        """

        self.y_scaler = StandardScaler().fit(y_train)
        y_train = self.y_scaler.transform(y_train)
        y_test = self.y_scaler.transform(y_test)

        logging.debug('...Making the dragonnet')
        model_dragonnet = make_dragonnet(n_covariates=x_train.shape[1], reg_l2=reg_l2, u1=u1, u2=u2, u3=u3)
        loss = dragonnet_loss_binarycross() # knob_loss

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        yt_train = np.concatenate([y_train, t_train], 1)
        # Define optimization parameters.
        compile_metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]
        model_dragonnet.compile(optimizer=Adam(lr=1e-3), loss=loss, metrics=compile_metrics)

        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                              min_delta=1e-8, cooldown=0, min_lr=0)
        ]
        # Fit model
        model_dragonnet.fit(x_train, yt_train, callbacks=adam_callbacks, validation_split=val_split,
                            epochs=epochs_adam, batch_size=batch_size, verbose=verbose)

        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                              min_delta=0., cooldown=0, min_lr=0)
        ]

        sgd_lr = 1e-5
        momentum = 0.9
        model_dragonnet.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), loss=loss,
                                metrics=compile_metrics)
        model_dragonnet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                            validation_split=val_split,
                            epochs=epochs_sgd,
                            batch_size=batch_size, verbose=verbose)
        yt_test_pred = model_dragonnet.predict(x_test)
        yt_train_pred = model_dragonnet.predict(x_train)
        all_pred = model_dragonnet.predict(np.concatenate(self.X, axis=0))

        K.clear_session()

        return yt_test_pred, yt_train_pred, all_pred

    def ate(self, ):
        y = self.y_scaler.transform(self.y)

        for i in range(len(self.treatments_columns)):
            t_col = self.treatments_columns[i]
            if dataset == 'train':
                t = self.X_train[:, [t_col]]
            if dataset == 'test':
                t = self.X_test[:, [t_col]]
            if dataset == 'all':
                t = np.concatenate([self.X_train[:, [t_col]], self.X_test[:, [t_col]]], axis=0)
            yt_hat = self.all_output_list[i]

            q_t0, q_t1, g, t, y_dragon, x, eps = self.split_output(yt_hat, t, y, self.y_scaler, X)

            psi_n, psi_tmle, initial_loss, final_loss, g_loss = self.get_estimate(q_t0, q_t1, g, t, y_dragon,
                                                                                  truncate_level=0.01)
            # print('TCOL',psi_n,q_t0, q_t1)
            self.simple_ate.append(psi_n)
            self.tmle_ate.append(psi_tmle)

        return self.simple_ate, self.tmle_ate

    def split_output(self, yt_hat, t, y, y_scaler, x):
        q_t0 = self.y_scaler.inverse_transform(yt_hat[:, 0].copy())
        q_t1 = self.y_scaler.inverse_transform(yt_hat[:, 1].copy())
        g = yt_hat[:, 2].copy()

        if yt_hat.shape[1] == 4:
            eps = yt_hat[:, 3][0]
        else:
            eps = np.zeros_like(yt_hat[:, 2])

        y = y_scaler.inverse_transform(y.copy())
        # var = "average propensity for treated: {} and untreated: {}".format(g[t.squeeze() == 1.].mean(),
        #                                                                    g[t.squeeze() == 0.].mean())
        # print(var)

        # return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'index': index, 'eps': eps}
        return q_t0, q_t1, g, t, y, x, eps

    def get_estimate(self, q_t0, q_t1, g, t, y_dragon, truncate_level=0.01):
        """
        getting the back door adjustment & TMLE estimation
        """

        psi_n = psi_naive(q_t0, q_t1, g, t, y_dragon, truncate_level=truncate_level)
        psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(q_t0, q_t1, g, t,
                                                                                                  y_dragon,
                                                                                                  truncate_level=truncate_level)
        return psi_n, psi_tmle, initial_loss, final_loss, g_loss

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------
    list type, with optimal cutoff value
    https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    """

    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


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


def track_epsilon(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))


class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]


def make_dragonnet(n_covariates, reg_l2, u1=200, u2=100, u3=1):
    """
    Neural net predictive model. The dragon has three heads.
    Parameters:
        n_covariates: number of covariates
        u1: 3-layer representation component with u1 units
        u2: 2-layer outcome prediction +
        u3: final-layer of outcome prediction

    :return:
    """
    t_l1 = 0.
    t_l2 = reg_l2
    inputs = Input(shape=(n_covariates,), name='input')

    # representation
    x = Dense(units=u1, activation='elu', kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=u1, activation='elu', kernel_initializer='RandomNormal')(x)
    x = Dense(units=u1, activation='elu', kernel_initializer='RandomNormal')(x)

    t_predictions = Dense(units=1, activation='sigmoid')(x)

    # HYPOTHESIS
    y0_hidden = Dense(units=u2, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=u2, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=u2, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=u2, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=u3, activation=None, kernel_regularizer=regularizers.l2(reg_l2),
                           name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=u3, activation=None, kernel_regularizer=regularizers.l2(reg_l2),
                           name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    logging.debug(epsilons)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model

