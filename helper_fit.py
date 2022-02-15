import pandas as pd
from baselines.baseline1 import Baseline
from new_method import NewMethod
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def fit(loader_train, loader_val, loader_test, loader_all, params, model_name, n_covariates):
    """Fit models.
    :param loader_train:
    :param loader_val:
    :param loader_test:
    :param loader_all:
    :param params: keys: units1, units2, units3, binary_target, seed, batch_size, epochs_adam, epochs_sgd, reg_l2
    :return:
    """
    logging.debug('Model: Fitting {}', params['model_name'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(params['seed'])
    tf.random.set_seed(self.seed)
    np.random.seed(self.seed)

    if model_name == 'dragonnet':
        model = dragonnet(n_covariates=n_covariates, units1=params['units1'], units2=params['units2'],
                          units3=params['units3'], binary_target=params['binary_target'],
                          type_original=params['type_original'])

        if params['binary_target']:
            criterion = []  # TODO

        logging.debug('...Making the dragonnet')
        if torch.cuda.is_available():
            model.to(device)

        optimizer1 = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
        opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer1, gamma=params["gamma"])

        loss_train, loss_val = np.zeros(params['max_epochs']), np.zeros(params['max_epochs'])
        metric_y_train, metric_y_val = [], []
        metric_t_train, metric_t_val = [], []

        if params['type_target'] == 'binary':
            best_val_metric = 0
        else:
            best_val_metric = 999

        print('... Training')

        best_epoch = 0

        optimizer2 = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])



        yt_train = np.concatenate([y_train, t_train], 1)
        # Define optimization parameters.
        compile_metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]
        model_dragonnet.compile(optimizer=Adam(lr=1e-3), loss=loss, metrics=compile_metrics)
        verbose = 0  # Don't output log into the standard output stream
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