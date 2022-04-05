""" CEVAE

We adaopted the existing implementation available in rik-helwegen/CEVAE_pytorch.
We used the implemented model, and modified the training setup.

Reference:
https://github.com/rik-helwegen/CEVAE_pytorch
# TODO: add paper
"""
import logging
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch import optim
from torch.distributions import normal

# Local
import utils
import helper_tensorboard as ht
from CEVAE_pytorch.networks import p_x_z, p_t_z, p_y_zt, q_t_x, q_y_xt, q_z_tyx

logger = logging.getLogger(__name__)

"""
# set evaluator objects
    evaluator_train = Evaluator(yalltr, talltr, y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                mu0=np.concatenate([mu0tr, mu0va], axis=0), mu1=np.concatenate([mu1tr, mu1va], axis=0))
    evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)
"""


class cevae:
    def __init__(self, n_covariates, binfeat, contfeat, z_dim=20, h_dim=64, ):
        x_dim = n_covariates  # len(binfeats) + len(contfeats)
        self.p_x_z_dist = p_x_z(dim_in=z_dim, nh=3, dim_h=h_dim, dim_out_bin=len(binfeat),
                                dim_out_con=len(contfeat))  # .cuda()
        self.p_t_z_dist = p_t_z(dim_in=z_dim, nh=1, dim_h=h_dim, dim_out=1)  # .cuda()
        self.p_y_zt_dist = p_y_zt(dim_in=z_dim, nh=3, dim_h=h_dim, dim_out=1)  # .cuda()
        self.q_t_x_dist = q_t_x(dim_in=x_dim, nh=1, dim_h=h_dim, dim_out=1)  # .cuda()
        # t is not feed into network, therefore not increasing input size (y is fed).
        self.q_y_xt_dist = q_y_xt(dim_in=x_dim, nh=3, dim_h=h_dim, dim_out=1)  # .cuda()
        self.q_z_tyx_dist = q_z_tyx(dim_in=len(binfeat) + len(contfeat) + 1, nh=3, dim_h=h_dim,
                                    dim_out=z_dim)  # .cuda()
        self.p_z_dist = normal.Normal(torch.zeros(z_dim), torch.ones(z_dim))
        self.binfeat = binfeat
        self.contfeat = contfeat

    def parameters(self):
        params = list(self.p_x_z_dist.parameters()) + list(self.p_t_z_dist.parameters()) + list(
            self.p_y_zt_dist.parameters()) + list(self.q_t_x_dist.parameters()) + list(
            self.q_y_xt_dist.parameters()) + list(self.q_z_tyx_dist.parameters())
        return params


def cevae_pred(model, batch, device):
    xy = torch.cat((batch[0], batch[1]), 1)
    z_infer = model.q_z_tyx_dist(xy=xy.to(device), t=batch[2].to(device))
    # use a single sample to approximate expectation in lowerbound
    z_infer_sample = z_infer.sample()
    # p(x|z)
    x_bin, x_con = model.p_x_z_dist(z_infer_sample)
    # p(t|z)
    t = model.p_t_z_dist(z_infer_sample)
    # p(y|t,z)
    # for training use t_train, in out-of-sample prediction this becomes t_infer
    y = model.p_y_zt_dist(z_infer_sample, batch[2].to(device))
    # AUXILIARY LOSS
    # q(t|x)
    t_infer = model.q_t_x_dist(batch[0].to(device))
    # q(y|x,t)
    y_infer = model.q_y_xt_dist(batch[0].to(device), batch[2].to(device))

    pred = {'x_bin': x_bin,
            'x_con': x_con,
            't': t,
            'y': y,
            't_infer': t_infer,
            'y_infer': y_infer,
            'p_z_dist': model.p_z_dist,
            'z_infer': z_infer,
            'z_infer_sample':z_infer_sample
            }

    return pred


def criterion_l1l2(batch, predictions, device='cpu', binfeat=None, contfeat=None):
    x_bin = batch[0][:, binfeat]
    x_con = batch[0][:, :len(contfeat)]
    l2 = predictions['x_con'].log_prob(x_con.to(device)).sum(1)
    l1 = predictions['x_bin'].log_prob(x_bin.to(device)).sum(1)
    return l1, l2


def criterion_t(batch, predictions, device='cpu'):
    # Reconstr_t
    return predictions['t'].log_prob(batch[2].to(device)).mean()


def criterion_y(batch, predictions, device='cpu'):
    # Reconstr_y
    return predictions['y'].log_prob(batch[1].to(device)).mean()


def criterion_kl(batch, predictions, device='cpu'):
    # REGULARIZATION LOSS
    # p(z) - q(z|x,t,y)
    # approximate KL
    batch = torch.Tensor(batch).to(device)
    return (predictions['p_z_dist'].log_prob(batch) - predictions['z_infer'].log_prob(batch)).sum(1)


def criterion_l6l7(batch, predictions, device='cpu'):
    l6 = predictions['t_infer'].log_prob(batch[2].to(device)).mean()
    l7 = predictions['y_infer'].log_prob(batch[1].to(device)).mean()
    return l6, l7


def _calculate_criterion_cevae(criterion_function, batch, predictions, device='cpu',
                               alpha=[1, 1, 0], episilon=0.001,
                               contfeat=None, binfeat=None, z_infer_sample=None):
    l1, l2 = criterion_function[0](batch=batch, predictions=predictions, device=device,
                                   binfeat=binfeat, contfeat=contfeat)
    lt = criterion_function[1](batch=batch, predictions=predictions, device=device)
    ly = criterion_function[2](batch=batch, predictions=predictions, device=device)
    l5 = criterion_function[3](batch=z_infer_sample, predictions=predictions, device=device)
    l6, l7 = criterion_function[4](batch=batch, predictions=predictions, device=device)

    return l1, l2, lt, ly, l5, l6, l7


def _calculate_metric_cevae(metric_functions, batch, predictions):
    metrics_t = metric_functions[0](batch=batch,
                                    predictions=predictions)
    metrics_y = metric_functions[1](batch=batch,
                                    predictions=predictions)
    return metrics_t, metrics_y


def metric_function_cevae_t(batch, predictions):
    dif = batch[2] - predictions['t'].mean.cpu().detach().numpy()
    return np.abs(dif).mean()


def metric_function_cevae_y(batch, predictions):
    pred = predictions['y'].mean.cpu().detach().numpy().reshape(-1,1)
    return mean_squared_error(batch[1], pred)


def init_qz(qz, loader_train, device):
    """
    Initialize qz towards outputting standard normal distributions
    - with standard torch init of weights the gradients tend to explode after first update step
    """

    optimizer = optim.Adam(qz.parameters(), lr=0.001)

    for i in range(50):
        batch = next(iter(loader_train))
        xy = torch.cat((batch[0], batch[1]), 1)
        z_infer = qz(xy=xy.to(device), t=batch[2].to(device))

        # KL(q_z|p_z) mean approx, to be minimized
        # KLqp = (z_infer.log_prob(z_infer.mean) - pz.log_prob(z_infer.mean)).sum(1)
        # Analytic KL
        KLqp = (-torch.log(z_infer.stddev) + 1 / 2 * (z_infer.variance + z_infer.mean ** 2 - 1)).sum(1)

        objective = torch.mean(KLqp)
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    return qz



def fit_cevae(epochs,
              model,
              loader_train,
              loader_test,
              optimizer,
              criterion,
              metric_functions,
              use_validation,
              use_tensorboard,
              device,
              loader_val=None,
              alpha=[],
              path_logger='',
              config_name='',
              home_dir='',
              episilon=0.001,
              weight_1=1):
    """
        Fit implementation: Contain epochs and batch iterator, optimization steps, and eval.
    :param episilon:
    :param home_dir:
    :param config_name:
    :param path_logger:
    :param metric_functions:
    :param use_tensorboard:
    :param loader_test:
    :param epochs: integer
    :param model: nn.Module
    :param loader_train: DataLoader
    :param optimizer: torch.optim
    :param criterion: List of criterions
    :param metric_functions: List of metrics
    :param use_validation: Bool
    :param device: torch.device
    :param loader_val: DataLoader (Optional)
    :param alpha: alphas to balance losses, torch.Tensor

    :return: model: nn.Module
    :return: loss: dictionary with all losses calculated
    :return: metrics: disctionary with all metrics calcualted
    """

    logger.debug('...starting')

    # init q_z inference
    model.q_z_tyx_dist = init_qz(model.q_z_tyx_dist, loader_train, device=device)

    # use prefetch_generator and tqdm for iterating through data
    # pbar = tqdm(enumerate(BackgroundGenerator(train_data_loader, ...)),
    #            total=len(train_data_loader))
    # start_time = time.time()

    if use_tensorboard:
        writer_tensorboard = ht.TensorboardWriter(path_logger=path_logger,
                                                  config_name=config_name,
                                                  home_dir=home_dir)

    if len(alpha) == 0:
        alpha = torch.ones(len(criterion))
    elif not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha)

    loss_train_t, metric_train_t = np.zeros(epochs), np.zeros(epochs)
    loss_train_y, metric_train_y = np.zeros(epochs), np.zeros(epochs)

    loss_val_t, metric_val_t = np.zeros(epochs), np.zeros(epochs)
    loss_val_y, metric_val_y = np.zeros(epochs), np.zeros(epochs)

    # All losses.
    loss_train_all, loss_val_all = np.zeros(epochs), np.zeros(epochs)

    for e in range(epochs):
        torch.cuda.empty_cache()
        _metrics_t, _metrics_y = [], []
        _loss_t, _loss_y, _loss_all = [], [], []
        for i, batch in enumerate(loader_train):
            optimizer.zero_grad()

            # inferred distribution over z
            predictions = cevae_pred(model, batch, device=device)

            l1, l2, lt, ly, l5, l6, l7 = _calculate_criterion_cevae(criterion_function=criterion,
                                                                    batch=batch,
                                                                    predictions=predictions,
                                                                    device=device,
                                                                    alpha=alpha,
                                                                    episilon=episilon,
                                                                    binfeat=model.binfeat,
                                                                    contfeat=model.contfeat,
                                                                    z_infer_sample=predictions['z_infer_sample']
                                                                    )
            # Total objective
            # inner sum to calculate loss per item, torch.mean over batch
            loss_batch = - torch.mean(l1 + l2 + lt + ly + l5 + l6 + l7)
            loss_batch.backward()
            optimizer.step()

            _loss_t.append(lt.cpu().detach().numpy())
            _loss_y.append(ly.cpu().detach().numpy())
            _loss_all.append(loss_batch.cpu().detach().numpy())
            metrics_batch_t, metrics_batch_y = _calculate_metric_cevae(metric_functions=metric_functions,
                                                                       batch=batch,
                                                                       predictions=predictions)
            _metrics_t.append(metrics_batch_t)
            _metrics_y.append(metrics_batch_y)

        loss_train_t[e] = np.mean(_loss_t)
        loss_train_y[e] = np.mean(_loss_y)
        loss_train_all[e] = np.mean(_loss_all)
        metric_train_t[e] = np.mean(_metrics_t)
        metric_train_y[e] = np.mean(_metrics_y)
        # print('epoch', e, loss_train_t[e], loss_train_y[e])
        if use_validation:
            batch = next(iter(loader_val))
            predictions = cevae_pred(model, batch, device=device)

            l1, l2, lt, ly, l5, l6, l7 = _calculate_criterion_cevae(criterion_function=criterion,
                                                                    batch=batch,
                                                                    predictions=predictions,
                                                                    device=device,
                                                                    alpha=alpha,
                                                                    episilon=episilon,
                                                                    binfeat=model.binfeat,
                                                                    contfeat=model.contfeat,
                                                                    z_infer_sample=predictions['z_infer_sample']
                                                                    )
            loss_val_all[e] = - torch.mean(l1 + l2 + lt + ly + l5 + l6 + l7).cpu().detach().numpy()
            loss_val_t[e], loss_val_y[e] = lt.cpu().detach().numpy(), ly.cpu().detach().numpy()
            metric_val_t[e], metric_val_y[e] = _calculate_metric_cevae(metric_functions=metric_functions,
                                                                       batch=batch,
                                                                       predictions=predictions)
        else:
            loss_val_t[e], loss_val_y[e], loss_val_all[e] = None, None, None
            metric_val_t[e], metric_val_y[e] = None, None

        if use_tensorboard:
            values = {'loss_train_t': loss_train_t[e], 'loss_train_y': loss_train_y[e],
                      'loss_train_all': loss_train_all[e], 'metric_train_t': metric_train_t[e],
                      'metric_train_y': metric_train_y[e]}
            writer_tensorboard = ht.update_tensorboar(writer_tensorboard, values, e)
            values = {'loss_val_t': loss_val_t[e], 'loss_val_y': loss_val_y[e],
                      'loss_val_all': loss_val_all[e],
                      'metric_val_t': metric_val_t[e], 'metric_val_y': metric_val_y[e]}
            writer_tensorboard = ht.update_tensorboar(writer_tensorboard, values, e)

    # Metrics on testins set
    batch = next(iter(loader_test))
    # predictions = model(batch[0].to(device))
    predictions = cevae_pred(model, batch, device=device)
    metric_test_t, metric_test_y = _calculate_metric_cevae(metric_functions=metric_functions,
                                                           batch=batch,
                                                           predictions=predictions,
                                                           )

    loss = {'loss_train_t': loss_train_t,
            'loss_val_t': loss_val_t,
            'loss_train_y': loss_train_y,
            'loss_val_y': loss_val_y,
            'loss_val_all': loss_val_all,
            'loss_train_all': loss_train_all,
            }
    metrics = {'metric_train_t': metric_train_t,
               'metric_val_t': metric_val_t,
               'metric_train_y': metric_train_y,
               'metric_val_y': metric_val_y,
               'metric_test_t': metric_test_t,
               'metric_test_t': metric_test_t,
               'metric_test_y': metric_test_y
               }

    return model, loss, metrics
