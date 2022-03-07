from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger(__name__)

class TensorboardWriter:
    def __init__(self, path_logger, name_config):
        date = self.get_date()
        #home_dir = os.getenv("HOME")
        home_dir = '/content'
        full_path = home_dir + "/" + path_logger + "/" + date + "/" + name_config + "/"
        logger.debug('Tensorboard folder path - %s', full_path)
        self.writer = SummaryWriter(log_dir=full_path)

    # Add day, month and year to path
    def get_date(self):
        now = datetime.now()  # Current date and time (Hour, minute)
        date = now.strftime("%Y_%m_%d_%H_%M")
        return date

    def add_scalar(self, name_metric, value_metric, epoch):
        self.writer.add_scalar(name_metric, value_metric, epoch)

    def end_writer(self):
        # Make sure all pending events have been written to disk
        self.writer.flush()


def update_tensorboar(writer_tensorboard, values, e, set='train'):
    names = ['loss_t_' + set, 'loss_y_' + set, 'loss_ty_' + set,
             'roc_t_' + set, 'mse_y_' + set]
    assert len(values) == len(names)
    for i in range(len(names)):
        writer_tensorboard.add_scalar(names[i], values[i], e)
    writer_tensorboard.end_writer()
    return writer_tensorboard