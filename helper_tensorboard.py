from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger(__name__)


class TensorboardWriter:
    """Creates tensorboard.
    One folder per config/b/seed
    """
    def __init__(self, path_logger, config_name, home_dir='/content'):
        date = self.get_date()
        full_path = home_dir + "/" + path_logger + "/" + config_name + "_" + date + "/"
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


def update_tensorboar(writer_tensorboard, values, e):
    for key in values.keys():
        writer_tensorboard.add_scalar(key, values[key], e)
    writer_tensorboard.end_writer()
    return writer_tensorboard
