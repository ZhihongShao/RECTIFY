import logging

from tensorboardX import SummaryWriter

import torch

def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter(
        "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)

logger = logging.getLogger()
setup_logger(logger)
log_on_rank_0 = True

def config_logger(log_on_rank_0_only: bool = True):
    assert isinstance(log_on_rank_0_only, bool)
    global log_on_rank_0
    log_on_rank_0 = log_on_rank_0_only

def log(message, log_on_rank_0_only: bool = None):
    """If distributed is initialized print only on rank 0."""
    log_on_rank_0_only = log_on_rank_0 if not isinstance(log_on_rank_0_only, bool) else log_on_rank_0_only
    if log_on_rank_0_only:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(message)
    else:
        logger.info(message)

tb_writer = None

def log_on_tensorboard(global_step, **kwargs):
    global tb_writer
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        if tb_writer is None:
            tb_writer = SummaryWriter()
        for key, val in kwargs.items():
            tb_writer.add_scalar(key, val, global_step)
