import argparse
import logging
import yaml
from pathlib import Path

import coloredlogs
import numpy as np
import random
import torch
from tape import ProteinBertForValuePrediction

LOG_FORMAT = "[%(asctime)s] [PID %(process)d] %(levelname)s:\t%(filename)s:%(funcName)s():%(lineno)s:\t%(message)s"


def get_args():
    """
    Standard parsing of command line argument specifying experiment config file
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c',
                        '--config',
                        required=True,
                        type=str,
                        help='Experiment YAML config file')
    args = parser.parse_args()
    args.exp_dir = Path(args.config).parent
    with open(args.config, 'r') as config_file:
        args.config = yaml.load(config_file, Loader=yaml.FullLoader)

    for path in args.config['files']:
        args.config['files'][path] = str(args.exp_dir / args.config['files'][path])
    return args


def get_logger(logger_id=__name__):
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
    logger = logging.getLogger(logger_id)
    coloredlogs.install(fmt=LOG_FORMAT, level='INFO', logger=logger)
    return logger


class LogCapture(object):
    """
    Wrapper class to monkeypatch logging inside a given module.
    Append all log messages to LocCapture.output for later inspection.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output = []

    def log(self, *args, **kwargs):
        self.logger.log(*args, **kwargs)
        self.output.append(args)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)
        self.output.append(args)

    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)
        self.output.append(args)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)
        self.output.append(args)


def load_model(experiment_config: dict) -> ProteinBertForValuePrediction:
    """
    Load model specified in experiment YAML config file
    and set random seeds and
    """
    RND_SEED = 42
    torch.manual_seed(RND_SEED)
    np.random.seed(RND_SEED)
    random.seed(RND_SEED)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(RND_SEED)
    else:
        logger = get_logger()
        logger.warning('PyTorch running on CPUs')

    model = ProteinBertForValuePrediction.from_pretrained(
        str(experiment_config['files']['model_checkpoint']),
        output_attentions=True
    )
    model.eval()
    return model
