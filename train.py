import argparse
import collections
import torch
import torch.nn as nn
import numpy as np
import data_loader.data_loaders as module_data
import model.metric as module_metric
from model.model import AdvancedEnsembleModel
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = AdvancedEnsembleModel(num_classes=config['arch']['args'].get('num_classes', 2))
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # setup loss, metrics, optimizer
    criterion = nn.CrossEntropyLoss()
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())

    # learning rate scheduler
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Initialize the Trainer
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
