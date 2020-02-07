import sys
sys.path.append('./')

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader as Loader
from data_loader.Dataset import Datasets
from model import model_rnn
from logger import set_logger
import logging
from config import option
import argparse
import torch
from trainer import trainer_Dual_RNN


def make_dataloader(opt):
    # make train's dataloader
    
    train_dataset = Datasets(
        opt['datasets']['train']['dataroot_mix'],
        [opt['datasets']['train']['dataroot_targets'][0],
         opt['datasets']['train']['dataroot_targets'][1]],
        **opt['datasets']['audio_setting'])
    train_dataloader = Loader(train_dataset,
                              batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                              num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                              shuffle=opt['datasets']['dataloader_setting']['shuffle'])
    
    # make validation dataloader
    
    val_dataset = Datasets(
        opt['datasets']['val']['dataroot_mix'],
        [opt['datasets']['val']['dataroot_targets'][0],
         opt['datasets']['val']['dataroot_targets'][1]],
        **opt['datasets']['audio_setting'])
    val_dataloader = Loader(val_dataset,
                            batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                            num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                            shuffle=False)
    
    return train_dataloader, val_dataloader


def make_optimizer(params, opt):
    optimizer = getattr(torch.optim, opt['optim']['name'])
    if opt['optim']['name'] == 'Adam':
        optimizer = optimizer(
            params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'])
    else:
        optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']
                              ['weight_decay'], momentum=opt['optim']['momentum'])

    return optimizer


def train():
    parser = argparse.ArgumentParser(
        description='Parameters for training Dual-Path-RNN')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt)
    set_logger.setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
    logger = logging.getLogger(opt['logger']['name'])
    # build model
    logger.info("Building the model of Dual-Path-RNN")
    Dual_Path_RNN = model_rnn.Dual_RNN_model(**opt['Dual_Path_RNN'])
    # build optimizer
    logger.info("Building the optimizer of Dual-Path-RNN")
    optimizer = make_optimizer(Dual_Path_RNN.parameters(), opt)
    # build dataloader
    logger.info('Building the dataloader of Dual-Path-RNN')
    train_dataloader, val_dataloader = make_dataloader(opt)

    logger.info('Train Datasets Length: {}, Val Datasets Length: {}'.format(
        len(train_dataloader), len(val_dataloader)))
    # build scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=opt['scheduler']['factor'],
        patience=opt['scheduler']['patience'],
        verbose=True, min_lr=opt['scheduler']['min_lr'])
    
    # build trainer
    logger.info('Building the Trainer of Dual-Path-RNN')
    trainer = trainer_Dual_RNN.Trainer(train_dataloader, val_dataloader, Dual_Path_RNN, optimizer, scheduler, opt)
    trainer.run()


if __name__ == "__main__":
    train()
