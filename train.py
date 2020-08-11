"""
Learnable generative compression model [1] implemented in Pytorch.

Example usage:
python3 train.py -h

[1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
    arXiv:2006.09965 (2020).
"""
import numpy as np
import os, glob, time, datetime
import logging, pickle, argparse
import functools, itertools

from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Custom modules
from hific.model import HificModel
from hific.utils import helpers, initialization, datasets, math
from default_config import hific_args, mse_lpips_args, directories, ModelModes, ModelTypes

# go fast boi!!
# Optimizes cuda kernels by benchmarking - no dynamic input sizes!
torch.backends.cudnn.benchmark = True

def create_model(args, device, logger, storage, storage_test):

    start_time = time.time()
    model = HificModel(args, logger, storage, storage_test, model_type=args.model_type)
    logger.info(model)

    logger.info('Trainable parameters:')
    for n, p in model.named_parameters():
        logger.info('{} - {}'.format(n, p.shape))

    logger.info("Number of trainable parameters: {}".format(helpers.count_parameters(model)))
    logger.info("Estimated size (under fp32): {:.3f} MB".format(helpers.count_parameters(model) * 4. / 10**6))

    logger.info('Model init {:.3f}s'.format(time.time() - start_time))

    return model

def optimize_loss(loss, opt, retain_graph=False):
    loss.backward(retain_graph=retain_graph)
    opt.step()
    opt.zero_grad()

def optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt):
    compression_loss.backward()
    amortization_opt.step()
    hyperlatent_likelihood_opt.step()
    amortization_opt.zero_grad()
    hyperlatent_likelihood_opt.zero_grad()

def test(args, model, epoch, idx, data, test_data, device, epoch_test_loss, storage, best_test_loss, 
         start_time, epoch_start_time, logger, train_writer, test_writer):

    model.eval()  
    with torch.no_grad():
        data = data.to(device, dtype=torch.float)

        losses, intermediates = model(data, return_intermediates=True, writeout=False)
        helpers.save_images(train_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
            fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TRAIN_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))

        test_data = test_data.to(device, dtype=torch.float)
        losses, intermediates = model(test_data, return_intermediates=True)
        helpers.save_images(test_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
            fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TEST_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))
    
        compression_loss = losses['compression'] 
        epoch_test_loss.append(compression_loss.item())
        mean_test_loss = np.mean(epoch_test_loss)
        
        best_test_loss = helpers.log(model, storage, epoch, idx, mean_test_loss, compression_loss.item(), 
                                     best_test_loss, start_time, epoch_start_time, 
                                     batch_size=data.shape[0], header='[TEST]', 
                                     logger=logger, writer=test_writer)
        
    return best_test_loss, epoch_test_loss


def train(args, model, train_loader, test_loader, device, storage, storage_test, logger):

    best_loss, best_test_loss, mean_epoch_loss = np.inf, np.inf, np.inf     
    test_loader_iter = iter(test_loader)
    model.train()
    start_time = time.time()
    current_D_steps, train_generator = 0, True
    train_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'train'))
    test_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'test'))

    amortization_parameters = itertools.chain.from_iterable(
        [am.parameters() for am in model.amortization_models])
    hyperlatent_likelihood_parameters = model.Hyperprior.hyperlatent_likelihood.parameters()

    amortization_opt = torch.optim.Adam(amortization_parameters,
        lr=args.learning_rate)
    hyperlatent_likelihood_opt = torch.optim.Adam(hyperlatent_likelihood_parameters, 
        lr=args.learning_rate)
    optimizers = dict(amort=amortization_opt, hyper=hyperlatent_likelihood_opt)

    # Contingency
    logger.info('Optimizing over:')
    amortization_params, hyperlatent_likelihood_params, discriminator_params = list(), list(), list()
    # for name, param in model.named_parameters():
    #     logger.info(name)
    #     if 'hyperlatent_likelihood' in name:
    #         logger.info('Adding {} to hyperlatent likelihood params'.format(name))
    #         hyperlatent_likelihood_params.append(param)
    #     if ('hyperlatent_likelihood' not in name) and ('Discriminator' not in name):
    #         logger.info('Adding {} to amortization params'.format(name))
    #         amortization_params.append(param)
    #     if 'Discriminator' in name:
    #         logger.info('Adding {} to discriminator params'.format(name))
    #         discriminator_params.append(param)
    # param_groups = {'amortization': amortization_params, 'hyperlatent_likelihood': hyperlatent_likelihood_params}

    if model.use_discriminator is True:
        # param_groups['discriminator'] = discriminator_params
        discriminator_parameters = model.Discriminator.parameters()
        disc_opt = torch.optim.Adam(discriminator_parameters, lr=args.learning_rate)
        optimizers['disc'] = disc_opt

    
    for epoch in trange(args.n_epochs, desc='Epoch'):

        epoch_loss, epoch_test_loss = [], []  
        epoch_start_time = time.time()
        
        if epoch > 0:
            ckpt_path = helpers.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
        
        for idx, data in enumerate(tqdm(train_loader, desc='Train'), 0):

            data = data.to(device, dtype=torch.float)
            
            try:
                if model.use_discriminator is True:
                    # Train D for D_steps, then G, using distinct batches
                    losses = model(data, train_generator=train_generator)
                    compression_loss = losses['compression']
                    disc_loss = losses['disc']

                    if train_generator is True:
                        optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt)
                        train_generator = False
                    else:
                        optimize_loss(disc_loss, disc_opt)
                        current_D_steps += 1

                        if current_D_steps == args.discriminator_steps:
                            current_D_steps = 0
                            train_generator = True

                        continue
                else:
                    # Rate, distortion, perceptual only
                    losses = model(data, train_generator=True)
                    compression_loss = losses['compression']
                    optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt)

            except KeyboardInterrupt:
                if model.step_counter > args.log_interval+1:
                    logger.warning('Exiting, saving ...')
                    ckpt_path = helpers.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
                    return model, ckpt_path
                else:
                    return model, None

            if idx % args.log_interval == 1:
                epoch_loss.append(compression_loss.item())
                mean_epoch_loss = np.mean(epoch_loss)

                best_loss = helpers.log(model, storage, epoch, idx, mean_epoch_loss, compression_loss.item(),
                                best_loss, start_time, epoch_start_time, batch_size=data.shape[0],
                                logger=logger, writer=train_writer)
                try:
                    test_data = test_loader_iter.next()
                except StopIteration:
                    test_loader_iter = iter(test_loader)
                    test_data = test_loader_iter.next()

                best_test_loss, epoch_test_loss = test(args, model, epoch, idx, data, test_data, device, epoch_test_loss, storage_test,
                     best_test_loss, start_time, epoch_start_time, logger, train_writer, test_writer)

                with open(os.path.join(args.storage_save, 'storage_{}_tmp.pkl'.format(args.name)), 'wb') as handle:
                    pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

                model.train()

                # LR scheduling
                helpers.update_lr(args, amortization_opt, model.step_counter, logger)
                helpers.update_lr(args, hyperlatent_likelihood_opt, model.step_counter, logger)
                if model.use_discriminator is True:
                    helpers.update_lr(args, disc_opt, model.step_counter, logger)

                if model.step_counter > args.n_steps:
                    logger.info('Reached step limit [args.n_steps = {}]'.format(args.n_steps))
                    break

            if idx % args.save_interval == 1:
                ckpt_path = helpers.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)

        # End epoch
        mean_epoch_loss = np.mean(epoch_loss)
        mean_epoch_test_loss = np.mean(epoch_test_loss)

        logger.info('===>> Epoch {} | Mean train loss: {:.3f} | Mean test loss: {:.3f}'.format(epoch, 
            mean_epoch_loss, mean_epoch_test_loss))    

        if model.step_counter > args.n_steps:
            break
    
    with open(os.path.join(args.storage_save, 'storage_{}_{:%Y_%m_%d_%H:%M:%S}.pkl'.format(args.name, datetime.datetime.now())), 'wb') as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ckpt_path = helpers.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
    args.ckpt = ckpt_path
    logger.info("Training complete. Time elapsed: {:.3f} s. Number of steps: {}".format((time.time()-start_time), model.step_counter))
    
    return model, ckpt_path


if __name__ == '__main__':

    description = "Learnable generative compression."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options - see `default_config.py` for full options
    general = parser.add_argument_group('General options')
    general.add_argument("-n", "--name", default=None, help="Identifier for checkpoints and metrics.")
    general.add_argument("-mt", "--model_type", required=True, choices=(ModelTypes.COMPRESSION, ModelTypes.COMPRESSION_GAN), 
        help="Type of model - with or without GAN component")
    general.add_argument("-regime", "--regime", choices=('low','med','high'), default='low', help="Set target bit rate - Low (0.14), Med (0.30), High (0.45)")
    general.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU ID.")
    general.add_argument("-log_intv", "--log_interval", type=int, default=100, help="Number of steps between logs.")
    general.add_argument("-save_intv", "--save_interval", type=int, default=50000, help="Number of steps between checkpoints.")
    general.add_argument("-multigpu", "--multigpu", help="Toggle data parallel capability using torch DataParallel", action="store_true")
    general.add_argument('-bs', '--batch_size', type=int, default=8, help='input batch size for training')
    general.add_argument('--save', type=str, default='experiments', help='Parent directory for stored information (checkpoints, logs, etc.)')

    # Optimization-related options
    optim_args = parser.add_argument_group("Optimization-related options")
    optim_args.add_argument('-steps', '--n_steps', type=int, default=2e6, 
        help="Number of gradient steps. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument('-epochs', '--n_epochs', type=int, default=10, 
        help="Number of passes over training dataset. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Optimizer learning rate.")
    optim_args.add_argument("-wd", "--weight_decay", type=float, default=1e-6, help="Coefficient of L2 regularization.")

    cmd_args = parser.parse_args()

    if cmd_args.gpu != 0:
        torch.cuda.set_device(cmd_args.gpu)

    if cmd_args.model_type == ModelTypes.COMPRESSION:
        args = mse_lpips_args
    elif cmd_args.model_type == ModelTypes.COMPRESSION_GAN:
        args = hific_args

    start_time = time.time()
    device = helpers.get_device()

    # Override default arguments from config file with provided command line arguments
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d, cmd_args_d = dictify(args), vars(cmd_args)
    args_d.update(cmd_args_d)
    args = helpers.Struct(**args_d)
    args = helpers.setup_generic_signature(args, special_info=args.model_type)
    args.target_rate = args.target_rate_map[args.regime]
    args.lambda_A = args.lambda_A_map[args.regime]  # High rate

    logger = helpers.logger_setup(logpath=os.path.join(args.snapshot, 'logs'), filepath=os.path.abspath(__file__))
    logger.info('MODEL TYPE: {}'.format(args.model_type))
    logger.info('MODEL MODE: {}'.format(args.model_mode))
    logger.info('BITRATE REGIME: {}'.format(args.regime))
    logger.info('SAVING LOGS/CHECKPOINTS/RECORDS TO {}'.format(args.snapshot))
    logger.info('USING DEVICE {}'.format(device))
    logger.info('USING GPU ID {}'.format(args.gpu))
    logger.info('USING DATASET: {}'.format(args.dataset))

    test_loader = datasets.get_dataloaders(args.dataset,
                                root=args.dataset_path,
                                batch_size=args.batch_size,
                                logger=logger,
                                mode='validation',
                                shuffle=True)

    train_loader = datasets.get_dataloaders(args.dataset,
                                root=args.dataset_path,
                                batch_size=args.batch_size,
                                logger=logger,
                                mode='train',
                                shuffle=True)


    args.n_data = len(train_loader.dataset)
    args.image_dims = train_loader.dataset.image_dims
    logger.info('Input Dimensions: {}'.format(args.image_dims))

    storage = defaultdict(list)
    storage_test = defaultdict(list)
    model = create_model(args, device, logger, storage, storage_test)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and args.multigpu is True:
        logger.info('Using {} GPUs.'.format(n_gpus))
        model = nn.DataParallel(model)

    model = model.to(device)
    logger.info('Using device {}'.format(device))

    metadata = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    logger.info(metadata)

    """
    Train
    """
    model, ckpt_path = train(args, model, train_loader, test_loader, device, storage, storage_test, logger)

    """
    TODO
    Generate metrics
    """
