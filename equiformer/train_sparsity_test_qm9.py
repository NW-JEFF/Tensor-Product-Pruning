import datetime
import itertools
import pickle
import subprocess
import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader

import os
from logger import FileLogger
import wandb
from pathlib import Path

from datasets.pyg.qm9 import QM9
#from torch_geometric.datasets import QM9
#from torch_geometric.nn import SchNet

# AMP
from contextlib import suppress
from timm.utils import NativeScaler

import nets
from nets import model_entrypoint

from timm.utils import ModelEmaV2
from timm.scheduler import create_scheduler
from optim_factory import create_optimizer

from engine import train_one_epoch, evaluate, compute_stats

# distributed training
import utils

ModelEma = ModelEmaV2



def main(args):

    utils.init_distributed_mode(args)
    is_main_process = (args.rank == 0)

    _log = FileLogger(is_master=is_main_process, is_rank0=is_main_process, output_dir=args.output_dir)
    _log.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ''' Dataset '''
    train_dataset = QM9(args.data_path, 'train', feature_type=args.feature_type)
    val_dataset   = QM9(args.data_path, 'valid', feature_type=args.feature_type)
    test_dataset  = QM9(args.data_path, 'test', feature_type=args.feature_type)
    _log.info('Training set mean: {}, std:{}'.format(
        train_dataset.mean(args.target), train_dataset.std(args.target)))
    # calculate dataset stats
    task_mean, task_std = 0, 1
    if args.standardize:
        task_mean, task_std = train_dataset.mean(args.target), train_dataset.std(args.target)
    norm_factor = [task_mean, task_std]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ''' Network '''
    create_model = model_entrypoint(args.model_name)
    model = create_model(irreps_in=args.input_irreps, 
        radius=args.radius, num_basis=args.num_basis, 
        out_channels=args.output_channels, 
        task_mean=task_mean, 
        task_std=task_std, 
        atomref=None, #train_dataset.atomref(args.target),
        drop_path=args.drop_path)
    _log.info(model)
    model = model.to(device)
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)

    # distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info('Number of params: {}'.format(n_parameters))

    ''' Optimizer and LR Scheduler '''
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = None #torch.nn.MSELoss() #torch.nn.L1Loss() # torch.nn.MSELoss() 
    if args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'l2':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError

    ''' AMP (from timm) '''
    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
    
    ''' Data Loader '''
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
                train_dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True
            )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
            sampler=sampler_train, num_workers=args.workers, pin_memory=args.pin_mem, 
            drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
            shuffle=True, num_workers=args.workers, pin_memory=args.pin_mem, 
            drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    ''' Compute stats '''
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return
    
    seq_num = str(args.model_seq)

    # initialize wandb
    wandb.init(project="Equiformer_sparsity_" + f"target{args.target}", name=seq_num, config=args)

    # prune parameters according to a specified threshold
    thresholds = np.linspace(0, 0.1, 50)
    test_MAEs = []
    sparsities = []
    for t in thresholds:
        # (!) Check file path
        trained_paras = torch.load('/mnt/workspace/linchen/nanxiang/my_segnn/equiformer/saved_models/equiformer_'+seq_num+'.pt')

        # In case the model is obtained from torch.nn.utils.prune, this ensures modules have the right parameters
        modified_trained_paras = trained_paras.copy()
        for k, v in trained_paras.items():
            if k.endswith(".tp.weight_orig"):
                new_k = k.replace(".tp.weight_orig", ".tp.weight")
                mask_key = k.replace(".tp.weight_orig", ".tp.weight_mask")
                mask = trained_paras[mask_key].detach().cpu()
                zero_mask = torch.nonzero(mask==0)
                v[zero_mask] = 0
                modified_trained_paras[new_k] = v
        trained_paras = modified_trained_paras
        keys_to_remove = [k for k in modified_trained_paras if k.endswith(".tp.weight_orig") or k.endswith(".tp.weight_mask")]
        for k in keys_to_remove:
            del trained_paras[k]

        total_para = 0
        non_zero_para = 0
        for k, v in trained_paras.items():
            if "tp.weight" in k:
                v[v.abs() < t] = 0
                total_para += v.numel()
                non_zero_para += len(v.nonzero())

        # compute sparsity
        sparsity = (total_para - non_zero_para) / total_para
        sparsities.append(sparsity)

        # reload model using pruned parameters
        model.load_state_dict(trained_paras)

        # Final evaluation on test set
        test_err, test_loss = evaluate(model, norm_factor, args.target, test_loader, device, 
            amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log)
        test_MAEs.append(test_err)

        wandb.log({"test MAE": test_err, "Sparsity": sparsity})
        
    print(test_MAEs)
    print(sparsities)

    wandb.finish()