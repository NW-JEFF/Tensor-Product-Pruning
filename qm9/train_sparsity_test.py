"""Examine sparsity vs test MAE after L1 pruning on a trained model. Set parameters below thresholds to 0 and evaluate test MAEs."""

import wandb
import torch
import torch.nn as nn
import torch.distributed as dist

from qm9.dataset import QM9
from qm9.evaluate import evaluate
import utils
import numpy as np


def train_sparsity_test(gpu, model, args):
    if args.gpus == 0:
        device = 'cpu'
    else:
        device = 'cuda:' + str(gpu)
        if args.gpus > 1:
            dist.init_process_group("nccl", rank=gpu, world_size=args.gpus)
            torch.cuda.set_device(gpu)

    model = model.to(device)
    if args.gpus > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)

    # Create datasets and dataloaders
    train_loader = utils.make_dataloader(QM9(args.root, args.target, args.radius, "train", args.lmax_attr,
                                             feature_type=args.feature_type), args.batch_size, args.num_workers, args.gpus, gpu)
    test_loader = utils.make_dataloader(QM9(args.root, args.target, args.radius, "test", args.lmax_attr,
                                            feature_type=args.feature_type), args.batch_size, args.num_workers, args.gpus, gpu, train=False)

    # Get train set statistics
    target_mean, target_mad = train_loader.dataset.calc_stats()

    # Set up optimizer and loss function
    criterion = nn.L1Loss()

    wandb.init(project="Sparsity-Test-" + args.dataset + "-" + args.target, name=args.ID, config=args)

    # trained model's sequence number
    seq_num = str(args.model_seq)

    # prune parameters according to a specified threshold
    thresholds = np.linspace(0, 0.1, 50)
    test_MAEs = []
    sparsities = []
    for t in thresholds:
        # (!) Check file path
        trained_paras = torch.load('/mnt/workspace/linchen/nanxiang/my_segnn/saved models/segnn_qm9_alpha_'+seq_num+'_cuda:0.pt')

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
        test_MAE = evaluate(model, test_loader, criterion, device, args.gpus, target_mean, target_mad)
        test_MAEs.append(test_MAE)

        wandb.log({args.target + " test MAE": test_MAE, "Sparsity": sparsity})
        
    print(test_MAEs)
    print(sparsities)

    if gpu == 0:
        wandb.finish()


