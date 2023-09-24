"""Hard-code L1 pruning on a trained model and continue training."""

import wandb
import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.nn.utils import prune

from qm9.dataset import QM9
from qm9.evaluate import evaluate
import utils


def train_prune_retrain(gpu, model, args):
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

    torch.manual_seed(0)
    np.random.seed(0)

    # Create datasets and dataloaders
    train_loader = utils.make_dataloader(QM9(args.root, args.target, args.radius, "train", args.lmax_attr,
                                             feature_type=args.feature_type), args.batch_size, args.num_workers, args.gpus, gpu)
    valid_loader = utils.make_dataloader(QM9(args.root, args.target, args.radius, "valid", args.lmax_attr,
                                             feature_type=args.feature_type), args.batch_size, args.num_workers, args.gpus, gpu, train=False)
    test_loader = utils.make_dataloader(QM9(args.root, args.target, args.radius, "test", args.lmax_attr,
                                            feature_type=args.feature_type), args.batch_size, args.num_workers, args.gpus, gpu, train=False)

    # Get train set statistics
    target_mean, target_mad = train_loader.dataset.calc_stats()

    # If reinitialize, keep the setting, else set up smaller stepsizes to continue training
    if args.reinitialize:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=[int(0.8*(args.epochs)), int(0.9*(args.epochs))], verbose=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*1.1, verbose=True)
    criterion = nn.L1Loss()

    # Logging parameters
    target = args.target
    best_valid_MAE = 1e30
    i = 0
    N_samples = 0
    loss_sum = 0
    train_MAE_sum = 0

    # Load trained parameters
    seq_num = str(args.model_seq)
    trained_paras = torch.load('/mnt/workspace/linchen/nanxiang/my_segnn/saved models/segnn_qm9_alpha_'+seq_num+'_cuda:0.pt')

    assert args.reinitialize in ["random", "reuse"]
    if args.reinitialize == "random":
        # directly take the current random reinitialization
        init_paras = model.state_dict()
    elif args.reinitialize == "reuse":
        # load the initialization parameters of the corresponding run
        init_paras = torch.load('/mnt/workspace/linchen/nanxiang/my_segnn/saved models/segnn_qm9_alpha_'+seq_num+'_init.pt')

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

    # Prune parameters with magnitude less than the specified threshold
    if args.prune_threshold:
        for k, v in trained_paras.items():
            if "tp.weight" in k:
                v[v.abs() < args.prune_threshold] = 0

    if args.reinitialize:
        model.load_state_dict(init_paras)
    else:
        model.load_state_dict(trained_paras)

    # create a mask according to zero entries in trained_paras
    message_layers = [segnn_layer.message_layer_1.tp for segnn_layer in model.layers]
    message_layers += [segnn_layer.message_layer_2.tp for segnn_layer in model.layers]
    update_layers = [segnn_layer.update_layer_1.tp for segnn_layer in model.layers]
    update_layers += [segnn_layer.update_layer_2.tp for segnn_layer in model.layers]
    other_layers = [model.embedding_layer.tp, model.pre_pool1.tp, model.pre_pool2.tp,
                    model.post_pool1.tp, model.post_pool2.tp]
    layers_to_prune = message_layers + update_layers + other_layers

    message_keys = [f"layers.{i}.message_layer_{j}.tp.weight" for j in range(1,3) for i in range(7)]
    update_keys = [f"layers.{i}.update_layer_{j}.tp.weight" for j in range(1,3) for i in range(7)]
    other_keys = ["embedding_layer.tp.weight", "pre_pool1.tp.weight", "pre_pool2.tp.weight", 
                  "post_pool1.tp.weight", "post_pool2.tp.weight"]
    keys = message_keys + update_keys + other_keys

    for idx, layer in enumerate(layers_to_prune):
        key = keys[idx]
        value = trained_paras[key]
        prune_mask = (value!=0).float()
        prune.custom_from_mask(layer, name="weight", mask=prune_mask)


    # Init wandb
    if args.log and gpu == 0:
        wandb.init(project="L1-Prune-Retrain-" + args.dataset + "-" + args.target, name=args.ID, config=args)

    # Let's start!
    if gpu == 0:
        print("Training:", args.ID)
    for epoch in range(args.epochs):
        # Set epoch so shuffling works right in distributed mode.
        if args.gpus > 1:
            train_loader.sampler.set_epoch(epoch)
        # Training loop

        for step, graph in enumerate(train_loader):
            # Forward & Backward.
            graph = graph.to(device)
            out = model(graph).squeeze()
            
            loss = criterion(out, (graph.y - target_mean)/target_mad)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            i += 1
            N_samples += graph.y.size(0)
            loss_sum += loss
            train_MAE_sum += criterion(out.detach()*target_mad + target_mean, graph.y)*graph.y.size(0)

            # Report
            if i % args.print == 0:
                print("epoch:%2d  step:%4d  loss: %0.4f  train MAE:%0.4f" %
                      (epoch, step, loss_sum/i, train_MAE_sum/N_samples))

                if args.log and gpu == 0:
                    wandb.log({"loss": loss_sum/i, target + " train MAE": train_MAE_sum /
                               N_samples})

                i = 0
                N_samples = 0
                loss_sum = 0
                train_MAE_sum = 0

        # Evaluate on validation set
        valid_MAE = evaluate(model, valid_loader, criterion, device, args.gpus, target_mean, target_mad)

        # Save best validation model; ID is indicated with "retrain"
        if valid_MAE < best_valid_MAE:
            best_valid_MAE = valid_MAE
            utils.save_model(model, args.save_dir, args.ID, device)
        if gpu == 0:
            print("VALIDATION: epoch:%2d  step:%4d  %s-MAE:%0.4f" %
                  (epoch, step, target, valid_MAE))
            if args.log:
                wandb.log({target + " val MAE": valid_MAE})

        # Adapt learning rate
        scheduler.step()

    # Final evaluation on test set
    model = utils.load_model(model, args.save_dir, args.ID, device)
    test_MAE = evaluate(model, test_loader, criterion, device, args.gpus, target_mean, target_mad, )
    if gpu == 0:
        print("TEST: epoch:%2d  step:%4d  %s-MAE:%0.4f" %
              (epoch, step, target, test_MAE))
        if args.log:
            wandb.log({target + " test MAE": test_MAE})
            wandb.save(os.path.join(args.save_dir, args.ID + "_" + device + ".pt"))

    if args.log and gpu == 0:
        wandb.finish()
    if args.gpus > 1:
        dist.destroy_process_group()
