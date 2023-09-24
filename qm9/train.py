import wandb
import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import prune

from qm9.dataset import QM9
from qm9.evaluate import evaluate
import utils


def train(gpu, model, args):
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

    # save the initialization parameters
    init_paras = model.state_dict()
    torch.save(init_paras, os.path.join(args.save_dir, args.ID + "_init.pt"))

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

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(0.8*(args.epochs)), int(0.9*(args.epochs))], verbose=True)
    criterion = nn.L1Loss()

    # Logging parameters
    target = args.target
    best_valid_MAE = 1e30
    i = 0
    N_samples = 0
    loss_sum = 0
    train_MAE_sum = 0


    # Parameters for iterative pruning
    remain_rate = 0.2  # target proportion of remaining weights
    prune_interval = 17  # number of epochs to wait until next pruning
    max_num_prune = 4  # max number of prunings allowed
    num_prune = min(args.epochs // prune_interval, max_num_prune)  # number of prunings that will occur
    prune_ratio = 1 - remain_rate ** (1/num_prune)  # proportion of weights to prune each time


    # (!) Random pruning at initialization
    # message_parameters_to_prune = [(segnn_layer.message_layer_1.tp, "weight") for segnn_layer in model.layers]
    # message_parameters_to_prune += [(segnn_layer.message_layer_2.tp, "weight") for segnn_layer in model.layers]
    # update_parameters_to_prune = [(segnn_layer.update_layer_1.tp, "weight") for segnn_layer in model.layers]
    # update_parameters_to_prune += [(segnn_layer.update_layer_2.tp, "weight") for segnn_layer in model.layers]
    # other_layers = [model.embedding_layer, model.pre_pool1, model.pre_pool2,
    #                 model.post_pool1, model.post_pool2]
    # other_parameters_to_prune = [(layer.tp, "weight") for layer in other_layers]
    # parameters_to_prune = message_parameters_to_prune + update_parameters_to_prune + other_parameters_to_prune
    # parameters_to_prune = tuple(parameters_to_prune)
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     pruning_method = prune.RandomUnstructured,
    #     amount = 1 - remain_rate,
    # )


    # Init wandb
    if args.log and gpu == 0:
        wandb.init(project="SEGNN " + args.dataset + " " + args.target, name=args.ID, config=args)

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


            # (!) add extra l1 loss
            if args.l1_weight:
                sparsity_loss = 0
                for n,m in model.named_parameters():
                    if 'tp.weight' in n:
                        sparsity_loss += m.abs().mean()
                loss += sparsity_loss * args.l1_weight
            

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


        # (!) Iterative Magnitude Pruning for each 10 epochs, for at most num_prune times
        if (epoch+1) % prune_interval == 0 and (epoch+1)//prune_interval <= num_prune:
            message_parameters_to_prune = [(segnn_layer.message_layer_1.tp, "weight") for segnn_layer in model.layers]
            message_parameters_to_prune += [(segnn_layer.message_layer_2.tp, "weight") for segnn_layer in model.layers]
            update_parameters_to_prune = [(segnn_layer.update_layer_1.tp, "weight") for segnn_layer in model.layers]
            update_parameters_to_prune += [(segnn_layer.update_layer_2.tp, "weight") for segnn_layer in model.layers]
            other_layers = [model.embedding_layer, model.pre_pool1, model.pre_pool2,
                            model.post_pool1, model.post_pool2]
            other_parameters_to_prune = [(layer.tp, "weight") for layer in other_layers]
            parameters_to_prune = message_parameters_to_prune + update_parameters_to_prune + other_parameters_to_prune
            parameters_to_prune = tuple(parameters_to_prune)
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method = prune.L1Unstructured,
                amount = prune_ratio,
            )

        
        # Evaluate on validation set
        valid_MAE = evaluate(model, valid_loader, criterion, device, args.gpus, target_mean, target_mad)

        # Save best validation model
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
