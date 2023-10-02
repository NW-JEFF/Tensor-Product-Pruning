# Pruning SEGNN and Equiformer

This repo is adapted from the code framework for [SEGNN](https://github.com/RobDHess/Steerable-E3-GNN) and [Equiformer](https://github.com/atomicarchitects/equiformer#training).

The pruning techniques are implemented only on the QM9 dataset. Functionalities, file structures, environments, and installation steps from the original repo are all preserved.


#### SEGNN
To run SEGNN baseline,
```bash
python main.py --dataset=qm9 --epochs=50 --target=alpha --radius=2 --model=segnn --lmax_h=2 --lmax_attr=3 --layers=7 --subspace_type=weightbalanced --norm=instance --batch_size=128 --gpu=1 --weight_decay=1e-8 --pool=avg 
```


#### Equiformer
To run Equiformer baseline, go to the folder `equiformer` and run
```bash
sh scripts/train/qm9/equiformer/target@1.sh
```


#### Extra Functionalities for SEGNN

Specify L1 loss by appending the parser argument `--l1_weight` followed by a number while running `main.py`.

Test MAE vs Sparsity: Plot the test MAE vs Sparsity based on a trained network by pruning its learned weights associated with the fully connected tensor product. Specify the name of your model in `main_sparsity_test.py` and `qm9.train_sparsity_test.py`. To run, remember to append the key parser arguments you used in the training. For example,

```bash
python main_sparsity_test.py --dataset=qm9 --epochs=50 --target=alpha --radius=2 --model=segnn --lmax_h=3 --lmax_attr=4 --layers=7 --subspace_type=weightbalanced --norm=instance --batch_size=128 --gpu=1 --weight_decay=1e-8 --pool=avg --log=True --model_seq 366652
```

For L1 Pruning followed by retraining, run `main_prune_retrain.py`. For example,

```bash
python main_prune_retrain.py --dataset=qm9 --epochs=250 --target=alpha --radius=2 --model=segnn --lmax_h=2 --lmax_attr=3 --layers=7 --subspace_type=weightbalanced --norm=instance --batch_size=128 --gpu=1 --weight_decay=1e-8 --pool=avg --model_seq=884207 --reinitialize=random --prune_threshold=0.009 --log=True
```

Some plotting functions are implemented under the folder `model_performance`.


#### Extra Functionalities for Equiformer

Specify L1 loss by appending parser arguments `--l1_weight` followed by a number to the shell script `scripts/train/qm9/equiformer/target@1.sh`.

Plot MAE vs Sparsity by running `sh scripts/train/qm9/equiformer/target@1_sparsity.sh`. Remember to append the key parser arguments you used in the training to the script.

