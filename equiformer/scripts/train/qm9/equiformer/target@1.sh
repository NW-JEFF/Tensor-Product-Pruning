#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
source activate th102_cu113_tgconda

python main_qm9.py \
    --output-dir 'models/qm9/equiformer/se_l2/target@1/' \
    --model-name 'graph_attention_transformer_nonlinear_l2' \
    --input-irreps '5x0e' \
    --target 1 \
    --data-path 'datasets/qm9' \
    --feature-type 'one_hot' \
    --batch-size 128 \
    --radius 5.0 \
    --num-basis 128 \
    --drop-path 0.0 \
    --lr 5e-4 \
    --min-lr 1e-6 \
    --no-model-ema \
    --no-amp \
    --warmup-epochs 1 \
    --epochs 50 \
    --log \
    --weight-decay 0 \
    --l1_weight 5e-3
