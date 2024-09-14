#!/bin/bash

datasets=("Mpro" "USP7" "CATS" "HSP90" "HIF2A" "MCL1" "SYK")
embedding_types=("ChemBERTa" "ECFP8")
test_sizes=(0.9 0.8 0.7)
random_seeds=(1234 123 12)
lr=0.1
epochs=50


for dataset in "${datasets[@]}"; do
    for embedding in "${embedding_types[@]}"; do
        for test_size in "${test_sizes[@]}"; do
            for seed in "${random_seeds[@]}"; do
                echo "Training GP on $dataset with $embedding embeddings, test_size=$test_size, random_seed=$seed"
                python scripts/train_gp.py --dataset=$dataset --embedding_type=$embedding --test_size=$test_size --random_seed=$seed --lr=$lr --epochs=$epochs
            done
        done
    done
done