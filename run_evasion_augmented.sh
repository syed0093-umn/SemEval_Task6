#!/bin/bash
# Evasion (Subtask 2): Augmented Data + DeBERTa-base Training Pipeline
set -e

# Step 1: Generate augmented data
echo "Generating augmented evasion data..."
python3 data_augmentation/generate_evasion_data.py \
    --method hybrid \
    --output_dir ./QEvasion_evasion_augmented \
    --seed 42

# Step 2: Train DeBERTa-base on augmented data
echo "Training DeBERTa-v3-base on augmented data..."
python3 train_deberta_evasion_augmented.py \
    --data_dir ./QEvasion_evasion_augmented \
    --learning_rate 3e-5 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_epochs 8 \
    --focal_gamma 2.5 \
    --patience 4 \
    --synthetic_weight 0.5

# Step 3: Generate eval predictions
echo "Generating eval predictions..."
python3 predict_eval_evasion_augmented.py

echo "Pipeline complete. Upload submission_eval_evasion_augmented.zip to Codabench (Subtask 2)."
