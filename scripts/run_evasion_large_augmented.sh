#!/bin/bash
# Evasion Subtask 2: DeBERTa-LARGE + Augmented Data
set -e

# Step 1: Generate augmented data (skip if already exists)
if [ -d "./QEvasion_evasion_augmented" ]; then
    echo "Augmented data already exists, skipping generation."
else
    echo "Generating augmented evasion data..."
    python3 data_augmentation/generate_evasion_data.py \
        --method hybrid \
        --output_dir ./QEvasion_evasion_augmented \
        --seed 42
fi

# Step 2: Train DeBERTa-large on augmented data + predict eval
echo "Training DeBERTa-v3-large on augmented data..."
python3 training/train_deberta_large_evasion_augmented.py \
    --data_dir ./QEvasion_evasion_augmented \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_epochs 10 \
    --focal_gamma 2.5 \
    --patience 4 \
    --synthetic_weight 0.5

echo "Done. Submit submission_eval_evasion_large_augmented.zip to Codabench (Subtask 2)."
