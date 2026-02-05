#!/bin/bash
# Subtask 2: DeBERTa-v3-LARGE for 9-Class Evasion Prediction

python3 training/train_deberta_large_evasion.py \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_epochs 6 \
    --warmup_ratio 0.15 \
    --weight_decay 0.01 \
    --llrd_alpha 0.9 \
    --max_length 512 \
    --focal_gamma 2.5 \
    --patience 3

echo "Done. Submit submission_evasion.zip to Codabench (Subtask 2)."
