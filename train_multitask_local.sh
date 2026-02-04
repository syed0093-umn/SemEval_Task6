#!/bin/bash
# Multi-Task DeBERTa training with GPT-3.5 features + evasion labels + metadata

# Check GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

python3 train_multitask_deberta.py \
  --learning_rate 2e-5 \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_epochs 8 \
  --warmup_ratio 0.15 \
  --patience 4 \
  --use_gpt_features \
  --use_metadata \
  --multitask

if [ $? -eq 0 ]; then
    if [ -f "prediction_multitask" ]; then
        echo "Predictions: $(wc -l < prediction_multitask) samples"
    fi
    echo "Submission: submission_multitask.zip"
else
    echo "ERROR: Training failed."
    exit 1
fi
