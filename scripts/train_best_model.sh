#!/bin/bash
# Train DeBERTa-v3-base with best configuration

# Check GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Train
python3 training/train_deberta_improved.py \
  --learning_rate 3e-5 \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_epochs 6 \
  --warmup_ratio 0.15 \
  --weight_decay 0.01 \
  --llrd_alpha 0.9 \
  --patience 3 \
  --max_length 512 \
  --reinit_layers 0

if [ $? -eq 0 ]; then
    if [ -f "prediction_deberta_improved_lr3e-05" ]; then
        echo "Predictions: $(wc -l < prediction_deberta_improved_lr3e-05) samples"
    fi
    cp prediction_deberta_improved_lr3e-05 prediction
    zip -q submission_deberta_improved.zip prediction
    echo "Submission created: submission_deberta_improved.zip"
else
    echo "ERROR: Training failed."
    echo "If OOM, try: python3 training/train_deberta_improved.py --batch_size 4 --gradient_accumulation_steps 8 --learning_rate 3e-5"
    exit 1
fi
