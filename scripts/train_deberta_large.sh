#!/bin/bash
# Train DeBERTa-v3-large (requires ~16GB GPU memory)

# Check GPU memory
python3 -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB') if torch.cuda.is_available() else print('No GPU available')" 2>/dev/null

python3 train_deberta_improved.py \
  --learning_rate 2e-5 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_epochs 6 \
  --warmup_ratio 0.15 \
  --weight_decay 0.01 \
  --llrd_alpha 0.9 \
  --patience 3 \
  --max_length 512

cp prediction_deberta_improved_lr2e-05 prediction
zip -q submission_deberta_large.zip prediction
echo "Submission created: submission_deberta_large.zip"
