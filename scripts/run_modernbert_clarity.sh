#!/bin/bash
# Subtask 1: ModernBERT-large for 3-Class Clarity Classification
# Requires transformers >= 4.48.0

# Check transformers version
TRANSFORMERS_VERSION=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null)
REQUIRED_VERSION="4.48.0"

if [ -n "$TRANSFORMERS_VERSION" ]; then
    if ! python3 -c "from packaging import version; exit(0 if version.parse('$TRANSFORMERS_VERSION') >= version.parse('$REQUIRED_VERSION') else 1)" 2>/dev/null; then
        echo "WARNING: ModernBERT requires transformers >= $REQUIRED_VERSION (current: $TRANSFORMERS_VERSION)"
        echo "Upgrade with: pip install --upgrade transformers"
        read -p "Continue anyway? (y/N) " response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

python3 training/train_modernbert_clarity.py \
    --model_size large \
    --learning_rate 2e-5 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_epochs 8 \
    --warmup_ratio 0.1 \
    --focal_gamma 2.0 \
    --patience 3 \
    --use_gpt_features \
    "$@"

echo "Done. Submit submission_modernbert_large.zip to Codabench."
