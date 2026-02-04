#!/bin/bash
# Hyperparameter search for DeBERTa-v3: tests 5 configurations

mkdir -p search_results

# Config 1: LR=2e-5
echo "[1/5] LR=2e-5..."
python3 train_deberta_improved.py \
  --learning_rate 2e-5 \
  --llrd_alpha 0.9 \
  --warmup_ratio 0.15 \
  --gradient_accumulation_steps 4 \
  --num_epochs 6 \
  --patience 3 \
  > search_results/config1_lr2e5.log 2>&1

# Config 2: LR=3e-5
echo "[2/5] LR=3e-5..."
python3 train_deberta_improved.py \
  --learning_rate 3e-5 \
  --llrd_alpha 0.9 \
  --warmup_ratio 0.15 \
  --gradient_accumulation_steps 4 \
  --num_epochs 6 \
  --patience 3 \
  > search_results/config2_lr3e5.log 2>&1

# Config 3: LR=5e-5
echo "[3/5] LR=5e-5..."
python3 train_deberta_improved.py \
  --learning_rate 5e-5 \
  --llrd_alpha 0.9 \
  --warmup_ratio 0.15 \
  --gradient_accumulation_steps 4 \
  --num_epochs 6 \
  --patience 3 \
  > search_results/config3_lr5e5.log 2>&1

# Config 4: LR=3e-5 with stronger LLRD
echo "[4/5] LR=3e-5, LLRD=0.85..."
python3 train_deberta_improved.py \
  --learning_rate 3e-5 \
  --llrd_alpha 0.85 \
  --warmup_ratio 0.20 \
  --gradient_accumulation_steps 6 \
  --num_epochs 8 \
  --patience 4 \
  > search_results/config4_lr3e5_llrd085.log 2>&1

# Config 5: LR=3e-5 with layer reinit
echo "[5/5] LR=3e-5, reinit=1..."
python3 train_deberta_improved.py \
  --learning_rate 3e-5 \
  --llrd_alpha 0.9 \
  --warmup_ratio 0.15 \
  --gradient_accumulation_steps 4 \
  --num_epochs 6 \
  --patience 3 \
  --reinit_layers 1 \
  > search_results/config5_lr3e5_reinit.log 2>&1

# Extract results
echo ""
echo "Config | LR   | LLRD | Reinit | Dev F1"
echo "-------|------|------|--------|-------"

for i in 1 2 3 4 5; do
    logfile="search_results/config${i}*.log"
    if [ -f $logfile ]; then
        f1=$(grep "DeBERTa-v3-base Improved F1:" $logfile | tail -1 | awk '{print $4}')
        case $i in
            1) echo "  $i    | 2e-5 | 0.9  | No     | $f1" ;;
            2) echo "  $i    | 3e-5 | 0.9  | No     | $f1" ;;
            3) echo "  $i    | 5e-5 | 0.9  | No     | $f1" ;;
            4) echo "  $i    | 3e-5 | 0.85 | No     | $f1" ;;
            5) echo "  $i    | 3e-5 | 0.9  | Yes    | $f1" ;;
        esac
    fi
done

# Find best configuration and create submission
best_f1=0
best_config=""
best_pred=""

for pred_file in prediction_deberta_improved_*; do
    if [ -f "$pred_file" ]; then
        lognum=$(echo $pred_file | grep -o "config[0-9]" | grep -o "[0-9]")
        if [ -n "$lognum" ]; then
            logfile="search_results/config${lognum}*.log"
            f1=$(grep "DeBERTa-v3-base Improved F1:" $logfile 2>/dev/null | tail -1 | awk '{print $4}')
            if (( $(echo "$f1 > $best_f1" | bc -l) )); then
                best_f1=$f1
                best_config="Config $lognum"
                best_pred=$pred_file
            fi
        fi
    fi
done

echo ""
echo "Best: $best_config (F1=$best_f1)"

if [ -n "$best_pred" ]; then
    cp $best_pred prediction
    zip -q submission_deberta_best.zip prediction
    echo "Submission created: submission_deberta_best.zip"
fi

echo "Logs saved in: search_results/"
