#!/bin/bash
# Hierarchical Evasion Classification Pipeline (Subtask 2)
#
# Runs Stage 2 classifiers (Ambivalent + NonReply) on top of Stage 1.
#
# Usage:
#   ./run_hierarchical_evasion.sh           # Full pipeline (train + predict)
#   ./run_hierarchical_evasion.sh --predict # Prediction only (skip training)

set -e

# Check for Stage 1 model
if [ ! -f "deberta_focal_features_best.pt" ]; then
    echo "ERROR: Stage 1 model not found: deberta_focal_features_best.pt"
    exit 1
fi

if [ "$1" == "--predict" ]; then
    # Prediction-only mode
    if [ ! -f "stage2_ambivalent_best.pt" ] || [ ! -f "stage2_nonreply_best.pt" ]; then
        echo "ERROR: Stage 2 models not found. Train first: python3 training/train_hierarchical_stage2.py"
        exit 1
    fi
    python3 evaluation/predict_hierarchical_evasion.py
else
    # Full pipeline: train + predict
    python3 training/train_hierarchical_stage2.py
    python3 evaluation/predict_hierarchical_evasion.py
fi

echo "Done. Submit submission_hierarchical.zip to Codabench (Subtask 2)."
