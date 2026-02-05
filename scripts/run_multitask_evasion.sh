#!/bin/bash
# Multi-Task DeBERTa-LARGE: Subtask 2 (9-class) with Subtask 1 (3-class) auxiliary

python3 training/train_multitask_large_evasion.py

echo "Done. Submit submission_multitask_evasion.zip to Codabench (Subtask 2)."
