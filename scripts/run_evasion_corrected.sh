#!/bin/bash
# Subtask 2: 9-Class Evasion with corrected label mapping
#   Partial/half-answer -> Partial
#   Declining to answer -> Declining
#   Claims ignorance    -> Ignorance

python3 training/train_evasion_corrected.py "$@"

echo "Done. Submit submission_evasion_corrected.zip to Codabench."
