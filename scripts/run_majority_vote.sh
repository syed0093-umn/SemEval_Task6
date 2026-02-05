#!/bin/bash
# Majority vote baseline from annotator labels

python3 evaluation/predict_majority_vote.py

echo "Done. Output: submission_majority_vote.zip"
echo "Verify competition rules allow using annotator labels before submitting."
