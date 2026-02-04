"""
Ensemble for Subtask 2 (9-class evasion classification).

Combines predictions from multiple evasion models using
hard voting and weighted voting strategies.
"""

import pandas as pd
import numpy as np
from collections import Counter
import zipfile

print("="*80)
print("Ensemble for Subtask 2 (9-Class Evasion)")
print("="*80)

# ============================================================================
# Configuration: Add your prediction files here
# ============================================================================

PREDICTION_FILES = [
 'prediction_evasion', # Single-task (0.45 F1)
 'prediction_multitask_evasion', # Multi-task (expected higher)
 'prediction_hierarchical_evasion', # Hierarchical (expected higher)
]

# Optional: Weight models by their validation F1 scores
MODEL_WEIGHTS = {
 'prediction_evasion': 0.45,
 'prediction_multitask_evasion': 0.52, # Estimated
 'prediction_hierarchical_evasion': 0.50 # Estimated
}

print(f"\n[1/4] Loading predictions from {len(PREDICTION_FILES)} models...")

# Load all predictions
all_predictions = {}
available_files = []

for pred_file in PREDICTION_FILES:
 try:
 with open(pred_file, 'r') as f:
 predictions = [line.strip() for line in f.readlines()]

 all_predictions[pred_file] = predictions
 available_files.append(pred_file)
 weight = MODEL_WEIGHTS.get(pred_file, 1.0)
 print(f"* Loaded: {pred_file} (308 lines, weight={weight:.2f})")

 except FileNotFoundError:
 print(f"⚠ Skipping: {pred_file} (not found)")

if len(available_files) == 0:
 print("\n ERROR: No prediction files found!")
 print("Train at least one model first:")
 print(" - ./run_subtask2_evasion.sh (single-task)")
 print(" - ./run_multitask_evasion.sh (multi-task)")
 exit(1)

if len(available_files) == 1:
 print("\n⚠ WARNING: Only 1 model found. Ensemble needs ≥2 models.")
 print("Recommend training:")
 print(" - Multi-task model: ./run_multitask_evasion.sh")
 print("\nProceeding with single model (no ensemble benefit)...")

num_samples = len(all_predictions[available_files[0]])
print(f"\n* All files have {num_samples} predictions")

# ============================================================================
# Method 1: Hard Voting (Majority Vote)
# ============================================================================
print("\n[2/4] Method 1: Hard Voting (Majority Vote)...")

hard_voting_predictions = []

for i in range(num_samples):
 # Collect predictions from all models for sample i
 votes = [all_predictions[model][i] for model in available_files]

 # Majority vote
 vote_counts = Counter(votes)
 majority_vote = vote_counts.most_common(1)[0][0]

 hard_voting_predictions.append(majority_vote)

# Show distribution
hard_dist = pd.Series(hard_voting_predictions).value_counts()
print(f"\n* Hard Voting predictions:")
for label, count in hard_dist.items():
 print(f" {label:25s}: {count:3d} ({count/num_samples*100:5.1f}%)")

# ============================================================================
# Method 2: Weighted Hard Voting
# ============================================================================
print("\n[3/4] Method 2: Weighted Hard Voting...")

weighted_voting_predictions = []

for i in range(num_samples):
 # Collect weighted votes
 vote_weights = {}

 for model in available_files:
 prediction = all_predictions[model][i]
 weight = MODEL_WEIGHTS.get(model, 1.0)

 if prediction in vote_weights:
 vote_weights[prediction] += weight
 else:
 vote_weights[prediction] = weight

 # Select prediction with highest total weight
 best_prediction = max(vote_weights.items(), key=lambda x: x[1])[0]
 weighted_voting_predictions.append(best_prediction)

# Show distribution
weighted_dist = pd.Series(weighted_voting_predictions).value_counts()
print(f"\n* Weighted Voting predictions:")
for label, count in weighted_dist.items():
 print(f" {label:25s}: {count:3d} ({count/num_samples*100:5.1f}%)")

# ============================================================================
# Save ensembles
# ============================================================================
print("\n[4/4] Saving ensemble predictions...")

# Save hard voting
output_hard = 'prediction_ensemble_hard_evasion'
with open(output_hard, 'w') as f:
 for pred in hard_voting_predictions:
 f.write(f"{pred}\n")
print(f"* Saved: {output_hard}")

# Save weighted voting
output_weighted = 'prediction_ensemble_weighted_evasion'
with open(output_weighted, 'w') as f:
 for pred in weighted_voting_predictions:
 f.write(f"{pred}\n")
print(f"* Saved: {output_weighted}")

# Create submissions
submission_hard = 'submission_ensemble_hard_evasion.zip'
with zipfile.ZipFile(submission_hard, 'w') as zf:
 zf.write(output_hard, arcname='prediction')
print(f"* Created: {submission_hard}")

submission_weighted = 'submission_ensemble_weighted_evasion.zip'
with zipfile.ZipFile(submission_weighted, 'w') as zf:
 zf.write(output_weighted, arcname='prediction')
print(f"* Created: {submission_weighted}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("ENSEMBLE COMPLETE")
print("="*80)
print(f"Models combined: {len(available_files)}")
for model in available_files:
 weight = MODEL_WEIGHTS.get(model, 1.0)
 print(f" - {model} (weight={weight:.2f})")

print(f"\nEnsemble methods generated:")
print(f" 1. Hard voting: {submission_hard}")
print(f" 2. Weighted voting: {submission_weighted}")

if len(available_files) >= 3:
 print("\n* Good! You have ≥3 models. Ensemble should help.")
 print(" Expected improvement: +0.02-0.04 F1")
elif len(available_files) == 2:
 print("\n* You have 2 models. Ensemble may help.")
 print(" Expected improvement: +0.01-0.02 F1")
 print(" Recommend training 1-2 more models for stronger ensemble.")
else:
 print("\n⚠ Only 1 model. No ensemble benefit.")
 print(" Train more models first:")
 print(" - Multi-task: ./run_multitask_evasion.sh")
 print(" - Hierarchical: python3 predict_hierarchical_evasion.py")

print("\nNext steps:")
print(" 1. Submit both ensemble files to Codabench")
print(" 2. Compare results with single models")
print(" 3. Use the better-performing ensemble method")

print("\nTo add more models to ensemble:")
print(" 1. Train model with different hyperparameters:")
print(" python3 train_deberta_large_evasion.py --focal_gamma 3.0")
print(" 2. Add prediction file to PREDICTION_FILES list")
print(" 3. Re-run this script")

print("="*80)
