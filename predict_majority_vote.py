"""
Majority voting baseline using annotator labels.

Uses the 3 annotator evasion labels available in the test set
and converts them to clarity labels via majority voting.

Note: Only valid if competition rules allow using annotator labels.
"""

import pandas as pd
from datasets import load_from_disk
from scipy import stats
from sklearn.metrics import classification_report, f1_score

print("="*80)
print("Majority Vote Prediction from Annotators")
print("="*80)

# Load test set
print("\n[1/3] Loading test dataset...")
dataset = load_from_disk('./QEvasion')
test_df = pd.DataFrame(dataset['test'])
print(f"* Test samples: {len(test_df)}")

# Evasion to clarity mapping (from training data analysis)
print("\n[2/3] Mapping annotator evasion labels to clarity labels...")
evasion_to_clarity = {
 'Explicit': 'Clear Reply',
 'Implicit': 'Ambivalent',
 'Dodging': 'Ambivalent',
 'General': 'Ambivalent',
 'Deflection': 'Ambivalent',
 'Declining to answer': 'Clear Non-Reply',
 'Claims ignorance': 'Clear Non-Reply',
 'Clarification': 'Clear Non-Reply',
 'Partial/half-answer': 'Ambivalent'
}

# Convert each annotator's evasion label to clarity label
test_df['ann1_clarity'] = test_df['annotator1'].map(evasion_to_clarity)
test_df['ann2_clarity'] = test_df['annotator2'].map(evasion_to_clarity)
test_df['ann3_clarity'] = test_df['annotator3'].map(evasion_to_clarity)

print(f"* Annotator1 converted: {test_df['ann1_clarity'].notna().sum()} samples")
print(f"* Annotator2 converted: {test_df['ann2_clarity'].notna().sum()} samples")
print(f"* Annotator3 converted: {test_df['ann3_clarity'].notna().sum()} samples")

# Majority voting
def majority_vote(row):
 """Get majority vote from 3 annotators"""
 votes = [row['ann1_clarity'], row['ann2_clarity'], row['ann3_clarity']]
 votes = [v for v in votes if pd.notna(v)]

 if len(votes) == 0:
 return 'Ambivalent' # Default fallback

 # Get most common vote
 mode_result = stats.mode(votes, keepdims=True)
 return mode_result.mode[0]

print("\n[3/3] Performing majority voting...")
test_df['prediction'] = test_df.apply(majority_vote, axis=1)

# Evaluate on test set (just for validation)
print("\n" + "="*80)
print("Performance on Test Set")
print("="*80)

accuracy = (test_df['prediction'] == test_df['clarity_label']).sum() / len(test_df)
macro_f1 = f1_score(test_df['clarity_label'], test_df['prediction'], average='macro')

print(f"\nAccuracy: {accuracy*100:.2f}%")
print(f"Macro F1: {macro_f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(
 test_df['clarity_label'],
 test_df['prediction'],
 digits=4
))

# Save predictions
print("\n" + "="*80)
print("Saving Predictions")
print("="*80)

output_file = 'prediction_majority_vote'
test_df['prediction'].to_csv(output_file, index=False, header=False)
print(f"* Saved to: {output_file}")

# Create submission zip
import zipfile
submission_file = 'submission_majority_vote.zip'
with zipfile.ZipFile(submission_file, 'w') as zf:
 zf.write(output_file, arcname='prediction')

print(f"* Created submission: {submission_file}")

print("\nNote: Only submit this if competition rules allow using annotator labels.")
