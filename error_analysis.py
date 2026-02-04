"""
Error Analysis for BERT Model
Understanding what works and what doesn't
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

print("="*80)
print("ERROR ANALYSIS: BERT Model on Political Evasion Detection")
print("="*80)

# Load dataset
print("\n[1/7] Loading dataset and model...")
dataset = load_from_disk('./QEvasion')
test_df = pd.DataFrame(dataset['test'])

# Load BERT model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('./bert_base_uncased_model')
model = AutoModelForSequenceClassification.from_pretrained('./bert_base_uncased_model')
model.to(device)
model.eval()

label_list = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

print(f"* Model loaded: {model.config.name_or_path}")
print(f"* Test samples: {len(test_df)}")

# Prepare text
def prepare_text_transformer(row):
 question = row['question']
 answer = row['interview_answer']
 return f"Question: {question} [SEP] Answer: {answer}"

test_df['text'] = test_df.apply(prepare_text_transformer, axis=1)
test_df['label_id'] = test_df['clarity_label'].map(label2id)

# Get predictions
print("\n[2/7] Getting model predictions...")
predictions = []
true_labels = []
prediction_probs = []

for idx, row in test_df.iterrows():
 text = row['text']
 true_label = row['label_id']

 encoding = tokenizer(
 text,
 max_length=512,
 padding='max_length',
 truncation=True,
 return_tensors='pt'
 )

 with torch.no_grad():
 input_ids = encoding['input_ids'].to(device)
 attention_mask = encoding['attention_mask'].to(device)
 outputs = model(input_ids=input_ids, attention_mask=attention_mask)

 probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
 pred = np.argmax(probs)

 predictions.append(pred)
 true_labels.append(true_label)
 prediction_probs.append(probs)

test_df['predicted_id'] = predictions
test_df['predicted_label'] = [id2label[p] for p in predictions]
test_df['true_label'] = [id2label[t] for t in true_labels]
test_df['prediction_probs'] = prediction_probs
test_df['confidence'] = [max(probs) for probs in prediction_probs]
test_df['correct'] = test_df['predicted_id'] == test_df['label_id']

print(f"* Predictions complete")
print(f" Overall Accuracy: {test_df['correct'].mean():.4f}")

# ============================================================================
# CONFUSION MATRIX
# ============================================================================
print("\n[3/7] Generating confusion matrix...")

cm = confusion_matrix(true_labels, predictions)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Print confusion matrix
print("\nConfusion Matrix (counts):")
print(" Predicted")
print(" Amb CNR CR")
for i, label in enumerate(['Ambivalent', 'Clear Non-Reply', 'Clear Reply']):
 print(f"{label:17} {cm[i][0]:5} {cm[i][1]:6} {cm[i][2]:5}")

print("\nConfusion Matrix (normalized):")
print(" Predicted")
print(" Amb CNR CR")
for i, label in enumerate(['Ambivalent', 'Clear Non-Reply', 'Clear Reply']):
 print(f"{label:17} {cm_normalized[i][0]:5.2f} {cm_normalized[i][1]:6.2f} {cm_normalized[i][2]:5.2f}")

# Detailed classification report
print("\n" + classification_report(true_labels, predictions, target_names=label_list))

# ============================================================================
# TEXT LENGTH ANALYSIS
# ============================================================================
print("\n[4/7] Text length analysis...")

test_df['question_len'] = test_df['question'].str.len()
test_df['answer_len'] = test_df['interview_answer'].str.len()
test_df['total_len'] = test_df['question_len'] + test_df['answer_len']

# ============================================================================
# LINGUISTIC PATTERN ANALYSIS
# ============================================================================
print("\n[5/7] Linguistic pattern analysis...")

# Evasion indicators
evasion_patterns = {
 "hedging": r"\b(maybe|perhaps|possibly|probably|might|could|would seem)\b",
 "uncertainty": r"\b(don't know|not sure|unclear|uncertain|difficult to say)\b",
 "deflection": r"\b(depends|complicated|complex issue|various factors)\b",
 "yes_but": r"\b(yes,? but|well,?|however,?)\b",
 "question_dodge": r"\b(good question|interesting question|let me)\b"
}

for pattern_name, pattern in evasion_patterns.items():
 test_df[f'has_{pattern_name}'] = test_df['interview_answer'].str.lower().str.contains(pattern, regex=True, na=False)

# ============================================================================
# ERROR BREAKDOWN BY CLASS
# ============================================================================
print("\n[6/7] Error breakdown by class...")

errors_df = test_df[~test_df['correct']].copy()
correct_df = test_df[test_df['correct']].copy()
print(f"\nTotal errors: {len(errors_df)} / {len(test_df)} ({len(errors_df)/len(test_df)*100:.1f}%)")

# Errors by true class
print("\nErrors by True Class:")
for label in label_list:
 class_df = test_df[test_df['true_label'] == label]
 class_errors = errors_df[errors_df['true_label'] == label]
 error_rate = len(class_errors) / len(class_df) * 100 if len(class_df) > 0 else 0
 print(f"\n {label}:")
 print(f" Total: {len(class_df)}, Errors: {len(class_errors)} ({error_rate:.1f}%)")

 if len(class_errors) > 0:
 error_types = class_errors['predicted_label'].value_counts()
 print(f" Confused as:")
 for pred_label, count in error_types.items():
 print(f" → {pred_label}: {count} ({count/len(class_errors)*100:.1f}%)")

# ============================================================================
# CONFIDENCE ANALYSIS
# ============================================================================
print("\n[7/7] Confidence analysis...")

print(f"\nCorrect predictions:")
print(f" Mean confidence: {correct_df['confidence'].mean():.4f}")
print(f" Median confidence: {correct_df['confidence'].median():.4f}")

print(f"\nIncorrect predictions:")
print(f" Mean confidence: {errors_df['confidence'].mean():.4f}")
print(f" Median confidence: {errors_df['confidence'].median():.4f}")

# Low confidence correct predictions
low_conf_threshold = 0.5
low_conf_correct = correct_df[correct_df['confidence'] < low_conf_threshold]
print(f"\nLow-confidence correct predictions (< {low_conf_threshold}):")
print(f" Count: {len(low_conf_correct)} ({len(low_conf_correct)/len(correct_df)*100:.1f}% of correct)")

# High confidence errors
high_conf_threshold = 0.7
high_conf_errors = errors_df[errors_df['confidence'] > high_conf_threshold]
print(f"\nHigh-confidence errors (> {high_conf_threshold}):")
print(f" Count: {len(high_conf_errors)} ({len(high_conf_errors)/len(errors_df)*100:.1f}% of errors)")

print(f"\nAnswer length comparison:")
print(f" Correct - Mean: {correct_df['answer_len'].mean():.0f} chars, Median: {correct_df['answer_len'].median():.0f} chars")
print(f" Errors - Mean: {errors_df['answer_len'].mean():.0f} chars, Median: {errors_df['answer_len'].median():.0f} chars")

print("\nEvasion pattern prevalence in Correct vs Incorrect predictions:")
print("\n{:20} {:>10} {:>10} {:>10}".format("Pattern", "Correct %", "Error %", "Diff"))
print("-" * 55)
for pattern_name in evasion_patterns.keys():
 col = f'has_{pattern_name}'
 correct_pct = correct_df[col].mean() * 100
 error_pct = errors_df[col].mean() * 100
 diff = error_pct - correct_pct
 print("{:20} {:>9.1f}% {:>9.1f}% {:>+9.1f}%".format(pattern_name, correct_pct, error_pct, diff))

# ============================================================================
# EXAMPLE ERRORS
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE ERRORS")
print("="*80)

# Get 2 examples from each error type
error_types = errors_df.groupby(['true_label', 'predicted_label'])

for (true_label, pred_label), group in error_types:
 if len(group) > 0:
 print(f"\n{'='*80}")
 print(f"TRUE: {true_label} → PREDICTED: {pred_label}")
 print(f"Count: {len(group)}")
 print("="*80)

 # Show up to 2 examples
 for idx, row in group.head(2).iterrows():
 print(f"\nExample {idx + 1}:")
 print(f" Confidence: {row['confidence']:.3f}")
 print(f" Question: {row['question'][:200]}...")
 print(f" Answer: {row['interview_answer'][:300]}...")
 print()

# ============================================================================
# SAVE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SAVING ANALYSIS RESULTS")
print("="*80)

# Save full analysis to CSV
analysis_df = test_df[[
 'question', 'interview_answer', 'true_label', 'predicted_label',
 'correct', 'confidence', 'answer_len'
]].copy()

# Add probability columns
for i, label in enumerate(label_list):
 analysis_df[f'prob_{label}'] = [probs[i] for probs in test_df['prediction_probs']]

analysis_df.to_csv('bert_error_analysis.csv', index=False)
print("* Saved: bert_error_analysis.csv")

# Save error summary
with open('bert_error_summary.txt', 'w') as f:
 f.write("BERT Model Error Analysis Summary\n")
 f.write("="*80 + "\n\n")

 f.write(f"Overall Accuracy: {test_df['correct'].mean():.4f}\n")
 f.write(f"Total Errors: {len(errors_df)} / {len(test_df)} ({len(errors_df)/len(test_df)*100:.1f}%)\n\n")

 f.write("Confusion Matrix (counts):\n")
 f.write(" Predicted\n")
 f.write(" Amb CNR CR\n")
 for i, label in enumerate(['Ambivalent', 'Clear Non-Reply', 'Clear Reply']):
 f.write(f"{label:17} {cm[i][0]:5} {cm[i][1]:6} {cm[i][2]:5}\n")

 f.write("\n" + classification_report(true_labels, predictions, target_names=label_list))

 f.write("\n\nError Breakdown by Class:\n")
 for label in label_list:
 class_df = test_df[test_df['true_label'] == label]
 class_errors = errors_df[errors_df['true_label'] == label]
 error_rate = len(class_errors) / len(class_df) * 100 if len(class_df) > 0 else 0
 f.write(f"\n{label}:\n")
 f.write(f" Total: {len(class_df)}, Errors: {len(class_errors)} ({error_rate:.1f}%)\n")

 if len(class_errors) > 0:
 error_types = class_errors['predicted_label'].value_counts()
 f.write(f" Confused as:\n")
 for pred_label, count in error_types.items():
 f.write(f" → {pred_label}: {count} ({count/len(class_errors)*100:.1f}%)\n")

print("* Saved: bert_error_summary.txt")

# Save high-confidence errors for manual review
high_conf_errors_detailed = errors_df[errors_df['confidence'] > 0.7][[
 'question', 'interview_answer', 'true_label', 'predicted_label', 'confidence'
]].copy()
high_conf_errors_detailed.to_csv('high_confidence_errors.csv', index=False)
print(f"* Saved: high_confidence_errors.csv ({len(high_conf_errors_detailed)} examples)")

print("\n" + "="*80)
print("ERROR ANALYSIS COMPLETE!")
print("="*80)
print("\nKey Findings:")
print(f"1. Overall accuracy: {test_df['correct'].mean():.1%}")
print(f"2. Mean confidence (correct): {correct_df['confidence'].mean():.3f}")
print(f"3. Mean confidence (errors): {errors_df['confidence'].mean():.3f}")
print(f"4. High-confidence errors: {len(high_conf_errors)} ({len(high_conf_errors)/len(errors_df)*100:.1f}% of errors)")
print("\nFiles generated:")
print(" - bert_error_analysis.csv (full analysis)")
print(" - bert_error_summary.txt (summary report)")
print(" - high_confidence_errors.csv (errors to review)")
