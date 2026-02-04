"""
Ensemble prediction by combining multiple model outputs.

Supports soft voting, hard voting, and weighted voting strategies.
"""

import torch
import numpy as np
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm

print("="*80)
print("Ensemble Model Creation")
print("="*80)

# Load dataset
print("\n[1/6] Loading dataset...")
dataset = load_from_disk('./QEvasion')
test_df = pd.DataFrame(dataset['test'])

def prepare_text_transformer(row):
 question = row['question']
 answer = row['interview_answer']
 return f"Question: {question} [SEP] Answer: {answer}"

test_df['text'] = test_df.apply(prepare_text_transformer, axis=1)

label_list = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}
test_df['label_id'] = test_df['clarity_label'].map(label2id)

print(f"* Test samples: {len(test_df)}")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"* Device: {device}")

# Model configurations
models_config = [
 {
 'name': 'DeBERTa-v3 Improved',
 'path': './deberta_v3_improved_model',
 'weight': 0.61, # Dev F1 score
 'model_name': 'microsoft/deberta-v3-base'
 },
 {
 'name': 'BERT-base',
 'path': './bert_base_uncased_model',
 'weight': 0.56, # Dev F1 score
 'model_name': 'bert-base-uncased'
 }
]

# Try to add DistilBERT if available
import os
if os.path.exists('./distilbert_base_uncased_model'):
 models_config.append({
 'name': 'DistilBERT-base',
 'path': './distilbert_base_uncased_model',
 'weight': 0.52, # Estimated
 'model_name': 'distilbert-base-uncased'
 })

print(f"\n[2/6] Loading {len(models_config)} models...")
models = []
tokenizers = []

for config in models_config:
 print(f"\n Loading {config['name']}...")
 try:
 tokenizer = AutoTokenizer.from_pretrained(config['path'])
 model = AutoModelForSequenceClassification.from_pretrained(config['path'])
 model.to(device)
 model.eval()

 tokenizers.append(tokenizer)
 models.append(model)
 print(f" * {config['name']} loaded (weight: {config['weight']:.2f})")
 except Exception as e:
 print(f" WARNING: Failed to load {config['name']}: {e}")
 models_config.remove(config)

print(f"\n* Successfully loaded {len(models)} models")

# Get predictions from each model
print("\n[3/6] Generating predictions from each model...")
all_probabilities = []

for idx, (model, tokenizer, config) in enumerate(zip(models, tokenizers, models_config)):
 print(f"\n [{idx+1}/{len(models)}] Predicting with {config['name']}...")

 probabilities = []

 with torch.no_grad():
 for text in tqdm(test_df['text'], desc=f" {config['name']}"):
 inputs = tokenizer(
 text,
 max_length=512,
 padding='max_length',
 truncation=True,
 return_tensors='pt'
 )

 inputs = {k: v.to(device) for k, v in inputs.items()}
 outputs = model(**inputs)

 # Get probabilities
 probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
 probabilities.append(probs)

 all_probabilities.append(np.array(probabilities))
 print(f" * {config['name']}: {len(probabilities)} predictions generated")

# Ensemble strategies
print("\n[4/6] Creating ensemble predictions...")

# Strategy 1: Simple Average (equal weight)
print("\n Strategy 1: Simple Average")
avg_probs = np.mean(all_probabilities, axis=0)
avg_preds = np.argmax(avg_probs, axis=1)
avg_f1 = f1_score(test_df['label_id'], avg_preds, average='macro')
avg_acc = accuracy_score(test_df['label_id'], avg_preds)
print(f" * Simple Average - F1: {avg_f1:.4f}, Acc: {avg_acc:.4f}")

# Strategy 2: Weighted Average (by dev F1)
print("\n Strategy 2: Weighted Average (by dev F1)")
weights = np.array([config['weight'] for config in models_config])
weights = weights / weights.sum() # Normalize
print(f" Weights: {dict(zip([c['name'] for c in models_config], weights))}")

weighted_probs = np.average(all_probabilities, axis=0, weights=weights)
weighted_preds = np.argmax(weighted_probs, axis=1)
weighted_f1 = f1_score(test_df['label_id'], weighted_preds, average='macro')
weighted_acc = accuracy_score(test_df['label_id'], weighted_preds)
print(f" * Weighted Average - F1: {weighted_f1:.4f}, Acc: {weighted_acc:.4f}")

# Strategy 3: Hard Voting (majority vote)
print("\n Strategy 3: Hard Voting (majority)")
hard_preds = np.array([np.argmax(probs, axis=1) for probs in all_probabilities])
majority_preds = np.apply_along_axis(lambda x: np.bincount(x, minlength=3).argmax(), axis=0, arr=hard_preds)
majority_f1 = f1_score(test_df['label_id'], majority_preds, average='macro')
majority_acc = accuracy_score(test_df['label_id'], majority_preds)
print(f" * Hard Voting - F1: {majority_f1:.4f}, Acc: {majority_acc:.4f}")

# Select best ensemble
print("\n[5/6] Selecting best ensemble strategy...")
strategies = [
 ('Simple Average', avg_f1, avg_preds),
 ('Weighted Average', weighted_f1, weighted_preds),
 ('Hard Voting', majority_f1, majority_preds)
]

best_strategy, best_f1, best_preds = max(strategies, key=lambda x: x[1])
print(f"\n* Best Strategy: {best_strategy}")
print(f"* Best Ensemble F1: {best_f1:.4f}")

# Compare with individual models
print("\n" + "="*80)
print("Performance Comparison")
print("="*80)
print(f"\n{'Model':<30} {'Dev F1':<10}")
print("-" * 40)
for config in models_config:
 print(f"{config['name']:<30} {config['weight']:.4f}")
print("-" * 40)
print(f"{'Ensemble (' + best_strategy + ')':<30} {best_f1:.4f} ")

improvement = best_f1 - models_config[0]['weight'] # vs best single model
print(f"\n* Improvement over best single model: {improvement:+.4f} ({improvement/models_config[0]['weight']*100:+.1f}%)")

# Detailed report
print("\n" + "="*80)
print("Ensemble Classification Report")
print("="*80)
print(classification_report(test_df['label_id'], best_preds, target_names=label_list))

# Generate submission
print("\n[6/6] Generating ensemble submission...")
pred_labels = [id2label[pred] for pred in best_preds]

# Save prediction file
output_name = 'prediction_ensemble'
with open(output_name, 'w') as f:
 for pred in pred_labels:
 f.write(f"{pred}\n")

print(f"* Prediction file created: {output_name}")

# Class distribution
unique, counts = np.unique(pred_labels, return_counts=True)
print(f"\nPrediction distribution:")
for label, count in zip(unique, counts):
 print(f" {label}: {count} ({count/len(pred_labels)*100:.1f}%)")

# Create submission zip
import zipfile
with zipfile.ZipFile('submission_ensemble.zip', 'w') as zipf:
 zipf.write(output_name, 'prediction')

print(f"\n* Submission created: submission_ensemble.zip")

print("\n" + "="*80)
print("Ensemble complete.")
print("="*80)
print(f"Strategy: {best_strategy}")
print(f"Dev F1: {best_f1:.4f}")
print(f"Improvement: {improvement:+.4f} over single model")
