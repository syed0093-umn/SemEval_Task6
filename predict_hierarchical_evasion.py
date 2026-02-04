"""
Hierarchical evasion classification inference pipeline.

Combines a clarity classifier (Stage 1) with evasion-type classifiers (Stage 2):
 - Clear Reply -> "Explicit" (deterministic mapping)
 - Ambivalent -> Ambivalent classifier (5 classes)
 - Clear Non-Reply -> Non-reply classifier (3 classes)

Requires:
 - Stage 1 model: deberta_focal_features_best.pt (or deberta_large_best.pt)
 - Stage 2 models: stage2_ambivalent_best.pt, stage2_nonreply_best.pt
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import zipfile

print("=" * 80)
print("Hierarchical Evasion Prediction")
print("=" * 80)

# Parse arguments
parser = argparse.ArgumentParser(description='Predict evasion techniques using hierarchical model')
parser.add_argument('--stage1_model', type=str, default='deberta_focal_features_best.pt',
 help='Path to Stage 1 (clarity) model')
parser.add_argument('--stage1_type', type=str, default='base', choices=['base', 'large'],
 help='Stage 1 model type (base or large)')
parser.add_argument('--stage2_type', type=str, default='base', choices=['base', 'large'],
 help='Stage 2 model type (base or large)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')

args = parser.parse_args()

# Set model names based on type
STAGE1_MODEL_NAME = 'microsoft/deberta-v3-large' if args.stage1_type == 'large' else 'microsoft/deberta-v3-base'
STAGE2_MODEL_NAME = 'microsoft/deberta-v3-large' if args.stage2_type == 'large' else 'microsoft/deberta-v3-base'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Stage 1 model: {args.stage1_model} ({args.stage1_type})")
print(f"Stage 2 models: stage2_ambivalent_best.pt, stage2_nonreply_best.pt ({args.stage2_type})")

# ============================================================================
# Label Mappings
# ============================================================================

# Stage 1: Clarity labels
CLARITY_LABELS = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
CLARITY_LABEL2ID = {label: idx for idx, label in enumerate(CLARITY_LABELS)}
CLARITY_ID2LABEL = {idx: label for label, idx in CLARITY_LABEL2ID.items()}

# Stage 2: Ambivalent evasion labels (5 classes)
AMBIVALENT_LABELS = ['Deflection', 'Dodging', 'General', 'Implicit', 'Partial/half-answer']
AMBIVALENT_ID2LABEL = {idx: label for idx, label in enumerate(AMBIVALENT_LABELS)}

# Stage 2: NonReply evasion labels (3 classes)
NONREPLY_LABELS = ['Claims ignorance', 'Clarification', 'Declining to answer']
NONREPLY_ID2LABEL = {idx: label for idx, label in enumerate(NONREPLY_LABELS)}

# Output label name mapping (training data -> submission format)
LABEL_NAME_MAP = {
 'Claims ignorance': 'Ignorance',
 'Declining to answer': 'Declining',
 'Partial/half-answer': 'Partial',
 'Explicit': 'Explicit',
 'Implicit': 'Implicit',
 'Dodging': 'Dodging',
 'General': 'General',
 'Deflection': 'Deflection',
 'Clarification': 'Clarification'
}

# ============================================================================
# Model Classes
# ============================================================================

class DeBERTaWithFeatures(nn.Module):
 """Stage 1 model with boolean features (for clarity classification)."""
 def __init__(self, model_name, num_labels, bool_feature_dim=2):
 super().__init__()
 self.deberta = AutoModel.from_pretrained(model_name)
 self.config = self.deberta.config
 hidden_size = self.config.hidden_size

 self.feature_processor = nn.Sequential(
 nn.Linear(bool_feature_dim, 16),
 nn.ReLU(),
 nn.Dropout(0.1)
 )
 self.classifier = nn.Sequential(
 nn.Dropout(0.1),
 nn.Linear(hidden_size + 16, num_labels)
 )

 def forward(self, input_ids, attention_mask, bool_features):
 outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
 pooled = outputs.last_hidden_state[:, 0, :]
 feature_embed = self.feature_processor(bool_features)
 combined = torch.cat([pooled, feature_embed], dim=1)
 logits = self.classifier(combined)
 return logits

class DeBERTaClassifier(nn.Module):
 """Stage 2 simple classifier (for evasion classification)."""
 def __init__(self, model_name, num_labels):
 super().__init__()
 self.deberta = AutoModel.from_pretrained(model_name)
 self.config = self.deberta.config
 self.dropout = nn.Dropout(0.1)
 self.classifier = nn.Linear(self.config.hidden_size, num_labels)

 def forward(self, input_ids, attention_mask):
 outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
 pooled = outputs.last_hidden_state[:, 0, :]
 pooled = self.dropout(pooled)
 logits = self.classifier(pooled)
 return logits

# ============================================================================
# Dataset Classes
# ============================================================================

class ClarityDataset(Dataset):
 """Dataset for Stage 1 (clarity) with boolean features."""
 def __init__(self, texts, bool_features, tokenizer, max_length=512):
 self.texts = texts
 self.bool_features = bool_features
 self.tokenizer = tokenizer
 self.max_length = max_length

 def __len__(self):
 return len(self.texts)

 def __getitem__(self, idx):
 encoding = self.tokenizer(
 str(self.texts[idx]),
 max_length=self.max_length,
 padding='max_length',
 truncation=True,
 return_tensors='pt'
 )
 return {
 'input_ids': encoding['input_ids'].flatten(),
 'attention_mask': encoding['attention_mask'].flatten(),
 'bool_features': torch.tensor(self.bool_features[idx], dtype=torch.float)
 }

class EvasionDataset(Dataset):
 """Dataset for Stage 2 (evasion)."""
 def __init__(self, texts, tokenizer, max_length=512):
 self.texts = texts
 self.tokenizer = tokenizer
 self.max_length = max_length

 def __len__(self):
 return len(self.texts)

 def __getitem__(self, idx):
 encoding = self.tokenizer(
 str(self.texts[idx]),
 max_length=self.max_length,
 padding='max_length',
 truncation=True,
 return_tensors='pt'
 )
 return {
 'input_ids': encoding['input_ids'].flatten(),
 'attention_mask': encoding['attention_mask'].flatten()
 }

# ============================================================================
# Load Data
# ============================================================================
print("\n[1/5] Loading test data...")

dataset = load_from_disk('./QEvasion')
test_df = pd.DataFrame(dataset['test'])

def prepare_text(row):
 return f"Question: {row['question']} [SEP] Answer: {row['interview_answer']}"

def extract_bool_features(row):
 return [float(row['affirmative_questions']), float(row['multiple_questions'])]

test_df['text'] = test_df.apply(prepare_text, axis=1)
test_df['bool_features'] = test_df.apply(extract_bool_features, axis=1)

print(f"Test samples: {len(test_df)}")

# ============================================================================
# Load Models
# ============================================================================
print("\n[2/5] Loading Stage 1 model (clarity)...")

# Load Stage 1 tokenizer and model
stage1_tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL_NAME)
stage1_model = DeBERTaWithFeatures(
 model_name=STAGE1_MODEL_NAME,
 num_labels=3,
 bool_feature_dim=2
).to(device)
stage1_model.load_state_dict(torch.load(args.stage1_model, map_location=device))
stage1_model.eval()
print(f" Loaded: {args.stage1_model}")

print("\n[3/5] Loading Stage 2 models (evasion)...")

# Load Stage 2 tokenizer
stage2_tokenizer = AutoTokenizer.from_pretrained(STAGE2_MODEL_NAME)

# Load Ambivalent classifier
ambivalent_model = DeBERTaClassifier(
 model_name=STAGE2_MODEL_NAME,
 num_labels=5
).to(device)
ambivalent_model.load_state_dict(torch.load('stage2_ambivalent_best.pt', map_location=device))
ambivalent_model.eval()
print(f" Loaded: stage2_ambivalent_best.pt (5 classes)")

# Load NonReply classifier
nonreply_model = DeBERTaClassifier(
 model_name=STAGE2_MODEL_NAME,
 num_labels=3
).to(device)
nonreply_model.load_state_dict(torch.load('stage2_nonreply_best.pt', map_location=device))
nonreply_model.eval()
print(f" Loaded: stage2_nonreply_best.pt (3 classes)")

# ============================================================================
# Stage 1: Predict Clarity
# ============================================================================
print("\n[4/5] Running Stage 1: Clarity prediction...")

clarity_dataset = ClarityDataset(
 texts=test_df['text'].tolist(),
 bool_features=test_df['bool_features'].tolist(),
 tokenizer=stage1_tokenizer,
 max_length=args.max_length
)
clarity_loader = DataLoader(clarity_dataset, batch_size=args.batch_size, shuffle=False)

clarity_predictions = []
with torch.no_grad():
 for batch in tqdm(clarity_loader, desc="Stage 1"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)

 logits = stage1_model(input_ids, attention_mask, bool_features)
 preds = torch.argmax(logits, dim=1)
 clarity_predictions.extend(preds.cpu().numpy())

clarity_labels = [CLARITY_ID2LABEL[p] for p in clarity_predictions]
test_df['clarity_pred'] = clarity_labels

print(f" Clarity distribution:")
for label in CLARITY_LABELS:
 count = (test_df['clarity_pred'] == label).sum()
 print(f" {label}: {count} ({count/len(test_df)*100:.1f}%)")

# ============================================================================
# Stage 2: Predict Evasion Based on Clarity
# ============================================================================
print("\n[5/5] Running Stage 2: Evasion prediction...")

# Initialize final predictions
final_evasion = [''] * len(test_df)

# Clear Reply -> Always "Explicit"
clear_reply_mask = test_df['clarity_pred'] == 'Clear Reply'
for idx in test_df[clear_reply_mask].index:
 final_evasion[idx] = 'Explicit'
print(f" Clear Reply -> Explicit: {clear_reply_mask.sum()} samples")

# Ambivalent -> Use ambivalent classifier
ambivalent_mask = test_df['clarity_pred'] == 'Ambivalent'
if ambivalent_mask.sum() > 0:
 ambivalent_texts = test_df[ambivalent_mask]['text'].tolist()
 ambivalent_indices = test_df[ambivalent_mask].index.tolist()

 ambivalent_dataset = EvasionDataset(
 texts=ambivalent_texts,
 tokenizer=stage2_tokenizer,
 max_length=args.max_length
 )
 ambivalent_loader = DataLoader(ambivalent_dataset, batch_size=args.batch_size, shuffle=False)

 ambivalent_preds = []
 with torch.no_grad():
 for batch in tqdm(ambivalent_loader, desc="Stage 2 (Ambivalent)"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)

 logits = ambivalent_model(input_ids, attention_mask)
 preds = torch.argmax(logits, dim=1)
 ambivalent_preds.extend(preds.cpu().numpy())

 for i, idx in enumerate(ambivalent_indices):
 label = AMBIVALENT_ID2LABEL[ambivalent_preds[i]]
 final_evasion[idx] = label

 print(f" Ambivalent predictions: {len(ambivalent_preds)}")
 for label in AMBIVALENT_LABELS:
 count = sum(1 for p in ambivalent_preds if AMBIVALENT_ID2LABEL[p] == label)
 print(f" {label}: {count}")

# Clear Non-Reply -> Use nonreply classifier
nonreply_mask = test_df['clarity_pred'] == 'Clear Non-Reply'
if nonreply_mask.sum() > 0:
 nonreply_texts = test_df[nonreply_mask]['text'].tolist()
 nonreply_indices = test_df[nonreply_mask].index.tolist()

 nonreply_dataset = EvasionDataset(
 texts=nonreply_texts,
 tokenizer=stage2_tokenizer,
 max_length=args.max_length
 )
 nonreply_loader = DataLoader(nonreply_dataset, batch_size=args.batch_size, shuffle=False)

 nonreply_preds = []
 with torch.no_grad():
 for batch in tqdm(nonreply_loader, desc="Stage 2 (NonReply)"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)

 logits = nonreply_model(input_ids, attention_mask)
 preds = torch.argmax(logits, dim=1)
 nonreply_preds.extend(preds.cpu().numpy())

 for i, idx in enumerate(nonreply_indices):
 label = NONREPLY_ID2LABEL[nonreply_preds[i]]
 final_evasion[idx] = label

 print(f" NonReply predictions: {len(nonreply_preds)}")
 for label in NONREPLY_LABELS:
 count = sum(1 for p in nonreply_preds if NONREPLY_ID2LABEL[p] == label)
 print(f" {label}: {count}")

# ============================================================================
# Convert to Submission Format and Save
# ============================================================================
print("\n" + "=" * 80)
print("Creating Submission")
print("=" * 80)

# Convert labels to submission format
submission_labels = [LABEL_NAME_MAP[label] for label in final_evasion]

print(f"\nFinal label distribution (submission format):")
from collections import Counter
label_counts = Counter(submission_labels)
for label in sorted(label_counts.keys()):
 count = label_counts[label]
 print(f" {label}: {count} ({count/len(submission_labels)*100:.1f}%)")

# Verify all 9 classes are covered
expected_labels = {'Explicit', 'Implicit', 'Dodging', 'General', 'Deflection',
 'Partial', 'Declining', 'Ignorance', 'Clarification'}
missing = expected_labels - set(submission_labels)
if missing:
 print(f"\nWARNING: Missing classes in predictions: {missing}")
else:
 print(f"\nAll 9 evasion classes are represented.")

# Save prediction file
output_file = 'prediction_hierarchical'
with open(output_file, 'w') as f:
 for label in submission_labels:
 f.write(f"{label}\n")
print(f"\nSaved predictions: {output_file}")

# Create submission zip
submission_file = 'submission_hierarchical.zip'
with zipfile.ZipFile(submission_file, 'w') as zf:
 zf.write(output_file, arcname='prediction')
print(f"Created submission: {submission_file}")

print("\n" + "=" * 80)
print("Prediction complete.")
print("=" * 80)
print(f"Submit {submission_file} to Codabench Subtask 2")
