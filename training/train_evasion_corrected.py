"""
Subtask 2: DeBERTa-v3-LARGE for 9-Class Evasion Prediction
WITH CORRECTED LABEL NAMES

Critical fix: Training data labels must be mapped to evaluation labels:
- Partial/half-answer → Partial
- Declining to answer → Declining
- Claims ignorance → Ignorance

Also handles multi-label nature of evaluation (predicts single label but
evaluation can have multiple valid labels per sample).
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
 AutoTokenizer,
 AutoModel,
 get_cosine_schedule_with_warmup
)
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm
import time
import argparse
import zipfile

print("="*80)
print("SUBTASK 2: 9-Class Evasion (CORRECTED LABELS)")
print("="*80)

# CRITICAL: Label mapping from training data to evaluation format
LABEL_MAPPING = {
 'Partial/half-answer': 'Partial',
 'Declining to answer': 'Declining',
 'Claims ignorance': 'Ignorance',
 'Explicit': 'Explicit',
 'Implicit': 'Implicit',
 'Dodging': 'Dodging',
 'General': 'General',
 'Deflection': 'Deflection',
 'Clarification': 'Clarification',
}

# Parse arguments
parser = argparse.ArgumentParser(description='Subtask 2: DeBERTa-LARGE 9-Class Evasion')
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=8)
parser.add_argument('--warmup_ratio', type=float, default=0.15)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--llrd_alpha', type=float, default=0.9)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--focal_gamma', type=float, default=2.5)
parser.add_argument('--patience', type=int, default=4)
parser.add_argument('--use_gpt_features', action='store_true', default=True)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Device: {device}")
if torch.cuda.is_available():
 print(f" GPU: {torch.cuda.get_device_name(0)}")

print(f"\n{'='*80}")
print("HYPERPARAMETERS")
print(f"{'='*80}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Batch Size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation_steps})")
print(f"Max Epochs: {args.num_epochs}")
print(f"Focal Gamma: {args.focal_gamma}")
print(f"Use GPT Features: {args.use_gpt_features}")

# Load dataset
print("\n[1/9] Loading dataset...")
dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Map training labels to evaluation format
print("\n[2/9] Mapping labels to evaluation format...")
train_df['evasion_label_mapped'] = train_df['evasion_label'].map(LABEL_MAPPING)

# Verify mapping
print("Label mapping applied:")
for old, new in LABEL_MAPPING.items():
 if old != new:
 count = (train_df['evasion_label'] == old).sum()
 print(f" {old:25s} -> {new:15s} ({count} samples)")

# Use corrected labels
EVASION_LABELS = sorted(train_df['evasion_label_mapped'].unique())
label2id = {label: idx for idx, label in enumerate(EVASION_LABELS)}
id2label = {idx: label for label, idx in label2id.items()}

train_df['label_id'] = train_df['evasion_label_mapped'].map(label2id)
test_df['label_id'] = 0 # Dummy

print(f"\n Corrected 9 evasion classes:")
for label, idx in label2id.items():
 count = (train_df['evasion_label_mapped'] == label).sum()
 print(f" {idx}. {label:20s}: {count:4d} ({count/len(train_df)*100:5.1f}%)")

# Prepare text with optional GPT features
print("\n[3/9] Preparing text...")
def prepare_text(row):
 question = row['question']
 answer = row['interview_answer']
 text = f"Question: {question} [SEP] Answer: {answer}"

 if args.use_gpt_features and pd.notna(row.get('gpt3.5_summary', None)):
 summary = str(row['gpt3.5_summary'])[:150]
 text += f" [SEP] Summary: {summary}"

 return text

train_df['text'] = train_df.apply(prepare_text, axis=1)
test_df['text'] = test_df.apply(prepare_text, axis=1)

# Boolean features
def extract_bool_features(row):
 return [
 float(row.get('affirmative_questions', False)),
 float(row.get('multiple_questions', False))
 ]

train_df['bool_features'] = train_df.apply(extract_bool_features, axis=1)
test_df['bool_features'] = test_df.apply(extract_bool_features, axis=1)

# Calculate Focal Loss weights
print("\n[4/9] Calculating class weights...")
class_counts = train_df['label_id'].value_counts().sort_index()
total = len(train_df)
focal_alpha = torch.tensor([total / (len(EVASION_LABELS) * count) for count in class_counts],
 dtype=torch.float).to(device)
focal_alpha = focal_alpha / focal_alpha.sum() * len(EVASION_LABELS)

print(f"Focal Loss alpha weights: {focal_alpha.cpu().numpy()}")

# Focal Loss
class FocalLoss(nn.Module):
 def __init__(self, alpha=None, gamma=2.5):
 super().__init__()
 self.alpha = alpha
 self.gamma = gamma

 def forward(self, logits, targets):
 ce = nn.functional.cross_entropy(logits, targets, reduction='none')
 pt = torch.exp(-ce)
 focal = (1 - pt) ** self.gamma * ce
 if self.alpha is not None:
 focal = self.alpha[targets] * focal
 return focal.mean()

# Model
class DeBERTaEvasion(nn.Module):
 def __init__(self, model_name, num_labels, bool_dim=2):
 super().__init__()
 self.deberta = AutoModel.from_pretrained(model_name)
 hidden = self.deberta.config.hidden_size

 self.feat_proj = nn.Sequential(
 nn.Linear(bool_dim, 16),
 nn.ReLU(),
 nn.Dropout(0.1)
 )

 self.classifier = nn.Sequential(
 nn.Dropout(0.2),
 nn.Linear(hidden + 16, hidden // 2),
 nn.GELU(),
 nn.Dropout(0.2),
 nn.Linear(hidden // 2, num_labels)
 )

 def forward(self, input_ids, attention_mask, bool_features):
 out = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
 cls = out.last_hidden_state[:, 0, :]
 feat = self.feat_proj(bool_features)
 combined = torch.cat([cls, feat], dim=1)
 return self.classifier(combined)

# Dataset
class EvasionDataset(Dataset):
 def __init__(self, texts, labels, bool_feats, tokenizer, max_len=512):
 self.texts = texts
 self.labels = labels
 self.bool_feats = bool_feats
 self.tokenizer = tokenizer
 self.max_len = max_len

 def __len__(self):
 return len(self.texts)

 def __getitem__(self, idx):
 enc = self.tokenizer(
 str(self.texts[idx]),
 max_length=self.max_len,
 padding='max_length',
 truncation=True,
 return_tensors='pt'
 )
 return {
 'input_ids': enc['input_ids'].flatten(),
 'attention_mask': enc['attention_mask'].flatten(),
 'bool_features': torch.tensor(self.bool_feats[idx], dtype=torch.float),
 'labels': torch.tensor(self.labels[idx], dtype=torch.long)
 }

# LLRD optimizer
def get_llrd_params(model, lr, wd, alpha):
 no_decay = ["bias", "LayerNorm.weight"]
 num_layers = model.deberta.config.num_hidden_layers
 params = []

 # Embeddings
 params.append({
 "params": [p for n, p in model.deberta.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
 "weight_decay": wd, "lr": lr * (alpha ** (num_layers + 1))
 })
 params.append({
 "params": [p for n, p in model.deberta.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0, "lr": lr * (alpha ** (num_layers + 1))
 })

 # Layers
 for i in range(num_layers):
 layer = model.deberta.encoder.layer[i]
 layer_lr = lr * (alpha ** (num_layers - i))
 params.append({
 "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
 "weight_decay": wd, "lr": layer_lr
 })
 params.append({
 "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0, "lr": layer_lr
 })

 # Classifier (full LR)
 for component in [model.feat_proj, model.classifier]:
 params.append({
 "params": [p for n, p in component.named_parameters() if not any(nd in n for nd in no_decay)],
 "weight_decay": wd, "lr": lr
 })
 params.append({
 "params": [p for n, p in component.named_parameters() if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0, "lr": lr
 })

 return params

# Training
def train_epoch(model, loader, optimizer, scheduler, criterion, device, accum_steps):
 model.train()
 total_loss = 0
 preds, labels = [], []
 optimizer.zero_grad()

 for step, batch in enumerate(tqdm(loader, desc="Training")):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)
 label = batch['labels'].to(device)

 logits = model(input_ids, attention_mask, bool_features)
 loss = criterion(logits, label) / accum_steps
 loss.backward()

 if (step + 1) % accum_steps == 0:
 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
 optimizer.step()
 scheduler.step()
 optimizer.zero_grad()

 total_loss += loss.item() * accum_steps
 preds.extend(torch.argmax(logits, 1).cpu().numpy())
 labels.extend(label.cpu().numpy())

 return total_loss / len(loader), f1_score(labels, preds, average='macro')

def eval_epoch(model, loader, device):
 model.eval()
 preds, labels = [], []

 with torch.no_grad():
 for batch in tqdm(loader, desc="Evaluating"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)
 label = batch['labels'].to(device)

 logits = model(input_ids, attention_mask, bool_features)
 preds.extend(torch.argmax(logits, 1).cpu().numpy())
 labels.extend(label.cpu().numpy())

 return f1_score(labels, preds, average='macro'), preds, labels

# Initialize
print("\n[5/9] Initializing model...")
model_name = 'microsoft/deberta-v3-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = DeBERTaEvasion(model_name, num_labels=len(EVASION_LABELS)).to(device)

print(f"Model: {model_name} (434M params)")
print(f"Output classes: {len(EVASION_LABELS)}")

# Datasets
print("\n[6/9] Creating datasets...")
train_dataset = EvasionDataset(
 train_df['text'].tolist(),
 train_df['label_id'].tolist(),
 train_df['bool_features'].tolist(),
 tokenizer, args.max_length
)
test_dataset = EvasionDataset(
 test_df['text'].tolist(),
 test_df['label_id'].tolist(),
 test_df['bool_features'].tolist(),
 tokenizer, args.max_length
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# Optimizer
print("\n[7/9] Setting up optimizer...")
optimizer_params = get_llrd_params(model, args.learning_rate, args.weight_decay, args.llrd_alpha)
optimizer = torch.optim.AdamW(optimizer_params)

total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
warmup_steps = int(total_steps * args.warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)

print(f"Total steps: {total_steps}")
print(f"Warmup steps: {warmup_steps}")

# Training loop
print("\n[8/9] Training...")
print("="*80)

best_f1 = 0
best_epoch = 0
patience_counter = 0
start_time = time.time()

for epoch in range(args.num_epochs):
 print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
 print("-" * 40)

 train_loss, train_f1 = train_epoch(
 model, train_loader, optimizer, scheduler, criterion,
 device, args.gradient_accumulation_steps
 )

 val_f1, val_preds, val_labels = eval_epoch(model, test_loader, device)

 print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
 print(f"Val F1: {val_f1:.4f}")

 if val_f1 > best_f1:
 best_f1 = val_f1
 best_epoch = epoch + 1
 patience_counter = 0
 torch.save(model.state_dict(), 'evasion_corrected_best.pt')
 print(f"New best. Saved.")
 else:
 patience_counter += 1
 print(f"No improvement ({patience_counter}/{args.patience})")
 if patience_counter >= args.patience:
 print("Early stopping.")
 break

training_time = time.time() - start_time
print(f"\nTraining complete. Best F1: {best_f1:.4f} (epoch {best_epoch})")
print(f"Time: {training_time/60:.1f} min")

# Generate predictions
print("\n[9/9] Generating predictions...")
model.load_state_dict(torch.load('evasion_corrected_best.pt'))
model.eval()

test_preds = []
with torch.no_grad():
 for batch in tqdm(test_loader, desc="Predicting"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)

 logits = model(input_ids, attention_mask, bool_features)
 test_preds.extend(torch.argmax(logits, 1).cpu().numpy())

# Convert to CORRECTED label names
pred_labels = [id2label[p] for p in test_preds]

# Distribution
print("\nPrediction distribution:")
dist = pd.Series(pred_labels).value_counts()
for label, count in dist.items():
 print(f" {label:20s}: {count:3d} ({count/len(pred_labels)*100:5.1f}%)")

# Save
output_file = 'prediction_evasion_corrected'
with open(output_file, 'w') as f:
 for p in pred_labels:
 f.write(f"{p}\n")
print(f"\nSaved: {output_file}")

# Zip
zip_file = 'submission_evasion_corrected.zip'
with zipfile.ZipFile(zip_file, 'w') as zf:
 zf.write(output_file, arcname='prediction')
print(f"Created: {zip_file}")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Best Val F1: {best_f1:.4f}")
print(f"Labels are now in CORRECT format (Partial, Declining, Ignorance)")
print("="*80)
