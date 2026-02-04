"""
ModernBERT for 3-class clarity classification.

ModernBERT (2024) improvements over DeBERTa:
- Rotary Positional Embeddings (RoPE)
- Flash Attention 2 for efficiency
- Alternating attention patterns
- 8192 token context length
- Better pre-training on modern data

Model variants:
- answerdotai/ModernBERT-base (139M params)
- answerdotai/ModernBERT-large (395M params)

REQUIRES: transformers >= 4.48.0
"""

import sys
import subprocess

# Check transformers version
try:
 import transformers
 version = tuple(map(int, transformers.__version__.split('.')[:2]))
 if version < (4, 48):
 print("="*70)
 print("ERROR: ModernBERT requires transformers >= 4.48.0")
 print(f"Current version: {transformers.__version__}")
 print()
 print("To upgrade, run:")
 print(" pip install --upgrade transformers")
 print()
 print("Or to install specific version:")
 print(" pip install transformers>=4.48.0")
 print("="*70)
 sys.exit(1)
except Exception as e:
 print(f"Warning: Could not check transformers version: {e}")

from datasets import load_from_disk
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
 AutoTokenizer,
 AutoModel,
 AutoConfig,
 get_cosine_schedule_with_warmup
)
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm
import time
import argparse
import zipfile

print("="*80)
print("ModernBERT for 3-Class Clarity Classification")
print("="*80)

# Parse arguments
parser = argparse.ArgumentParser(description='ModernBERT Clarity Classification')
parser.add_argument('--model_size', type=str, default='large', choices=['base', 'large'],
 help='Model size: base (139M) or large (395M)')
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=8)
parser.add_argument('--warmup_ratio', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--focal_gamma', type=float, default=2.0)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--use_gpt_features', action='store_true', default=True,
 help='Include GPT-3.5 summary in input')

args = parser.parse_args()

# Model selection
if args.model_size == 'large':
 MODEL_NAME = 'answerdotai/ModernBERT-large'
 MODEL_PARAMS = '395M'
else:
 MODEL_NAME = 'answerdotai/ModernBERT-base'
 MODEL_PARAMS = '139M'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
 print(f"GPU: {torch.cuda.get_device_name(0)}")
 print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\n{'='*80}")
print("CONFIGURATION")
print(f"{'='*80}")
print(f"Model: {MODEL_NAME} ({MODEL_PARAMS} params)")
print(f"Learning Rate: {args.learning_rate}")
print(f"Batch Size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation_steps})")
print(f"Max Epochs: {args.num_epochs}")
print(f"Max Length: {args.max_length}")
print(f"Focal Gamma: {args.focal_gamma}")
print(f"Use GPT Features: {args.use_gpt_features}")

# Load dataset
print("\n[1/9] Loading dataset...")
dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Prepare text
print("\n[2/9] Preparing text...")
def prepare_text(row, use_gpt=True):
 question = row['question']
 answer = row['interview_answer']
 text = f"Question: {question} [SEP] Answer: {answer}"

 if use_gpt and pd.notna(row.get('gpt3.5_summary', None)):
 summary = str(row['gpt3.5_summary'])[:200]
 text += f" [SEP] Summary: {summary}"

 return text

train_df['text'] = train_df.apply(lambda r: prepare_text(r, args.use_gpt_features), axis=1)
test_df['text'] = test_df.apply(lambda r: prepare_text(r, args.use_gpt_features), axis=1)

# Boolean features
def extract_bool_features(row):
 return [
 float(row.get('affirmative_questions', False)),
 float(row.get('multiple_questions', False))
 ]

train_df['bool_features'] = train_df.apply(extract_bool_features, axis=1)
test_df['bool_features'] = test_df.apply(extract_bool_features, axis=1)

print(f"Boolean features: affirmative_questions, multiple_questions")

# Labels
LABELS = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for label, idx in label2id.items()}

train_df['label_id'] = train_df['clarity_label'].map(label2id)
test_df['label_id'] = 0 # Dummy for test

print(f"\n3 Clarity classes:")
for label, idx in label2id.items():
 count = (train_df['clarity_label'] == label).sum()
 print(f" {idx}. {label:20s}: {count:4d} ({count/len(train_df)*100:5.1f}%)")

# Class weights for Focal Loss
print("\n[3/9] Calculating class weights...")
class_counts = train_df['label_id'].value_counts().sort_index()
total = len(train_df)
focal_alpha = torch.tensor([total / (len(LABELS) * count) for count in class_counts],
 dtype=torch.float).to(device)
focal_alpha = focal_alpha / focal_alpha.sum() * len(LABELS)
print(f"Focal alpha: {focal_alpha.cpu().numpy()}")

# Focal Loss
class FocalLoss(nn.Module):
 def __init__(self, alpha=None, gamma=2.0):
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

# ModernBERT Model with Features
class ModernBERTWithFeatures(nn.Module):
 def __init__(self, model_name, num_labels, bool_dim=2):
 super().__init__()
 self.config = AutoConfig.from_pretrained(model_name)
 self.encoder = AutoModel.from_pretrained(model_name)
 hidden_size = self.config.hidden_size

 # Feature processor
 self.feat_proj = nn.Sequential(
 nn.Linear(bool_dim, 32),
 nn.GELU(),
 nn.Dropout(0.1)
 )

 # Classifier with larger hidden layer
 self.classifier = nn.Sequential(
 nn.Dropout(0.2),
 nn.Linear(hidden_size + 32, hidden_size // 2),
 nn.GELU(),
 nn.Dropout(0.15),
 nn.Linear(hidden_size // 2, num_labels)
 )

 def forward(self, input_ids, attention_mask, bool_features):
 outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

 # Use [CLS] token (first token)
 cls_output = outputs.last_hidden_state[:, 0, :]

 # Process features
 feat_embed = self.feat_proj(bool_features)

 # Combine and classify
 combined = torch.cat([cls_output, feat_embed], dim=1)
 logits = self.classifier(combined)

 return logits

# Dataset
class ClarityDataset(Dataset):
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

# LLRD optimizer for ModernBERT
def get_llrd_params(model, lr, wd, alpha=0.9):
 """
 Layer-wise Learning Rate Decay for ModernBERT.
 Handles different layer access patterns across model architectures.
 """
 no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm"]

 num_layers = model.config.num_hidden_layers
 params = []

 # Find encoder layers - ModernBERT uses different structure
 encoder_layers = None
 embeddings = None

 # Try different access patterns for embeddings
 if hasattr(model.encoder, 'embeddings'):
 embeddings = model.encoder.embeddings
 elif hasattr(model.encoder, 'embed_tokens'):
 embeddings = model.encoder.embed_tokens

 # Try different access patterns for layers
 if hasattr(model.encoder, 'encoder') and hasattr(model.encoder.encoder, 'layers'):
 encoder_layers = model.encoder.encoder.layers
 elif hasattr(model.encoder, 'encoder') and hasattr(model.encoder.encoder, 'layer'):
 encoder_layers = model.encoder.encoder.layer
 elif hasattr(model.encoder, 'layers'):
 encoder_layers = model.encoder.layers
 elif hasattr(model.encoder, 'layer'):
 encoder_layers = model.encoder.layer

 if encoder_layers is None:
 raise ValueError("Could not find encoder layers in model architecture")

 # Embeddings (lowest LR)
 if embeddings is not None:
 embed_lr = lr * (alpha ** (num_layers + 1))
 params.append({
 "params": [p for n, p in embeddings.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": wd, "lr": embed_lr
 })
 params.append({
 "params": [p for n, p in embeddings.named_parameters()
 if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0, "lr": embed_lr
 })

 # Encoder layers with LLRD
 for i in range(len(encoder_layers)):
 layer = encoder_layers[i]
 layer_lr = lr * (alpha ** (num_layers - i))

 params.append({
 "params": [p for n, p in layer.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": wd, "lr": layer_lr
 })
 params.append({
 "params": [p for n, p in layer.named_parameters()
 if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0, "lr": layer_lr
 })

 # Feature projector and classifier (full LR)
 for component in [model.feat_proj, model.classifier]:
 params.append({
 "params": [p for n, p in component.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": wd, "lr": lr
 })
 params.append({
 "params": [p for n, p in component.named_parameters()
 if any(nd in n for nd in no_decay)],
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

def eval_epoch(model, loader, criterion, device):
 model.eval()
 total_loss = 0
 preds, labels = [], []

 with torch.no_grad():
 for batch in tqdm(loader, desc="Evaluating"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)
 label = batch['labels'].to(device)

 logits = model(input_ids, attention_mask, bool_features)
 loss = criterion(logits, label)

 total_loss += loss.item()
 preds.extend(torch.argmax(logits, 1).cpu().numpy())
 labels.extend(label.cpu().numpy())

 f1 = f1_score(labels, preds, average='macro')
 acc = accuracy_score(labels, preds)
 return total_loss / len(loader), acc, f1, preds, labels

# Initialize model
print("\n[4/9] Initializing ModernBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ModernBERTWithFeatures(MODEL_NAME, num_labels=len(LABELS)).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {MODEL_NAME}")
print(f"Total parameters: {total_params:,}")

# Datasets
print("\n[5/9] Creating datasets...")
train_dataset = ClarityDataset(
 train_df['text'].tolist(),
 train_df['label_id'].tolist(),
 train_df['bool_features'].tolist(),
 tokenizer, args.max_length
)
test_dataset = ClarityDataset(
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
print("\n[6/9] Setting up optimizer...")
try:
 optimizer_params = get_llrd_params(model, args.learning_rate, args.weight_decay)
except Exception as e:
 print(f"LLRD setup failed ({e}), using standard optimizer")
 optimizer_params = [
 {"params": [p for n, p in model.named_parameters() if "bias" not in n and "norm" not in n.lower()],
 "weight_decay": args.weight_decay, "lr": args.learning_rate},
 {"params": [p for n, p in model.named_parameters() if "bias" in n or "norm" in n.lower()],
 "weight_decay": 0.0, "lr": args.learning_rate}
 ]

optimizer = torch.optim.AdamW(optimizer_params)

total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
warmup_steps = int(total_steps * args.warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)

print(f"Total steps: {total_steps}")
print(f"Warmup steps: {warmup_steps}")

# Training loop
print("\n[7/9] Training...")
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

 val_loss, val_acc, val_f1, val_preds, val_labels = eval_epoch(
 model, test_loader, criterion, device
 )

 print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
 print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

 if val_f1 > best_f1:
 best_f1 = val_f1
 best_epoch = epoch + 1
 patience_counter = 0
 torch.save(model.state_dict(), 'modernbert_clarity_best.pt')
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

# Final evaluation
print("\n[8/9] Final evaluation...")
model.load_state_dict(torch.load('modernbert_clarity_best.pt'))
val_loss, val_acc, val_f1, val_preds, val_labels = eval_epoch(
 model, test_loader, criterion, device
)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Accuracy: {val_acc:.4f}")
print(f"Macro F1: {val_f1:.4f}")

print("\nClassification Report:")
print(classification_report(val_labels, val_preds, target_names=LABELS, digits=4))

# Generate predictions
print("\n[9/9] Generating predictions...")
model.eval()
test_preds = []
with torch.no_grad():
 for batch in tqdm(test_loader, desc="Predicting"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)

 logits = model(input_ids, attention_mask, bool_features)
 test_preds.extend(torch.argmax(logits, 1).cpu().numpy())

pred_labels = [id2label[p] for p in test_preds]

# Distribution
print("\nPrediction distribution:")
dist = pd.Series(pred_labels).value_counts()
for label, count in dist.items():
 print(f" {label:20s}: {count:3d} ({count/len(pred_labels)*100:5.1f}%)")

# Save
output_file = f'prediction_modernbert_{args.model_size}'
with open(output_file, 'w') as f:
 for p in pred_labels:
 f.write(f"{p}\n")
print(f"\nSaved: {output_file}")

# Zip
zip_file = f'submission_modernbert_{args.model_size}.zip'
with zipfile.ZipFile(zip_file, 'w') as zf:
 zf.write(output_file, arcname='prediction')
print(f"Created: {zip_file}")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Model: {MODEL_NAME}")
print(f"Best Val F1: {best_f1:.4f}")
print(f"DeBERTa-LARGE baseline: 0.61-0.63")
print(f"Improvement: {best_f1 - 0.62:+.4f}")
print("="*80)
