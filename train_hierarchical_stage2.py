"""
Hierarchical Evasion Classification - Stage 2 Training

Trains two separate classifiers for fine-grained evasion detection:
1. Ambivalent classifier: 5 classes (Dodging, Implicit, General, Deflection, Partial)
2. NonReply classifier: 3 classes (Declining, Ignorance, Clarification)

Clear Reply always maps to "Explicit" (no classifier needed).

Usage:
 python train_hierarchical_stage2.py [options]

This script trains Stage 2 classifiers. Use predict_hierarchical_evasion.py
for full inference combining Stage 1 + Stage 2.
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
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import argparse
import os

print("=" * 80)
print("HIERARCHICAL EVASION CLASSIFICATION - STAGE 2 TRAINING")
print("=" * 80)

# Parse arguments
parser = argparse.ArgumentParser(description='Train Stage 2 hierarchical classifiers')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation')
parser.add_argument('--num_epochs', type=int, default=8, help='Max epochs')
parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
parser.add_argument('--model_size', type=str, default='base', choices=['base', 'large'],
 help='Model size: base or large')

args = parser.parse_args()

# Set model based on size
if args.model_size == 'large':
 MODEL_NAME = 'microsoft/deberta-v3-large'
 print(f"\nUsing LARGE model (434M params)")
else:
 MODEL_NAME = 'microsoft/deberta-v3-base'
 print(f"\nUsing BASE model (184M params)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
 print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# Label Mappings
# ============================================================================

# Ambivalent -> 5 evasion classes
AMBIVALENT_LABELS = ['Deflection', 'Dodging', 'General', 'Implicit', 'Partial/half-answer']
AMBIVALENT_LABEL2ID = {label: idx for idx, label in enumerate(AMBIVALENT_LABELS)}
AMBIVALENT_ID2LABEL = {idx: label for label, idx in AMBIVALENT_LABEL2ID.items()}

# NonReply -> 3 evasion classes
NONREPLY_LABELS = ['Claims ignorance', 'Clarification', 'Declining to answer']
NONREPLY_LABEL2ID = {label: idx for idx, label in enumerate(NONREPLY_LABELS)}
NONREPLY_ID2LABEL = {idx: label for label, idx in NONREPLY_LABEL2ID.items()}

# Output label name mapping (training data -> submission format)
LABEL_NAME_MAP = {
 'Claims ignorance': 'Ignorance',
 'Declining to answer': 'Declining',
 'Partial/half-answer': 'Partial',
 # These stay the same
 'Explicit': 'Explicit',
 'Implicit': 'Implicit',
 'Dodging': 'Dodging',
 'General': 'General',
 'Deflection': 'Deflection',
 'Clarification': 'Clarification'
}

print(f"\nLabel mappings:")
print(f" Ambivalent classes (5): {AMBIVALENT_LABELS}")
print(f" NonReply classes (3): {NONREPLY_LABELS}")

# ============================================================================
# Load and Filter Data
# ============================================================================
print("\n[1/6] Loading and filtering dataset...")

dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])

# Prepare text
def prepare_text(row):
 return f"Question: {row['question']} [SEP] Answer: {row['interview_answer']}"

train_df['text'] = train_df.apply(prepare_text, axis=1)

# Split by clarity category
ambivalent_df = train_df[train_df['clarity_label'] == 'Ambivalent'].copy()
nonreply_df = train_df[train_df['clarity_label'] == 'Clear Non-Reply'].copy()

# Create label IDs
ambivalent_df['label_id'] = ambivalent_df['evasion_label'].map(AMBIVALENT_LABEL2ID)
nonreply_df['label_id'] = nonreply_df['evasion_label'].map(NONREPLY_LABEL2ID)

print(f" Ambivalent samples: {len(ambivalent_df)}")
for label in AMBIVALENT_LABELS:
 count = (ambivalent_df['evasion_label'] == label).sum()
 print(f" - {label}: {count} ({count/len(ambivalent_df)*100:.1f}%)")

print(f"\n NonReply samples: {len(nonreply_df)}")
for label in NONREPLY_LABELS:
 count = (nonreply_df['evasion_label'] == label).sum()
 print(f" - {label}: {count} ({count/len(nonreply_df)*100:.1f}%)")

# ============================================================================
# Dataset and Model Classes
# ============================================================================

class EvasionDataset(Dataset):
 def __init__(self, texts, labels, tokenizer, max_length=512):
 self.texts = texts
 self.labels = labels
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
 'labels': torch.tensor(self.labels[idx], dtype=torch.long)
 }

class DeBERTaClassifier(nn.Module):
 def __init__(self, model_name, num_labels):
 super().__init__()
 self.deberta = AutoModel.from_pretrained(model_name)
 self.config = self.deberta.config
 self.dropout = nn.Dropout(0.1)
 self.classifier = nn.Linear(self.config.hidden_size, num_labels)

 def forward(self, input_ids, attention_mask):
 outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
 pooled = outputs.last_hidden_state[:, 0, :] # [CLS] token
 pooled = self.dropout(pooled)
 logits = self.classifier(pooled)
 return logits

class FocalLoss(nn.Module):
 def __init__(self, alpha=None, gamma=2.0):
 super().__init__()
 self.alpha = alpha
 self.gamma = gamma

 def forward(self, logits, targets):
 ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
 p_t = torch.exp(-ce_loss)
 focal_term = (1 - p_t) ** self.gamma
 if self.alpha is not None:
 alpha_t = self.alpha[targets]
 focal_loss = alpha_t * focal_term * ce_loss
 else:
 focal_loss = focal_term * ce_loss
 return focal_loss.mean()

# ============================================================================
# Training Functions
# ============================================================================

def compute_class_weights(df, label_col, num_classes):
 """Compute inverse frequency weights for focal loss."""
 counts = df[label_col].value_counts().sort_index()
 total = len(df)
 weights = torch.tensor([total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
 dtype=torch.float)
 return weights

def train_classifier(name, train_df, label2id, id2label, num_classes,
 tokenizer, model_name, device, args):
 """Train a single classifier."""
 print(f"\n{'='*80}")
 print(f"TRAINING {name.upper()} CLASSIFIER ({num_classes} classes)")
 print(f"{'='*80}")

 # Create train/val split (90/10)
 train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
 val_size = int(len(train_df) * 0.1)
 val_df = train_df[:val_size]
 train_subset = train_df[val_size:]

 print(f"Train samples: {len(train_subset)}, Val samples: {len(val_df)}")

 # Create datasets
 train_dataset = EvasionDataset(
 texts=train_subset['text'].tolist(),
 labels=train_subset['label_id'].tolist(),
 tokenizer=tokenizer,
 max_length=args.max_length
 )
 val_dataset = EvasionDataset(
 texts=val_df['text'].tolist(),
 labels=val_df['label_id'].tolist(),
 tokenizer=tokenizer,
 max_length=args.max_length
 )

 train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
 val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

 # Initialize model
 model = DeBERTaClassifier(model_name, num_classes).to(device)

 # Class weights for focal loss
 class_weights = compute_class_weights(train_subset, 'label_id', num_classes).to(device)
 focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)

 # Optimizer and scheduler
 optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
 weight_decay=args.weight_decay)
 num_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
 warmup_steps = int(num_steps * args.warmup_ratio)
 scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps)

 best_f1 = 0
 patience_counter = 0
 save_path = f'stage2_{name.lower()}_best.pt'

 for epoch in range(args.num_epochs):
 # Training
 model.train()
 total_loss = 0
 train_preds, train_labels = [], []
 optimizer.zero_grad()

 progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
 for step, batch in enumerate(progress):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 labels = batch['labels'].to(device)

 logits = model(input_ids, attention_mask)
 loss = focal_loss(logits, labels) / args.gradient_accumulation_steps
 loss.backward()

 if (step + 1) % args.gradient_accumulation_steps == 0:
 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
 optimizer.step()
 scheduler.step()
 optimizer.zero_grad()

 total_loss += loss.item() * args.gradient_accumulation_steps
 train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
 train_labels.extend(labels.cpu().numpy())
 progress.set_postfix({'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}'})

 train_f1 = f1_score(train_labels, train_preds, average='macro')

 # Validation
 model.eval()
 val_preds, val_labels = [], []
 with torch.no_grad():
 for batch in val_loader:
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 labels = batch['labels'].to(device)

 logits = model(input_ids, attention_mask)
 val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
 val_labels.extend(labels.cpu().numpy())

 val_f1 = f1_score(val_labels, val_preds, average='macro')

 print(f" Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

 if val_f1 > best_f1:
 best_f1 = val_f1
 patience_counter = 0
 torch.save(model.state_dict(), save_path)
 print(f" -> New best. Saved to {save_path}")
 else:
 patience_counter += 1
 if patience_counter >= args.patience:
 print(f" Early stopping after {epoch+1} epochs")
 break

 # Load best and show final report
 model.load_state_dict(torch.load(save_path))
 model.eval()
 val_preds, val_labels = [], []
 with torch.no_grad():
 for batch in val_loader:
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 labels = batch['labels'].to(device)
 logits = model(input_ids, attention_mask)
 val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
 val_labels.extend(labels.cpu().numpy())

 label_names = [id2label[i] for i in range(num_classes)]
 print(f"\nFinal {name} Classification Report:")
 print(classification_report(val_labels, val_preds, target_names=label_names, digits=4))

 return best_f1

# ============================================================================
# Main Training
# ============================================================================
print("\n[2/6] Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\n[3/6] Training Ambivalent classifier (5 classes)...")
ambivalent_f1 = train_classifier(
 name="ambivalent",
 train_df=ambivalent_df,
 label2id=AMBIVALENT_LABEL2ID,
 id2label=AMBIVALENT_ID2LABEL,
 num_classes=5,
 tokenizer=tokenizer,
 model_name=MODEL_NAME,
 device=device,
 args=args
)

print("\n[4/6] Training NonReply classifier (3 classes)...")
nonreply_f1 = train_classifier(
 name="nonreply",
 train_df=nonreply_df,
 label2id=NONREPLY_LABEL2ID,
 id2label=NONREPLY_ID2LABEL,
 num_classes=3,
 tokenizer=tokenizer,
 model_name=MODEL_NAME,
 device=device,
 args=args
)

print("\n" + "=" * 80)
print("STAGE 2 TRAINING COMPLETE")
print("=" * 80)
print(f"Ambivalent classifier F1: {ambivalent_f1:.4f}")
print(f"NonReply classifier F1: {nonreply_f1:.4f}")
print(f"\nSaved models:")
print(f" - stage2_ambivalent_best.pt")
print(f" - stage2_nonreply_best.pt")
print(f"\nNext step: Run predict_hierarchical_evasion.py to generate predictions")
print("=" * 80)
