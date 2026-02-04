"""
Subtask 2: DeBERTa-v3-LARGE for 9-Class Evasion with Augmented Data

Key fixes over previous training:
- Proper train/val split (previous version used dummy test labels for validation)
- DeBERTa-large on augmented data for better minority class handling
- Correct Codabench label format
- Synthetic sample downweighting
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
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import time
import argparse
import zipfile

print("=" * 80)
print("SUBTASK 2: DeBERTa-v3-LARGE + Augmented Data (with proper validation)")
print("=" * 80)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./QEvasion_evasion_augmented')
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--warmup_ratio', type=float, default=0.15)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--llrd_alpha', type=float, default=0.9)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--focal_gamma', type=float, default=2.5)
parser.add_argument('--patience', type=int, default=4)
parser.add_argument('--synthetic_weight', type=float, default=0.5)
parser.add_argument('--val_fold', type=int, default=0,
 help='Which fold to use as validation (0-4)')
parser.add_argument('--n_folds', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
 print(f"GPU: {torch.cuda.get_device_name(0)}")
 print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

print(f"\nHyperparameters:")
for k, v in vars(args).items():
 print(f" {k}: {v}")

# ============================================================================
# Load dataset
# ============================================================================
print("\n[1/8] Loading augmented dataset...")
dataset = load_from_disk(args.data_dir)
full_train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"Full training samples: {len(full_train_df)}")
print(f"Test samples: {len(test_df)}")

if 'is_synthetic' in full_train_df.columns:
 n_syn = full_train_df['is_synthetic'].sum()
 print(f" Original: {len(full_train_df) - n_syn}, Synthetic: {n_syn}")
else:
 full_train_df['is_synthetic'] = False

# ============================================================================
# Prepare text and features
# ============================================================================
print("\n[2/8] Preparing text and features...")

def prepare_text(row):
 return f"Question: {row['question']} [SEP] Answer: {row['interview_answer']}"

def extract_features(row):
 return [
 float(row['affirmative_questions']) if pd.notna(row.get('affirmative_questions')) else 0.0,
 float(row['multiple_questions']) if pd.notna(row.get('multiple_questions')) else 0.0,
 ]

full_train_df['text'] = full_train_df.apply(prepare_text, axis=1)
test_df['text'] = test_df.apply(prepare_text, axis=1)
full_train_df['bool_features'] = full_train_df.apply(extract_features, axis=1)
test_df['bool_features'] = test_df.apply(extract_features, axis=1)

# ============================================================================
# Label mapping (sorted, consistent)
# ============================================================================
print("\n[3/8] Preparing labels...")
label_list = sorted(full_train_df['evasion_label'].unique())
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}
print(f"Classes: {label_list}")

full_train_df['label_id'] = full_train_df['evasion_label'].map(label2id)

# Codabench submission label mapping
submission_label_map = {
 'Claims ignorance': 'Ignorance',
 'Clarification': 'Clarification',
 'Declining to answer': 'Declining',
 'Deflection': 'Deflection',
 'Dodging': 'Dodging',
 'Explicit': 'Explicit',
 'General': 'General',
 'Implicit': 'Implicit',
 'Partial/half-answer': 'Partial',
}

# ============================================================================
# PROPER train/val split using stratified k-fold on ORIGINAL samples only
# ============================================================================
print(f"\n[4/8] Creating train/val split (fold {args.val_fold}/{args.n_folds})...")

# Only split original samples; synthetic samples always go to training
original_mask = ~full_train_df['is_synthetic']
original_df = full_train_df[original_mask].copy()
synthetic_df = full_train_df[full_train_df['is_synthetic']].copy()

skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
splits = list(skf.split(original_df, original_df['label_id']))
train_idx, val_idx = splits[args.val_fold]

val_df = original_df.iloc[val_idx].copy()
train_original = original_df.iloc[train_idx].copy()

# Add synthetic samples to training only
train_df = pd.concat([train_original, synthetic_df], ignore_index=True)
train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

print(f" Train: {len(train_df)} ({len(train_original)} original + {len(synthetic_df)} synthetic)")
print(f" Val: {len(val_df)} (original only)")

# Per-sample weights
sample_weights = []
for _, row in train_df.iterrows():
 sample_weights.append(args.synthetic_weight if row.get('is_synthetic', False) else 1.0)
sample_weights = torch.tensor(sample_weights, dtype=torch.float)

# Focal loss alpha from training set class distribution
class_counts = train_df['label_id'].value_counts().sort_index()
total = len(train_df)
focal_alpha = torch.tensor([1.0 / (count / total) for count in class_counts],
 dtype=torch.float).to(device)
focal_alpha = focal_alpha / focal_alpha.sum() * len(focal_alpha)

print("\nClass distribution (train) and focal weights:")
for idx in range(len(label_list)):
 label = id2label[idx]
 count = class_counts.get(idx, 0)
 weight = focal_alpha[idx].item()
 print(f" {label:25s}: {count:4d} | weight: {weight:.3f}")

print(f"\nVal class distribution:")
for idx in range(len(label_list)):
 label = id2label[idx]
 count = (val_df['label_id'] == idx).sum()
 print(f" {label:25s}: {count:4d}")

# ============================================================================
# Model, Loss, Dataset definitions
# ============================================================================
class FocalLoss(nn.Module):
 def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
 super().__init__()
 self.alpha = alpha
 self.gamma = gamma
 self.reduction = reduction

 def forward(self, logits, targets):
 ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
 p_t = torch.exp(-ce_loss)
 focal_term = (1 - p_t) ** self.gamma
 if self.alpha is not None:
 alpha_t = self.alpha[targets]
 focal_loss = alpha_t * focal_term * ce_loss
 else:
 focal_loss = focal_term * ce_loss
 return focal_loss.mean() if self.reduction == 'mean' else focal_loss

class DeBERTaWithFeatures(nn.Module):
 def __init__(self, model_name, num_labels, bool_feature_dim):
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
 return self.classifier(combined)

class EvasionDataset(Dataset):
 def __init__(self, texts, labels, bool_features, tokenizer, max_length=512,
 sample_weights=None):
 self.texts = texts
 self.labels = labels
 self.bool_features = bool_features
 self.tokenizer = tokenizer
 self.max_length = max_length
 self.sample_weights = sample_weights

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
 item = {
 'input_ids': encoding['input_ids'].flatten(),
 'attention_mask': encoding['attention_mask'].flatten(),
 'bool_features': torch.tensor(self.bool_features[idx], dtype=torch.float),
 'labels': torch.tensor(self.labels[idx], dtype=torch.long),
 }
 if self.sample_weights is not None:
 item['sample_weight'] = self.sample_weights[idx]
 return item

# ============================================================================
# LLRD
# ============================================================================
def get_optimizer_grouped_parameters(model, learning_rate, weight_decay, llrd_alpha):
 no_decay = ["bias", "LayerNorm.weight"]
 num_layers = model.config.num_hidden_layers
 groups = []

 # Embeddings
 groups.append({
 "params": [p for n, p in model.deberta.embeddings.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": weight_decay,
 "lr": learning_rate * (llrd_alpha ** (num_layers + 1))
 })
 groups.append({
 "params": [p for n, p in model.deberta.embeddings.named_parameters()
 if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0,
 "lr": learning_rate * (llrd_alpha ** (num_layers + 1))
 })

 # Encoder layers
 for layer_idx in range(num_layers):
 layer = model.deberta.encoder.layer[layer_idx]
 layer_lr = learning_rate * (llrd_alpha ** (num_layers - layer_idx))
 groups.append({
 "params": [p for n, p in layer.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": weight_decay, "lr": layer_lr
 })
 groups.append({
 "params": [p for n, p in layer.named_parameters()
 if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0, "lr": layer_lr
 })

 # Heads
 for module in [model.feature_processor, model.classifier]:
 groups.append({
 "params": [p for n, p in module.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": weight_decay, "lr": learning_rate
 })
 groups.append({
 "params": [p for n, p in module.named_parameters()
 if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0, "lr": learning_rate
 })

 return groups

# ============================================================================
# Train / Eval
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device, focal_loss,
 grad_accum, use_sample_weights=False):
 model.train()
 total_loss = 0
 preds_all, labels_all = [], []
 optimizer.zero_grad()

 for step, batch in enumerate(tqdm(dataloader, desc="Training")):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)
 labels = batch['labels'].to(device)

 logits = model(input_ids=input_ids, attention_mask=attention_mask,
 bool_features=bool_features)

 if use_sample_weights and 'sample_weight' in batch:
 ce = nn.functional.cross_entropy(logits, labels, reduction='none')
 p_t = torch.exp(-ce)
 fl = (1 - p_t) ** focal_loss.gamma
 if focal_loss.alpha is not None:
 fl = focal_loss.alpha[labels] * fl
 fl = fl * ce * batch['sample_weight'].to(device)
 loss = fl.mean()
 else:
 loss = focal_loss(logits, labels)

 loss = loss / grad_accum
 loss.backward()

 if (step + 1) % grad_accum == 0:
 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 optimizer.step()
 scheduler.step()
 optimizer.zero_grad()

 total_loss += loss.item() * grad_accum
 preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
 labels_all.extend(labels.cpu().numpy())

 return total_loss / len(dataloader), f1_score(labels_all, preds_all, average='macro')

def eval_epoch(model, dataloader, device, focal_loss):
 model.eval()
 total_loss = 0
 preds_all, labels_all = [], []

 with torch.no_grad():
 for batch in tqdm(dataloader, desc="Evaluating"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)
 labels = batch['labels'].to(device)

 logits = model(input_ids=input_ids, attention_mask=attention_mask,
 bool_features=bool_features)
 loss = focal_loss(logits, labels)
 total_loss += loss.item()

 preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
 labels_all.extend(labels.cpu().numpy())

 f1 = f1_score(labels_all, preds_all, average='macro')
 return total_loss / len(dataloader), f1, preds_all, labels_all

# ============================================================================
# Main
# ============================================================================
print("\n[5/8] Initializing model...")
model_name = 'microsoft/deberta-v3-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = DeBERTaWithFeatures(model_name=model_name, num_labels=len(label_list), bool_feature_dim=2)
model = model.to(device)
print(f"Model: {model_name} ({sum(p.numel() for p in model.parameters()):,} params)")

use_sw = args.synthetic_weight < 1.0

train_dataset = EvasionDataset(
 train_df['text'].tolist(), train_df['label_id'].tolist(),
 train_df['bool_features'].tolist(), tokenizer, args.max_length,
 sample_weights=sample_weights if use_sw else None
)
val_dataset = EvasionDataset(
 val_df['text'].tolist(), val_df['label_id'].tolist(),
 val_df['bool_features'].tolist(), tokenizer, args.max_length
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

print("\n[6/8] Setting up optimizer...")
opt_params = get_optimizer_grouped_parameters(model, args.learning_rate, args.weight_decay, args.llrd_alpha)
optimizer = torch.optim.AdamW(opt_params)

num_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
warmup_steps = int(num_steps * args.warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps)
focal_loss = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)

print(f"Steps: {num_steps}, Warmup: {warmup_steps}")

# ============================================================================
# Training loop
# ============================================================================
print("\n[7/8] Training...")
print("=" * 80)

checkpoint_path = 'deberta_large_evasion_augmented_best.pt'
best_f1 = 0
best_epoch = 0
patience_counter = 0
start_time = time.time()

for epoch in range(args.num_epochs):
 print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
 print("-" * 80)

 train_loss, train_f1 = train_epoch(
 model, train_loader, optimizer, scheduler, device, focal_loss,
 args.gradient_accumulation_steps, use_sample_weights=use_sw
 )

 val_loss, val_f1, val_preds, val_labels = eval_epoch(
 model, val_loader, device, focal_loss
 )

 print(f"\nTrain Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
 print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

 # Per-class breakdown on val
 print("\nPer-class val F1:")
 for idx in range(len(label_list)):
 tp = sum(1 for p, g in zip(val_preds, val_labels) if p == idx and g == idx)
 fp = sum(1 for p, g in zip(val_preds, val_labels) if p == idx and g != idx)
 fn = sum(1 for p, g in zip(val_preds, val_labels) if p != idx and g == idx)
 prec = tp / (tp + fp) if tp + fp > 0 else 0
 rec = tp / (tp + fn) if tp + fn > 0 else 0
 f1c = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
 print(f" {id2label[idx]:25s}: P={prec:.3f} R={rec:.3f} F1={f1c:.3f}")

 if val_f1 > best_f1:
 best_f1 = val_f1
 best_epoch = epoch + 1
 patience_counter = 0
 torch.save(model.state_dict(), checkpoint_path)
 print(f"\nNew best val F1. Saved to {checkpoint_path}")
 else:
 patience_counter += 1
 print(f"\nNo improvement ({patience_counter}/{args.patience})")
 if patience_counter >= args.patience:
 print("Early stopping triggered.")
 break

training_time = time.time() - start_time
print(f"\n{'=' * 80}")
print(f"Training complete. Best val F1: {best_f1:.4f} (epoch {best_epoch})")
print(f"Time: {training_time / 60:.1f} min")

# ============================================================================
# Predict on eval dataset
# ============================================================================
print("\n[8/8] Generating eval predictions...")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

eval_df = pd.read_csv('clarity_task_evaluation_dataset.csv')
print(f"Eval samples: {len(eval_df)}")

eval_df['text'] = eval_df.apply(prepare_text, axis=1)
eval_df['bool_features'] = eval_df.apply(extract_features, axis=1)

eval_dataset = EvasionDataset(
 eval_df['text'].tolist(), [0] * len(eval_df),
 eval_df['bool_features'].tolist(), tokenizer, args.max_length
)
eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

eval_preds = []
with torch.no_grad():
 for batch in tqdm(eval_loader, desc="Eval inference"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)
 logits = model(input_ids=input_ids, attention_mask=attention_mask,
 bool_features=bool_features)
 eval_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

pred_labels = [submission_label_map[id2label[p]] for p in eval_preds]
print(f"Distribution: {pd.Series(pred_labels).value_counts().to_dict()}")

# Save
output_file = 'prediction_eval_evasion_large_augmented'
with open(output_file, 'w') as f:
 for label in pred_labels:
 f.write(label + '\n')

submission_zip = 'submission_eval_evasion_large_augmented.zip'
with zipfile.ZipFile(submission_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
 zf.write(output_file, arcname='prediction')

print(f"\nSaved: {output_file}")
print(f"Created: {submission_zip}")
print(f"\nBest val F1: {best_f1:.4f}")
print(f"Upload {submission_zip} to Codabench (Subtask 2)")
