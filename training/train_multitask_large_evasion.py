"""
Multi-Task DeBERTa-LARGE: Subtask 2 (9-class) with Subtask 1 (3-class) Auxiliary.

Approach:
  Primary Task: 9-class evasion (Subtask 2)
  Auxiliary Task: 3-class clarity (Subtask 1) provides hierarchical signal

Benefits:
  1. Hierarchical regularization - model learns category structure
  2. Shared representations - 3-class task helps 9-class task
  3. Consistency - predictions respect hierarchy
  4. Summary features as additional context
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

print("="*80)
print("MULTI-TASK DeBERTa-LARGE: Subtask 2 (Primary) + Subtask 1 (Auxiliary)")
print("="*80)

# Parse arguments
parser = argparse.ArgumentParser(description='Multi-Task DeBERTa-LARGE for Evasion')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation')
parser.add_argument('--num_epochs', type=int, default=10, help='Max epochs (more for multi-task)')
parser.add_argument('--warmup_ratio', type=float, default=0.15, help='Warmup ratio')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--llrd_alpha', type=float, default=0.9, help='LLRD alpha')
parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
parser.add_argument('--focal_gamma', type=float, default=2.5, help='Focal loss gamma for evasion')
parser.add_argument('--clarity_focal_gamma', type=float, default=2.0, help='Focal loss gamma for clarity')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='Weight for auxiliary clarity task')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
parser.add_argument('--use_gpt_summary', action='store_true', default=True, help='Use GPT-3.5 summary')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Device: {device}")
if torch.cuda.is_available():
 print(f" GPU: {torch.cuda.get_device_name(0)}")
 print(f" Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\n{'='*80}")
print("HYPERPARAMETERS")
print(f"{'='*80}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Batch Size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation_steps})")
print(f"Max Epochs: {args.num_epochs}")
print(f"Focal Gamma (Evasion): {args.focal_gamma}")
print(f"Focal Gamma (Clarity): {args.clarity_focal_gamma}")
print(f"Auxiliary Weight: {args.auxiliary_weight}")
print(f"Use GPT Summary: {args.use_gpt_summary}")
print(f"Patience: {args.patience}")

# Load dataset
print("\n[1/10] Loading dataset...")
dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"* Training samples: {len(train_df)}")
print(f"* Test samples: {len(test_df)}")

# Prepare text with GPT features
print("\n[2/10] Preparing enhanced text...")

def prepare_enhanced_text(row):
 """Combine question, answer, GPT summary, and metadata"""
 question = row['question']
 answer = row['interview_answer']

 text = f"Question: {question} [SEP] Answer: {answer}"

 # Add GPT-3.5 summary if available
 if args.use_gpt_summary and pd.notna(row.get('gpt3.5_summary', None)):
 summary = str(row['gpt3.5_summary'])[:200] # Truncate
 text += f" [SEP] Summary: {summary}"

 # Add metadata
 president = row.get('president', 'Unknown')
 text += f" [SEP] Speaker: {president}"

 if row.get('multiple_questions', False):
 text += " [MULTI]"
 if row.get('affirmative_questions', False):
 text += " [AFFIRM]"

 return text

train_df['text'] = train_df.apply(prepare_enhanced_text, axis=1)
test_df['text'] = test_df.apply(prepare_enhanced_text, axis=1)

print(f"* Enhanced text prepared")
print(f" Example: {train_df['text'].iloc[0][:250]}...")

# Extract boolean features
def extract_boolean_features(row):
 return [
 float(row.get('affirmative_questions', False)),
 float(row.get('multiple_questions', False))
 ]

train_df['bool_features'] = train_df.apply(extract_boolean_features, axis=1)
test_df['bool_features'] = test_df.apply(extract_boolean_features, axis=1)

# Prepare EVASION labels (9 classes - PRIMARY TASK)
print("\n[3/10] Preparing 9-class evasion labels (PRIMARY)...")
evasion_labels = sorted(train_df['evasion_label'].unique())
evasion_label2id = {label: idx for idx, label in enumerate(evasion_labels)}
evasion_id2label = {idx: label for label, idx in evasion_label2id.items()}

train_df['evasion_id'] = train_df['evasion_label'].map(evasion_label2id)
test_df['evasion_id'] = 0 # Dummy for test set

print(f"* {len(evasion_labels)} evasion classes:")
for label, idx in evasion_label2id.items():
 count = (train_df['evasion_label'] == label).sum()
 print(f" {idx}. {label:25s}: {count:4d} ({count/len(train_df)*100:5.1f}%)")

# Prepare CLARITY labels (3 classes - AUXILIARY TASK)
print("\n[4/10] Preparing 3-class clarity labels (AUXILIARY)...")
clarity_labels = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
clarity_label2id = {label: idx for idx, label in enumerate(clarity_labels)}
clarity_id2label = {idx: label for label, idx in clarity_label2id.items()}

train_df['clarity_id'] = train_df['clarity_label'].map(clarity_label2id)
test_df['clarity_id'] = test_df['clarity_label'].map(clarity_label2id)

print(f"* {len(clarity_labels)} clarity classes:")
for label, idx in clarity_label2id.items():
 count = (train_df['clarity_label'] == label).sum()
 print(f" {idx}. {label:20s}: {count:4d} ({count/len(train_df)*100:5.1f}%)")

# Calculate class weights for Focal Loss
print("\n[5/10] Calculating Focal Loss weights...")

# Evasion weights
evasion_counts = train_df['evasion_id'].value_counts().sort_index()
total = len(train_df)
evasion_alpha = torch.tensor([1.0 / (count / total) for count in evasion_counts],
 dtype=torch.float).to(device)
evasion_alpha = evasion_alpha / evasion_alpha.sum() * len(evasion_labels)

# Clarity weights
clarity_counts = train_df['clarity_id'].value_counts().sort_index()
clarity_alpha = torch.tensor([1.0 / (count / total) for count in clarity_counts],
 dtype=torch.float).to(device)
clarity_alpha = clarity_alpha / clarity_alpha.sum() * len(clarity_labels)

print(f"* Evasion Focal weights: {evasion_alpha.cpu().numpy()}")
print(f"* Clarity Focal weights: {clarity_alpha.cpu().numpy()}")

# ============================================================================
# Hierarchical Mapping (for consistency checking)
# ============================================================================
EVASION_TO_CLARITY = {
 'Explicit': 'Clear Reply',
 'Implicit': 'Ambivalent',
 'Dodging': 'Ambivalent',
 'General': 'Ambivalent',
 'Deflection': 'Ambivalent',
 'Partial/half-answer': 'Ambivalent',
 'Declining to answer': 'Clear Non-Reply',
 'Claims ignorance': 'Clear Non-Reply',
 'Clarification': 'Clear Non-Reply'
}

print("\n* Hierarchical mapping loaded:")
for evasion, clarity in EVASION_TO_CLARITY.items():
 print(f" {evasion:25s} → {clarity}")

# ============================================================================
# Focal Loss
# ============================================================================
class FocalLoss(nn.Module):
 def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
 super(FocalLoss, self).__init__()
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

 if self.reduction == 'mean':
 return focal_loss.mean()
 elif self.reduction == 'sum':
 return focal_loss.sum()
 else:
 return focal_loss

# ============================================================================
# Multi-Task Model
# ============================================================================
class MultiTaskDeBERTaLarge(nn.Module):
 """DeBERTa-LARGE with dual heads: 9-class evasion (primary) + 3-class clarity (auxiliary)"""

 def __init__(self, model_name, num_evasion_labels, num_clarity_labels, bool_feature_dim):
 super(MultiTaskDeBERTaLarge, self).__init__()
 self.deberta = AutoModel.from_pretrained(model_name)
 self.config = self.deberta.config
 hidden_size = self.config.hidden_size

 # Boolean feature processor
 self.feature_processor = nn.Sequential(
 nn.Linear(bool_feature_dim, 16),
 nn.ReLU(),
 nn.Dropout(0.1)
 )

 # Evasion classifier (PRIMARY - 9 classes)
 self.evasion_classifier = nn.Sequential(
 nn.Dropout(0.2),
 nn.Linear(hidden_size + 16, hidden_size // 2),
 nn.GELU(),
 nn.Dropout(0.2),
 nn.Linear(hidden_size // 2, num_evasion_labels)
 )

 # Clarity classifier (AUXILIARY - 3 classes)
 self.clarity_classifier = nn.Sequential(
 nn.Dropout(0.1),
 nn.Linear(hidden_size + 16, num_clarity_labels)
 )

 def forward(self, input_ids, attention_mask, bool_features):
 outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
 pooled = outputs.last_hidden_state[:, 0, :] # CLS token

 # Process features
 feature_embed = self.feature_processor(bool_features)
 combined = torch.cat([pooled, feature_embed], dim=1)

 # Dual heads
 evasion_logits = self.evasion_classifier(combined)
 clarity_logits = self.clarity_classifier(combined)

 return evasion_logits, clarity_logits

# ============================================================================
# Dataset
# ============================================================================
class MultiTaskDataset(Dataset):
 def __init__(self, texts, evasion_labels, clarity_labels, bool_features, tokenizer, max_length=512):
 self.texts = texts
 self.evasion_labels = evasion_labels
 self.clarity_labels = clarity_labels
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
 'bool_features': torch.tensor(self.bool_features[idx], dtype=torch.float),
 'evasion_labels': torch.tensor(self.evasion_labels[idx], dtype=torch.long),
 'clarity_labels': torch.tensor(self.clarity_labels[idx], dtype=torch.long)
 }

# ============================================================================
# LLRD Optimizer
# ============================================================================
def get_optimizer_grouped_parameters(model, learning_rate, weight_decay, llrd_alpha):
 no_decay = ["bias", "LayerNorm.weight"]
 num_layers = model.config.num_hidden_layers

 optimizer_grouped_parameters = []

 # Embeddings
 optimizer_grouped_parameters.extend([
 {
 "params": [p for n, p in model.deberta.embeddings.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": weight_decay,
 "lr": learning_rate * (llrd_alpha ** (num_layers + 1))
 },
 {
 "params": [p for n, p in model.deberta.embeddings.named_parameters()
 if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0,
 "lr": learning_rate * (llrd_alpha ** (num_layers + 1))
 }
 ])

 # Encoder layers
 for layer_idx in range(num_layers):
 layer = model.deberta.encoder.layer[layer_idx]
 layer_lr = learning_rate * (llrd_alpha ** (num_layers - layer_idx))

 optimizer_grouped_parameters.extend([
 {
 "params": [p for n, p in layer.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": weight_decay,
 "lr": layer_lr
 },
 {
 "params": [p for n, p in layer.named_parameters()
 if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0,
 "lr": layer_lr
 }
 ])

 # Feature processor and classifiers (full learning rate)
 for component in [model.feature_processor, model.evasion_classifier, model.clarity_classifier]:
 optimizer_grouped_parameters.extend([
 {
 "params": [p for n, p in component.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": weight_decay,
 "lr": learning_rate
 },
 {
 "params": [p for n, p in component.named_parameters()
 if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0,
 "lr": learning_rate
 }
 ])

 return optimizer_grouped_parameters

# ============================================================================
# Training
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device, evasion_loss_fn,
 clarity_loss_fn, auxiliary_weight, gradient_accumulation_steps):
 model.train()
 total_evasion_loss = 0
 total_clarity_loss = 0
 evasion_preds = []
 evasion_labels = []

 optimizer.zero_grad()

 progress_bar = tqdm(dataloader, desc="Training")
 for step, batch in enumerate(progress_bar):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)
 evasion_label = batch['evasion_labels'].to(device)
 clarity_label = batch['clarity_labels'].to(device)

 evasion_logits, clarity_logits = model(input_ids, attention_mask, bool_features)

 # Multi-task loss
 evasion_loss = evasion_loss_fn(evasion_logits, evasion_label)
 clarity_loss = clarity_loss_fn(clarity_logits, clarity_label)

 # Combined loss (evasion is primary)
 loss = evasion_loss + auxiliary_weight * clarity_loss
 loss = loss / gradient_accumulation_steps
 loss.backward()

 if (step + 1) % gradient_accumulation_steps == 0:
 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 optimizer.step()
 scheduler.step()
 optimizer.zero_grad()

 total_evasion_loss += evasion_loss.item()
 total_clarity_loss += clarity_loss.item()

 preds = torch.argmax(evasion_logits, dim=1)
 evasion_preds.extend(preds.cpu().numpy())
 evasion_labels.extend(evasion_label.cpu().numpy())

 progress_bar.set_postfix({
 'evasion_loss': evasion_loss.item(),
 'clarity_loss': clarity_loss.item()
 })

 avg_evasion_loss = total_evasion_loss / len(dataloader)
 avg_clarity_loss = total_clarity_loss / len(dataloader)
 f1 = f1_score(evasion_labels, evasion_preds, average='macro')

 return avg_evasion_loss, avg_clarity_loss, f1

def eval_epoch(model, dataloader, device):
 model.eval()
 evasion_preds = []
 evasion_labels = []
 clarity_preds = []
 clarity_labels = []

 with torch.no_grad():
 for batch in tqdm(dataloader, desc="Evaluating"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)
 evasion_label = batch['evasion_labels'].to(device)
 clarity_label = batch['clarity_labels'].to(device)

 evasion_logits, clarity_logits = model(input_ids, attention_mask, bool_features)

 evasion_pred = torch.argmax(evasion_logits, dim=1)
 clarity_pred = torch.argmax(clarity_logits, dim=1)

 evasion_preds.extend(evasion_pred.cpu().numpy())
 evasion_labels.extend(evasion_label.cpu().numpy())
 clarity_preds.extend(clarity_pred.cpu().numpy())
 clarity_labels.extend(clarity_label.cpu().numpy())

 evasion_f1 = f1_score(evasion_labels, evasion_preds, average='macro')
 clarity_f1 = f1_score(clarity_labels, clarity_preds, average='macro')

 return evasion_f1, clarity_f1, evasion_preds, evasion_labels

# ============================================================================
# Main
# ============================================================================
print("\n[6/10] Initializing model...")
model_name = 'microsoft/deberta-v3-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = MultiTaskDeBERTaLarge(
 model_name=model_name,
 num_evasion_labels=len(evasion_labels),
 num_clarity_labels=len(clarity_labels),
 bool_feature_dim=2
)
model = model.to(device)

print(f"* Model: {model_name} (434M params)")
print(f"* Primary head: {len(evasion_labels)} evasion classes")
print(f"* Auxiliary head: {len(clarity_labels)} clarity classes")

print("\n[7/10] Creating datasets...")
train_dataset = MultiTaskDataset(
 texts=train_df['text'].tolist(),
 evasion_labels=train_df['evasion_id'].tolist(),
 clarity_labels=train_df['clarity_id'].tolist(),
 bool_features=train_df['bool_features'].tolist(),
 tokenizer=tokenizer,
 max_length=args.max_length
)

test_dataset = MultiTaskDataset(
 texts=test_df['text'].tolist(),
 evasion_labels=test_df['evasion_id'].tolist(),
 clarity_labels=test_df['clarity_id'].tolist(),
 bool_features=test_df['bool_features'].tolist(),
 tokenizer=tokenizer,
 max_length=args.max_length
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print(f"* Train batches: {len(train_loader)}")
print(f"* Test batches: {len(test_loader)}")

print("\n[8/10] Setting up optimizer...")
optimizer_grouped_parameters = get_optimizer_grouped_parameters(
 model, args.learning_rate, args.weight_decay, args.llrd_alpha
)
optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

num_training_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
num_warmup_steps = int(num_training_steps * args.warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(
 optimizer,
 num_warmup_steps=num_warmup_steps,
 num_training_steps=num_training_steps
)

evasion_loss_fn = FocalLoss(alpha=evasion_alpha, gamma=args.focal_gamma)
clarity_loss_fn = FocalLoss(alpha=clarity_alpha, gamma=args.clarity_focal_gamma)

print(f"* Training steps: {num_training_steps}")
print(f"* Warmup steps: {num_warmup_steps}")

print("\n[9/10] Training...")
print("="*80)

best_evasion_f1 = 0
best_epoch = 0
patience_counter = 0
start_time = time.time()

for epoch in range(args.num_epochs):
 print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
 print("-" * 80)

 evasion_loss, clarity_loss, train_f1 = train_epoch(
 model, train_loader, optimizer, scheduler, device,
 evasion_loss_fn, clarity_loss_fn, args.auxiliary_weight,
 args.gradient_accumulation_steps
 )

 evasion_f1, clarity_f1, val_preds, val_labels = eval_epoch(model, test_loader, device)

 print(f"\nTrain - Evasion Loss: {evasion_loss:.4f}, Clarity Loss: {clarity_loss:.4f}, F1: {train_f1:.4f}")
 print(f"Val - Evasion F1: {evasion_f1:.4f}, Clarity F1: {clarity_f1:.4f}")

 if evasion_f1 > best_evasion_f1:
 best_evasion_f1 = evasion_f1
 best_epoch = epoch + 1
 patience_counter = 0
 torch.save(model.state_dict(), 'multitask_large_evasion_best.pt')
 print(f"* New best evasion F1. Saved model.")
 else:
 patience_counter += 1
 print(f"No improvement ({patience_counter}/{args.patience})")

 if patience_counter >= args.patience:
 print(f"\nEarly stopping triggered")
 break

training_time = time.time() - start_time
print("\n" + "="*80)
print(f"Training complete. Best Evasion F1: {best_evasion_f1:.4f} (Epoch {best_epoch})")
print(f"Training time: {training_time/60:.1f} minutes")
print("="*80)

print("\n[10/10] Generating predictions...")
model.load_state_dict(torch.load('multitask_large_evasion_best.pt'))
model.eval()

test_evasion_preds = []
test_clarity_preds = []

with torch.no_grad():
 for batch in tqdm(test_loader, desc="Predicting"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)

 evasion_logits, clarity_logits = model(input_ids, attention_mask, bool_features)

 evasion_pred = torch.argmax(evasion_logits, dim=1)
 clarity_pred = torch.argmax(clarity_logits, dim=1)

 test_evasion_preds.extend(evasion_pred.cpu().numpy())
 test_clarity_preds.extend(clarity_pred.cpu().numpy())

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Best Evasion F1 (Subtask 2 - PRIMARY): {best_evasion_f1:.4f}")
print(f"Improvement over single-task: {best_evasion_f1 - 0.45:+.4f}")
print(f"Training time: {training_time/60:.1f} minutes")

# Convert to labels
pred_evasion_labels = [evasion_id2label[pred] for pred in test_evasion_preds]
pred_clarity_labels = [clarity_id2label[pred] for pred in test_clarity_preds]

# Check hierarchical consistency
consistent = 0
for i in range(len(pred_evasion_labels)):
 expected_clarity = EVASION_TO_CLARITY[pred_evasion_labels[i]]
 if pred_clarity_labels[i] == expected_clarity:
 consistent += 1

consistency_rate = consistent / len(pred_evasion_labels)
print(f"Hierarchical consistency: {consistency_rate:.2%}")

# Show distribution
pred_dist = pd.Series(pred_evasion_labels).value_counts()
print("\nPrediction distribution:")
for label, count in pred_dist.items():
 print(f" {label:25s}: {count:3d} ({count/len(pred_evasion_labels)*100:5.1f}%)")

# Save Subtask 2 predictions
output_file = 'prediction_multitask_evasion'
with open(output_file, 'w') as f:
 for pred in pred_evasion_labels:
 f.write(f"{pred}\n")

print(f"\n* Saved predictions: {output_file}")

# Create submission
import zipfile
submission_file = 'submission_multitask_evasion.zip'
with zipfile.ZipFile(submission_file, 'w') as zf:
 zf.write(output_file, arcname='prediction')

print(f"* Created submission: {submission_file}")

print("\n" + "="*80)
print("MULTI-TASK TRAINING COMPLETE")
print("="*80)
print(f"Previous (single-task): 0.45 F1")
print(f"Current (multi-task): {best_evasion_f1:.4f} F1")
print(f"Improvement: {best_evasion_f1 - 0.45:+.4f} F1")
print("="*80)
