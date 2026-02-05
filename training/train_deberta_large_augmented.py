"""
DeBERTa-v3-LARGE with Augmented Data Training

Based on train_deberta_large.py with modifications for:
1. Loading augmented dataset with --data_dir argument
2. Synthetic sample weight downweighting with --synthetic_weight argument
3. Tracking is_synthetic field in dataset

Key differences from base model:
- Larger model (434M params vs 184M)
- Lower learning rate (2e-5 vs 3e-5)
- Smaller batch size (4 vs 8) with more gradient accumulation (8 vs 4)

Usage:
 python train_deberta_large_augmented.py --data_dir ./QEvasion_augmented --synthetic_weight 0.5
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
print("DeBERTa-v3-LARGE with AUGMENTED DATA + Focal Loss")
print("="*80)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='DeBERTa-LARGE with Augmented Data + Focal Loss')
parser.add_argument('--data_dir', type=str, default='./QEvasion_augmented',
 help='Path to augmented dataset (default: ./QEvasion_augmented)')
parser.add_argument('--synthetic_weight', type=float, default=0.5,
 help='Weight for synthetic samples in loss (default: 0.5)')
parser.add_argument('--learning_rate', type=float, default=2e-5,
 help='Learning rate (default: 2e-5, lower for large model)')
parser.add_argument('--batch_size', type=int, default=4,
 help='Batch size (default: 4, smaller for memory)')
parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
 help='Gradient accumulation (default: 8, maintain effective batch=32)')
parser.add_argument('--num_epochs', type=int, default=6, help='Max epochs (default: 6)')
parser.add_argument('--warmup_ratio', type=float, default=0.15, help='Warmup ratio (default: 0.15)')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
parser.add_argument('--llrd_alpha', type=float, default=0.9, help='LLRD alpha (default: 0.9)')
parser.add_argument('--max_length', type=int, default=512, help='Max sequence length (default: 512)')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma (default: 2.0)')
parser.add_argument('--patience', type=int, default=3, help='Early stopping patience (default: 3)')
parser.add_argument('--output_prefix', type=str, default='deberta_large_augmented',
 help='Prefix for output files (default: deberta_large_augmented)')

args = parser.parse_args()

# Use command-line arguments
DATA_DIR = args.data_dir
SYNTHETIC_WEIGHT = args.synthetic_weight
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
NUM_EPOCHS = args.num_epochs
WARMUP_RATIO = args.warmup_ratio
WEIGHT_DECAY = args.weight_decay
LLRD_ALPHA = args.llrd_alpha
MAX_LENGTH = args.max_length
FOCAL_GAMMA = args.focal_gamma
PATIENCE = args.patience
OUTPUT_PREFIX = args.output_prefix
FOCAL_ALPHA = None # Will compute from class distribution

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
 print(f" GPU: {torch.cuda.get_device_name(0)}")
 print(f" Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\n{'='*80}")
print("HYPERPARAMETERS (LARGE MODEL)")
print(f"{'='*80}")
print(f"Model: DeBERTa-v3-LARGE (434M params)")
print(f"Data Directory: {DATA_DIR}")
print(f"Synthetic Sample Weight: {SYNTHETIC_WEIGHT}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Batch Size (per device): {BATCH_SIZE}")
print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Max Epochs: {NUM_EPOCHS}")
print(f"Warmup Ratio: {WARMUP_RATIO:.1%}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"LLRD Alpha: {LLRD_ALPHA}")
print(f"Early Stopping Patience: {PATIENCE}")
print(f"Max Length: {MAX_LENGTH}")
print(f"Focal Loss Gamma: {FOCAL_GAMMA}")

# Load dataset
print(f"\n[1/9] Loading augmented dataset from {DATA_DIR}...")
dataset = load_from_disk(DATA_DIR)
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Check for synthetic samples
if 'is_synthetic' in train_df.columns:
 n_synthetic = train_df['is_synthetic'].sum()
 n_original = len(train_df) - n_synthetic
 print(f" Original samples: {n_original}")
 print(f" Synthetic samples: {n_synthetic}")
else:
 print(" Note: is_synthetic column not found, treating all as original")
 train_df['is_synthetic'] = False

# Prepare text
print("\n[2/9] Preparing text and features...")
def prepare_text_transformer(row):
 """Format: Question: {q} [SEP] Answer: {a}"""
 question = row['question']
 answer = row['interview_answer']
 return f"Question: {question} [SEP] Answer: {answer}"

train_df['text'] = train_df.apply(prepare_text_transformer, axis=1)
test_df['text'] = test_df.apply(prepare_text_transformer, axis=1)

# Extract boolean features (only the 2 statistically significant ones)
def extract_boolean_features(row):
 """Extract 2 statistically significant boolean features as floats"""
 return [
 float(row.get('affirmative_questions', 0)),
 float(row.get('multiple_questions', 0))
 ]

train_df['bool_features'] = train_df.apply(extract_boolean_features, axis=1)
test_df['bool_features'] = test_df.apply(extract_boolean_features, axis=1)

print(f"Boolean feature dimension: 2")
if 'affirmative_questions' in train_df.columns:
 print(f" - affirmative_questions: {train_df['affirmative_questions'].sum()} / {len(train_df)}")
if 'multiple_questions' in train_df.columns:
 print(f" - multiple_questions: {train_df['multiple_questions'].sum()} / {len(train_df)}")

# Create label mapping
label_list = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

train_df['label_id'] = train_df['clarity_label'].map(label2id)
test_df['label_id'] = test_df['clarity_label'].map(label2id)

print(f"Label mapping: {label2id}")

# Calculate class weights for Focal Loss alpha
class_counts = train_df['label_id'].value_counts().sort_index()
total = len(train_df)
# Focal Loss alpha: inverse frequency
focal_alpha = torch.tensor([1.0 / (count / total) for count in class_counts],
 dtype=torch.float).to(device)
focal_alpha = focal_alpha / focal_alpha.sum() * len(focal_alpha) # Normalize

print(f"\nClass distribution:")
for label, count in zip(label_list, class_counts):
 synthetic_count = len(train_df[(train_df['clarity_label'] == label) & (train_df['is_synthetic'] == True)])
 print(f" {label}: {count} ({count/total*100:.1f}%) [{synthetic_count} synthetic]")
print(f"Focal Loss alpha: {focal_alpha.cpu().numpy()}")

# ============================================================================
# Focal Loss Implementation with Sample Weights
# ============================================================================
class FocalLossWithSampleWeights(nn.Module):
 """
 Focal Loss with per-sample weights for downweighting synthetic samples.
 """
 def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
 super(FocalLossWithSampleWeights, self).__init__()
 self.alpha = alpha
 self.gamma = gamma
 self.reduction = reduction

 def forward(self, logits, targets, sample_weights=None):
 ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
 p_t = torch.exp(-ce_loss)
 focal_term = (1 - p_t) ** self.gamma

 if self.alpha is not None:
 alpha_t = self.alpha[targets]
 focal_loss = alpha_t * focal_term * ce_loss
 else:
 focal_loss = focal_term * ce_loss

 if sample_weights is not None:
 focal_loss = focal_loss * sample_weights

 if self.reduction == 'mean':
 if sample_weights is not None:
 return focal_loss.sum() / sample_weights.sum()
 return focal_loss.mean()
 elif self.reduction == 'sum':
 return focal_loss.sum()
 else:
 return focal_loss

# ============================================================================
# Custom Model with Boolean Features
# ============================================================================
class DeBERTaWithFeatures(nn.Module):
 """DeBERTa model that accepts additional boolean features."""
 def __init__(self, model_name, num_labels, bool_feature_dim):
 super(DeBERTaWithFeatures, self).__init__()
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

# Custom Dataset with Sample Weights
class ClarityDatasetWithWeights(Dataset):
 def __init__(self, texts, labels, bool_features, is_synthetic, tokenizer,
 max_length=512, synthetic_weight=0.5):
 self.texts = texts
 self.labels = labels
 self.bool_features = bool_features
 self.is_synthetic = is_synthetic
 self.tokenizer = tokenizer
 self.max_length = max_length
 self.synthetic_weight = synthetic_weight

 def __len__(self):
 return len(self.texts)

 def __getitem__(self, idx):
 text = str(self.texts[idx])
 label = self.labels[idx]
 bool_feat = self.bool_features[idx]
 is_synth = self.is_synthetic[idx]

 encoding = self.tokenizer(
 text,
 max_length=self.max_length,
 padding='max_length',
 truncation=True,
 return_tensors='pt'
 )

 sample_weight = self.synthetic_weight if is_synth else 1.0

 return {
 'input_ids': encoding['input_ids'].flatten(),
 'attention_mask': encoding['attention_mask'].flatten(),
 'bool_features': torch.tensor(bool_feat, dtype=torch.float),
 'labels': torch.tensor(label, dtype=torch.long),
 'sample_weight': torch.tensor(sample_weight, dtype=torch.float),
 'is_synthetic': torch.tensor(is_synth, dtype=torch.bool)
 }

# ============================================================================
# Layer-wise Learning Rate Decay (LLRD)
# ============================================================================
def get_optimizer_grouped_parameters(model, learning_rate, weight_decay, llrd_alpha):
 """Apply layer-wise learning rate decay"""
 no_decay = ["bias", "LayerNorm.weight"]
 num_layers = model.config.num_hidden_layers

 optimizer_grouped_parameters = []

 # Embeddings
 optimizer_grouped_parameters.append({
 "params": [p for n, p in model.deberta.embeddings.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": weight_decay,
 "lr": learning_rate * (llrd_alpha ** (num_layers + 1))
 })
 optimizer_grouped_parameters.append({
 "params": [p for n, p in model.deberta.embeddings.named_parameters()
 if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0,
 "lr": learning_rate * (llrd_alpha ** (num_layers + 1))
 })

 # Encoder layers (apply LLRD)
 for layer_idx in range(num_layers):
 layer = model.deberta.encoder.layer[layer_idx]
 layer_lr = learning_rate * (llrd_alpha ** (num_layers - layer_idx))

 optimizer_grouped_parameters.append({
 "params": [p for n, p in layer.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": weight_decay,
 "lr": layer_lr
 })
 optimizer_grouped_parameters.append({
 "params": [p for n, p in layer.named_parameters()
 if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0,
 "lr": layer_lr
 })

 # Feature processor
 optimizer_grouped_parameters.append({
 "params": [p for n, p in model.feature_processor.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": weight_decay,
 "lr": learning_rate
 })
 optimizer_grouped_parameters.append({
 "params": [p for n, p in model.feature_processor.named_parameters()
 if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0,
 "lr": learning_rate
 })

 # Classifier head
 optimizer_grouped_parameters.append({
 "params": [p for n, p in model.classifier.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": weight_decay,
 "lr": learning_rate
 })
 optimizer_grouped_parameters.append({
 "params": [p for n, p in model.classifier.named_parameters()
 if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0,
 "lr": learning_rate
 })

 return optimizer_grouped_parameters

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device, focal_loss,
 gradient_accumulation_steps):
 model.train()
 total_loss = 0
 predictions = []
 true_labels = []

 optimizer.zero_grad()

 progress_bar = tqdm(dataloader, desc="Training")
 for step, batch in enumerate(progress_bar):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)
 labels = batch['labels'].to(device)
 sample_weights = batch['sample_weight'].to(device)

 logits = model(input_ids=input_ids, attention_mask=attention_mask,
 bool_features=bool_features)

 loss = focal_loss(logits, labels, sample_weights)

 loss = loss / gradient_accumulation_steps
 loss.backward()

 if (step + 1) % gradient_accumulation_steps == 0:
 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 optimizer.step()
 scheduler.step()
 optimizer.zero_grad()

 total_loss += loss.item() * gradient_accumulation_steps

 preds = torch.argmax(logits, dim=1)
 predictions.extend(preds.cpu().numpy())
 true_labels.extend(labels.cpu().numpy())

 progress_bar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})

 avg_loss = total_loss / len(dataloader)
 f1 = f1_score(true_labels, predictions, average='macro')

 return avg_loss, f1

# Validation function
def eval_epoch(model, dataloader, device, focal_loss):
 model.eval()
 total_loss = 0
 predictions = []
 true_labels = []

 with torch.no_grad():
 for batch in tqdm(dataloader, desc="Evaluating"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)
 labels = batch['labels'].to(device)

 logits = model(input_ids=input_ids, attention_mask=attention_mask,
 bool_features=bool_features)

 loss = focal_loss(logits, labels, sample_weights=None)
 total_loss += loss.item()

 preds = torch.argmax(logits, dim=1)
 predictions.extend(preds.cpu().numpy())
 true_labels.extend(labels.cpu().numpy())

 avg_loss = total_loss / len(dataloader)
 accuracy = accuracy_score(true_labels, predictions)
 f1 = f1_score(true_labels, predictions, average='macro')

 return avg_loss, accuracy, f1, predictions, true_labels

# ============================================================================
# Main Training Loop
# ============================================================================
print("\n[3/9] Initializing tokenizer and model...")
model_name = 'microsoft/deberta-v3-large'
print(f"Using LARGE model: {model_name} (434M params, 2.4x larger than base)")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = DeBERTaWithFeatures(
 model_name=model_name,
 num_labels=len(label_list),
 bool_feature_dim=2
)
model = model.to(device)

print(f"Model: {model_name}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n[4/9] Creating datasets and dataloaders...")

train_is_synthetic = train_df['is_synthetic'].tolist() if 'is_synthetic' in train_df.columns else [False] * len(train_df)
test_is_synthetic = [False] * len(test_df)

train_dataset = ClarityDatasetWithWeights(
 texts=train_df['text'].tolist(),
 labels=train_df['label_id'].tolist(),
 bool_features=train_df['bool_features'].tolist(),
 is_synthetic=train_is_synthetic,
 tokenizer=tokenizer,
 max_length=MAX_LENGTH,
 synthetic_weight=SYNTHETIC_WEIGHT
)

test_dataset = ClarityDatasetWithWeights(
 texts=test_df['text'].tolist(),
 labels=test_df['label_id'].tolist(),
 bool_features=test_df['bool_features'].tolist(),
 is_synthetic=test_is_synthetic,
 tokenizer=tokenizer,
 max_length=MAX_LENGTH,
 synthetic_weight=1.0
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

print("\n[5/9] Setting up optimizer and scheduler...")
optimizer_grouped_parameters = get_optimizer_grouped_parameters(
 model, LEARNING_RATE, WEIGHT_DECAY, LLRD_ALPHA
)
optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

num_training_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(
 optimizer,
 num_warmup_steps=num_warmup_steps,
 num_training_steps=num_training_steps
)

print(f"Total training steps: {num_training_steps}")
print(f"Warmup steps: {num_warmup_steps}")

# Initialize Focal Loss with sample weight support
focal_loss = FocalLossWithSampleWeights(alpha=focal_alpha, gamma=FOCAL_GAMMA)

print("\n[6/9] Training model...")
print("="*80)

best_f1 = 0
best_epoch = 0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
 print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
 print("-" * 80)

 train_loss, train_f1 = train_epoch(
 model, train_loader, optimizer, scheduler, device, focal_loss,
 GRADIENT_ACCUMULATION_STEPS
 )

 val_loss, val_accuracy, val_f1, val_preds, val_labels = eval_epoch(
 model, test_loader, device, focal_loss
 )

 print(f"\nTrain Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
 print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")

 if val_f1 > best_f1:
 best_f1 = val_f1
 best_epoch = epoch + 1
 patience_counter = 0

 model_path = f'{OUTPUT_PREFIX}_best.pt'
 torch.save(model.state_dict(), model_path)
 print(f"New best F1. Saved model to {model_path}")
 else:
 patience_counter += 1
 print(f"No improvement ({patience_counter}/{PATIENCE})")

 if patience_counter >= PATIENCE:
 print(f"\nEarly stopping triggered after {epoch + 1} epochs")
 break

print("\n" + "="*80)
print(f"Training complete. Best F1: {best_f1:.4f} (Epoch {best_epoch})")
print("="*80)

# Load best model for final evaluation
print("\n[7/9] Loading best model for final evaluation...")
model.load_state_dict(torch.load(f'{OUTPUT_PREFIX}_best.pt'))

val_loss, val_accuracy, val_f1, val_preds, val_labels = eval_epoch(
 model, test_loader, device, focal_loss
)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Accuracy: {val_accuracy:.4f}")
print(f"Macro F1: {val_f1:.4f}")

print("\nClassification Report:")
print(classification_report(val_labels, val_preds, target_names=label_list, digits=4))

# Generate predictions
print("\n[8/9] Generating predictions...")
pred_labels = [id2label[pred] for pred in val_preds]

# Save predictions
output_file = f'prediction_{OUTPUT_PREFIX}'
pd.Series(pred_labels).to_csv(output_file, index=False, header=False)
print(f"Saved predictions: {output_file}")

# Create submission zip
print("\n[9/9] Creating submission file...")
import zipfile
submission_file = f'submission_{OUTPUT_PREFIX}.zip'
with zipfile.ZipFile(submission_file, 'w') as zf:
 zf.write(output_file, arcname='prediction')

print(f"Created submission: {submission_file}")

print("\n" + "="*80)
print("DeBERTa-LARGE AUGMENTED DATA TRAINING COMPLETE")
print("="*80)
print(f"Best Validation F1: {best_f1:.4f}")
print(f"Submission file: {submission_file}")

# Improvement analysis
baseline_f1 = 0.67 # Large model baseline
improvement = best_f1 - baseline_f1
print(f"\nImprovement over large model baseline (0.67): {improvement:+.4f} ({improvement/baseline_f1*100:+.1f}%)")

if best_f1 >= 0.69:
 print(f"Reached target threshold (0.69)")
elif best_f1 >= 0.68:
 print(f"Close to target threshold")
 print(f" Gap remaining: {0.69 - best_f1:.4f}")
else:
 print(f"Below expected improvement")
 print(f" Gap remaining: {0.69 - best_f1:.4f}")

print("="*80)
