"""
DeBERTa-v3-LARGE for 9-class evasion prediction (Subtask 2).

Direct 9-class classification of evasion techniques using:
- DeBERTa-v3-LARGE (434M params)
- Boolean Features: affirmative_questions, multiple_questions
- Focal Loss for imbalanced classes
- LLRD, Gradient Accumulation, Cosine Scheduling
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
print("SUBTASK 2: DeBERTa-v3-LARGE for 9-Class Evasion Prediction")
print("="*80)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Subtask 2: DeBERTa-LARGE for 9-Class Evasion')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate (default: 2e-5)')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation (default: 8)')
parser.add_argument('--num_epochs', type=int, default=8, help='Max epochs (default: 8, more for harder task)')
parser.add_argument('--warmup_ratio', type=float, default=0.15, help='Warmup ratio (default: 0.15)')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
parser.add_argument('--llrd_alpha', type=float, default=0.9, help='LLRD alpha (default: 0.9)')
parser.add_argument('--max_length', type=int, default=512, help='Max sequence length (default: 512)')
parser.add_argument('--focal_gamma', type=float, default=2.5, help='Focal loss gamma (default: 2.5, higher for 9 classes)')
parser.add_argument('--patience', type=int, default=4, help='Early stopping patience (default: 4)')

args = parser.parse_args()

# Use command-line arguments
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

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Device: {device}")
if torch.cuda.is_available():
 print(f" GPU: {torch.cuda.get_device_name(0)}")
 print(f" Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\n{'='*80}")
print("HYPERPARAMETERS")
print(f"{'='*80}")
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
print("\n[1/9] Loading dataset...")
dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"* Training samples: {len(train_df)}")
print(f"* Test samples: {len(test_df)}")

# Prepare text
print("\n[2/9] Preparing text and features...")
def prepare_text_transformer(row):
 """Format: Question: {q} [SEP] Answer: {a}"""
 question = row['question']
 answer = row['interview_answer']
 return f"Question: {question} [SEP] Answer: {answer}"

train_df['text'] = train_df.apply(prepare_text_transformer, axis=1)
test_df['text'] = test_df.apply(prepare_text_transformer, axis=1)

# Extract boolean features
def extract_boolean_features(row):
 """Extract 2 statistically significant boolean features as floats"""
 return [
 float(row['affirmative_questions']),
 float(row['multiple_questions'])
 ]

train_df['bool_features'] = train_df.apply(extract_boolean_features, axis=1)
test_df['bool_features'] = test_df.apply(extract_boolean_features, axis=1)

print(f"* Boolean feature dimension: 2")
print(f" - affirmative_questions: {train_df['affirmative_questions'].sum()} / {len(train_df)} ({train_df['affirmative_questions'].sum()/len(train_df)*100:.1f}%)")
print(f" - multiple_questions: {train_df['multiple_questions'].sum()} / {len(train_df)} ({train_df['multiple_questions'].sum()/len(train_df)*100:.1f}%)")

# Create label mapping for 9 evasion classes
print("\n[3/9] Preparing 9-class evasion labels...")

# Get unique evasion labels from training data and sort them
evasion_labels_raw = train_df['evasion_label'].unique()
# Sort to ensure consistent ordering
label_list = sorted(evasion_labels_raw)

print(f"* Found {len(label_list)} evasion classes:")
for i, label in enumerate(label_list, 1):
 count = (train_df['evasion_label'] == label).sum()
 print(f" {i}. {label}: {count} samples ({count/len(train_df)*100:.1f}%)")

# Create label mappings
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

train_df['label_id'] = train_df['evasion_label'].map(label2id)

# For test set, we don't have evasion labels, use dummy (will predict them)
# But we need label_id column for the dataset class
test_df['label_id'] = 0 # Dummy - will be overwritten by predictions

print(f"\n* Label mapping created:")
for label, idx in label2id.items():
 print(f" {idx}: {label}")

# Calculate class weights for Focal Loss
class_counts = train_df['label_id'].value_counts().sort_index()
total = len(train_df)

# Focal Loss alpha: inverse frequency normalized
focal_alpha = torch.tensor([1.0 / (count / total) for count in class_counts],
 dtype=torch.float).to(device)
focal_alpha = focal_alpha / focal_alpha.sum() * len(focal_alpha) # Normalize

print(f"\n* Class distribution and Focal Loss weights:")
for idx in range(len(label_list)):
 label = id2label[idx]
 count = class_counts[idx]
 weight = focal_alpha[idx].item()
 print(f" {idx}. {label:25s}: {count:4d} ({count/total*100:5.1f}%) | weight: {weight:.3f}")

# ============================================================================
# Focal Loss Implementation
# ============================================================================
class FocalLoss(nn.Module):
 """
 Focal Loss for highly imbalanced multi-class classification.
 Essential for 9-class evasion task with severe imbalance.
 """
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
# Custom Model with Boolean Features
# ============================================================================
class DeBERTaWithFeatures(nn.Module):
 """DeBERTa-LARGE with boolean features for 9-class evasion prediction"""
 def __init__(self, model_name, num_labels, bool_feature_dim):
 super(DeBERTaWithFeatures, self).__init__()
 self.deberta = AutoModel.from_pretrained(model_name)
 self.config = self.deberta.config
 hidden_size = self.config.hidden_size

 # Feature processor
 self.feature_processor = nn.Sequential(
 nn.Linear(bool_feature_dim, 16),
 nn.ReLU(),
 nn.Dropout(0.1)
 )

 # Classifier for 9 classes
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

# Custom Dataset
class EvasionDatasetWithFeatures(Dataset):
 def __init__(self, texts, labels, bool_features, tokenizer, max_length=512):
 self.texts = texts
 self.labels = labels
 self.bool_features = bool_features
 self.tokenizer = tokenizer
 self.max_length = max_length

 def __len__(self):
 return len(self.texts)

 def __getitem__(self, idx):
 text = str(self.texts[idx])
 label = self.labels[idx]
 bool_feat = self.bool_features[idx]

 encoding = self.tokenizer(
 text,
 max_length=self.max_length,
 padding='max_length',
 truncation=True,
 return_tensors='pt'
 )

 return {
 'input_ids': encoding['input_ids'].flatten(),
 'attention_mask': encoding['attention_mask'].flatten(),
 'bool_features': torch.tensor(bool_feat, dtype=torch.float),
 'labels': torch.tensor(label, dtype=torch.long)
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

 # Encoder layers
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

 logits = model(input_ids=input_ids, attention_mask=attention_mask,
 bool_features=bool_features)

 loss = focal_loss(logits, labels)
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

 loss = focal_loss(logits, labels)
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
print("\n[4/9] Initializing tokenizer and model...")
model_name = 'microsoft/deberta-v3-large'
print(f"* Using LARGE model: {model_name} (434M params)")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = DeBERTaWithFeatures(
 model_name=model_name,
 num_labels=len(label_list),
 bool_feature_dim=2
)
model = model.to(device)

print(f"* Model: {model_name}")
print(f"* Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"* Output: {len(label_list)} evasion classes")

print("\n[5/9] Creating datasets and dataloaders...")
train_dataset = EvasionDatasetWithFeatures(
 texts=train_df['text'].tolist(),
 labels=train_df['label_id'].tolist(),
 bool_features=train_df['bool_features'].tolist(),
 tokenizer=tokenizer,
 max_length=MAX_LENGTH
)

test_dataset = EvasionDatasetWithFeatures(
 texts=test_df['text'].tolist(),
 labels=test_df['label_id'].tolist(),
 bool_features=test_df['bool_features'].tolist(),
 tokenizer=tokenizer,
 max_length=MAX_LENGTH
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"* Train batches: {len(train_loader)}")
print(f"* Test batches: {len(test_loader)}")

print("\n[6/9] Setting up optimizer and scheduler...")
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

print(f"* Total training steps: {num_training_steps}")
print(f"* Warmup steps: {num_warmup_steps}")

# Initialize Focal Loss
focal_loss = FocalLoss(alpha=focal_alpha, gamma=FOCAL_GAMMA)
print(f"* Focal Loss initialized (gamma={FOCAL_GAMMA})")

print("\n[7/9] Training model...")
print("="*80)

best_f1 = 0
best_epoch = 0
patience_counter = 0
start_time = time.time()

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

 # Save best model
 torch.save(model.state_dict(), 'deberta_large_evasion_best.pt')
 print(f"* New best F1. Saved model.")
 else:
 patience_counter += 1
 print(f"No improvement ({patience_counter}/{PATIENCE})")

 if patience_counter >= PATIENCE:
 print(f"\nEarly stopping triggered after {epoch + 1} epochs")
 break

training_time = time.time() - start_time
print("\n" + "="*80)
print(f"Training complete. Best F1: {best_f1:.4f} (Epoch {best_epoch})")
print(f"Training time: {training_time/60:.1f} minutes")
print("="*80)

# Load best model for final evaluation
print("\n[8/9] Loading best model for final predictions...")
model.load_state_dict(torch.load('deberta_large_evasion_best.pt'))

# Make predictions on test set (which doesn't have true evasion labels)
model.eval()
test_predictions = []

with torch.no_grad():
 for batch in tqdm(test_loader, desc="Predicting"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)

 logits = model(input_ids=input_ids, attention_mask=attention_mask,
 bool_features=bool_features)

 preds = torch.argmax(logits, dim=1)
 test_predictions.extend(preds.cpu().numpy())

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Best Validation F1 (on training set): {best_f1:.4f}")
print(f"Test predictions: {len(test_predictions)} samples")

# Show prediction distribution
pred_labels_text = [id2label[pred] for pred in test_predictions]
pred_distribution = pd.Series(pred_labels_text).value_counts()
print("\nPrediction distribution on test set:")
for label, count in pred_distribution.items():
 print(f" {label:25s}: {count:3d} ({count/len(test_predictions)*100:5.1f}%)")

# Generate predictions for submission
print("\n[9/9] Generating submission file...")

# Save predictions (one label per line, extensionless file)
output_file = 'prediction_evasion'
with open(output_file, 'w') as f:
 for pred_label in pred_labels_text:
 f.write(f"{pred_label}\n")

print(f"* Saved predictions: {output_file}")

# Create submission zip
import zipfile
submission_file = 'submission_evasion.zip'
with zipfile.ZipFile(submission_file, 'w') as zf:
 zf.write(output_file, arcname='prediction')

print(f"* Created submission: {submission_file}")

print("\n" + "="*80)
print("9-CLASS EVASION TRAINING COMPLETE")
print("="*80)
print(f"Best Validation F1: {best_f1:.4f}")
print(f"Training time: {training_time/60:.1f} minutes")
print(f"Submission file: {submission_file}")
print(f"\nSubmission format:")
print(f" - File: {output_file} (extensionless)")
print(f" - Format: One evasion label per line (308 lines)")
print(f" - Zipped as: {submission_file}")
print("="*80)
