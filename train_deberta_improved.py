"""
DeBERTa-v3-base with advanced fine-tuning:
- Layer-wise Learning Rate Decay (LLRD)
- Gradient Accumulation for larger effective batch size
- Cosine Annealing with Warmup
- Early Stopping
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
 AutoTokenizer,
 AutoModelForSequenceClassification,
 get_cosine_schedule_with_warmup
)
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm
import time
import argparse

print("="*80)
print("DeBERTa-v3-base with Advanced Optimization")
print("="*80)

# Parse arguments for hyperparameter search
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=3e-5,
 help='Peak learning rate (default: 3e-5)')
parser.add_argument('--batch_size', type=int, default=8,
 help='Per-device batch size (default: 8)')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
 help='Gradient accumulation steps (default: 4, effective batch=32)')
parser.add_argument('--num_epochs', type=int, default=6,
 help='Maximum epochs (default: 6)')
parser.add_argument('--warmup_ratio', type=float, default=0.15,
 help='Warmup ratio (default: 0.15 = 15%)')
parser.add_argument('--weight_decay', type=float, default=0.01,
 help='Weight decay (default: 0.01)')
parser.add_argument('--llrd_alpha', type=float, default=0.9,
 help='Layer-wise LR decay factor (default: 0.9)')
parser.add_argument('--patience', type=int, default=3,
 help='Early stopping patience (default: 3)')
parser.add_argument('--max_length', type=int, default=512,
 help='Max sequence length (default: 512)')
parser.add_argument('--reinit_layers', type=int, default=0,
 help='Number of top layers to reinitialize (default: 0)')

args = parser.parse_args()

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Device: {device}")
if torch.cuda.is_available():
 print(f" GPU: {torch.cuda.get_device_name(0)}")
 print(f" Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Print hyperparameters
print(f"\n{'='*80}")
print("HYPERPARAMETERS")
print(f"{'='*80}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Batch Size (per device): {args.batch_size}")
print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
print(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
print(f"Max Epochs: {args.num_epochs}")
print(f"Warmup Ratio: {args.warmup_ratio:.1%}")
print(f"Weight Decay: {args.weight_decay}")
print(f"LLRD Alpha: {args.llrd_alpha}")
print(f"Early Stopping Patience: {args.patience}")
print(f"Max Length: {args.max_length}")
print(f"Reinit Layers: {args.reinit_layers}")

# Load dataset
print("\n[1/8] Loading dataset...")
dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"* Training samples: {len(train_df)}")
print(f"* Test samples: {len(test_df)}")

# Prepare text
print("\n[2/8] Preparing text for DeBERTa...")
def prepare_text_transformer(row):
 """Format: Question: {q} [SEP] Answer: {a}"""
 question = row['question']
 answer = row['interview_answer']
 return f"Question: {question} [SEP] Answer: {answer}"

train_df['text'] = train_df.apply(prepare_text_transformer, axis=1)
test_df['text'] = test_df.apply(prepare_text_transformer, axis=1)

# Create label mapping
label_list = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

train_df['label_id'] = train_df['clarity_label'].map(label2id)
test_df['label_id'] = test_df['clarity_label'].map(label2id)

print(f"* Label mapping: {label2id}")

# Calculate class weights for imbalance
class_counts = train_df['label_id'].value_counts().sort_index()
total = len(train_df)
class_weights = torch.tensor([total / (len(class_counts) * count) for count in class_counts],
 dtype=torch.float).to(device)
print(f"\n* Class distribution:")
for label, count in zip(label_list, class_counts):
 print(f" {label}: {count} ({count/total*100:.1f}%)")
print(f"* Class weights: {class_weights.cpu().numpy()}")

# Custom Dataset
class ClarityDataset(Dataset):
 def __init__(self, texts, labels, tokenizer, max_length=512):
 self.texts = texts
 self.labels = labels
 self.tokenizer = tokenizer
 self.max_length = max_length

 def __len__(self):
 return len(self.texts)

 def __getitem__(self, idx):
 text = str(self.texts[idx])
 label = self.labels[idx]

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
 'labels': torch.tensor(label, dtype=torch.long)
 }

# ============================================================================
# Layer-wise Learning Rate Decay (LLRD)
# ============================================================================
def get_optimizer_grouped_parameters(model, learning_rate, weight_decay, llrd_alpha):
 """
 Apply layer-wise learning rate decay:
 - Lower layers get lower learning rates (they're more general)
 - Higher layers get higher learning rates (they're more task-specific)
 """
 no_decay = ["bias", "LayerNorm.weight"]

 # Get the number of layers
 num_layers = model.config.num_hidden_layers

 # Group parameters by layer
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

 # Classifier head (highest learning rate)
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

 # Pooler if exists
 if hasattr(model.deberta, 'pooler') and model.deberta.pooler is not None:
 optimizer_grouped_parameters.append({
 "params": [p for n, p in model.deberta.pooler.named_parameters()
 if not any(nd in n for nd in no_decay)],
 "weight_decay": weight_decay,
 "lr": learning_rate
 })
 optimizer_grouped_parameters.append({
 "params": [p for n, p in model.deberta.pooler.named_parameters()
 if any(nd in n for nd in no_decay)],
 "weight_decay": 0.0,
 "lr": learning_rate
 })

 return optimizer_grouped_parameters

# Training function with gradient accumulation
def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights,
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
 labels = batch['labels'].to(device)

 outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

 # Apply class weights
 loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
 loss = loss_fct(outputs.logits, labels)

 # Scale loss for gradient accumulation
 loss = loss / gradient_accumulation_steps

 total_loss += loss.item() * gradient_accumulation_steps

 loss.backward()

 # Update weights every gradient_accumulation_steps
 if (step + 1) % gradient_accumulation_steps == 0:
 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
 optimizer.step()
 scheduler.step()
 optimizer.zero_grad()

 preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
 predictions.extend(preds)
 true_labels.extend(labels.cpu().numpy())

 progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

 avg_loss = total_loss / len(dataloader)
 f1 = f1_score(true_labels, predictions, average='macro')

 return avg_loss, f1

# Evaluation function
def evaluate(model, dataloader, device):
 model.eval()
 predictions = []
 true_labels = []

 with torch.no_grad():
 for batch in tqdm(dataloader, desc="Evaluating"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 labels = batch['labels'].to(device)

 outputs = model(input_ids=input_ids, attention_mask=attention_mask)

 preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
 predictions.extend(preds)
 true_labels.extend(labels.cpu().numpy())

 return predictions, true_labels

# ============================================================================
# TRAIN DeBERTa-v3-base with Advanced Optimization
# ============================================================================

model_name = 'microsoft/deberta-v3-base'

print(f"\n{'='*80}")
print(f"Training: {model_name}")
print(f"{'='*80}")

# Initialize tokenizer and model
print(f"\n[3/8] Loading {model_name} tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
 model_name,
 num_labels=3,
 id2label=id2label,
 label2id=label2id,
 problem_type="single_label_classification"
)

# Reinitialize top layers if specified
if args.reinit_layers > 0:
 print(f"\n[3b/8] Reinitializing top {args.reinit_layers} layer(s)...")
 num_layers = model.config.num_hidden_layers
 for layer_idx in range(num_layers - args.reinit_layers, num_layers):
 for module in model.deberta.encoder.layer[layer_idx].modules():
 if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
 module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
 if isinstance(module, torch.nn.Linear) and module.bias is not None:
 module.bias.data.zero_()
 print(f"* Reinitialized layers {num_layers - args.reinit_layers} to {num_layers - 1}")

model.to(device)

print(f"* Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

# Create datasets
print(f"\n[4/8] Creating datasets...")
train_dataset = ClarityDataset(
 train_df['text'].values,
 train_df['label_id'].values,
 tokenizer,
 max_length=args.max_length
)
test_dataset = ClarityDataset(
 test_df['text'].values,
 test_df['label_id'].values,
 tokenizer,
 max_length=args.max_length
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

print(f"* Train batches: {len(train_loader)}")
print(f"* Test batches: {len(test_loader)}")

# Optimizer with LLRD
print(f"\n[5/8] Setting up optimizer with Layer-wise LR Decay...")
optimizer_grouped_parameters = get_optimizer_grouped_parameters(
 model,
 args.learning_rate,
 args.weight_decay,
 args.llrd_alpha
)

optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

# Calculate learning rates for each layer
num_layers = model.config.num_hidden_layers
print(f"\n* Layer-wise Learning Rates (LLRD alpha={args.llrd_alpha}):")
print(f" Embeddings: {args.learning_rate * (args.llrd_alpha ** (num_layers + 1)):.2e}")
for layer_idx in range(num_layers):
 layer_lr = args.learning_rate * (args.llrd_alpha ** (num_layers - layer_idx))
 if layer_idx < 3 or layer_idx >= num_layers - 3:
 print(f" Layer {layer_idx:2d}: {layer_lr:.2e}")
 elif layer_idx == 3:
 print(f" ...")
print(f" Classifier: {args.learning_rate:.2e}")

# Scheduler: Cosine with Warmup
total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.num_epochs
warmup_steps = int(total_steps * args.warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(
 optimizer,
 num_warmup_steps=warmup_steps,
 num_training_steps=total_steps
)

print(f"\n* Scheduler: Cosine with Warmup")
print(f"* Warmup steps: {warmup_steps} ({args.warmup_ratio:.1%})")
print(f"* Total steps: {total_steps}")

# Training loop with early stopping
print(f"\n[6/8] Training for up to {args.num_epochs} epochs (early stopping patience={args.patience})...")
start_time = time.time()

best_f1 = 0
best_model_state = None
best_epoch = 0
patience_counter = 0

history = {
 'train_loss': [],
 'train_f1': [],
 'val_f1': [],
 'val_acc': []
}

for epoch in range(args.num_epochs):
 print(f"\n{'='*80}")
 print(f"Epoch {epoch + 1}/{args.num_epochs}")
 print(f"{'='*80}")

 train_loss, train_f1 = train_epoch(
 model, train_loader, optimizer, scheduler, device,
 class_weights, args.gradient_accumulation_steps
 )
 print(f"\n* Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")

 history['train_loss'].append(train_loss)
 history['train_f1'].append(train_f1)

 # Evaluate on test set (dev set)
 test_preds, test_labels = evaluate(model, test_loader, device)
 test_f1 = f1_score(test_labels, test_preds, average='macro')
 test_acc = accuracy_score(test_labels, test_preds)

 history['val_f1'].append(test_f1)
 history['val_acc'].append(test_acc)

 print(f"* Dev F1: {test_f1:.4f}, Dev Acc: {test_acc:.4f}")

 # Save best model
 if test_f1 > best_f1:
 best_f1 = test_f1
 best_model_state = model.state_dict().copy()
 best_epoch = epoch + 1
 patience_counter = 0
 print(f" New best F1: {best_f1:.4f} (saved)")
 else:
 patience_counter += 1
 print(f" No improvement for {patience_counter} epoch(s)")

 if patience_counter >= args.patience:
 print(f"\nEarly stopping triggered after {epoch + 1} epochs")
 break

train_time = time.time() - start_time
print(f"\n{'='*80}")
print(f"* Training completed in {train_time/60:.1f} minutes")
print(f"* Best model from epoch {best_epoch} with F1: {best_f1:.4f}")

# Load best model
model.load_state_dict(best_model_state)

# Final evaluation
print(f"\n[7/8] Final evaluation with best model...")
test_preds, test_labels = evaluate(model, test_loader, device)

test_f1 = f1_score(test_labels, test_preds, average='macro')
test_acc = accuracy_score(test_labels, test_preds)

print(f"\n{'='*80}")
print("DeBERTa-v3-base IMPROVED RESULTS")
print(f"{'='*80}")
print(f"\nBest Model (Epoch {best_epoch}):")
print(f" Dev F1 (Macro): {test_f1:.4f}")
print(f" Dev Accuracy: {test_acc:.4f}")
print(f" Training Time: {train_time/60:.1f} minutes")
print(f"\nDetailed Classification Report:")
print(classification_report(test_labels, test_preds, target_names=label_list))

# Training history
print(f"\n{'='*80}")
print("TRAINING HISTORY")
print(f"{'='*80}")
print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Train F1':<10} {'Dev F1':<10} {'Dev Acc':<10}")
print("-" * 50)
for i in range(len(history['train_loss'])):
 marker = " " if i == best_epoch - 1 else ""
 print(f"{i+1:<6} {history['train_loss'][i]:<12.4f} {history['train_f1'][i]:<10.4f} "
 f"{history['val_f1'][i]:<10.4f} {history['val_acc'][i]:<10.4f}{marker}")

# ============================================================================
# COMPARISON WITH PREVIOUS MODELS
# ============================================================================
print(f"\n{'='*80}")
print("MODEL COMPARISON")
print(f"{'='*80}")

comparison_df = pd.DataFrame([
 {'Model': 'Logistic Regression', 'Dev F1': 0.4476, 'Test Score': 0.45, 'Type': 'Classical ML'},
 {'Model': 'SVM Linear', 'Dev F1': 0.4270, 'Test Score': 0.43, 'Type': 'Classical ML'},
 {'Model': 'Random Forest', 'Dev F1': 0.4256, 'Test Score': '~0.43', 'Type': 'Classical ML'},
 {'Model': 'DistilBERT', 'Dev F1': 0.5158, 'Test Score': 'TBD', 'Type': 'Transformer'},
 {'Model': 'BERT-base', 'Dev F1': 0.5628, 'Test Score': 0.56, 'Type': 'Transformer'},
 {'Model': 'DeBERTa-v3 Basic', 'Dev F1': '?', 'Test Score': 'TBD', 'Type': 'Transformer'},
 {'Model': 'DeBERTa-v3 Improved', 'Dev F1': test_f1, 'Test Score': 'TBD', 'Type': 'Transformer'}
])

print("\n")
print(comparison_df.to_string(index=False))

improvement_over_bert = test_f1 - 0.5628
improvement_over_baseline = test_f1 - 0.4476

print(f"\n* DeBERTa-v3-base Improved F1: {test_f1:.4f}")
print(f"* Improvement over BERT: {improvement_over_bert:+.4f} ({improvement_over_bert/0.5628*100:+.1f}%)")
print(f"* Improvement over baseline: {improvement_over_baseline:+.4f} ({improvement_over_baseline/0.4476*100:+.1f}%)")

if test_f1 > 0.5628:
 print(f" NEW Best model. Beats BERT by {improvement_over_bert:.4f}")
elif test_f1 > 0.56:
 print(f"* Competitive with BERT")
else:
 print(f"  Below BERT, may need further tuning")

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print(f"\n[8/8] Generating submission...")
print(f"{'='*80}")

# Convert predictions to labels
pred_labels = [id2label[pred] for pred in test_preds]

# Save prediction file
output_name = f'prediction_deberta_improved_lr{args.learning_rate:.0e}'
with open(output_name, 'w') as f:
 for pred in pred_labels:
 f.write(f"{pred}\n")

print(f"* Prediction file created: {output_name}")

# Class distribution
unique, counts = np.unique(pred_labels, return_counts=True)
print(f"\nPrediction distribution:")
for label, count in zip(unique, counts):
 print(f" {label}: {count} ({count/len(pred_labels)*100:.1f}%)")

# Save model
print(f"\n* Saving model...")
save_dir = './deberta_v3_improved_model'
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"* Model saved to: {save_dir}")

# Save training config
config_path = f'{save_dir}/training_config.txt'
with open(config_path, 'w') as f:
 f.write(f"DeBERTa-v3-base Improved Training Configuration\n")
 f.write(f"{'='*60}\n\n")
 f.write(f"Learning Rate: {args.learning_rate}\n")
 f.write(f"Batch Size: {args.batch_size}\n")
 f.write(f"Gradient Accumulation: {args.gradient_accumulation_steps}\n")
 f.write(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}\n")
 f.write(f"Epochs: {best_epoch} (early stopped)\n")
 f.write(f"Warmup Ratio: {args.warmup_ratio}\n")
 f.write(f"Weight Decay: {args.weight_decay}\n")
 f.write(f"LLRD Alpha: {args.llrd_alpha}\n")
 f.write(f"Max Length: {args.max_length}\n")
 f.write(f"Reinit Layers: {args.reinit_layers}\n\n")
 f.write(f"Results:\n")
 f.write(f"Dev F1: {test_f1:.4f}\n")
 f.write(f"Dev Acc: {test_acc:.4f}\n")
 f.write(f"Training Time: {train_time/60:.1f} min\n")

print(f"* Training config saved to: {config_path}")

# If this is best model, update main prediction file
if test_f1 > 0.5628:
 with open('prediction', 'w') as f:
 for pred in pred_labels:
 f.write(f"{pred}\n")
 print(f"\n* Updated main prediction file")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Model: DeBERTa-v3-base Improved")
print(f"Dev F1: {test_f1:.4f}")
print(f"\nHyperparameters used:")
print(f" LR: {args.learning_rate:.0e}, Batch: {args.batch_size}x{args.gradient_accumulation_steps}, "
 f"LLRD: {args.llrd_alpha}, Warmup: {args.warmup_ratio:.1%}")
