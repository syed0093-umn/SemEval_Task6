"""
DeBERTa-v3-base fine-tuning for clarity classification.
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
 AutoTokenizer,
 AutoModelForSequenceClassification,
 AdamW,
 get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm
import time

print("="*80)
print("DeBERTa-v3-base")
print("Transformer model for clarity classification")
print("="*80)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Device: {device}")
if torch.cuda.is_available():
 print(f" GPU: {torch.cuda.get_device_name(0)}")
 print(f" Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load dataset
print("\n[1/7] Loading dataset...")
dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"* Training samples: {len(train_df)}")
print(f"* Test samples: {len(test_df)}")

# Prepare text
print("\n[2/7] Preparing text for DeBERTa...")
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

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights):
 model.train()
 total_loss = 0
 predictions = []
 true_labels = []

 progress_bar = tqdm(dataloader, desc="Training")
 for batch in progress_bar:
 optimizer.zero_grad()

 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 labels = batch['labels'].to(device)

 outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

 # Apply class weights
 loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
 loss = loss_fct(outputs.logits, labels)

 total_loss += loss.item()

 loss.backward()
 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
 optimizer.step()
 scheduler.step()

 preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
 predictions.extend(preds)
 true_labels.extend(labels.cpu().numpy())

 progress_bar.set_postfix({'loss': loss.item()})

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
# TRAIN DeBERTa-v3-base
# ============================================================================

model_name = 'microsoft/deberta-v3-base'
num_epochs = 4
batch_size = 8 # Smaller batch for larger model
learning_rate = 2e-5

print(f"\n{'='*80}")
print(f"Training: {model_name}")
print(f"{'='*80}")

# Initialize tokenizer and model
print(f"\n[3/7] Loading {model_name} tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
 model_name,
 num_labels=3,
 id2label=id2label,
 label2id=label2id,
 problem_type="single_label_classification"
)
model.to(device)

print(f"* Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

# Create datasets
print(f"\n[4/7] Creating datasets...")
train_dataset = ClarityDataset(
 train_df['text'].values,
 train_df['label_id'].values,
 tokenizer,
 max_length=512
)
test_dataset = ClarityDataset(
 test_df['text'].values,
 test_df['label_id'].values,
 tokenizer,
 max_length=512
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(f"* Train batches: {len(train_loader)}")
print(f"* Test batches: {len(test_loader)}")

# Optimizer and scheduler
print(f"\n[5/7] Setting up optimizer and scheduler...")
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

total_steps = len(train_loader) * num_epochs
warmup_steps = total_steps // 10 # 10% warmup

scheduler = get_linear_schedule_with_warmup(
 optimizer,
 num_warmup_steps=warmup_steps,
 num_training_steps=total_steps
)

print(f"* Learning rate: {learning_rate}")
print(f"* Batch size: {batch_size}")
print(f"* Warmup steps: {warmup_steps}")
print(f"* Total steps: {total_steps}")

# Training loop
print(f"\n[6/7] Training for {num_epochs} epochs...")
start_time = time.time()

best_f1 = 0
best_model_state = None
best_epoch = 0

for epoch in range(num_epochs):
 print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

 train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights)
 print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")

 # Evaluate on test set
 test_preds, test_labels = evaluate(model, test_loader, device)
 test_f1 = f1_score(test_labels, test_preds, average='macro')
 test_acc = accuracy_score(test_labels, test_preds)

 print(f"Test F1: {test_f1:.4f}, Test Acc: {test_acc:.4f}")

 # Save best model
 if test_f1 > best_f1:
 best_f1 = test_f1
 best_model_state = model.state_dict().copy()
 best_epoch = epoch + 1
 print(f"* New best F1: {best_f1:.4f}")

train_time = time.time() - start_time
print(f"\n* Training completed in {train_time/60:.1f} minutes")
print(f"* Best model from epoch {best_epoch}")

# Load best model
model.load_state_dict(best_model_state)

# Final evaluation
print(f"\n[7/7] Final evaluation with best model...")
test_preds, test_labels = evaluate(model, test_loader, device)

test_f1 = f1_score(test_labels, test_preds, average='macro')
test_acc = accuracy_score(test_labels, test_preds)

print(f"\n{'='*80}")
print("DeBERTa-v3-base RESULTS")
print(f"{'='*80}")
print(f"\nBest Model (Epoch {best_epoch}):")
print(f" Dev F1 (Macro): {test_f1:.4f}")
print(f" Dev Accuracy: {test_acc:.4f}")
print(f" Training Time: {train_time/60:.1f} minutes")
print(f"\nDetailed Classification Report:")
print(classification_report(test_labels, test_preds, target_names=label_list))

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
 {'Model': 'DeBERTa-v3-base', 'Dev F1': test_f1, 'Test Score': 'TBD', 'Type': 'Transformer'}
])

print("\n")
print(comparison_df.to_string(index=False))

improvement_over_bert = test_f1 - 0.5628
improvement_over_baseline = test_f1 - 0.4476

print(f"\n* DeBERTa-v3-base F1: {test_f1:.4f}")
print(f"* Improvement over BERT: {improvement_over_bert:+.4f} ({improvement_over_bert/0.5628*100:+.1f}%)")
print(f"* Improvement over baseline: {improvement_over_baseline:+.4f} ({improvement_over_baseline/0.4476*100:+.1f}%)")

if test_f1 > 0.5628:
 print(f" NEW Best model. Beats BERT by {improvement_over_bert:.4f}")
elif test_f1 > 0.56:
 print(f"* Competitive with BERT, slight improvement")
else:
 print(f"  Slightly below BERT, but still beats baseline")

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print(f"\n{'='*80}")
print("GENERATING SUBMISSION")
print(f"{'='*80}")

# Convert predictions to labels
pred_labels = [id2label[pred] for pred in test_preds]

# Save prediction file
with open('prediction_deberta', 'w') as f:
 for pred in pred_labels:
 f.write(f"{pred}\n")

print(f"* Prediction file created: prediction_deberta")

# Class distribution
unique, counts = np.unique(pred_labels, return_counts=True)
print(f"\nPrediction distribution:")
for label, count in zip(unique, counts):
 print(f" {label}: {count} ({count/len(pred_labels)*100:.1f}%)")

# Save model
print(f"\n* Saving model...")
model.save_pretrained('./deberta_v3_base_model')
tokenizer.save_pretrained('./deberta_v3_base_model')
print(f"* Model saved to: ./deberta_v3_base_model")

# If this is best model, update main prediction file
if test_f1 > 0.5628:
 with open('prediction', 'w') as f:
 for pred in pred_labels:
 f.write(f"{pred}\n")
 print(f"\n* Updated main prediction file")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Model: DeBERTa-v3-base")
print(f"Dev F1: {test_f1:.4f}")
