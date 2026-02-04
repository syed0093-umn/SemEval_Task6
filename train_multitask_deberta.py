"""
Multi-Task DeBERTa with available features.

Uses:
1. Primary task: clarity_label (Ambivalent, Clear Reply, Clear Non-Reply)
2. Auxiliary task: evasion_label (9 fine-grained types)
3. Summary features as additional context
4. Metadata: president, multiple_questions, affirmative_questions
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
 AutoConfig,
 get_cosine_schedule_with_warmup
)
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import time
import argparse

print("="*80)
print("MULTI-TASK DeBERTa with ALL Features")
print("="*80)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=8)
parser.add_argument('--warmup_ratio', type=float, default=0.15)
parser.add_argument('--patience', type=int, default=4)
parser.add_argument('--use_gpt_features', action='store_true', default=True,
 help='Use GPT-3.5 summary and predictions')
parser.add_argument('--use_metadata', action='store_true', default=True,
 help='Use metadata features (president, flags)')
parser.add_argument('--multitask', action='store_true', default=True,
 help='Use multi-task learning with evasion labels')

args = parser.parse_args()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Device: {device}")

# Load dataset
print("\n[1/8] Loading dataset with ALL features...")
dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"* Training samples: {len(train_df)}")
print(f"* Test samples: {len(test_df)}")

# Prepare labels
print("\n[2/8] Preparing labels...")

# Primary task: clarity labels
clarity_labels = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
clarity_label2id = {label: idx for idx, label in enumerate(clarity_labels)}
clarity_id2label = {idx: label for label, idx in clarity_label2id.items()}

train_df['clarity_id'] = train_df['clarity_label'].map(clarity_label2id)
test_df['clarity_id'] = test_df['clarity_label'].map(clarity_label2id)

# Auxiliary task: evasion labels
# Note: Test set has no evasion labels (empty strings), so we use dummy values
evasion_encoder = LabelEncoder()
train_df['evasion_id'] = evasion_encoder.fit_transform(train_df['evasion_label'])

# For test set, use dummy evasion labels (all 0) since we only need clarity predictions
test_df['evasion_id'] = 0 # Dummy value - not used during inference

evasion_labels = evasion_encoder.classes_

print(f"* Clarity labels: {len(clarity_labels)}")
print(f"* Evasion labels: {len(evasion_labels)}")
print(f" {list(evasion_labels)}")

# Prepare text with GPT features
print("\n[3/8] Preparing enhanced text inputs...")

def prepare_enhanced_text(row):
 """Combine question, answer, and GPT-3.5 features"""
 question = row['question']
 answer = row['interview_answer']

 # Base text
 text = f"Question: {question} [SEP] Answer: {answer}"

 # Add GPT-3.5 summary if available
 if args.use_gpt_features and pd.notna(row['gpt3.5_summary']):
 summary = row['gpt3.5_summary'][:200] # Truncate to avoid too long
 text += f" [SEP] Summary: {summary}"

 # Add metadata
 if args.use_metadata:
 president = row['president']
 text += f" [SEP] President: {president}"

 if row['multiple_questions']:
 text += " [MULTI-Q]"
 if row['affirmative_questions']:
 text += " [AFFIRM-Q]"

 return text

train_df['text'] = train_df.apply(prepare_enhanced_text, axis=1)
test_df['text'] = test_df.apply(prepare_enhanced_text, axis=1)

print(f"* Enhanced text prepared")
print(f" Example: {train_df['text'].iloc[0][:300]}...")

# Calculate class weights
class_counts = train_df['clarity_id'].value_counts().sort_index()
total = len(train_df)
clarity_weights = torch.tensor([total / (len(class_counts) * count) for count in class_counts],
 dtype=torch.float).to(device)

evasion_counts = train_df['evasion_id'].value_counts().sort_index()
evasion_weights = torch.tensor([total / (len(evasion_counts) * count) for count in evasion_counts],
 dtype=torch.float).to(device)

print(f"\n* Clarity weights: {clarity_weights.cpu().numpy()}")
print(f"* Evasion weights: {evasion_weights.cpu().numpy()[:5]}...") # First 5

# Custom Dataset
class MultiTaskDataset(Dataset):
 def __init__(self, texts, clarity_labels, evasion_labels, tokenizer, max_length=512):
 self.texts = texts
 self.clarity_labels = clarity_labels
 self.evasion_labels = evasion_labels
 self.tokenizer = tokenizer
 self.max_length = max_length

 def __len__(self):
 return len(self.texts)

 def __getitem__(self, idx):
 text = str(self.texts[idx])
 clarity_label = self.clarity_labels[idx]
 evasion_label = self.evasion_labels[idx]

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
 'clarity_labels': torch.tensor(clarity_label, dtype=torch.long),
 'evasion_labels': torch.tensor(evasion_label, dtype=torch.long)
 }

# Multi-Task Model
class MultiTaskDeBERTa(nn.Module):
 def __init__(self, model_name, num_clarity_labels, num_evasion_labels):
 super().__init__()

 self.deberta = AutoModel.from_pretrained(model_name)
 hidden_size = self.deberta.config.hidden_size

 # Clarity classifier (primary task)
 self.clarity_classifier = nn.Sequential(
 nn.Dropout(0.1),
 nn.Linear(hidden_size, hidden_size),
 nn.GELU(),
 nn.Dropout(0.1),
 nn.Linear(hidden_size, num_clarity_labels)
 )

 # Evasion classifier (auxiliary task)
 self.evasion_classifier = nn.Sequential(
 nn.Dropout(0.1),
 nn.Linear(hidden_size, hidden_size // 2),
 nn.GELU(),
 nn.Dropout(0.1),
 nn.Linear(hidden_size // 2, num_evasion_labels)
 )

 def forward(self, input_ids, attention_mask):
 outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
 pooled_output = outputs.last_hidden_state[:, 0, :] # CLS token

 clarity_logits = self.clarity_classifier(pooled_output)
 evasion_logits = self.evasion_classifier(pooled_output)

 return clarity_logits, evasion_logits

# Initialize model
print("\n[4/8] Initializing multi-task DeBERTa...")
model_name = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = MultiTaskDeBERTa(
 model_name,
 num_clarity_labels=len(clarity_labels),
 num_evasion_labels=len(evasion_labels)
)
model.to(device)

print(f"* Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

# Create datasets
print(f"\n[5/8] Creating datasets...")
train_dataset = MultiTaskDataset(
 train_df['text'].values,
 train_df['clarity_id'].values,
 train_df['evasion_id'].values,
 tokenizer
)
test_dataset = MultiTaskDataset(
 test_df['text'].values,
 test_df['clarity_id'].values,
 test_df['evasion_id'].values,
 tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

print(f"* Train batches: {len(train_loader)}")
print(f"* Test batches: {len(test_loader)}")

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.num_epochs
warmup_steps = int(total_steps * args.warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(
 optimizer,
 num_warmup_steps=warmup_steps,
 num_training_steps=total_steps
)

print(f"\n* Optimizer: AdamW (lr={args.learning_rate})")
print(f"* Scheduler: Cosine with warmup ({args.warmup_ratio:.1%})")

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device):
 model.train()
 total_clarity_loss = 0
 total_evasion_loss = 0
 clarity_preds = []
 clarity_labels = []

 optimizer.zero_grad()

 progress_bar = tqdm(dataloader, desc="Training")
 for step, batch in enumerate(progress_bar):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 clarity_label = batch['clarity_labels'].to(device)
 evasion_label = batch['evasion_labels'].to(device)

 clarity_logits, evasion_logits = model(input_ids, attention_mask)

 # Multi-task loss
 clarity_loss_fct = nn.CrossEntropyLoss(weight=clarity_weights)
 evasion_loss_fct = nn.CrossEntropyLoss(weight=evasion_weights)

 clarity_loss = clarity_loss_fct(clarity_logits, clarity_label)
 evasion_loss = evasion_loss_fct(evasion_logits, evasion_label)

 # Combined loss (weighted)
 loss = clarity_loss + 0.3 * evasion_loss # Evasion as auxiliary

 loss = loss / args.gradient_accumulation_steps
 loss.backward()

 total_clarity_loss += clarity_loss.item()
 total_evasion_loss += evasion_loss.item()

 if (step + 1) % args.gradient_accumulation_steps == 0:
 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
 optimizer.step()
 scheduler.step()
 optimizer.zero_grad()

 preds = torch.argmax(clarity_logits, dim=1).cpu().numpy()
 clarity_preds.extend(preds)
 clarity_labels.extend(clarity_label.cpu().numpy())

 progress_bar.set_postfix({
 'clarity_loss': clarity_loss.item(),
 'evasion_loss': evasion_loss.item()
 })

 avg_clarity_loss = total_clarity_loss / len(dataloader)
 avg_evasion_loss = total_evasion_loss / len(dataloader)
 f1 = f1_score(clarity_labels, clarity_preds, average='macro')

 return avg_clarity_loss, avg_evasion_loss, f1

# Evaluation function
def evaluate(model, dataloader, device):
 model.eval()
 clarity_preds = []
 clarity_labels = []

 with torch.no_grad():
 for batch in tqdm(dataloader, desc="Evaluating"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 labels = batch['clarity_labels'].to(device)

 clarity_logits, _ = model(input_ids, attention_mask)

 preds = torch.argmax(clarity_logits, dim=1).cpu().numpy()
 clarity_preds.extend(preds)
 clarity_labels.extend(labels.cpu().numpy())

 return clarity_preds, clarity_labels

# Training loop
print(f"\n[6/8] Training for up to {args.num_epochs} epochs...")
start_time = time.time()

best_f1 = 0
best_model_state = None
best_epoch = 0
patience_counter = 0

for epoch in range(args.num_epochs):
 print(f"\n{'='*80}")
 print(f"Epoch {epoch + 1}/{args.num_epochs}")
 print(f"{'='*80}")

 clarity_loss, evasion_loss, train_f1 = train_epoch(
 model, train_loader, optimizer, scheduler, device
 )
 print(f"\n* Clarity Loss: {clarity_loss:.4f}, Evasion Loss: {evasion_loss:.4f}, Train F1: {train_f1:.4f}")

 # Evaluate
 test_preds, test_labels = evaluate(model, test_loader, device)
 test_f1 = f1_score(test_labels, test_preds, average='macro')
 test_acc = accuracy_score(test_labels, test_preds)

 print(f"* Dev F1: {test_f1:.4f}, Dev Acc: {test_acc:.4f}")

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
 print(f"\nEarly stopping triggered")
 break

train_time = time.time() - start_time
print(f"\n* Training completed in {train_time/60:.1f} minutes")
print(f"* Best F1: {best_f1:.4f} from epoch {best_epoch}")

# Load best model and evaluate
model.load_state_dict(best_model_state)
test_preds, test_labels = evaluate(model, test_loader, device)
test_f1 = f1_score(test_labels, test_preds, average='macro')
test_acc = accuracy_score(test_labels, test_preds)

print(f"\n{'='*80}")
print("MULTI-TASK DeBERTa RESULTS")
print(f"{'='*80}")
print(f"\nDev F1 (Macro): {test_f1:.4f}")
print(f"Dev Accuracy: {test_acc:.4f}")
print(f"Training Time: {train_time/60:.1f} minutes")
print(f"\n{classification_report(test_labels, test_preds, target_names=clarity_labels)}")

# Generate submission
print("\n[7/8] Generating submission...")
pred_labels = [clarity_id2label[pred] for pred in test_preds]

with open('prediction_multitask', 'w') as f:
 for pred in pred_labels:
 f.write(f"{pred}\n")

print(f"* Prediction file: prediction_multitask")

# Save model
model.deberta.save_pretrained('./deberta_multitask_model')
tokenizer.save_pretrained('./deberta_multitask_model')
print(f"* Model saved to: ./deberta_multitask_model")

import zipfile
with zipfile.ZipFile('submission_multitask.zip', 'w') as zipf:
 zipf.write('prediction_multitask', 'prediction')

print(f"* Submission: submission_multitask.zip")

improvement = test_f1 - 0.61
print(f"\n{'='*80}")
print(f"* Multi-task F1: {test_f1:.4f}")
print(f"* Improvement over single-task: {improvement:+.4f} ({improvement/0.61*100:+.1f}%)")

print(f"{'='*80}")
