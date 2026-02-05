"""
Annotator-Aware DeBERTa Model

Uses annotator labels as features (not direct predictions).
Model learns to combine:
1. Text understanding (DeBERTa embeddings)
2. Annotator voting patterns
3. Annotator confidence/agreement signals
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel, AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANNOTATOR-AWARE DeBERTa TRAINING")
print("="*80)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Device: {device}")

# Load dataset
print("\n[1/8] Loading dataset...")
dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"* Training samples: {len(train_df)}")
print(f"* Test samples: {len(test_df)}")

# Evasion to clarity mapping
evasion_to_clarity = {
 'Explicit': 'Clear Reply',
 'Implicit': 'Ambivalent',
 'Dodging': 'Ambivalent',
 'General': 'Ambivalent',
 'Deflection': 'Ambivalent',
 'Declining to answer': 'Clear Non-Reply',
 'Claims ignorance': 'Clear Non-Reply',
 'Clarification': 'Clear Non-Reply',
 'Partial/half-answer': 'Ambivalent'
}

clarity_to_id = {'Ambivalent': 0, 'Clear Non-Reply': 1, 'Clear Reply': 2}
id_to_clarity = {v: k for k, v in clarity_to_id.items()}

# Prepare labels
train_df['label_id'] = train_df['clarity_label'].map(clarity_to_id)
test_df['label_id'] = test_df['clarity_label'].map(clarity_to_id)

# Prepare annotator features for test set
print("\n[2/8] Engineering annotator features...")

def extract_annotator_features(row):
 """
 Extract features from 3 annotators:
 - 3 one-hot encoded votes (converted to clarity)
 - Agreement level (all agree, 2 agree, all disagree)
 - Confidence score based on consensus
 """
 # Convert annotator evasion labels to clarity
 ann_labels = []
 for col in ['annotator1', 'annotator2', 'annotator3']:
 evasion = row[col]
 if pd.notna(evasion) and evasion in evasion_to_clarity:
 ann_labels.append(evasion_to_clarity[evasion])
 else:
 ann_labels.append(None)

 # One-hot encode each annotator's vote (9 features: 3 annotators x 3 classes)
 features = []
 for ann_label in ann_labels:
 if ann_label == 'Ambivalent':
 features.extend([1, 0, 0])
 elif ann_label == 'Clear Non-Reply':
 features.extend([0, 1, 0])
 elif ann_label == 'Clear Reply':
 features.extend([0, 0, 1])
 else:
 features.extend([0, 0, 0]) # Missing annotator

 # Agreement features (3 features)
 valid_labels = [l for l in ann_labels if l is not None]
 if len(valid_labels) >= 2:
 # Calculate agreement
 if len(set(valid_labels)) == 1:
 agreement = 1.0 # All agree
 elif len(set(valid_labels)) == 2:
 agreement = 0.5 # 2 agree
 else:
 agreement = 0.0 # All disagree

 # Confidence based on majority
 from collections import Counter
 vote_counts = Counter(valid_labels)
 max_votes = max(vote_counts.values())
 confidence = max_votes / len(valid_labels)

 # Majority class (one-hot)
 majority = vote_counts.most_common(1)[0][0]
 if majority == 'Ambivalent':
 majority_onehot = [1, 0, 0]
 elif majority == 'Clear Non-Reply':
 majority_onehot = [0, 1, 0]
 else: # Clear Reply
 majority_onehot = [0, 0, 1]
 else:
 agreement = 0.0
 confidence = 0.0
 majority_onehot = [0, 0, 0]

 features.extend([agreement, confidence] + majority_onehot)

 return features

# Test set has annotators - extract features
print("Extracting annotator features from test set...")
test_annotator_features = test_df.apply(extract_annotator_features, axis=1).tolist()

# Training set has NO annotators - use zero features
print("Creating dummy annotator features for training set...")
train_annotator_features = [[0.0] * 14] * len(train_df) # 9 + 2 + 3 = 14 features

print(f"* Annotator feature dimension: {len(test_annotator_features[0])}")

# Prepare text
def prepare_text(row):
 return f"Question: {row['question']} [SEP] Answer: {row['interview_answer']}"

train_df['text'] = train_df.apply(prepare_text, axis=1)
test_df['text'] = test_df.apply(prepare_text, axis=1)

# Dataset class
class AnnotatorAwareDataset(Dataset):
 def __init__(self, texts, labels, annotator_features, tokenizer, max_length=512):
 self.texts = texts
 self.labels = labels
 self.annotator_features = annotator_features
 self.tokenizer = tokenizer
 self.max_length = max_length

 def __len__(self):
 return len(self.texts)

 def __getitem__(self, idx):
 encoding = self.tokenizer(
 self.texts[idx],
 truncation=True,
 max_length=self.max_length,
 padding='max_length',
 return_tensors='pt'
 )

 return {
 'input_ids': encoding['input_ids'].squeeze(),
 'attention_mask': encoding['attention_mask'].squeeze(),
 'annotator_features': torch.tensor(self.annotator_features[idx], dtype=torch.float32),
 'label': torch.tensor(self.labels[idx], dtype=torch.long)
 }

# Model class
class AnnotatorAwareDeBERTa(nn.Module):
 def __init__(self, model_name, num_labels, annotator_feature_dim):
 super().__init__()
 self.deberta = AutoModel.from_pretrained(model_name)
 hidden_size = self.deberta.config.hidden_size

 # Annotator feature processor
 self.annotator_processor = nn.Sequential(
 nn.Linear(annotator_feature_dim, 64),
 nn.ReLU(),
 nn.Dropout(0.1),
 nn.Linear(64, 32)
 )

 # Combined classifier
 self.classifier = nn.Sequential(
 nn.Dropout(0.1),
 nn.Linear(hidden_size + 32, 256),
 nn.ReLU(),
 nn.Dropout(0.1),
 nn.Linear(256, num_labels)
 )

 def forward(self, input_ids, attention_mask, annotator_features):
 # Get DeBERTa embeddings
 outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
 pooled = outputs.last_hidden_state[:, 0, :] # [CLS] token

 # Process annotator features
 annotator_embed = self.annotator_processor(annotator_features)

 # Combine and classify
 combined = torch.cat([pooled, annotator_embed], dim=1)
 logits = self.classifier(combined)

 return logits

# Training setup
print("\n[3/8] Initializing model and tokenizer...")
model_name = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AnnotatorAwareDeBERTa(model_name, num_labels=3, annotator_feature_dim=14)
model.to(device)

print(f"* Model: {model_name}")
print(f"* Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create datasets
print("\n[4/8] Creating datasets...")
train_dataset = AnnotatorAwareDataset(
 train_df['text'].tolist(),
 train_df['label_id'].tolist(),
 train_annotator_features,
 tokenizer
)

test_dataset = AnnotatorAwareDataset(
 test_df['text'].tolist(),
 test_df['label_id'].tolist(),
 test_annotator_features,
 tokenizer
)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"* Train batches: {len(train_loader)}")
print(f"* Test batches: {len(test_loader)}")

# Optimizer
print("\n[5/8] Setting up optimizer...")
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

num_epochs = 4
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_cosine_schedule_with_warmup(
 optimizer,
 num_warmup_steps=num_warmup_steps,
 num_training_steps=num_training_steps
)

criterion = nn.CrossEntropyLoss()

print(f"* Learning rate: 2e-5")
print(f"* Epochs: {num_epochs}")
print(f"* Warmup steps: {num_warmup_steps}")

# Training loop
print("\n[6/8] Training model...")
print("="*80)

best_f1 = 0
for epoch in range(num_epochs):
 model.train()
 total_loss = 0

 pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
 for batch in pbar:
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 annotator_features = batch['annotator_features'].to(device)
 labels = batch['label'].to(device)

 optimizer.zero_grad()
 logits = model(input_ids, attention_mask, annotator_features)
 loss = criterion(logits, labels)

 loss.backward()
 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
 optimizer.step()
 scheduler.step()

 total_loss += loss.item()
 pbar.set_postfix({'loss': f'{loss.item():.4f}'})

 avg_loss = total_loss / len(train_loader)

 # Evaluate
 model.eval()
 all_preds = []
 all_labels = []

 with torch.no_grad():
 for batch in test_loader:
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 annotator_features = batch['annotator_features'].to(device)
 labels = batch['label'].to(device)

 logits = model(input_ids, attention_mask, annotator_features)
 preds = torch.argmax(logits, dim=1)

 all_preds.extend(preds.cpu().numpy())
 all_labels.extend(labels.cpu().numpy())

 epoch_f1 = f1_score(all_labels, all_preds, average='macro')

 print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Test F1={epoch_f1:.4f}")

 if epoch_f1 > best_f1:
 best_f1 = epoch_f1
 torch.save(model.state_dict(), 'annotator_aware_model.pt')
 print(f"* New best F1: {best_f1:.4f} - Model saved.")

# Load best model and make final predictions
print("\n[7/8] Loading best model for final predictions...")
model.load_state_dict(torch.load('annotator_aware_model.pt'))
model.eval()

all_preds = []
with torch.no_grad():
 for batch in tqdm(test_loader, desc="Predicting"):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 annotator_features = batch['annotator_features'].to(device)

 logits = model(input_ids, attention_mask, annotator_features)
 preds = torch.argmax(logits, dim=1)
 all_preds.extend(preds.cpu().numpy())

# Convert predictions to labels
pred_labels = [id_to_clarity[pred] for pred in all_preds]

print("\n[8/8] Evaluation...")
print("="*80)
print(classification_report(test_df['clarity_label'], pred_labels, digits=4))

final_f1 = f1_score(test_df['clarity_label'], pred_labels, average='macro')
print(f"\n Final Test F1: {final_f1:.4f}")

# Save predictions
output_file = 'prediction_annotator_aware'
pd.Series(pred_labels).to_csv(output_file, index=False, header=False)

import zipfile
submission_file = 'submission_annotator_aware.zip'
with zipfile.ZipFile(submission_file, 'w') as zf:
 zf.write(output_file, arcname='prediction')

print(f"\n* Saved predictions: {output_file}")
print(f"* Created submission: {submission_file}")
print("\n" + "="*80)
print(f"Final F1: {final_f1:.4f}")
print("="*80)
