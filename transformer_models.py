"""
Experiment 4: Transformer Models (DistilBERT & BERT)
Fine-tuning pre-trained language models for evasion detection
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
import pickle
import time

print("="*80)
print("EXPERIMENT 4: Transformer Models (DistilBERT & BERT)")
print("="*80)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Device: {device}")
if torch.cuda.is_available():
 print(f" GPU: {torch.cuda.get_device_name(0)}")
 print(f" Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load dataset
print("\n[1/8] Loading dataset...")
dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"* Training samples: {len(train_df)}")
print(f"* Test samples: {len(test_df)}")

# Prepare text
print("\n[2/8] Preparing text for transformers...")
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

# Function to train a model
def train_model(model_name, num_epochs=4, batch_size=16, learning_rate=2e-5):
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
 model.to(device)

 print(f"* Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

 # Create datasets
 print(f"\n[4/8] Creating datasets...")
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
 print(f"\n[5/8] Setting up optimizer and scheduler...")
 optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

 total_steps = len(train_loader) * num_epochs
 warmup_steps = total_steps // 10 # 10% warmup

 scheduler = get_linear_schedule_with_warmup(
 optimizer,
 num_warmup_steps=warmup_steps,
 num_training_steps=total_steps
 )

 print(f"* Learning rate: {learning_rate}")
 print(f"* Warmup steps: {warmup_steps}")
 print(f"* Total steps: {total_steps}")

 # Training loop
 print(f"\n[6/8] Training for {num_epochs} epochs...")
 start_time = time.time()

 best_f1 = 0
 best_model_state = None

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
 print(f"* New best F1: {best_f1:.4f}")

 train_time = time.time() - start_time
 print(f"\n* Training completed in {train_time/60:.1f} minutes")

 # Load best model
 model.load_state_dict(best_model_state)

 # Final evaluation
 print(f"\n[7/8] Final evaluation with best model...")
 test_preds, test_labels = evaluate(model, test_loader, device)

 test_f1 = f1_score(test_labels, test_preds, average='macro')
 test_acc = accuracy_score(test_labels, test_preds)

 print(f"\nBest Model Results:")
 print(f" Dev F1 (Macro): {test_f1:.4f}")
 print(f" Dev Accuracy: {test_acc:.4f}")
 print(f"\nDetailed Classification Report:")
 print(classification_report(test_labels, test_preds, target_names=label_list))

 # Convert predictions to labels
 pred_labels = [id2label[pred] for pred in test_preds]

 return {
 'model_name': model_name,
 'model': model,
 'tokenizer': tokenizer,
 'test_f1': test_f1,
 'test_acc': test_acc,
 'predictions': pred_labels,
 'train_time': train_time
 }

# ============================================================================
# TRAIN MODELS
# ============================================================================

results = []

# Model 1: DistilBERT
print("\n" + "="*80)
print("MODEL 1: DistilBERT")
print("="*80)
distilbert_result = train_model(
 model_name='distilbert-base-uncased',
 num_epochs=4,
 batch_size=16,
 learning_rate=2e-5
)
results.append(distilbert_result)

# Model 2: BERT
print("\n" + "="*80)
print("MODEL 2: BERT")
print("="*80)
bert_result = train_model(
 model_name='bert-base-uncased',
 num_epochs=4,
 batch_size=8, # Smaller batch for larger model
 learning_rate=2e-5
)
results.append(bert_result)

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame([
 {'Model': 'Logistic Regression (Exp 1)', 'Dev F1': 0.4476, 'Accuracy': 0.5227,
 'Test Score': 0.45, 'Type': 'Classical ML'},
 {'Model': 'SVM Linear (Exp 2)', 'Dev F1': 0.4270, 'Accuracy': 0.5097,
 'Test Score': 0.43, 'Type': 'Classical ML'},
 {'Model': 'Random Forest (Exp 3)', 'Dev F1': 0.4256, 'Accuracy': 0.6136,
 'Test Score': '~0.43', 'Type': 'Classical ML'},
] + [
 {'Model': r['model_name'], 'Dev F1': r['test_f1'], 'Accuracy': r['test_acc'],
 'Test Score': 'TBD', 'Type': 'Transformer'}
 for r in results
])

print("\n")
print(comparison_df.to_string(index=False))

# Select best transformer model
best_result = max(results, key=lambda x: x['test_f1'])
best_model_name = best_result['model_name']
best_f1 = best_result['test_f1']
best_predictions = best_result['predictions']

print(f"\n* Best Transformer Model: {best_model_name} (F1: {best_f1:.4f})")

if best_f1 > 0.4476:
 print(f" BEATS ALL BASELINES! Improvement over LogReg: +{best_f1 - 0.4476:.4f}")
elif best_f1 > 0.43:
 print(f"* Better than SVM and RF, approaching LogReg baseline")
else:
 print(f"⚠ Similar to classical ML baselines")

# ============================================================================
# SAVE MODELS AND GENERATE SUBMISSIONS
# ============================================================================
print(f"\n[8/8] Saving models and generating submissions...")

for result in results:
 model_name_short = result['model_name'].replace('/', '_').replace('-', '_')

 # Save model and tokenizer
 result['model'].save_pretrained(f'./{model_name_short}_model')
 result['tokenizer'].save_pretrained(f'./{model_name_short}_model')
 print(f"* Saved {result['model_name']} model")

 # Generate prediction file
 pred_file = f'prediction_{model_name_short}'
 with open(pred_file, 'w') as f:
 for pred in result['predictions']:
 f.write(f"{pred}\n")
 print(f"* Generated {pred_file}")

# Generate submission for best model
with open('prediction', 'w') as f:
 for pred in best_predictions:
 f.write(f"{pred}\n")

print(f"\n* Best model prediction file: prediction (using {best_model_name})")

# Class distribution
unique, counts = np.unique(best_predictions, return_counts=True)
print(f"\nPrediction distribution ({best_model_name}):")
for label, count in zip(unique, counts):
 print(f" {label}: {count} ({count/len(best_predictions)*100:.1f}%)")

print("\n" + "="*80)
print("EXPERIMENT 4 COMPLETE!")
print("="*80)
print(f"Best Model: {best_model_name}")
print(f"Dev F1: {best_f1:.4f}")
print(f"Expected Test Score: ~{best_f1:.2f}")
print("\nNext steps:")
print("1. Create submission.zip with prediction file")
print("2. Submit to Codabench")
print("3. Update experimental log")
print("4. If needed, try RoBERTa or ensemble multiple transformers")
