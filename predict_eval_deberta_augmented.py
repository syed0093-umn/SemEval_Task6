"""
Generate predictions on the evaluation dataset using the trained
DeBERTa-v3-base model trained on augmented data.
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import zipfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# --- Model definition (must match training) ---
class DeBERTaWithFeatures(nn.Module):
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

class EvalDataset(Dataset):
 def __init__(self, texts, bool_features, tokenizer, max_length=512):
 self.texts = texts
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
 }

# --- Load evaluation data ---
print("Loading evaluation dataset...")
eval_df = pd.read_csv('clarity_task_evaluation_dataset.csv')
print(f"Evaluation samples: {len(eval_df)}")

def prepare_text(row):
 question = row['question']
 answer = row['interview_answer']
 return f"Question: {question} [SEP] Answer: {answer}"

eval_df['text'] = eval_df.apply(prepare_text, axis=1)

eval_df['bool_features'] = eval_df.apply(
 lambda row: [
 float(row['affirmative_questions']) if pd.notna(row['affirmative_questions']) else 0.0,
 float(row['multiple_questions']) if pd.notna(row['multiple_questions']) else 0.0,
 ], axis=1
)

# --- Load model ---
label_list = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
id2label = {idx: label for idx, label in enumerate(label_list)}

model_name = 'microsoft/deberta-v3-base'
print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

checkpoint = 'deberta_augmented_best.pt'
print(f"Loading model checkpoint: {checkpoint}")
model = DeBERTaWithFeatures(model_name=model_name, num_labels=3, bool_feature_dim=2)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model = model.to(device)
model.eval()

# --- Run inference ---
eval_dataset = EvalDataset(
 texts=eval_df['text'].tolist(),
 bool_features=eval_df['bool_features'].tolist(),
 tokenizer=tokenizer,
 max_length=512
)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

print("Running inference...")
all_preds = []
with torch.no_grad():
 for batch in tqdm(eval_loader):
 input_ids = batch['input_ids'].to(device)
 attention_mask = batch['attention_mask'].to(device)
 bool_features = batch['bool_features'].to(device)
 logits = model(input_ids=input_ids, attention_mask=attention_mask,
 bool_features=bool_features)
 preds = torch.argmax(logits, dim=1)
 all_preds.extend(preds.cpu().numpy())

pred_labels = [id2label[p] for p in all_preds]
print(f"Generated {len(pred_labels)} predictions")
print(f"Distribution: {pd.Series(pred_labels).value_counts().to_dict()}")

# --- Save prediction file (extensionless) ---
output_file = 'prediction_eval_deberta_augmented'
with open(output_file, 'w') as f:
 for label in pred_labels:
 f.write(label + '\n')
print(f"Saved: {output_file}")

# --- Create submission zip ---
submission_zip = 'submission_eval_deberta_augmented.zip'
with zipfile.ZipFile(submission_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
 zf.write(output_file, arcname='prediction')
print(f"Created: {submission_zip}")
print("Done. Upload submission_eval_deberta_augmented.zip to Codabench.")
