from datasets import load_from_disk
import pandas as pd

print("Loading dataset from ./QEvasion directory...")

# Load the dataset
dataset = load_from_disk('./QEvasion')

print(f"\n* Training set: {len(dataset['train'])} examples")
print(f"* Test set: {len(dataset['test'])} examples")

# Convert to pandas for easier exploration
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print("\n" + "="*80)
print("COLUMNS AVAILABLE:")
print("="*80)
for i, col in enumerate(train_df.columns, 1):
 print(f"{i:2d}. {col}")

print("\n" + "="*80)
print("SAMPLE ENTRY:")
print("="*80)
sample = train_df.iloc[0]
print(f"Question: {sample['question'][:150]}...")
print(f"Clarity Label: {sample['clarity_label']}")
print(f"Evasion Label: {sample['evasion_label']}")
print(f"President: {sample['president']}")

print("\n" + "="*80)
print("LABEL DISTRIBUTION:")
print("="*80)
print("\nClarity Labels:")
print(train_df['clarity_label'].value_counts())
print("\nEvasion Labels:")
print(train_df['evasion_label'].value_counts())

print("\n" + "="*80)
print("HOW TO USE:")
print("="*80)
print("""
# Load data:
from datasets import load_from_disk
import pandas as pd

dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Access columns:
questions = train_df['question']
labels = train_df['clarity_label']

# Filter examples:
dodging = train_df[train_df['evasion_label'] == 'Dodging']
""")
