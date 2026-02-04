"""
Synthetic Data Generation for Subtask 2 (9-Class Evasion Classification)

Generates synthetic training data to address class imbalance across 9 evasion classes:
- Top 3 classes (Explicit, Dodging, Implicit) kept unchanged
- Bottom 6 classes augmented to moderate target sizes

Target distribution (4,596 total from 3,448 original):
 - Explicit: 1,052 (unchanged)
 - Dodging: 706 (unchanged)
 - Implicit: 488 (unchanged)
 - General: 500 (+114)
 - Deflection: 500 (+119)
 - Declining to answer: 400 (+255)
 - Claims ignorance: 350 (+231)
 - Clarification: 300 (+208)
 - Partial/half-answer: 300 (+221)

Usage:
 python data_augmentation/generate_evasion_data.py --method hybrid --output_dir ./QEvasion_evasion_augmented
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
import random
import json
from typing import List, Dict

from data_augmentation.augmenters import get_augmenter

def parse_args():
 parser = argparse.ArgumentParser(
 description='Generate synthetic data for 9-class evasion rebalancing'
 )

 parser.add_argument('--data_dir', type=str, default='./QEvasion',
 help='Path to original dataset')
 parser.add_argument('--output_dir', type=str, default='./QEvasion_evasion_augmented',
 help='Path to save augmented dataset')

 parser.add_argument('--method', type=str, default='hybrid',
 choices=['eda', 'backtranslation', 'llm', 'hybrid'],
 help='Augmentation method to use')

 # Target sizes for each evasion class
 parser.add_argument('--target_explicit', type=int, default=1052)
 parser.add_argument('--target_dodging', type=int, default=706)
 parser.add_argument('--target_implicit', type=int, default=488)
 parser.add_argument('--target_general', type=int, default=500)
 parser.add_argument('--target_deflection', type=int, default=500)
 parser.add_argument('--target_declining', type=int, default=400)
 parser.add_argument('--target_ignorance', type=int, default=350)
 parser.add_argument('--target_clarification', type=int, default=300)
 parser.add_argument('--target_partial', type=int, default=300)

 parser.add_argument('--augmentations_per_sample', type=int, default=2,
 help='Number of augmentations to generate per sample')
 parser.add_argument('--llm_provider', type=str, default='openai',
 choices=['openai', 'anthropic'])
 parser.add_argument('--llm_model', type=str, default=None)
 parser.add_argument('--device', type=str, default='cuda',
 choices=['cuda', 'cpu'])
 parser.add_argument('--seed', type=int, default=42)

 return parser.parse_args()

def set_seed(seed: int):
 random.seed(seed)
 np.random.seed(seed)
 try:
 import torch
 torch.manual_seed(seed)
 if torch.cuda.is_available():
 torch.cuda.manual_seed_all(seed)
 except ImportError:
 pass

def load_dataset(data_dir: str) -> pd.DataFrame:
 print(f"\n[1/5] Loading dataset from {data_dir}...")
 dataset = load_from_disk(data_dir)
 train_df = pd.DataFrame(dataset['train'])

 print(f" Original training samples: {len(train_df)}")
 print("\n Current evasion class distribution:")
 class_counts = train_df['evasion_label'].value_counts()
 for label, count in class_counts.items():
 print(f" {label}: {count} ({count/len(train_df)*100:.1f}%)")

 return train_df

def calculate_augmentation_needs(train_df: pd.DataFrame, targets: Dict[str, int]) -> Dict[str, int]:
 class_counts = train_df['evasion_label'].value_counts()

 needs = {}
 for label, target in targets.items():
 original = class_counts.get(label, 0)
 needs[label] = max(0, target - original)

 print("\n Augmentation needs:")
 for label, need in needs.items():
 original = class_counts.get(label, 0)
 target = targets[label]
 print(f" {label}: {original} -> {target} (need {need} synthetic)")

 return needs

def augment_class(
 df: pd.DataFrame,
 class_label: str,
 num_needed: int,
 augmenter,
 augmentations_per_sample: int = 2
) -> List[Dict]:
 if num_needed <= 0:
 return []

 class_df = df[df['evasion_label'] == class_label].copy()

 if len(class_df) == 0:
 print(f" Warning: No samples found for class {class_label}")
 return []

 print(f"\n Augmenting {class_label} ({num_needed} samples needed)...")

 synthetic_samples = []
 samples_generated = 0

 iterations_needed = (num_needed // (len(class_df) * augmentations_per_sample)) + 1

 with tqdm(total=num_needed, desc=f" Generating {class_label}") as pbar:
 for iteration in range(iterations_needed):
 if samples_generated >= num_needed:
 break

 for idx, row in class_df.iterrows():
 if samples_generated >= num_needed:
 break

 original_answer = row['interview_answer']

 try:
 augmented_texts = augmenter.augment(
 original_answer,
 num_augments=augmentations_per_sample
 )
 except Exception as e:
 print(f" Warning: Augmentation failed: {e}")
 continue

 for aug_text in augmented_texts:
 if samples_generated >= num_needed:
 break

 if aug_text == original_answer:
 continue

 synthetic = row.to_dict()
 synthetic['interview_answer'] = aug_text
 synthetic['is_synthetic'] = True
 synthetic['original_idx'] = idx
 synthetic['augmentation_iteration'] = iteration

 synthetic_samples.append(synthetic)
 samples_generated += 1
 pbar.update(1)

 print(f" Generated {len(synthetic_samples)} synthetic samples for {class_label}")
 return synthetic_samples

def create_augmented_dataset(train_df: pd.DataFrame, synthetic_samples: List[Dict]):
 print(f"\n[4/5] Creating augmented dataset...")

 train_df = train_df.copy()
 train_df['is_synthetic'] = False
 train_df['original_idx'] = train_df.index
 train_df['augmentation_iteration'] = -1

 synthetic_df = pd.DataFrame(synthetic_samples)
 augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)
 augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

 print(f" Total samples: {len(augmented_df)}")
 print(f" Original: {len(train_df)}")
 print(f" Synthetic: {len(synthetic_samples)}")

 print("\n New evasion class distribution:")
 class_counts = augmented_df['evasion_label'].value_counts()
 for label, count in class_counts.items():
 synthetic_count = len(augmented_df[(augmented_df['evasion_label'] == label) &
 (augmented_df['is_synthetic'] == True)])
 print(f" {label}: {count} ({count/len(augmented_df)*100:.1f}%) "
 f"[{synthetic_count} synthetic]")

 return augmented_df

def save_dataset(augmented_df: pd.DataFrame, original_data_dir: str, output_dir: str):
 print(f"\n[5/5] Saving augmented dataset to {output_dir}...")

 original = load_from_disk(original_data_dir)

 original_columns = list(original['train'].features.keys())
 keep_columns = original_columns + ['is_synthetic', 'original_idx', 'augmentation_iteration']
 available_columns = [c for c in keep_columns if c in augmented_df.columns]
 augmented_df = augmented_df[available_columns]

 train_dataset = Dataset.from_pandas(augmented_df, preserve_index=False)
 test_dataset = original['test']

 augmented_dataset = DatasetDict({
 'train': train_dataset,
 'test': test_dataset
 })

 os.makedirs(output_dir, exist_ok=True)
 augmented_dataset.save_to_disk(output_dir)

 print(f" Saved to {output_dir}")

 metadata = {
 'original_train_size': len(original['train']),
 'augmented_train_size': len(train_dataset),
 'synthetic_samples': int(augmented_df['is_synthetic'].sum()),
 'test_size': len(test_dataset),
 'class_distribution': augmented_df['evasion_label'].value_counts().to_dict()
 }

 metadata_path = os.path.join(output_dir, 'augmentation_metadata.json')
 with open(metadata_path, 'w') as f:
 json.dump(metadata, f, indent=2)

 print(f" Saved metadata to {metadata_path}")
 return augmented_dataset

def main():
 args = parse_args()

 print("="*80)
 print("SYNTHETIC DATA GENERATION FOR SUBTASK 2 (9-CLASS EVASION)")
 print("="*80)

 set_seed(args.seed)
 print(f"\nRandom seed: {args.seed}")
 print(f"Augmentation method: {args.method}")

 train_df = load_dataset(args.data_dir)

 # Target sizes for each class
 targets = {
 'Explicit': args.target_explicit,
 'Dodging': args.target_dodging,
 'Implicit': args.target_implicit,
 'General': args.target_general,
 'Deflection': args.target_deflection,
 'Declining to answer': args.target_declining,
 'Claims ignorance': args.target_ignorance,
 'Clarification': args.target_clarification,
 'Partial/half-answer': args.target_partial,
 }

 print("\n[2/5] Calculating augmentation needs...")
 needs = calculate_augmentation_needs(train_df, targets)

 print(f"\n[3/5] Initializing {args.method} augmenter...")
 if args.method == 'llm':
 augmenter = get_augmenter(
 args.method,
 provider=args.llm_provider,
 model=args.llm_model
 )
 elif args.method in ['backtranslation', 'hybrid']:
 augmenter = get_augmenter(args.method, device=args.device)
 else:
 augmenter = get_augmenter(args.method)

 # Generate synthetic samples for each class that needs augmentation
 all_synthetic = []
 for class_label, num_needed in needs.items():
 if num_needed > 0:
 if args.method == 'llm' and hasattr(augmenter, 'class_label'):
 augmenter.class_label = class_label

 synthetic = augment_class(
 train_df,
 class_label,
 num_needed,
 augmenter,
 args.augmentations_per_sample
 )
 all_synthetic.extend(synthetic)

 augmented_df = create_augmented_dataset(train_df, all_synthetic)
 save_dataset(augmented_df, args.data_dir, args.output_dir)

 print("\n" + "="*80)
 print("SYNTHETIC DATA GENERATION COMPLETE!")
 print("="*80)
 print(f"\nOutput directory: {args.output_dir}")
 print(f"Total training samples: {len(augmented_df)}")
 print(f" Original: {len(train_df)}")
 print(f" Synthetic: {len(all_synthetic)}")

 print("\n Final evasion class distribution:")
 class_counts = augmented_df['evasion_label'].value_counts()
 for label in sorted(targets.keys()):
 count = class_counts.get(label, 0)
 print(f" {label}: {count} ({count/len(augmented_df)*100:.1f}%)")

 print(f"\nNext steps:")
 print(f" 1. Train: python train_deberta_evasion_augmented.py --data_dir {args.output_dir}")
 print(f" 2. Predict: python predict_eval_evasion_augmented.py")

 return augmented_df

if __name__ == '__main__':
 main()
