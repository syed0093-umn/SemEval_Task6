"""
Synthetic Data Generation for Subtask 1 (Clarity Classification)

Generates synthetic training data to address class imbalance:
- Current: Ambivalent 59.2%, Clear Reply 30.5%, Clear Non-Reply 10.3%
- Target (Strategy B - Moderate Balance):
 - Ambivalent: 2,041 (original, no augmentation)
 - Clear Reply: 1,500 (1,051 original + 449 synthetic)
 - Clear Non-Reply: 1,020 (355 original + 665 synthetic)
 - Total: 4,561 samples

Usage:
 python data_augmentation/generate_synthetic_data.py --method hybrid --output_dir ./QEvasion_augmented

Methods available:
 - eda: Fast EDA augmentation (synonym replacement, random operations)
 - backtranslation: Back-translation via German
 - llm: LLM paraphrasing (requires API key)
 - hybrid: Mix of EDA and back-translation (recommended)
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
from typing import List, Dict, Tuple

from data_augmentation.augmenters import (
 EDAugmenter, BackTranslationAugmenter, LLMParaphraser,
 HybridAugmenter, get_augmenter
)

def parse_args():
 parser = argparse.ArgumentParser(
 description='Generate synthetic data for class rebalancing'
 )

 # Data paths
 parser.add_argument('--data_dir', type=str, default='./QEvasion',
 help='Path to original dataset')
 parser.add_argument('--output_dir', type=str, default='./QEvasion_augmented',
 help='Path to save augmented dataset')

 # Augmentation method
 parser.add_argument('--method', type=str, default='hybrid',
 choices=['eda', 'backtranslation', 'llm', 'hybrid'],
 help='Augmentation method to use')

 # Target class sizes (Strategy B - Moderate Balance)
 parser.add_argument('--target_ambivalent', type=int, default=2041,
 help='Target size for Ambivalent class (no augmentation)')
 parser.add_argument('--target_clear_reply', type=int, default=1500,
 help='Target size for Clear Reply class')
 parser.add_argument('--target_clear_non_reply', type=int, default=1020,
 help='Target size for Clear Non-Reply class')

 # Augmentation parameters
 parser.add_argument('--augmentations_per_sample', type=int, default=2,
 help='Number of augmentations to generate per sample')

 # LLM settings (if using llm method)
 parser.add_argument('--llm_provider', type=str, default='openai',
 choices=['openai', 'anthropic'],
 help='LLM provider for paraphrasing')
 parser.add_argument('--llm_model', type=str, default=None,
 help='LLM model name')

 # Device settings
 parser.add_argument('--device', type=str, default='cuda',
 choices=['cuda', 'cpu'],
 help='Device for back-translation models')

 # Random seed
 parser.add_argument('--seed', type=int, default=42,
 help='Random seed for reproducibility')

 return parser.parse_args()

def set_seed(seed: int):
 """Set random seeds for reproducibility"""
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
 """Load the original dataset"""
 print(f"\n[1/5] Loading dataset from {data_dir}...")

 dataset = load_from_disk(data_dir)
 train_df = pd.DataFrame(dataset['train'])

 print(f" Original training samples: {len(train_df)}")

 # Print class distribution
 print("\n Current class distribution:")
 class_counts = train_df['clarity_label'].value_counts()
 for label, count in class_counts.items():
 print(f" {label}: {count} ({count/len(train_df)*100:.1f}%)")

 return train_df

def calculate_augmentation_needs(
 train_df: pd.DataFrame,
 target_clear_reply: int,
 target_clear_non_reply: int
) -> Dict[str, int]:
 """Calculate how many synthetic samples needed per class"""

 class_counts = train_df['clarity_label'].value_counts()

 needs = {
 'Ambivalent': 0, # No augmentation needed
 'Clear Reply': max(0, target_clear_reply - class_counts.get('Clear Reply', 0)),
 'Clear Non-Reply': max(0, target_clear_non_reply - class_counts.get('Clear Non-Reply', 0))
 }

 print("\n Augmentation needs:")
 for label, need in needs.items():
 original = class_counts.get(label, 0)
 target = {'Ambivalent': class_counts.get('Ambivalent', 0),
 'Clear Reply': target_clear_reply,
 'Clear Non-Reply': target_clear_non_reply}[label]
 print(f" {label}: {original} -> {target} (need {need} synthetic)")

 return needs

def augment_class(
 df: pd.DataFrame,
 class_label: str,
 num_needed: int,
 augmenter,
 augmentations_per_sample: int = 2
) -> List[Dict]:
 """Generate synthetic samples for a specific class"""

 if num_needed <= 0:
 return []

 # Filter to class
 class_df = df[df['clarity_label'] == class_label].copy()

 if len(class_df) == 0:
 print(f" Warning: No samples found for class {class_label}")
 return []

 print(f"\n Augmenting {class_label} ({num_needed} samples needed)...")

 synthetic_samples = []
 samples_generated = 0

 # Calculate how many times to iterate through the data
 iterations_needed = (num_needed // (len(class_df) * augmentations_per_sample)) + 1

 with tqdm(total=num_needed, desc=f" Generating {class_label}") as pbar:
 for iteration in range(iterations_needed):
 if samples_generated >= num_needed:
 break

 for idx, row in class_df.iterrows():
 if samples_generated >= num_needed:
 break

 # Create the text to augment (just the answer)
 original_answer = row['interview_answer']

 # Generate augmentations
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

 # Skip if identical to original
 if aug_text == original_answer:
 continue

 # Create synthetic sample (copy all fields, update answer)
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

def create_augmented_dataset(
 train_df: pd.DataFrame,
 synthetic_samples: List[Dict],
 output_dir: str
):
 """Create and save the augmented dataset"""

 print(f"\n[4/5] Creating augmented dataset...")

 # Add is_synthetic flag to original samples
 train_df = train_df.copy()
 train_df['is_synthetic'] = False
 train_df['original_idx'] = train_df.index
 train_df['augmentation_iteration'] = -1

 # Combine original and synthetic
 synthetic_df = pd.DataFrame(synthetic_samples)
 augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)

 # Shuffle
 augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

 print(f" Total samples: {len(augmented_df)}")
 print(f" Original: {len(train_df)}")
 print(f" Synthetic: {len(synthetic_samples)}")

 print("\n New class distribution:")
 class_counts = augmented_df['clarity_label'].value_counts()
 for label, count in class_counts.items():
 synthetic_count = len(augmented_df[(augmented_df['clarity_label'] == label) &
 (augmented_df['is_synthetic'] == True)])
 print(f" {label}: {count} ({count/len(augmented_df)*100:.1f}%) "
 f"[{synthetic_count} synthetic]")

 return augmented_df

def save_dataset(
 augmented_df: pd.DataFrame,
 original_data_dir: str,
 output_dir: str
):
 """Save the augmented dataset in HuggingFace format"""

 print(f"\n[5/5] Saving augmented dataset to {output_dir}...")

 # Load original dataset to get test set
 original = load_from_disk(original_data_dir)

 # Create augmented train dataset
 # Need to ensure columns match the original format
 original_columns = list(original['train'].features.keys())

 # Keep only original columns plus our new tracking columns
 keep_columns = original_columns + ['is_synthetic', 'original_idx', 'augmentation_iteration']
 available_columns = [c for c in keep_columns if c in augmented_df.columns]
 augmented_df = augmented_df[available_columns]

 # Convert to HuggingFace Dataset
 train_dataset = Dataset.from_pandas(augmented_df, preserve_index=False)
 test_dataset = original['test']

 # Create DatasetDict
 augmented_dataset = DatasetDict({
 'train': train_dataset,
 'test': test_dataset
 })

 # Save
 os.makedirs(output_dir, exist_ok=True)
 augmented_dataset.save_to_disk(output_dir)

 print(f" Saved to {output_dir}")

 # Save metadata
 metadata = {
 'original_train_size': len(original['train']),
 'augmented_train_size': len(train_dataset),
 'synthetic_samples': int(augmented_df['is_synthetic'].sum()),
 'test_size': len(test_dataset),
 'class_distribution': augmented_df['clarity_label'].value_counts().to_dict()
 }

 metadata_path = os.path.join(output_dir, 'augmentation_metadata.json')
 with open(metadata_path, 'w') as f:
 json.dump(metadata, f, indent=2)

 print(f" Saved metadata to {metadata_path}")

 return augmented_dataset

def main():
 args = parse_args()

 print("="*80)
 print("SYNTHETIC DATA GENERATION FOR SUBTASK 1")
 print("="*80)

 # Set seed
 set_seed(args.seed)
 print(f"\nRandom seed: {args.seed}")
 print(f"Augmentation method: {args.method}")

 # Load data
 train_df = load_dataset(args.data_dir)

 # Calculate needs
 print("\n[2/5] Calculating augmentation needs...")
 needs = calculate_augmentation_needs(
 train_df,
 args.target_clear_reply,
 args.target_clear_non_reply
 )

 # Initialize augmenter
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

 # Generate synthetic samples
 all_synthetic = []

 # Augment Clear Reply
 if needs['Clear Reply'] > 0:
 # For LLM augmenter, set class label
 if args.method == 'llm' and hasattr(augmenter, 'class_label'):
 augmenter.class_label = 'Clear Reply'

 synthetic = augment_class(
 train_df,
 'Clear Reply',
 needs['Clear Reply'],
 augmenter,
 args.augmentations_per_sample
 )
 all_synthetic.extend(synthetic)

 # Augment Clear Non-Reply
 if needs['Clear Non-Reply'] > 0:
 if args.method == 'llm' and hasattr(augmenter, 'class_label'):
 augmenter.class_label = 'Clear Non-Reply'

 synthetic = augment_class(
 train_df,
 'Clear Non-Reply',
 needs['Clear Non-Reply'],
 augmenter,
 args.augmentations_per_sample
 )
 all_synthetic.extend(synthetic)

 # Create combined dataset
 augmented_df = create_augmented_dataset(train_df, all_synthetic, args.output_dir)

 # Save
 save_dataset(augmented_df, args.data_dir, args.output_dir)

 print("\n" + "="*80)
 print("SYNTHETIC DATA GENERATION COMPLETE!")
 print("="*80)
 print(f"\nOutput directory: {args.output_dir}")
 print(f"Total training samples: {len(augmented_df)}")
 print(f" Original: {len(train_df)}")
 print(f" Synthetic: {len(all_synthetic)}")

 print("\n Final class distribution:")
 class_counts = augmented_df['clarity_label'].value_counts()
 for label in ['Ambivalent', 'Clear Reply', 'Clear Non-Reply']:
 count = class_counts.get(label, 0)
 print(f" {label}: {count} ({count/len(augmented_df)*100:.1f}%)")

 print("\nNext steps:")
 print(f" 1. Train DeBERTa-base: python train_deberta_augmented.py --data_dir {args.output_dir}")
 print(f" 2. Train DeBERTa-large: python train_deberta_large_augmented.py --data_dir {args.output_dir}")

 return augmented_df

if __name__ == '__main__':
 main()
