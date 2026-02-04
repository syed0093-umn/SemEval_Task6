"""
Data Augmentation Module for SemEval Task 6

This module provides text augmentation methods for addressing class imbalance:
- EDA (Easy Data Augmentation): Synonym replacement, random operations
- Back-Translation: Translation round-trip for paraphrasing
- LLM Paraphrasing: High-quality paraphrases via large language model APIs
- Hybrid: Combination of EDA and back-translation

Usage:
 from data_augmentation import get_augmenter

 augmenter = get_augmenter('hybrid')
 augmented_texts = augmenter.augment("Some text to augment", num_augments=3)
"""

from .augmenters import (
 BaseAugmenter,
 EDAugmenter,
 BackTranslationAugmenter,
 LLMParaphraser,
 HybridAugmenter,
 get_augmenter
)

__all__ = [
 'BaseAugmenter',
 'EDAugmenter',
 'BackTranslationAugmenter',
 'LLMParaphraser',
 'HybridAugmenter',
 'get_augmenter'
]
