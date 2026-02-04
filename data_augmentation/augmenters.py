"""
Data Augmentation Classes for Text Classification

Three augmentation methods:
1. EDA (Easy Data Augmentation) - Fast, no API costs
2. Back-Translation - Moderate quality, no API costs
3. LLM Paraphrasing - Highest quality, requires API keys
"""

import random
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class BaseAugmenter(ABC):
 """Base class for text augmentation"""

 @abstractmethod
 def augment(self, text: str, num_augments: int = 1) -> List[str]:
 """Generate augmented versions of the input text"""
 pass

 def augment_batch(self, texts: List[str], num_augments: int = 1) -> List[List[str]]:
 """Augment a batch of texts"""
 return [self.augment(text, num_augments) for text in texts]

class EDAugmenter(BaseAugmenter):
 """
 Easy Data Augmentation (EDA)

 Implements four simple operations:
 1. Synonym Replacement (SR) - Replace n words with synonyms
 2. Random Insertion (RI) - Insert n synonyms of random words
 3. Random Swap (RS) - Swap n pairs of words
 4. Random Deletion (RD) - Delete words with probability p

 Reference: Wei & Zou (2019) "EDA: Easy Data Augmentation Techniques"
 """

 def __init__(self,
 alpha_sr: float = 0.1, # % of words to replace with synonyms
 alpha_ri: float = 0.1, # % of words for random insertion
 alpha_rs: float = 0.1, # % of words for random swap
 p_rd: float = 0.1, # probability of random deletion
 use_nlpaug: bool = True):
 """
 Args:
 alpha_sr: Percent of words to replace with synonyms
 alpha_ri: Percent of words for random insertion
 alpha_rs: Percent of words for random swap
 p_rd: Probability of random deletion for each word
 use_nlpaug: Whether to use nlpaug library (recommended)
 """
 self.alpha_sr = alpha_sr
 self.alpha_ri = alpha_ri
 self.alpha_rs = alpha_rs
 self.p_rd = p_rd
 self.use_nlpaug = use_nlpaug

 if use_nlpaug:
 self._init_nlpaug()
 else:
 self._init_wordnet()

 def _init_nlpaug(self):
 """Initialize nlpaug augmenters"""
 try:
 import nlpaug.augmenter.word as naw

 # Synonym replacement using WordNet
 self.syn_aug = naw.SynonymAug(
 aug_src='wordnet',
 aug_min=1,
 aug_max=5,
 aug_p=self.alpha_sr
 )

 # Random word swap
 self.swap_aug = naw.RandomWordAug(
 action='swap',
 aug_min=1,
 aug_max=3,
 aug_p=self.alpha_rs
 )

 # Random word deletion
 self.delete_aug = naw.RandomWordAug(
 action='delete',
 aug_min=1,
 aug_max=3,
 aug_p=self.p_rd
 )

 self.nlpaug_available = True
 print(" EDA: Using nlpaug library")

 except ImportError:
 print(" Warning: nlpaug not available, using basic EDA")
 self.nlpaug_available = False
 self._init_wordnet()

 def _init_wordnet(self):
 """Initialize WordNet for synonym replacement"""
 try:
 from nltk.corpus import wordnet
 import nltk
 nltk.download('wordnet', quiet=True)
 nltk.download('omw-1.4', quiet=True)
 self.wordnet = wordnet
 self.wordnet_available = True
 except ImportError:
 self.wordnet_available = False
 print(" Warning: NLTK/WordNet not available")

 def _get_synonyms(self, word: str) -> List[str]:
 """Get synonyms for a word using WordNet"""
 if not hasattr(self, 'wordnet') or not self.wordnet_available:
 return []

 synonyms = set()
 for syn in self.wordnet.synsets(word):
 for lemma in syn.lemmas():
 synonym = lemma.name().replace('_', ' ')
 if synonym.lower() != word.lower():
 synonyms.add(synonym)
 return list(synonyms)

 def _synonym_replacement(self, words: List[str], n: int) -> List[str]:
 """Replace n words with synonyms"""
 new_words = words.copy()
 random_indices = list(range(len(words)))
 random.shuffle(random_indices)

 num_replaced = 0
 for idx in random_indices:
 if num_replaced >= n:
 break

 word = words[idx]
 synonyms = self._get_synonyms(word)
 if synonyms:
 new_words[idx] = random.choice(synonyms)
 num_replaced += 1

 return new_words

 def _random_insertion(self, words: List[str], n: int) -> List[str]:
 """Randomly insert n synonyms"""
 new_words = words.copy()
 for _ in range(n):
 if not new_words:
 break
 random_word = random.choice(new_words)
 synonyms = self._get_synonyms(random_word)
 if synonyms:
 random_synonym = random.choice(synonyms)
 random_idx = random.randint(0, len(new_words))
 new_words.insert(random_idx, random_synonym)
 return new_words

 def _random_swap(self, words: List[str], n: int) -> List[str]:
 """Randomly swap n pairs of words"""
 new_words = words.copy()
 for _ in range(n):
 if len(new_words) < 2:
 break
 idx1, idx2 = random.sample(range(len(new_words)), 2)
 new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
 return new_words

 def _random_deletion(self, words: List[str], p: float) -> List[str]:
 """Randomly delete words with probability p"""
 if len(words) == 1:
 return words
 new_words = [word for word in words if random.random() > p]
 if not new_words:
 return [random.choice(words)]
 return new_words

 def augment(self, text: str, num_augments: int = 1) -> List[str]:
 """Generate augmented versions of text using EDA"""
 if self.use_nlpaug and hasattr(self, 'nlpaug_available') and self.nlpaug_available:
 return self._augment_nlpaug(text, num_augments)
 else:
 return self._augment_basic(text, num_augments)

 def _augment_nlpaug(self, text: str, num_augments: int) -> List[str]:
 """Use nlpaug for augmentation"""
 augmented = []

 for _ in range(num_augments):
 # Randomly select an augmentation method
 aug_choice = random.choice(['synonym', 'swap', 'delete'])

 try:
 if aug_choice == 'synonym':
 aug_text = self.syn_aug.augment(text)
 elif aug_choice == 'swap':
 aug_text = self.swap_aug.augment(text)
 else:
 aug_text = self.delete_aug.augment(text)

 # nlpaug can return a list or string
 if isinstance(aug_text, list):
 aug_text = aug_text[0] if aug_text else text

 if aug_text and aug_text != text:
 augmented.append(aug_text)
 else:
 augmented.append(text)

 except Exception:
 augmented.append(text)

 return augmented

 def _augment_basic(self, text: str, num_augments: int) -> List[str]:
 """Basic EDA without nlpaug"""
 words = text.split()
 n_sr = max(1, int(self.alpha_sr * len(words)))
 n_ri = max(1, int(self.alpha_ri * len(words)))
 n_rs = max(1, int(self.alpha_rs * len(words)))

 augmented = []
 operations = ['sr', 'ri', 'rs', 'rd']

 for _ in range(num_augments):
 op = random.choice(operations)

 if op == 'sr':
 new_words = self._synonym_replacement(words, n_sr)
 elif op == 'ri':
 new_words = self._random_insertion(words, n_ri)
 elif op == 'rs':
 new_words = self._random_swap(words, n_rs)
 else:
 new_words = self._random_deletion(words, self.p_rd)

 augmented.append(' '.join(new_words))

 return augmented

class BackTranslationAugmenter(BaseAugmenter):
 """
 Back-Translation Augmentation

 Translates text to an intermediate language and back to English.
 This creates paraphrases while preserving semantics.

 Uses Helsinki-NLP/opus-mt models for translation.
 """

 def __init__(self,
 intermediate_lang: str = 'de', # German as default
 device: str = 'cuda'):
 """
 Args:
 intermediate_lang: Intermediate language code (de, fr, es, etc.)
 device: 'cuda' or 'cpu'
 """
 self.intermediate_lang = intermediate_lang
 self.device = device
 self._init_models()

 def _init_models(self):
 """Initialize translation models"""
 try:
 from transformers import MarianMTModel, MarianTokenizer
 import torch

 self.torch = torch

 # Check device availability
 if self.device == 'cuda' and not torch.cuda.is_available():
 self.device = 'cpu'
 print(" Warning: CUDA not available, using CPU for back-translation")

 # English -> Intermediate language
 en_to_lang_name = f"Helsinki-NLP/opus-mt-en-{self.intermediate_lang}"
 print(f" Loading {en_to_lang_name}...")
 self.en_to_lang_tokenizer = MarianTokenizer.from_pretrained(en_to_lang_name)
 self.en_to_lang_model = MarianMTModel.from_pretrained(en_to_lang_name).to(self.device)

 # Intermediate language -> English
 lang_to_en_name = f"Helsinki-NLP/opus-mt-{self.intermediate_lang}-en"
 print(f" Loading {lang_to_en_name}...")
 self.lang_to_en_tokenizer = MarianTokenizer.from_pretrained(lang_to_en_name)
 self.lang_to_en_model = MarianMTModel.from_pretrained(lang_to_en_name).to(self.device)

 self.models_available = True
 print(f" Back-translation: Using {self.intermediate_lang.upper()} as intermediate language")

 except Exception as e:
 print(f" Warning: Could not load translation models: {e}")
 self.models_available = False

 def _translate(self, text: str, model, tokenizer) -> str:
 """Translate text using a model"""
 inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
 inputs = {k: v.to(self.device) for k, v in inputs.items()}

 with self.torch.no_grad():
 translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)

 return tokenizer.decode(translated[0], skip_special_tokens=True)

 def augment(self, text: str, num_augments: int = 1) -> List[str]:
 """Generate augmented versions via back-translation"""
 if not self.models_available:
 return [text] * num_augments

 augmented = []

 for _ in range(num_augments):
 try:
 # English -> Intermediate
 intermediate = self._translate(text, self.en_to_lang_model, self.en_to_lang_tokenizer)

 # Intermediate -> English
 back_translated = self._translate(intermediate, self.lang_to_en_model, self.lang_to_en_tokenizer)

 # Only add if different from original
 if back_translated and back_translated != text:
 augmented.append(back_translated)
 else:
 augmented.append(text)

 except Exception as e:
 print(f" Warning: Back-translation failed: {e}")
 augmented.append(text)

 return augmented

 def augment_batch(self, texts: List[str], num_augments: int = 1) -> List[List[str]]:
 """Batch augmentation for efficiency"""
 if not self.models_available:
 return [[text] * num_augments for text in texts]

 results = []

 for text in texts:
 augmented = self.augment(text, num_augments)
 results.append(augmented)

 return results

class LLMParaphraser(BaseAugmenter):
 """
 LLM-based Paraphrasing using OpenAI API or Anthropic Claude

 Highest quality augmentation but requires API keys.
 Uses targeted prompts for class-specific paraphrasing.
 """

 def __init__(self,
 provider: str = 'openai', # 'openai' or 'anthropic'
 model: str = None,
 api_key: str = None,
 class_label: str = None):
 """
 Args:
 provider: 'openai' or 'anthropic'
 model: Model name (default: gpt-4 for openai, claude-3-sonnet for anthropic)
 api_key: API key (can also be set via environment variable)
 class_label: Target class for specialized prompts
 """
 self.provider = provider
 self.class_label = class_label
 self.api_key = api_key

 if model is None:
 self.model = 'gpt-4' if provider == 'openai' else 'claude-3-sonnet-20240229'
 else:
 self.model = model

 self._init_client()

 def _init_client(self):
 """Initialize API client"""
 import os

 if self.provider == 'openai':
 try:
 from openai import OpenAI
 api_key = self.api_key or os.environ.get('OPENAI_API_KEY')
 if api_key:
 self.client = OpenAI(api_key=api_key)
 self.client_available = True
 print(f" LLM Paraphraser: Using OpenAI {self.model}")
 else:
 print(" Warning: OPENAI_API_KEY not found")
 self.client_available = False
 except ImportError:
 print(" Warning: openai package not installed")
 self.client_available = False

 elif self.provider == 'anthropic':
 try:
 from anthropic import Anthropic
 api_key = self.api_key or os.environ.get('ANTHROPIC_API_KEY')
 if api_key:
 self.client = Anthropic(api_key=api_key)
 self.client_available = True
 print(f" LLM Paraphraser: Using Anthropic {self.model}")
 else:
 print(" Warning: ANTHROPIC_API_KEY not found")
 self.client_available = False
 except ImportError:
 print(" Warning: anthropic package not installed")
 self.client_available = False
 else:
 print(f" Warning: Unknown provider {self.provider}")
 self.client_available = False

 def _get_prompt(self, text: str, question: str = None, answer: str = None) -> str:
 """Generate appropriate prompt based on class label"""

 base_prompt = """You are an expert at paraphrasing text while preserving its meaning and tone.

Paraphrase the following text. Keep the same meaning but use different words and sentence structure.
Do not add new information or change the intent.
Return ONLY the paraphrased text, nothing else."""

 if self.class_label == 'Clear Reply':
 base_prompt = """You are an expert at paraphrasing political interview answers.

The following is a CLEAR REPLY - an answer that directly addresses the question asked.
Paraphrase this answer while:
1. Keeping it as a clear, direct response to the question
2. Preserving the specific policy positions or facts mentioned
3. Maintaining the same level of directness and clarity

Return ONLY the paraphrased answer, nothing else."""

 elif self.class_label == 'Clear Non-Reply':
 base_prompt = """You are an expert at paraphrasing political interview answers.

The following is a CLEAR NON-REPLY - an answer that clearly does NOT address the question asked.
Paraphrase this answer while:
1. Keeping it as a clear deflection/non-answer
2. Preserving the topic shift or redirect
3. Maintaining the evasive nature while using different words

Return ONLY the paraphrased answer, nothing else."""

 if question and answer:
 return f"""{base_prompt}

Question: {question}

Answer to paraphrase:
{answer}"""
 else:
 return f"""{base_prompt}

Text to paraphrase:
{text}"""

 def _call_openai(self, prompt: str) -> str:
 """Call OpenAI API"""
 response = self.client.chat.completions.create(
 model=self.model,
 messages=[{"role": "user", "content": prompt}],
 temperature=0.7,
 max_tokens=500
 )
 return response.choices[0].message.content.strip()

 def _call_anthropic(self, prompt: str) -> str:
 """Call Anthropic API"""
 response = self.client.messages.create(
 model=self.model,
 max_tokens=500,
 messages=[{"role": "user", "content": prompt}]
 )
 return response.content[0].text.strip()

 def augment(self, text: str, num_augments: int = 1,
 question: str = None, answer: str = None) -> List[str]:
 """Generate paraphrased versions using LLM"""
 if not self.client_available:
 return [text] * num_augments

 augmented = []

 for _ in range(num_augments):
 try:
 prompt = self._get_prompt(text, question, answer)

 if self.provider == 'openai':
 paraphrased = self._call_openai(prompt)
 else:
 paraphrased = self._call_anthropic(prompt)

 if paraphrased and paraphrased != text:
 augmented.append(paraphrased)
 else:
 augmented.append(text)

 except Exception as e:
 print(f" Warning: LLM paraphrasing failed: {e}")
 augmented.append(text)

 return augmented

class HybridAugmenter(BaseAugmenter):
 """
 Hybrid Augmenter that combines multiple augmentation methods.

 Uses a mix of EDA (fast) and back-translation (quality) to generate
 diverse augmentations while managing computational costs.
 """

 def __init__(self,
 eda_ratio: float = 0.6, # 60% EDA
 backtrans_ratio: float = 0.4, # 40% back-translation
 device: str = 'cuda'):
 """
 Args:
 eda_ratio: Proportion of augmentations from EDA
 backtrans_ratio: Proportion from back-translation
 device: Device for back-translation models
 """
 self.eda_ratio = eda_ratio
 self.backtrans_ratio = backtrans_ratio

 print("Initializing Hybrid Augmenter...")

 # Initialize augmenters
 self.eda = EDAugmenter()
 self.backtrans = BackTranslationAugmenter(device=device)

 def augment(self, text: str, num_augments: int = 1) -> List[str]:
 """Generate augmented versions using multiple methods"""
 augmented = []

 # Calculate how many from each method
 n_eda = max(1, int(num_augments * self.eda_ratio))
 n_backtrans = num_augments - n_eda

 # Get EDA augmentations
 if n_eda > 0:
 eda_augs = self.eda.augment(text, n_eda)
 augmented.extend(eda_augs)

 # Get back-translation augmentations
 if n_backtrans > 0 and self.backtrans.models_available:
 bt_augs = self.backtrans.augment(text, n_backtrans)
 augmented.extend(bt_augs)
 elif n_backtrans > 0:
 # Fallback to EDA if back-translation not available
 eda_augs = self.eda.augment(text, n_backtrans)
 augmented.extend(eda_augs)

 return augmented[:num_augments] # Ensure we return exactly num_augments

def get_augmenter(method: str, **kwargs) -> BaseAugmenter:
 """
 Factory function to get an augmenter by name.

 Args:
 method: 'eda', 'backtranslation', 'llm', or 'hybrid'
 **kwargs: Additional arguments passed to the augmenter

 Returns:
 BaseAugmenter instance
 """
 method = method.lower()

 if method == 'eda':
 return EDAugmenter(**kwargs)
 elif method in ['backtranslation', 'back_translation', 'bt']:
 return BackTranslationAugmenter(**kwargs)
 elif method == 'llm':
 return LLMParaphraser(**kwargs)
 elif method == 'hybrid':
 return HybridAugmenter(**kwargs)
 else:
 raise ValueError(f"Unknown augmentation method: {method}. "
 f"Choose from: eda, backtranslation, llm, hybrid")
