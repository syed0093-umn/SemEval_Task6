# Duluth at SemEval 2026 Task 6 - CLARITY

Political Question Evasion Detection: classifying Q&A responses from political interviews by their level of clarity.

## Task Description

This repository contains our system for [SemEval 2026 Task 6 (CLARITY)](https://ailsntua.github.io/SemEval2025-Task6/), which focuses on detecting question evasion in political interviews. The task uses the **QEvasion** dataset and involves two subtasks:

- **Subtask 1 (3-class)**: Classify responses as *Ambivalent*, *Clear Reply*, or *Clear Non-Reply*
- **Subtask 2 (9-class)**: Fine-grained evasion classification into 9 categories (Explicit, Implicit, Dodging, General, Deflection, Partial/half-answer, Declining to answer, Claims ignorance, Clarification)

**Evaluation metric**: Macro F1-Score

## Results

| Model | Subtask 1 F1 | Notes |
|-------|-------------|-------|
| TF-IDF + Logistic Regression | 0.45 | Baseline |
| SVM (Linear) | 0.43 | |
| BERT-base | 0.56 | |
| **DeBERTa-v3-base (improved)** | **0.61** | Best model, rank 17/40 |

See [docs/EXPERIMENTAL_LOG.md](docs/EXPERIMENTAL_LOG.md) for the full experiment history and [docs/ANALYSIS_INSIGHTS.md](docs/ANALYSIS_INSIGHTS.md) for BERT error analysis.

## Repository Structure

```
.
├── training/                              # Model training scripts
│   ├── train_deberta_improved.py          # Best model: DeBERTa-v3-base with LLRD + cosine scheduling
│   ├── train_deberta.py                   # Base DeBERTa-v3 training
│   ├── train_deberta_large.py             # DeBERTa-v3-LARGE variant
│   ├── train_deberta_augmented.py         # Training with augmented data
│   ├── train_deberta_focal_features.py    # Focal loss + boolean features
│   ├── train_deberta_large_evasion.py     # 9-class subtask (DeBERTa-LARGE)
│   ├── train_multitask_deberta.py         # Multi-task learning (3-class + 9-class)
│   ├── train_multitask_large_evasion.py   # Multi-task with DeBERTa-LARGE
│   ├── train_modernbert_clarity.py        # ModernBERT alternative
│   ├── train_annotator_aware.py           # Annotator-aware features
│   └── train_hierarchical_stage2.py       # Hierarchical two-stage classification
├── evaluation/                            # Prediction and evaluation
│   ├── predict_*.py                       # Prediction and evaluation scripts
│   ├── create_ensemble*.py                # Ensemble methods
│   └── ensemble_models.py                 # Traditional ML ensembles (RF, XGBoost)
├── baselines/                             # Baseline models
│   ├── baseline_tfidf.py                  # TF-IDF + Logistic Regression
│   ├── baseline_svm.py                    # SVM variants
│   └── transformer_models.py              # BERT / DistilBERT baselines
├── utils/                                 # Utilities
│   ├── download_data.py                   # Download QEvasion dataset
│   ├── scoring.py                         # Competition scoring script
│   ├── error_analysis.py                  # BERT error analysis
│   └── load_data.py                       # Dataset inspection
├── data_augmentation/                     # Data augmentation pipeline
│   ├── augmenters.py                      # EDA, back-translation, LLM paraphrasing
│   ├── generate_synthetic_data.py         # Subtask 1 augmentation
│   └── generate_evasion_data.py           # Subtask 2 augmentation
├── scripts/                               # Shell scripts for training pipelines
├── docs/                                  # Documentation
│   ├── EXPERIMENTAL_LOG.md                # Experiment tracking
│   └── ANALYSIS_INSIGHTS.md               # Error analysis findings
└── QEvasion/                              # Dataset (gitignored, download via utils/download_data.py)
```

## Setup

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM for LARGE models, 8GB+ for base models)

```bash
pip install -r requirements.txt
```

### Download Data

```bash
python utils/download_data.py
```

This downloads the [QEvasion dataset](https://huggingface.co/datasets/ailsntua/QEvasion) from HuggingFace to `./QEvasion/`.

## Training

### Best Model (DeBERTa-v3-base with advanced optimization)

```bash
python training/train_deberta_improved.py \
    --learning_rate 3e-5 \
    --llrd_alpha 0.9 \
    --warmup_ratio 0.15 \
    --gradient_accumulation_steps 4 \
    --num_epochs 6 \
    --patience 3
```

### DeBERTa-v3-LARGE

```bash
python training/train_deberta_large.py
```

### Subtask 2 (9-class evasion)

```bash
python training/train_deberta_large_evasion.py
```

### Data Augmentation

Generate synthetic training data to address class imbalance:

```bash
# Hybrid augmentation (EDA + back-translation)
python data_augmentation/generate_synthetic_data.py --method hybrid --output_dir ./QEvasion_augmented

# Train on augmented data
python training/train_deberta_augmented.py --data_dir ./QEvasion_augmented
```

## Key Techniques

- **Layer-wise Learning Rate Decay (LLRD)**: Different learning rates per transformer layer, with lower rates for earlier layers
- **Gradient Accumulation**: Effective batch size of 32 with batch size 8 and 4 accumulation steps
- **Cosine Annealing with Warmup**: Learning rate schedule with 15% linear warmup followed by cosine decay
- **Focal Loss**: Addresses class imbalance by down-weighting easy examples (used in DeBERTa-LARGE variants)
- **Data Augmentation**: EDA, back-translation, and LLM-based paraphrasing for minority class oversampling

## Dataset

The QEvasion dataset (Kalouli et al.) contains 3,448 training and 308 test examples of political Q&A pairs, each annotated with clarity labels by 3 independent annotators.

**Class distribution (training set)**:
- Ambivalent: 59.2%
- Clear Reply: 30.5%
- Clear Non-Reply: 10.3%

## License

This project is released for research and educational purposes.

## Acknowledgments

- SemEval 2026 Task 6 organizers
- [QEvasion dataset](https://huggingface.co/datasets/ailsntua/QEvasion) by Kalouli et al.
- Microsoft for the [DeBERTa-v3](https://huggingface.co/microsoft/deberta-v3-base) model

