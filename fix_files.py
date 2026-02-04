#!/usr/bin/env python3
"""Script to clean up AI-generated docstrings and comments in training scripts."""

import os

base = '/home/csgrads/syed0093/SemEval_Task6'

def read_file(name):
    with open(os.path.join(base, name), 'r') as f:
        return f.read()

def write_file(name, content):
    with open(os.path.join(base, name), 'w') as f:
        f.write(content)
    print(f"  Updated: {name}")

# ============================================================================
# File 1: train_deberta.py
# ============================================================================
print("Processing train_deberta.py...")
c = read_file('train_deberta.py')

c = c.replace(
    '"""\nExperiment 5: DeBERTa-v3-base\nState-of-the-art transformer for political evasion detection\nExpected to handle ambiguous Ambivalent \u2194 Clear Reply boundary better than BERT\n"""',
    '"""\nDeBERTa-v3-base fine-tuning for clarity classification.\n"""'
)
c = c.replace('print("EXPERIMENT 5: DeBERTa-v3-base")', 'print("DeBERTa-v3-base Training")')
c = c.replace('print("State-of-the-art transformer model")', 'print("Transformer model for clarity classification")')
c = c.replace(
    '    print(f" NEW BEST MODEL! Beats BERT by {improvement_over_bert:.4f}")',
    '    print(f"  New best model (beats BERT by {improvement_over_bert:.4f})")'
)
c = c.replace(
    '    print("4. Celebrate new best score! ")\n',
    ''
)
c = c.replace('print("EXPERIMENT 5 COMPLETE!")', 'print("TRAINING COMPLETE")')
c = c.replace(
    'print(f"\\n* Updated main prediction file (DeBERTa is new best!)")',
    'print(f"\\n* Updated main prediction file")'
)
c = c.replace("'Model': 'Logistic Regression (Exp 1)'", "'Model': 'Logistic Regression'")
c = c.replace("'Model': 'SVM Linear (Exp 2)'", "'Model': 'SVM Linear'")
c = c.replace("'Model': 'Random Forest (Exp 3)'", "'Model': 'Random Forest'")
c = c.replace("'Model': 'DistilBERT (Exp 4)'", "'Model': 'DistilBERT'")
c = c.replace("'Model': 'BERT-base (Exp 4b)'", "'Model': 'BERT-base'")
c = c.replace("'Model': 'DeBERTa-v3-base (Exp 5)'", "'Model': 'DeBERTa-v3-base'")
c = c.replace(
    'print(f"\u26a0 Slightly below BERT, but still beats baseline")',
    'print(f"  Slightly below BERT, but still beats baseline")'
)
c = c.replace('print("\\nNext steps:")\n', '')
c = c.replace('print("1. Create submission.zip with prediction file")\n', '')
c = c.replace('print("2. Submit to Codabench")\n', '')
c = c.replace('print("3. Update experimental log")\n', '')
c = c.replace('print(f"Expected Test Score: ~{test_f1:.2f}")\n', '')

write_file('train_deberta.py', c)

# ============================================================================
# File 2: train_deberta_improved.py
# ============================================================================
print("Processing train_deberta_improved.py...")
c = read_file('train_deberta_improved.py')

c = c.replace(
    '"""\nExperiment 5b: DeBERTa-v3-base with Advanced Hyperparameter Optimization\nImplements state-of-the-art fine-tuning techniques:\n- Layer-wise Learning Rate Decay (LLRD)\n- Gradient Accumulation for larger effective batch size\n- Cosine Annealing with Warmup\n- Early Stopping\n- Multiple learning rate search\n"""',
    '"""\nDeBERTa-v3-base with advanced fine-tuning:\n- Layer-wise Learning Rate Decay (LLRD)\n- Gradient Accumulation for larger effective batch size\n- Cosine Annealing with Warmup\n- Early Stopping\n"""'
)
c = c.replace('print("EXPERIMENT 5b: DeBERTa-v3-base with Advanced Optimization")', 'print("DeBERTa-v3-base with Advanced Optimization")')
c = c.replace(
    '    print(f" NEW BEST MODEL! Beats BERT by {improvement_over_bert:.4f}")',
    '    print(f"  New best model (beats BERT by {improvement_over_bert:.4f})")'
)
c = c.replace(
    '    print("4. Celebrate new best score! ")\n',
    ''
)
c = c.replace('print("EXPERIMENT 5b COMPLETE!")', 'print("TRAINING COMPLETE")')
c = c.replace(
    'print(f"\\n* Updated main prediction file (DeBERTa Improved is new best!)")',
    'print(f"\\n* Updated main prediction file")'
)
c = c.replace("'Model': 'Logistic Regression (Exp 1)'", "'Model': 'Logistic Regression'")
c = c.replace("'Model': 'SVM Linear (Exp 2)'", "'Model': 'SVM Linear'")
c = c.replace("'Model': 'Random Forest (Exp 3)'", "'Model': 'Random Forest'")
c = c.replace("'Model': 'DistilBERT (Exp 4)'", "'Model': 'DistilBERT'")
c = c.replace("'Model': 'BERT-base (Exp 4b)'", "'Model': 'BERT-base'")
c = c.replace("'Model': 'DeBERTa-v3 Basic (Exp 5)'", "'Model': 'DeBERTa-v3 Basic'")
c = c.replace("'Model': 'DeBERTa-v3 Improved (Exp 5b)'", "'Model': 'DeBERTa-v3 Improved'")
c = c.replace(
    'print(f"\u26a0 Below BERT, may need further tuning")',
    'print(f"  Below BERT, may need further tuning")'
)
c = c.replace('print("\\nNext steps:")\n', '')
c = c.replace('print("1. Create submission.zip with prediction file")\n', '')
c = c.replace('print("2. Submit to Codabench")\n', '')
c = c.replace('print("3. Update experimental log")\n', '')
c = c.replace('print(f"Expected Test Score: ~{test_f1:.2f}")\n', '')
c = c.replace('\u26d4 Early stopping triggered', 'Early stopping triggered')

write_file('train_deberta_improved.py', c)

# ============================================================================
# File 3: train_deberta_large.py
# ============================================================================
print("Processing train_deberta_large.py...")
c = read_file('train_deberta_large.py')

c = c.replace(
    '"""\nPhase 2 (Option 3): DeBERTa-v3-LARGE with Boolean Features + Focal Loss\n\nUpgrades from base (184M) to large (434M params) for more capacity.\nRetains all Phase 1 improvements:\n1. Boolean Features: affirmative_questions (p=0.004), multiple_questions (p=0.047)\n2. Focal Loss: Emphasizes hard examples (67% errors are Ambivalent\u2194Clear Reply)\n3. LLRD, Gradient Accumulation, Cosine Scheduling\n\nExpected improvement: +0.03-0.05 F1 (from 0.64 to ~0.67-0.69)\nTarget: Reach 0.69 F1 (top 10 threshold)\n"""',
    '"""\nDeBERTa-v3-LARGE with Boolean Features + Focal Loss\n\nUpgrades from base (184M) to large (434M params) for more capacity.\nIncludes:\n1. Boolean Features: affirmative_questions (p=0.004), multiple_questions (p=0.047)\n2. Focal Loss: Emphasizes hard examples (67% errors are Ambivalent vs Clear Reply)\n3. LLRD, Gradient Accumulation, Cosine Scheduling\n"""'
)
c = c.replace('print("PHASE 2: DeBERTa-v3-LARGE with Boolean Features + Focal Loss")', 'print("DeBERTa-v3-LARGE with Boolean Features + Focal Loss")')
c = c.replace("description='Phase 2: DeBERTa-LARGE with Boolean Features + Focal Loss'", "description='DeBERTa-LARGE with Boolean Features + Focal Loss'")
c = c.replace('print("PHASE 2 (DeBERTa-LARGE) COMPLETE!")', 'print("DeBERTa-LARGE TRAINING COMPLETE")')
c = c.replace('    print(f"2. Celebrate! ")\n', '')
c = c.replace(
    '    print(f" SUCCESS! Reached top 10 threshold (0.69)")',
    '    print(f"  Reached target threshold (0.69)")'
)
c = c.replace('\u26a0 No improvement', 'No improvement')
c = c.replace('\u26a0 Early stopping', 'Early stopping')
c = c.replace('\u26a0 Below expected', 'Below expected')
c = c.replace('    print(f"\\nNext steps:")\n', '')
c = c.replace('    print(f"1. Submit {submission_file} to Codabench")\n', '')
c = c.replace('    print(f"3. Optional: Can ensemble multiple large models for even higher score")\n', '')
c = c.replace('    print(f"2. Try hyperparameter tuning (different learning rates, focal_gamma)")\n', '')
c = c.replace('    print(f"3. Or ensemble 3-5 large models with different seeds/hyperparams")\n', '')
c = c.replace('    print(f"2. Ensemble 3-5 large models with different hyperparameters")\n', '')
c = c.replace('    print(f"3. Try: --learning_rate 3e-5 --focal_gamma 3.0")\n', '')
c = c.replace('print(f"\\n* Phase 1 (base model): 0.64 F1")', 'print(f"\\n* Base model: 0.64 F1")')
c = c.replace('print(f"* Phase 2 (large model): {best_f1:.4f} F1")', 'print(f"* Large model: {best_f1:.4f} F1")')
c = c.replace('phase1_f1 = 0.64', 'baseline_f1 = 0.64')
c = c.replace('improvement = best_f1 - phase1_f1', 'improvement = best_f1 - baseline_f1')
c = c.replace('improvement/phase1_f1', 'improvement/baseline_f1')

write_file('train_deberta_large.py', c)

# ============================================================================
# File 4: train_deberta_large_evasion.py
# ============================================================================
print("Processing train_deberta_large_evasion.py...")
c = read_file('train_deberta_large_evasion.py')

c = c.replace(
    '"""\nSubtask 2: DeBERTa-v3-LARGE for 9-Class Evasion Prediction\n\nDirect 9-class classification of evasion techniques:\n1. Explicit (Clear Reply)\n2. Implicit, 3. Dodging, 4. General, 5. Deflection, 6. Partial (Ambivalent)\n7. Declining, 8. Ignorance, 9. Clarification (Clear Non-Reply)\n\nUses:\n1. DeBERTa-v3-LARGE (434M params) for capacity to handle 9 classes\n2. Boolean Features: affirmative_questions, multiple_questions\n3. Focal Loss: CRITICAL for imbalanced 9 classes (largest=1052, smallest=79)\n4. LLRD, Gradient Accumulation, Cosine Scheduling\n\nExpected: 9-class is harder than 3-class, so F1 will be lower\nTarget: Maximize macro F1 across all 9 classes\n"""',
    '"""\nDeBERTa-v3-LARGE for 9-class evasion prediction (Subtask 2).\n\nDirect 9-class classification of evasion techniques using:\n- DeBERTa-v3-LARGE (434M params)\n- Boolean Features: affirmative_questions, multiple_questions\n- Focal Loss for imbalanced classes\n- LLRD, Gradient Accumulation, Cosine Scheduling\n"""'
)
c = c.replace('print("SUBTASK 2 (9-CLASS EVASION) COMPLETE!")', 'print("9-CLASS EVASION TRAINING COMPLETE")')
c = c.replace('\u26a0 No improvement', 'No improvement')
c = c.replace('\u26a0 Early stopping', 'Early stopping')
c = c.replace('print("\\nNext steps:")\n', '')
c = c.replace('print(f"1. Upload {submission_file} to Codabench (Subtask 2)")\n', '')
c = c.replace('print(f"2. Compare with Subtask 1 performance")\n', '')
c = c.replace('print(f"3. Consider multi-task approach if both tasks need improvement")\n', '')

write_file('train_deberta_large_evasion.py', c)

# ============================================================================
# File 5: train_deberta_focal_features.py
# ============================================================================
print("Processing train_deberta_focal_features.py...")
c = read_file('train_deberta_focal_features.py')

c = c.replace(
    '"""\nPhase 1: DeBERTa-v3-base with Boolean Features + Focal Loss\n\nCombines two improvements:\n1. Boolean Features: affirmative_questions (p=0.004), multiple_questions (p=0.047)\n2. Focal Loss: Emphasizes hard examples (67% errors are Ambivalent\u2194Clear Reply)\n3. Retains: LLRD, Gradient Accumulation, Cosine Scheduling\n\nExpected improvement: +0.04-0.05 F1 (from 0.61 to ~0.65-0.66)\n"""',
    '"""\nDeBERTa-v3-base with Boolean Features + Focal Loss\n\nCombines:\n1. Boolean Features: affirmative_questions (p=0.004), multiple_questions (p=0.047)\n2. Focal Loss: Emphasizes hard examples (67% errors are Ambivalent vs Clear Reply)\n3. LLRD, Gradient Accumulation, Cosine Scheduling\n"""'
)
c = c.replace('print("PHASE 1: DeBERTa-v3-base with Boolean Features + Focal Loss")', 'print("DeBERTa-v3-base with Boolean Features + Focal Loss")')
c = c.replace("description='Phase 1: DeBERTa with Boolean Features + Focal Loss'", "description='DeBERTa with Boolean Features + Focal Loss'")
c = c.replace('print("PHASE 1 COMPLETE!")', 'print("TRAINING COMPLETE")')
c = c.replace('    print(f"2. Celebrate! ")\n', '')
c = c.replace(
    '    print(f" SUCCESS! Reached top 10 threshold (0.69)")',
    '    print(f"  Reached target threshold (0.69)")'
)
c = c.replace('\u26a0 No improvement', 'No improvement')
c = c.replace('\u26a0 Early stopping', 'Early stopping')
c = c.replace('\u26a0 Below expected', 'Below expected')
c = c.replace('    print(f"\\nNext steps:")\n', '')
c = c.replace('    print(f"1. Submit {submission_file} to Codabench")\n', '')
c = c.replace('    print(f"2. Proceed to Phase 2 (Ensemble of 3-5 strong models)")\n', '')
c = c.replace('    print(f"2. Consider hyperparameter tuning or proceed to Phase 2")\n', '')

write_file('train_deberta_focal_features.py', c)

# ============================================================================
# File 6: train_deberta_augmented.py
# ============================================================================
print("Processing train_deberta_augmented.py...")
c = read_file('train_deberta_augmented.py')

# Docstring looks OK but clean "AUGMENTED DATA TRAINING COMPLETE!" and "SUCCESS!"
c = c.replace('print("AUGMENTED DATA TRAINING COMPLETE!")', 'print("AUGMENTED DATA TRAINING COMPLETE")')
c = c.replace(
    '    print(f"SUCCESS! Reached top 10 threshold (0.69)")',
    '    print(f"Reached target threshold (0.69)")'
)
c = c.replace(
    '    print(f"Good progress! Close to top 10 threshold")',
    '    print(f"Close to target threshold")'
)

write_file('train_deberta_augmented.py', c)

# ============================================================================
# File 7: train_deberta_large_augmented.py
# ============================================================================
print("Processing train_deberta_large_augmented.py...")
c = read_file('train_deberta_large_augmented.py')

# Docstring looks OK, but clean some prints
c = c.replace('print("DeBERTa-LARGE AUGMENTED DATA TRAINING COMPLETE!")', 'print("DeBERTa-LARGE AUGMENTED DATA TRAINING COMPLETE")')
c = c.replace(
    '    print(f"SUCCESS! Reached top 10 threshold (0.69)")',
    '    print(f"Reached target threshold (0.69)")'
)
c = c.replace(
    '    print(f"Very close to top 10 threshold!")',
    '    print(f"Close to target threshold")'
)

write_file('train_deberta_large_augmented.py', c)

# ============================================================================
# File 8: train_multitask_deberta.py
# ============================================================================
print("Processing train_multitask_deberta.py...")
c = read_file('train_multitask_deberta.py')

c = c.replace(
    '"""\nMulti-Task DeBERTa with ALL Available Features\nExpected improvement: +10-20% F1 (0.61 \u2192 0.70-0.80)\n\nUses:\n1. Primary task: clarity_label (Ambivalent, Clear Reply, Clear Non-Reply)\n2. Auxiliary task: evasion_label (9 fine-grained types)\n3. GPT-3.5 features: summary + prediction as additional context\n4. Metadata: president, multiple_questions, affirmative_questions\n\nThis is likely what top teams are doing!\n"""',
    '"""\nMulti-Task DeBERTa with available features.\n\nUses:\n1. Primary task: clarity_label (Ambivalent, Clear Reply, Clear Non-Reply)\n2. Auxiliary task: evasion_label (9 fine-grained types)\n3. Summary features as additional context\n4. Metadata: president, multiple_questions, affirmative_questions\n"""'
)
# Clean early stopping emoji
c = c.replace('\u26d4 Early stopping triggered', 'Early stopping triggered')
# Clean target line
c = c.replace('print(f"* Target: 0.70+ F1 to reach top 15")', '')

write_file('train_multitask_deberta.py', c)

# ============================================================================
# File 9: train_multitask_large_evasion.py
# ============================================================================
print("Processing train_multitask_large_evasion.py...")
c = read_file('train_multitask_large_evasion.py')

c = c.replace(
    '"""\nMulti-Task DeBERTa-LARGE: Subtask 2 (9-class) with Subtask 1 (3-class) Auxiliary\n\nKey Insight from Paper: "Using both levels in conjunction improves performance"\n\nApproach:\n Primary Task: 9-class evasion (Subtask 2) - what we submit\n Auxiliary Task: 3-class clarity (Subtask 1) - provides hierarchical signal\n\nBenefits:\n 1. Hierarchical regularization - model learns category structure\n 2. Shared representations - 3-class task helps 9-class task\n 3. Consistency - predictions respect hierarchy\n 4. GPT-3.5 features - expert knowledge from summaries\n\nExpected: 0.45 \u2192 0.50-0.55 F1 (+0.05-0.10 improvement)\nTarget: Reach top 5 (0.50 F1) or top 3 (0.53 F1)\n"""',
    '"""\nMulti-Task DeBERTa-LARGE: Subtask 2 (9-class) with Subtask 1 (3-class) Auxiliary.\n\nApproach:\n  Primary Task: 9-class evasion (Subtask 2)\n  Auxiliary Task: 3-class clarity (Subtask 1) provides hierarchical signal\n\nBenefits:\n  1. Hierarchical regularization - model learns category structure\n  2. Shared representations - 3-class task helps 9-class task\n  3. Consistency - predictions respect hierarchy\n  4. Summary features as additional context\n"""'
)
c = c.replace('print("MULTI-TASK TRAINING COMPLETE!")', 'print("MULTI-TASK TRAINING COMPLETE")')
c = c.replace('\u26a0 No improvement', 'No improvement')
c = c.replace('\u26a0 Early stopping', 'Early stopping')
# Clean target line at end
c = c.replace('print(f"\\nTarget: 0.50 F1 (top 5) | Gap: {0.50 - best_evasion_f1:.4f}")\n', '')
c = c.replace('print(f"Submit: {submission_file} to Codabench Subtask 2")\n', '')

write_file('train_multitask_large_evasion.py', c)

# ============================================================================
# File 10: train_modernbert_clarity.py
# ============================================================================
print("Processing train_modernbert_clarity.py...")
c = read_file('train_modernbert_clarity.py')

# Docstring looks mostly OK, just clean "Subtask 1:" prefix
c = c.replace(
    '"""\nSubtask 1: ModernBERT for 3-Class Clarity Classification',
    '"""\nModernBERT for 3-class clarity classification.'
)
c = c.replace("description='Subtask 1: ModernBERT Clarity Classification'", "description='ModernBERT Clarity Classification'")
c = c.replace('print("SUBTASK 1: ModernBERT for 3-Class Clarity Classification")', 'print("ModernBERT for 3-Class Clarity Classification")')
c = c.replace('print("DONE!")', 'print("TRAINING COMPLETE")')
c = c.replace('print(f"\\nSubmit {zip_file} to Codabench")\n', '')

write_file('train_modernbert_clarity.py', c)

# ============================================================================
# File 11: train_annotator_aware.py
# ============================================================================
print("Processing train_annotator_aware.py...")
c = read_file('train_annotator_aware.py')

c = c.replace(
    '"""\nSTRATEGY B: Annotator-Aware DeBERTa Model\nExpected F1: 0.80-0.90\n\nUses annotator labels as FEATURES (not direct predictions).\nModel learns to combine:\n1. Text understanding (DeBERTa embeddings)\n2. Annotator voting patterns\n3. Annotator confidence/agreement signals\n\nThis is DEFINITELY allowed - we\'re training a model, just using all available features.\n"""',
    '"""\nAnnotator-Aware DeBERTa Model\n\nUses annotator labels as features (not direct predictions).\nModel learns to combine:\n1. Text understanding (DeBERTa embeddings)\n2. Annotator voting patterns\n3. Annotator confidence/agreement signals\n"""'
)
# Clean "EXPECTED IMPROVEMENT" print
c = c.replace(
    'print(f"EXPECTED IMPROVEMENT: 0.61 \u2192 {final_f1:.2f} F1")',
    'print(f"Final F1: {final_f1:.4f}")'
)
# Clean "Model saved!" exclamation
c = c.replace(
    '        print(f"* New best F1: {best_f1:.4f} - Model saved!")',
    '        print(f"* New best F1: {best_f1:.4f} - Saved.")'
)

write_file('train_annotator_aware.py', c)

# ============================================================================
# File 12: train_hierarchical_stage2.py
# ============================================================================
print("Processing train_hierarchical_stage2.py...")
c = read_file('train_hierarchical_stage2.py')

# Docstring looks clean, just check for issues
c = c.replace('print("STAGE 2 TRAINING COMPLETE!")', 'print("STAGE 2 TRAINING COMPLETE")')

write_file('train_hierarchical_stage2.py', c)

# ============================================================================
# File 13: train_evasion_corrected.py
# ============================================================================
print("Processing train_evasion_corrected.py...")
c = read_file('train_evasion_corrected.py')

# Docstring looks factual and clean, just check prints
c = c.replace('print("DONE! Upload submission_evasion_corrected.zip to Codabench")', 'print("TRAINING COMPLETE")')

write_file('train_evasion_corrected.py', c)

# ============================================================================
# File 14: train_deberta_evasion_augmented.py
# ============================================================================
print("Processing train_deberta_evasion_augmented.py...")
c = read_file('train_deberta_evasion_augmented.py')

# Docstring looks clean and factual. Check prints.
c = c.replace('print("TRAINING COMPLETE!")', 'print("TRAINING COMPLETE")')
# Clean "Next steps"
c = c.replace('print(f"\\nNext steps:")\n', '')
c = c.replace('print(f"1. Upload {submission_file} to Codabench (Subtask 2)")\n', '')
c = c.replace('print(f"2. Run predict_eval_evasion_augmented.py for eval predictions")\n', '')

write_file('train_deberta_evasion_augmented.py', c)

# ============================================================================
# File 15: train_deberta_large_evasion_augmented.py
# ============================================================================
print("Processing train_deberta_large_evasion_augmented.py...")
c = read_file('train_deberta_large_evasion_augmented.py')

# Docstring looks clean. No major issues.
# Just ensure no exclamation marks in key prints
# This file looks fine already.

write_file('train_deberta_large_evasion_augmented.py', c)

print("\nAll 15 files processed successfully!")
