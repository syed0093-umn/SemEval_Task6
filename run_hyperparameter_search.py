"""
Hyperparameter Search Script for DeBERTa-v3-base
Tests multiple learning rate configurations to find the best one
"""

import subprocess
import pandas as pd
from datetime import datetime

print("="*80)
print("HYPERPARAMETER SEARCH FOR DeBERTa-v3-base")
print("="*80)

# Define hyperparameter configurations to test
configs = [
 {
 'name': 'Config 1: LR=2e-5 (BERT default)',
 'learning_rate': 2e-5,
 'llrd_alpha': 0.9,
 'warmup_ratio': 0.15,
 'gradient_accumulation_steps': 4
 },
 {
 'name': 'Config 2: LR=3e-5 (Recommended)',
 'learning_rate': 3e-5,
 'llrd_alpha': 0.9,
 'warmup_ratio': 0.15,
 'gradient_accumulation_steps': 4
 },
 {
 'name': 'Config 3: LR=5e-5 (Higher)',
 'learning_rate': 5e-5,
 'llrd_alpha': 0.9,
 'warmup_ratio': 0.15,
 'gradient_accumulation_steps': 4
 },
 {
 'name': 'Config 4: LR=3e-5 + Strong LLRD',
 'learning_rate': 3e-5,
 'llrd_alpha': 0.8, # Stronger decay
 'warmup_ratio': 0.15,
 'gradient_accumulation_steps': 4
 },
 {
 'name': 'Config 5: LR=3e-5 + Weak LLRD',
 'learning_rate': 3e-5,
 'llrd_alpha': 0.95, # Weaker decay
 'warmup_ratio': 0.15,
 'gradient_accumulation_steps': 4
 }
]

results = []

for i, config in enumerate(configs, 1):
 print(f"\n{'='*80}")
 print(f"Running {config['name']} ({i}/{len(configs)})")
 print(f"{'='*80}")
 print(f"Learning Rate: {config['learning_rate']:.0e}")
 print(f"LLRD Alpha: {config['llrd_alpha']}")
 print(f"Warmup Ratio: {config['warmup_ratio']}")
 print(f"Gradient Accumulation: {config['gradient_accumulation_steps']}")
 print()

 # Build command
 cmd = [
 'python3',
 'train_deberta_improved.py',
 '--learning_rate', str(config['learning_rate']),
 '--llrd_alpha', str(config['llrd_alpha']),
 '--warmup_ratio', str(config['warmup_ratio']),
 '--gradient_accumulation_steps', str(config['gradient_accumulation_steps']),
 '--num_epochs', '6',
 '--patience', '3'
 ]

 try:
 # Run training
 result = subprocess.run(cmd, capture_output=True, text=True)

 # Parse output to extract F1 score
 output = result.stdout

 # Look for the line "Dev F1 (Macro): X.XXXX"
 for line in output.split('\n'):
 if 'Dev F1 (Macro):' in line:
 f1_score = float(line.split(':')[1].strip())
 results.append({
 'Config': config['name'],
 'Learning Rate': config['learning_rate'],
 'LLRD Alpha': config['llrd_alpha'],
 'Warmup Ratio': config['warmup_ratio'],
 'Grad Accum': config['gradient_accumulation_steps'],
 'Dev F1': f1_score
 })
 print(f"\n* Completed: Dev F1 = {f1_score:.4f}")
 break

 except Exception as e:
 print(f"\n✗ Error running config: {e}")
 results.append({
 'Config': config['name'],
 'Learning Rate': config['learning_rate'],
 'LLRD Alpha': config['llrd_alpha'],
 'Warmup Ratio': config['warmup_ratio'],
 'Grad Accum': config['gradient_accumulation_steps'],
 'Dev F1': 0.0
 })

# Print results summary
print("\n" + "="*80)
print("HYPERPARAMETER SEARCH RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Dev F1', ascending=False)

print("\n")
print(results_df.to_string(index=False))

best_config = results_df.iloc[0]
print(f"\n{'='*80}")
print("BEST CONFIGURATION")
print(f"{'='*80}")
print(f"Config: {best_config['Config']}")
print(f"Learning Rate: {best_config['Learning Rate']:.0e}")
print(f"LLRD Alpha: {best_config['LLRD Alpha']}")
print(f"Warmup Ratio: {best_config['Warmup Ratio']}")
print(f"Gradient Accumulation: {int(best_config['Grad Accum'])}")
print(f"Dev F1: {best_config['Dev F1']:.4f}")

# Save results
results_df.to_csv('hyperparameter_search_results.csv', index=False)
print(f"\n* Results saved to: hyperparameter_search_results.csv")

# Compare with BERT
bert_f1 = 0.5628
improvement = best_config['Dev F1'] - bert_f1
print(f"\n{'='*80}")
print(f"COMPARISON WITH BERT")
print(f"{'='*80}")
print(f"BERT F1: {bert_f1:.4f}")
print(f"Best DeBERTa F1: {best_config['Dev F1']:.4f}")
print(f"Improvement: {improvement:+.4f} ({improvement/bert_f1*100:+.1f}%)")

if best_config['Dev F1'] > bert_f1:
 print(f"\n DeBERTa beats BERT!")
else:
 print(f"\n⚠ DeBERTa did not beat BERT, consider ensemble methods")

print("\nSearch completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
