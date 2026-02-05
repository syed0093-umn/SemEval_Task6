"""
Fix label name mismatch between training data and evaluation

Training data labels: Expected labels (evaluation):
- Partial/half-answer → Partial
- Declining to answer → Declining
- Claims ignorance → Ignorance

This mismatch causes ~0% F1 on these 3 classes!
"""

import sys
import zipfile

# Label mapping from training data format to evaluation format
LABEL_MAPPING = {
 'Partial/half-answer': 'Partial',
 'Declining to answer': 'Declining',
 'Claims ignorance': 'Ignorance',
 # These should be unchanged (but let's be explicit)
 'Explicit': 'Explicit',
 'Implicit': 'Implicit',
 'Dodging': 'Dodging',
 'General': 'General',
 'Deflection': 'Deflection',
 'Clarification': 'Clarification',
}

def fix_predictions(input_file, output_file):
 """Read predictions, fix label names, write corrected file"""

 print(f"Reading: {input_file}")
 with open(input_file, 'r') as f:
 predictions = [line.strip() for line in f.readlines()]

 print(f"Total predictions: {len(predictions)}")

 # Count changes
 changes = {}
 fixed_predictions = []

 for pred in predictions:
 if pred in LABEL_MAPPING:
 fixed = LABEL_MAPPING[pred]
 if pred != fixed:
 changes[pred] = changes.get(pred, 0) + 1
 fixed_predictions.append(fixed)
 else:
 print(f"WARNING: Unknown label '{pred}'")
 fixed_predictions.append(pred)

 # Report changes
 print(f"\nLabel changes made:")
 for old, count in changes.items():
 print(f" {old:25s} → {LABEL_MAPPING[old]:15s} ({count} times)")

 # Write fixed predictions
 print(f"\nWriting: {output_file}")
 with open(output_file, 'w') as f:
 for pred in fixed_predictions:
 f.write(f"{pred}\n")

 # Create submission zip
 zip_file = output_file.replace('prediction', 'submission') + '.zip'
 print(f"Creating: {zip_file}")
 with zipfile.ZipFile(zip_file, 'w') as zf:
 zf.write(output_file, arcname='prediction')

 return zip_file

if __name__ == '__main__':
 # Fix all existing prediction files
 prediction_files = [
 ('prediction_evasion', 'prediction_evasion_fixed'),
 ('prediction_multitask_evasion', 'prediction_multitask_evasion_fixed'),
 ]

 print("="*70)
 print("FIXING EVASION LABEL NAMES")
 print("="*70)

 for input_file, output_file in prediction_files:
 try:
 print(f"\n{'='*70}")
 zip_file = fix_predictions(input_file, output_file)
 print(f"* Created: {zip_file}")
 except FileNotFoundError:
 print(f"⚠ Skipping {input_file} (not found)")

 print("\n" + "="*70)
 print("DONE! Upload the *_fixed.zip files to Codabench")
 print("="*70)
