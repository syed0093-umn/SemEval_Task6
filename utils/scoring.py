import json
import os
import numpy as np

reference_dir = os.path.join('/app/input', 'ref')
prediction_dir = os.path.join('/app/input', 'res')
score_dir = '/app/output'
print('Reading prediction')

prediction = np.genfromtxt(os.path.join(prediction_dir, 'prediction'), delimiter='\n', dtype=str)
truth = np.genfromtxt(os.path.join(reference_dir, 'testing_label'), delimiter='\n', dtype=str)

truth_list = []

for row in truth:
 truth_list.append(row.split(", "))

def f1_for_class(gold_annotations, predictions, target_class):
 """
 Calculates Precision/Recall/F1 for only one class.

 gold_annotations: list of lists (or sets) with labels per sample
 predictions: list with one prediction per sample
 target_class: the class for which we want the F1
 """
 TP = FP = FN = 0

 for gold, pred in zip(gold_annotations, predictions):
 gold = set(gold)

 if pred == target_class and target_class in gold:
 TP += 1 # we correctly predicted target_class
 elif pred == target_class and target_class not in gold:
 FP += 1 # we predicted target_class but it was not in gold
 elif target_class in gold and pred not in gold:
 FN += 1 # the class was in gold but the sample is overall wrong

 precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
 recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
 f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

 return {"precision": precision, "recall": recall, "f1": f1, "tp": TP, "fp": FP, "fn": FN}


classes = list(set([str(r) for r in np.array(truth_list).flatten()]))

f1 = []
for cls in classes:
 f1.append(f1_for_class(truth_list, prediction, cls)["f1"])

f1 = float(np.mean(f1))

print('Scores:')
scores = {
 'f1': f1,
 'accuracy': f1,
}
print(scores)

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
 score_file.write(json.dumps(scores))
