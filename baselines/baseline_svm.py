"""
Experiment 2: Support Vector Machine (SVM)
Trying both Linear and RBF kernels for text classification
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import pickle
import time

print("="*80)
print("EXPERIMENT 2: Support Vector Machine (SVM)")
print("="*80)

# Load dataset
print("\n[1/6] Loading dataset...")
dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"* Training samples: {len(train_df)}")
print(f"* Test samples: {len(test_df)}")

# Prepare features (same as baseline)
print("\n[2/6] Preparing text features...")
def prepare_text(row):
 question = row['question']
 answer = row['interview_answer']
 return f"Question: {question} Answer: {answer}"

train_df['text'] = train_df.apply(prepare_text, axis=1)
test_df['text'] = test_df.apply(prepare_text, axis=1)

X_train = train_df['text'].values
y_train = train_df['clarity_label'].values
X_test = test_df['text'].values
y_test = test_df['clarity_label'].values

print(f"* Text features prepared")

# Create TF-IDF features (same as baseline for fair comparison)
print("\n[3/6] Creating TF-IDF features...")
tfidf = TfidfVectorizer(
 max_features=5000,
 ngram_range=(1, 2),
 min_df=2,
 max_df=0.95,
 stop_words='english'
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"* TF-IDF features created: {X_train_tfidf.shape}")

# ============================================================================
# SVM WITH LINEAR KERNEL
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: SVM with Linear Kernel")
print("="*80)

print("\n[4/6] Training Linear SVM...")
start_time = time.time()

svm_linear = SVC(
 kernel='linear',
 C=1.0, # Regularization parameter
 class_weight='balanced', # Handle class imbalance
 random_state=42,
 max_iter=1000
)

svm_linear.fit(X_train_tfidf, y_train)
train_time_linear = time.time() - start_time

print(f"* Linear SVM trained in {train_time_linear:.2f} seconds")

# Evaluate Linear SVM
y_pred_linear = svm_linear.predict(X_test_tfidf)

acc_linear = accuracy_score(y_test, y_pred_linear)
f1_linear = f1_score(y_test, y_pred_linear, average='macro')

print(f"\nLinear SVM Results:")
print(f" Accuracy: {acc_linear:.4f}")
print(f" F1-Score (Macro): {f1_linear:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_linear))

# Cross-validation for Linear SVM
print("\nCross-Validation (5-fold):")
cv_scores_linear = cross_val_score(svm_linear, X_train_tfidf, y_train,
 cv=5, scoring='f1_macro', n_jobs=-1)
print(f"CV F1-Scores: {cv_scores_linear}")
print(f"Mean CV F1: {cv_scores_linear.mean():.4f} (+/- {cv_scores_linear.std():.4f})")

# ============================================================================
# SVM WITH RBF KERNEL
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: SVM with RBF Kernel")
print("="*80)

print("\n[5/6] Training RBF SVM...")
start_time = time.time()

svm_rbf = SVC(
 kernel='rbf',
 C=1.0, # Regularization parameter
 gamma='scale', # Kernel coefficient (auto-adjusted)
 class_weight='balanced',
 random_state=42,
 max_iter=1000
)

svm_rbf.fit(X_train_tfidf, y_train)
train_time_rbf = time.time() - start_time

print(f"* RBF SVM trained in {train_time_rbf:.2f} seconds")

# Evaluate RBF SVM
y_pred_rbf = svm_rbf.predict(X_test_tfidf)

acc_rbf = accuracy_score(y_test, y_pred_rbf)
f1_rbf = f1_score(y_test, y_pred_rbf, average='macro')

print(f"\nRBF SVM Results:")
print(f" Accuracy: {acc_rbf:.4f}")
print(f" F1-Score (Macro): {f1_rbf:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_rbf))

# Cross-validation for RBF SVM
print("\nCross-Validation (5-fold):")
cv_scores_rbf = cross_val_score(svm_rbf, X_train_tfidf, y_train,
 cv=5, scoring='f1_macro', n_jobs=-1)
print(f"CV F1-Scores: {cv_scores_rbf}")
print(f"Mean CV F1: {cv_scores_rbf.mean():.4f} (+/- {cv_scores_rbf.std():.4f})")

# ============================================================================
# COMPARISON & MODEL SELECTION
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
 'Model': ['Logistic Regression (Exp 1)', 'SVM Linear', 'SVM RBF'],
 'Dev F1': [0.4476, f1_linear, f1_rbf],
 'CV F1': [0.4104, cv_scores_linear.mean(), cv_scores_rbf.mean()],
 'Accuracy': [0.5227, acc_linear, acc_rbf],
 'Training Time (s)': ['~1', f'{train_time_linear:.2f}', f'{train_time_rbf:.2f}']
})

print(comparison.to_string(index=False))

# Select best model
best_model_name = 'SVM Linear' if f1_linear >= f1_rbf else 'SVM RBF'
best_model = svm_linear if f1_linear >= f1_rbf else svm_rbf
best_f1 = max(f1_linear, f1_rbf)

print(f"\n* Best Model: {best_model_name} (F1: {best_f1:.4f})")

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("\n[6/6] Generating submission file...")

predictions = best_model.predict(X_test_tfidf)

with open('prediction', 'w') as f:
 for pred in predictions:
 f.write(f"{pred}\n")

print(f"* Prediction file created using {best_model_name}")
print(f" Sample predictions: {predictions[:5]}")

# Save models
with open('svm_linear.pkl', 'wb') as f:
 pickle.dump(svm_linear, f)
with open('svm_rbf.pkl', 'wb') as f:
 pickle.dump(svm_rbf, f)
print("\n* Models saved: svm_linear.pkl, svm_rbf.pkl")

print("\n" + "="*80)
print("EXPERIMENT 2 COMPLETE!")
print("="*80)
print(f"Selected Model: {best_model_name}")
print(f"Expected Test Score: ~{best_f1:.2f}")
print("\nNext steps:")
print("1. Create submission.zip")
print("2. Submit to Codabench")
print("3. Log results in EXPERIMENTAL_LOG.md")
