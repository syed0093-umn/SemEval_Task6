"""
Experiment 2 (Fixed): Support Vector Machine with proper preprocessing
Fixing convergence issues and improving performance
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
import pickle
import time

print("="*80)
print("EXPERIMENT 2 (FIXED): SVM with Proper Preprocessing")
print("="*80)

# Load dataset
print("\n[1/5] Loading dataset...")
dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"* Training samples: {len(train_df)}")
print(f"* Test samples: {len(test_df)}")

# Prepare features
print("\n[2/5] Preparing text features...")
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

# Create TF-IDF features
print("\n[3/5] Creating TF-IDF features with L2 normalization...")
tfidf = TfidfVectorizer(
 max_features=5000,
 ngram_range=(1, 2),
 min_df=2,
 max_df=0.95,
 stop_words='english',
 norm='l2', # L2 normalization for SVM
 use_idf=True,
 sublinear_tf=True # Use sublinear tf scaling
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"* TF-IDF features created: {X_train_tfidf.shape}")
print(f" Normalized: L2 norm applied")

# ============================================================================
# GRID SEARCH FOR BEST SVM HYPERPARAMETERS
# ============================================================================
print("\n[4/5] Grid search for optimal hyperparameters...")
print("Testing different C values and kernels...")

# Define parameter grid
param_grid = {
 'C': [0.1, 1.0, 10.0],
 'kernel': ['linear', 'rbf'],
 'gamma': ['scale', 'auto']
}

# Create SVM with more iterations
svm_base = SVC(
 class_weight='balanced',
 random_state=42,
 max_iter=5000, # Increased from 1000
 cache_size=500 # Increase cache for faster training
)

# Grid search
grid_search = GridSearchCV(
 svm_base,
 param_grid,
 cv=3, # 3-fold CV for speed
 scoring='f1_macro',
 n_jobs=-1,
 verbose=1
)

print("\nStarting grid search...")
start_time = time.time()
grid_search.fit(X_train_tfidf, y_train)
search_time = time.time() - start_time

print(f"\n* Grid search completed in {search_time:.2f} seconds")
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

# Get best model
best_svm = grid_search.best_estimator_

# ============================================================================
# EVALUATE BEST MODEL
# ============================================================================
print("\n[5/5] Evaluating best SVM model...")

y_pred = best_svm.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print("\n" + "="*80)
print("BEST SVM MODEL RESULTS")
print("="*80)
print(f"\nHyperparameters:")
for param, value in grid_search.best_params_.items():
 print(f" {param}: {value}")

print(f"\nPerformance:")
print(f" Dev Accuracy: {acc:.4f}")
print(f" Dev F1 (Macro): {f1:.4f}")
print(f" CV F1 (3-fold): {grid_search.best_score_:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# ============================================================================
# COMPARISON WITH BASELINE
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
 'Model': ['Logistic Regression', 'SVM (Original)', 'SVM (Fixed)'],
 'Dev F1': [0.4476, 0.2974, f1],
 'Accuracy': [0.5227, 0.3052, acc]
})

print(comparison.to_string(index=False))

improvement = f1 - 0.2974
print(f"\n* Improvement over original SVM: {improvement:+.4f}")

if f1 > 0.4476:
 print(f"* BEAT BASELINE! New best F1: {f1:.4f}")
elif f1 > 0.4:
 print(f"* Close to baseline (F1: {f1:.4f})")
else:
 print(f"⚠ Still below baseline, but improved from original SVM")

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSION")
print("="*80)

predictions = best_svm.predict(X_test_tfidf)

# Save prediction file
with open('prediction', 'w') as f:
 for pred in predictions:
 f.write(f"{pred}\n")

print(f"* Prediction file created")
print(f" Sample predictions: {predictions[:5]}")
print(f" Class distribution:")
unique, counts = np.unique(predictions, return_counts=True)
for label, count in zip(unique, counts):
 print(f" {label}: {count} ({count/len(predictions)*100:.1f}%)")

# Save model
with open('svm_best.pkl', 'wb') as f:
 pickle.dump(best_svm, f)
with open('tfidf_vectorizer_svm.pkl', 'wb') as f:
 pickle.dump(tfidf, f)

print("\n* Models saved: svm_best.pkl, tfidf_vectorizer_svm.pkl")

print("\n" + "="*80)
print("READY FOR SUBMISSION!")
print("="*80)
print(f"Expected Test Score: ~{f1:.2f}")
