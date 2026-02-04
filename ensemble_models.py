"""
Experiment 3: Ensemble Methods (Random Forest & XGBoost)
Testing tree-based models for comparison with linear models
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import pickle
import time

# Check if xgboost is available
try:
 import xgboost as xgb
 XGBOOST_AVAILABLE = True
except ImportError:
 XGBOOST_AVAILABLE = False
 print("⚠ XGBoost not installed. Will skip XGBoost model.")

print("="*80)
print("EXPERIMENT 3: Ensemble Methods")
print("="*80)

# Load dataset
print("\n[1/6] Loading dataset...")
dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"* Training samples: {len(train_df)}")
print(f"* Test samples: {len(test_df)}")

# Prepare features
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

# Create TF-IDF features
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

print(f"* TF-IDF features: {X_train_tfidf.shape}")

# Store all results
results = []

# ============================================================================
# MODEL 1: RANDOM FOREST
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: Random Forest")
print("="*80)

print("\n[4/6] Training Random Forest...")
start_time = time.time()

rf = RandomForestClassifier(
 n_estimators=100, # Number of trees
 max_depth=20, # Limit depth to prevent overfitting
 min_samples_split=10, # Minimum samples to split node
 min_samples_leaf=4, # Minimum samples in leaf
 class_weight='balanced', # Handle class imbalance
 random_state=42,
 n_jobs=-1, # Use all cores
 verbose=0
)

rf.fit(X_train_tfidf, y_train)
train_time_rf = time.time() - start_time

print(f"* Random Forest trained in {train_time_rf:.2f} seconds")

# Evaluate
y_pred_rf = rf.predict(X_test_tfidf)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='macro')

print(f"\nRandom Forest Results:")
print(f" Dev Accuracy: {acc_rf:.4f}")
print(f" Dev F1 (Macro): {f1_rf:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Cross-validation
print("\nCross-Validation (3-fold):")
cv_scores_rf = cross_val_score(rf, X_train_tfidf, y_train,
 cv=3, scoring='f1_macro', n_jobs=-1)
print(f"CV F1-Scores: {cv_scores_rf}")
print(f"Mean CV F1: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

# Feature importance (top 10)
feature_names = tfidf.get_feature_names_out()
importances = rf.feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]
print("\nTop 10 Most Important Features:")
for idx in top_indices:
 print(f" {feature_names[idx]}: {importances[idx]:.4f}")

results.append({
 'Model': 'Random Forest',
 'Dev F1': f1_rf,
 'CV F1': cv_scores_rf.mean(),
 'Accuracy': acc_rf,
 'Train Time': train_time_rf,
 'predictions': y_pred_rf
})

# Save model
with open('rf_classifier.pkl', 'wb') as f:
 pickle.dump(rf, f)

# ============================================================================
# MODEL 2: XGBOOST (if available)
# ============================================================================
if XGBOOST_AVAILABLE:
 print("\n" + "="*80)
 print("MODEL 2: XGBoost")
 print("="*80)

 print("\n[5/6] Training XGBoost...")
 start_time = time.time()

 # Encode labels for XGBoost
 from sklearn.preprocessing import LabelEncoder
 le = LabelEncoder()
 y_train_encoded = le.fit_transform(y_train)
 y_test_encoded = le.transform(y_test)

 xgb_clf = xgb.XGBClassifier(
 n_estimators=100,
 max_depth=6,
 learning_rate=0.1,
 subsample=0.8,
 colsample_bytree=0.8,
 random_state=42,
 n_jobs=-1,
 eval_metric='mlogloss'
 )

 xgb_clf.fit(X_train_tfidf, y_train_encoded)
 train_time_xgb = time.time() - start_time

 print(f"* XGBoost trained in {train_time_xgb:.2f} seconds")

 # Evaluate
 y_pred_xgb_encoded = xgb_clf.predict(X_test_tfidf)
 y_pred_xgb = le.inverse_transform(y_pred_xgb_encoded)

 acc_xgb = accuracy_score(y_test, y_pred_xgb)
 f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')

 print(f"\nXGBoost Results:")
 print(f" Dev Accuracy: {acc_xgb:.4f}")
 print(f" Dev F1 (Macro): {f1_xgb:.4f}")
 print("\nDetailed Classification Report:")
 print(classification_report(y_test, y_pred_xgb))

 # Cross-validation
 print("\nCross-Validation (3-fold):")
 cv_scores_xgb = cross_val_score(xgb_clf, X_train_tfidf, y_train_encoded,
 cv=3, scoring='f1_macro', n_jobs=-1)
 print(f"CV F1-Scores: {cv_scores_xgb}")
 print(f"Mean CV F1: {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std():.4f})")

 results.append({
 'Model': 'XGBoost',
 'Dev F1': f1_xgb,
 'CV F1': cv_scores_xgb.mean(),
 'Accuracy': acc_xgb,
 'Train Time': train_time_xgb,
 'predictions': y_pred_xgb
 })

 # Save model
 with open('xgb_classifier.pkl', 'wb') as f:
 pickle.dump(xgb_clf, f)
 with open('label_encoder.pkl', 'wb') as f:
 pickle.dump(le, f)

# ============================================================================
# COMPARISON WITH ALL MODELS
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame([
 {'Model': 'Logistic Regression (Exp 1)', 'Dev F1': 0.4476, 'CV F1': 0.4104,
 'Accuracy': 0.5227, 'Test Score': 0.45},
 {'Model': 'SVM Linear (Exp 2)', 'Dev F1': 0.4270, 'CV F1': 0.4220,
 'Accuracy': 0.5097, 'Test Score': 0.43},
] + [
 {'Model': r['Model'], 'Dev F1': r['Dev F1'], 'CV F1': r['CV F1'],
 'Accuracy': r['Accuracy'], 'Test Score': 'TBD'}
 for r in results
])

print("\n")
print(comparison_df.to_string(index=False))

# Select best model
best_result = max(results, key=lambda x: x['Dev F1'])
best_model_name = best_result['Model']
best_f1 = best_result['Dev F1']
best_predictions = best_result['predictions']

print(f"\n* Best Ensemble Model: {best_model_name} (F1: {best_f1:.4f})")

if best_f1 > 0.4476:
 print(f" BEATS LOGISTIC REGRESSION! Improvement: +{best_f1 - 0.4476:.4f}")
elif best_f1 > 0.43:
 print(f"* Better than SVM, close to LogReg baseline")
else:
 print(f"⚠ Below both baselines")

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("\n[6/6] Generating submission file...")

with open('prediction', 'w') as f:
 for pred in best_predictions:
 f.write(f"{pred}\n")

print(f"* Prediction file created using {best_model_name}")
print(f" Sample predictions: {best_predictions[:5]}")

# Class distribution
unique, counts = np.unique(best_predictions, return_counts=True)
print(f"\n Prediction distribution:")
for label, count in zip(unique, counts):
 print(f" {label}: {count} ({count/len(best_predictions)*100:.1f}%)")

print("\n" + "="*80)
print("EXPERIMENT 3 COMPLETE!")
print("="*80)
print(f"Best Model: {best_model_name}")
print(f"Expected Test Score: ~{best_f1:.2f}")
print("\nNext steps:")
print("1. Create submission.zip")
print("2. Submit to Codabench")
print("3. Update experimental log")
print("4. If not better than 0.45, move to transformers (Option B)")
