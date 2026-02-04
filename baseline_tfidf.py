"""
Baseline Model: TF-IDF + Logistic Regression
For CLARITY Task 1 - Clarity-level Classification
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import pickle

print("="*80)
print("BASELINE: TF-IDF + Logistic Regression")
print("="*80)

# Load dataset
print("\n[1/5] Loading dataset...")
dataset = load_from_disk('./QEvasion')
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"* Training samples: {len(train_df)}")
print(f"* Test samples: {len(test_df)}")

# Prepare features: Combine question + answer
print("\n[2/5] Preparing text features...")
def prepare_text(row):
 """Combine question and answer with special separator"""
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
print(f" Sample length: {len(X_train[0])} chars")

# Check label distribution
print("\n[3/5] Label distribution:")
print(train_df['clarity_label'].value_counts())

# Create TF-IDF features
print("\n[4/5] Creating TF-IDF vectorizer...")
tfidf = TfidfVectorizer(
 max_features=5000, # Limit to top 5000 features
 ngram_range=(1, 2), # Use unigrams and bigrams
 min_df=2, # Ignore terms that appear in less than 2 documents
 max_df=0.95, # Ignore terms that appear in more than 95% of documents
 stop_words='english' # Remove common English stop words
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"* TF-IDF features created")
print(f" Feature dimensions: {X_train_tfidf.shape}")
print(f" Vocabulary size: {len(tfidf.vocabulary_)}")

# Train Logistic Regression
print("\n[5/5] Training Logistic Regression...")
clf = LogisticRegression(
 max_iter=1000,
 random_state=42,
 class_weight='balanced', # Handle class imbalance
 C=1.0 # Regularization strength
)

clf.fit(X_train_tfidf, y_train)
print("* Model trained")

# Evaluate on test set
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

y_pred = clf.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1-Score (Macro): {f1_macro:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation on training set
print("\n" + "="*80)
print("CROSS-VALIDATION (5-fold)")
print("="*80)
cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=5, scoring='f1_macro')
print(f"CV F1-Scores: {cv_scores}")
print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Save model
print("\n[Saving model...]")
with open('tfidf_vectorizer.pkl', 'wb') as f:
 pickle.dump(tfidf, f)
with open('logreg_classifier.pkl', 'wb') as f:
 pickle.dump(clf, f)
print("* Model saved: tfidf_vectorizer.pkl, logreg_classifier.pkl")

# Generate predictions for submission
print("\n" + "="*80)
print("GENERATING SUBMISSION FILE")
print("="*80)

predictions = clf.predict(X_test_tfidf)

# Save predictions
with open('prediction', 'w') as f:
 for pred in predictions:
 f.write(f"{pred}\n")

print(f"* Prediction file created: 'prediction'")
print(f" Total predictions: {len(predictions)}")
print(f" Sample predictions: {predictions[:5]}")

print("\n" + "="*80)
print("BASELINE MODEL COMPLETE!")
print("="*80)
print("Next steps:")
print("1. Review results above")
print("2. Zip the 'prediction' file")
print("3. Submit to Codabench")
