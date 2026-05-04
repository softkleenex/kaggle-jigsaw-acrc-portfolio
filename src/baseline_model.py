"""
Jigsaw ACRC Baseline Model
==========================
This script implements a baseline model for the Jigsaw ACRC competition using:
- TF-IDF vectorization
- LightGBM classifier
- Combined features (body + rule + examples)
- Stratified 5-Fold Cross-Validation
- ROC-AUC scoring
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import time
import os
from datetime import datetime

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_data():
    """Load train, test, and sample submission data"""
    print("Loading data...")
    train_df = pd.read_csv('/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/data/train.csv')
    test_df = pd.read_csv('/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/data/test.csv')
    sample_submission = pd.read_csv('/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/data/sample_submission.csv')

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Target distribution:\n{train_df['rule_violation'].value_counts()}")

    return train_df, test_df, sample_submission

def create_combined_features(df):
    """
    Combine body, rule, and examples into a single text feature
    Feature engineering: body + rule + positive_example_1 + positive_example_2 + negative_example_1 + negative_example_2
    """
    print("Creating combined text features...")

    # Handle missing values (fill with empty string)
    text_columns = ['body', 'rule', 'positive_example_1', 'positive_example_2',
                   'negative_example_1', 'negative_example_2']

    for col in text_columns:
        df[col] = df[col].fillna('').astype(str)

    # Combine all text features with separators
    combined_text = (
        df['body'] + ' [SEP] ' +
        df['rule'] + ' [SEP] ' +
        df['positive_example_1'] + ' [SEP] ' +
        df['positive_example_2'] + ' [SEP] ' +
        df['negative_example_1'] + ' [SEP] ' +
        df['negative_example_2']
    )

    return combined_text

def create_tfidf_features(train_text, test_text, max_features=10000):
    """
    Create TF-IDF features from combined text
    """
    print(f"Creating TF-IDF features with max_features={max_features}...")

    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
        min_df=2,  # Minimum document frequency
        max_df=0.95,  # Maximum document frequency
        sublinear_tf=True,  # Apply sublinear tf scaling
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english'
    )

    # Fit on train and transform both train and test
    train_tfidf = tfidf.fit_transform(train_text)
    test_tfidf = tfidf.transform(test_text)

    print(f"TF-IDF feature shape: {train_tfidf.shape}")
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

    return train_tfidf, test_tfidf, tfidf

def train_with_cv(X, y, n_splits=5):
    """
    Train model with Stratified K-Fold Cross-Validation
    """
    print(f"\nTraining with {n_splits}-Fold Stratified Cross-Validation...")

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # Store CV scores and predictions
    cv_scores = []
    oof_predictions = np.zeros(len(y))

    # LightGBM parameters
    lgbm_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 1000,
        'early_stopping_rounds': 50
    }

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        # Split data
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # Train model
        model = LGBMClassifier(**lgbm_params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='auc'
        )

        # Predict on validation fold
        val_preds = model.predict_proba(X_val_fold)[:, 1]
        oof_predictions[val_idx] = val_preds

        # Calculate AUC score for this fold
        fold_auc = roc_auc_score(y_val_fold, val_preds)
        cv_scores.append(fold_auc)

        print(f"Fold {fold} AUC: {fold_auc:.6f}")

    # Calculate overall CV score
    overall_auc = roc_auc_score(y, oof_predictions)

    print("\n" + "="*60)
    print("Cross-Validation Results:")
    print("="*60)
    for fold, score in enumerate(cv_scores, 1):
        print(f"Fold {fold} AUC: {score:.6f}")
    print(f"\nMean CV AUC: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
    print(f"Overall CV AUC: {overall_auc:.6f}")
    print("="*60)

    return cv_scores, overall_auc, oof_predictions

def train_final_model(X_train, y_train, X_test):
    """
    Train final model on full training data and predict on test set
    """
    print("\nTraining final model on full training data...")

    # LightGBM parameters
    lgbm_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 1000
    }

    # Train final model
    final_model = LGBMClassifier(**lgbm_params)
    final_model.fit(X_train, y_train)

    # Predict on test set
    test_predictions = final_model.predict_proba(X_test)[:, 1]

    print("Final model training completed!")

    return final_model, test_predictions

def create_submission(test_df, predictions, sample_submission, output_path):
    """
    Create submission file
    """
    print(f"\nCreating submission file: {output_path}")

    # Create submission dataframe
    submission = sample_submission.copy()
    submission['rule_violation'] = predictions

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save submission
    submission.to_csv(output_path, index=False)

    print(f"Submission file saved successfully!")
    print(f"Submission shape: {submission.shape}")
    print(f"\nFirst few predictions:")
    print(submission.head(10))

    return submission

def main():
    """
    Main execution function
    """
    start_time = time.time()

    print("="*60)
    print("Jigsaw ACRC Baseline Model")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. Load data
    train_df, test_df, sample_submission = load_data()

    # 2. Create combined features
    train_text = create_combined_features(train_df)
    test_text = create_combined_features(test_df)

    # 3. Create TF-IDF features
    X_train, X_test, tfidf_vectorizer = create_tfidf_features(train_text, test_text, max_features=10000)
    y_train = train_df['rule_violation']

    # 4. Train with Cross-Validation
    cv_scores, overall_auc, oof_predictions = train_with_cv(X_train.toarray(), y_train, n_splits=5)

    # 5. Train final model and predict on test set
    final_model, test_predictions = train_final_model(X_train.toarray(), y_train, X_test.toarray())

    # 6. Create submission file
    submission_path = '/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/submissions/baseline_v1.csv'
    submission = create_submission(test_df, test_predictions, sample_submission, submission_path)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Model: LightGBM Classifier")
    print(f"Features: TF-IDF on combined text (body + rule + examples)")
    print(f"TF-IDF Parameters:")
    print(f"  - max_features: 10000")
    print(f"  - ngram_range: (1, 3)")
    print(f"  - min_df: 2")
    print(f"  - max_df: 0.95")
    print(f"\nCross-Validation:")
    print(f"  - Method: Stratified 5-Fold")
    print(f"  - Mean CV AUC: {np.mean(cv_scores):.6f}")
    print(f"  - Std CV AUC: {np.std(cv_scores):.6f}")
    print(f"  - Overall CV AUC: {overall_auc:.6f}")
    print(f"\nFeature Engineering:")
    print(f"  - Combined text: body + rule + positive_example_1 + positive_example_2 +")
    print(f"                   negative_example_1 + negative_example_2")
    print(f"  - Total features: {X_train.shape[1]}")
    print(f"\nExecution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Submission file: {submission_path}")
    print("="*60)

    return {
        'cv_scores': cv_scores,
        'mean_cv_auc': np.mean(cv_scores),
        'std_cv_auc': np.std(cv_scores),
        'overall_cv_auc': overall_auc,
        'elapsed_time': elapsed_time,
        'submission_path': submission_path
    }

if __name__ == '__main__':
    results = main()
