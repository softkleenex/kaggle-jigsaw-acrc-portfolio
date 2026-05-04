"""
Jigsaw ACRC Baseline v2 - Improved with Feature Engineering
=============================================================
Improvements over v1:
- Additional text length features
- Character-level features
- Word count features
- Improved LightGBM parameters
- XGBoost ensemble
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import time
import os
from datetime import datetime

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_data():
    print("Loading data...")
    train_df = pd.read_csv('/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/data/train.csv')
    test_df = pd.read_csv('/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/data/test.csv')
    sample_submission = pd.read_csv('/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/data/sample_submission.csv')

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df, sample_submission

def create_text_features(df):
    """Create additional text-based features"""
    print("Creating text features...")

    features = pd.DataFrame()

    # Fill NaN
    text_cols = ['body', 'rule', 'positive_example_1', 'positive_example_2',
                 'negative_example_1', 'negative_example_2']
    for col in text_cols:
        df[col] = df[col].fillna('').astype(str)

    # Length features
    features['body_len'] = df['body'].str.len()
    features['rule_len'] = df['rule'].str.len()
    features['pos1_len'] = df['positive_example_1'].str.len()
    features['pos2_len'] = df['positive_example_2'].str.len()
    features['neg1_len'] = df['negative_example_1'].str.len()
    features['neg2_len'] = df['negative_example_2'].str.len()

    # Word count features
    features['body_words'] = df['body'].str.split().str.len()
    features['rule_words'] = df['rule'].str.split().str.len()

    # Character features
    features['body_upper_ratio'] = df['body'].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
    features['body_digit_ratio'] = df['body'].apply(lambda x: sum(1 for c in x if c.isdigit()) / (len(x) + 1))
    features['body_punct_ratio'] = df['body'].apply(lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()) / (len(x) + 1))

    # Avg word length
    features['body_avg_word_len'] = df['body'].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)

    # Sentence count (rough estimate using periods)
    features['body_sentences'] = df['body'].str.count('\.') + 1

    print(f"Created {features.shape[1]} additional features")
    return features

def create_combined_text(df):
    """Combine all text fields"""
    text_cols = ['body', 'rule', 'positive_example_1', 'positive_example_2',
                 'negative_example_1', 'negative_example_2']

    for col in text_cols:
        df[col] = df[col].fillna('').astype(str)

    combined = (
        df['body'] + ' [SEP] ' +
        df['rule'] + ' [SEP] ' +
        df['positive_example_1'] + ' [SEP] ' +
        df['positive_example_2'] + ' [SEP] ' +
        df['negative_example_1'] + ' [SEP] ' +
        df['negative_example_2']
    )

    return combined

def create_tfidf_features(train_text, test_text, max_features=15000):
    print(f"Creating TF-IDF features (max_features={max_features})...")

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english'
    )

    train_tfidf = tfidf.fit_transform(train_text)
    test_tfidf = tfidf.transform(test_text)

    print(f"TF-IDF shape: {train_tfidf.shape}")
    return train_tfidf, test_tfidf, tfidf

def train_lgbm_cv(X, y, n_splits=5):
    print(f"\nLightGBM - {n_splits}-Fold CV...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = []
    oof_preds = np.zeros(len(y))

    lgbm_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,  # Increased from 31
        'learning_rate': 0.03,  # Decreased for more iterations
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 2000,  # Increased
        'early_stopping_rounds': 100
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        model = LGBMClassifier(**lgbm_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='auc'
        )

        val_preds = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = val_preds

        fold_auc = roc_auc_score(y_val_fold, val_preds)
        cv_scores.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.6f}")

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nLightGBM Mean CV AUC: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
    print(f"LightGBM Overall CV AUC: {overall_auc:.6f}")

    return cv_scores, overall_auc, oof_preds

def train_xgb_cv(X, y, n_splits=5):
    print(f"\nXGBoost - {n_splits}-Fold CV...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = []
    oof_preds = np.zeros(len(y))

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'max_depth': 7,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': RANDOM_STATE,
        'n_estimators': 2000,
        'early_stopping_rounds': 100,
        'verbosity': 0
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        model = XGBClassifier(**xgb_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )

        val_preds = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = val_preds

        fold_auc = roc_auc_score(y_val_fold, val_preds)
        cv_scores.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.6f}")

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nXGBoost Mean CV AUC: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
    print(f"XGBoost Overall CV AUC: {overall_auc:.6f}")

    return cv_scores, overall_auc, oof_preds

def train_final_models(X_train, y_train, X_test):
    print("\nTraining final models on full data...")

    # LightGBM
    lgbm = LGBMClassifier(
        objective='binary', metric='auc', boosting_type='gbdt',
        num_leaves=63, learning_rate=0.03, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.1, verbose=-1,
        random_state=RANDOM_STATE, n_estimators=2000
    )
    lgbm.fit(X_train, y_train)
    lgbm_preds = lgbm.predict_proba(X_test)[:, 1]

    # XGBoost
    xgb = XGBClassifier(
        objective='binary:logistic', eval_metric='auc', booster='gbtree',
        max_depth=7, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        random_state=RANDOM_STATE, n_estimators=2000, verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict_proba(X_test)[:, 1]

    # Ensemble (weighted average)
    ensemble_preds = 0.6 * lgbm_preds + 0.4 * xgb_preds

    print("Final models trained!")
    return ensemble_preds

def main():
    start_time = time.time()

    print("="*70)
    print("Jigsaw ACRC Baseline v2 - Improved Model")
    print("="*70)

    # Load data
    train_df, test_df, sample_submission = load_data()

    # Create text features
    train_text_feats = create_text_features(train_df)
    test_text_feats = create_text_features(test_df)

    # Create combined text
    train_text = create_combined_text(train_df)
    test_text = create_combined_text(test_df)

    # TF-IDF
    train_tfidf, test_tfidf, _ = create_tfidf_features(train_text, test_text, max_features=15000)

    # Combine features
    from scipy.sparse import hstack, csr_matrix
    X_train = hstack([train_tfidf, csr_matrix(train_text_feats.values)]).toarray()
    X_test = hstack([test_tfidf, csr_matrix(test_text_feats.values)]).toarray()
    y_train = train_df['rule_violation']

    print(f"\nFinal feature shape: {X_train.shape}")

    # Train LightGBM with CV
    lgbm_scores, lgbm_auc, lgbm_oof = train_lgbm_cv(X_train, y_train, n_splits=5)

    # Train XGBoost with CV
    xgb_scores, xgb_auc, xgb_oof = train_xgb_cv(X_train, y_train, n_splits=5)

    # Ensemble OOF predictions
    ensemble_oof = 0.6 * lgbm_oof + 0.4 * xgb_oof
    ensemble_auc = roc_auc_score(y_train, ensemble_oof)

    print(f"\n{'='*70}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*70}")
    print(f"LightGBM CV AUC: {lgbm_auc:.6f}")
    print(f"XGBoost CV AUC: {xgb_auc:.6f}")
    print(f"Ensemble CV AUC: {ensemble_auc:.6f}")
    print(f"{'='*70}")

    # Train final models and predict
    test_preds = train_final_models(X_train, y_train, X_test)

    # Create submission
    submission = sample_submission.copy()
    submission['rule_violation'] = test_preds

    output_path = '/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/submissions/baseline_v2.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)

    elapsed = time.time() - start_time

    print(f"\nSubmission saved: {output_path}")
    print(f"Execution time: {elapsed:.2f}s ({elapsed/60:.2f}m)")
    print(f"\nPrediction stats:")
    print(f"  Min: {test_preds.min():.6f}")
    print(f"  Max: {test_preds.max():.6f}")
    print(f"  Mean: {test_preds.mean():.6f}")
    print(f"  Median: {np.median(test_preds):.6f}")
    print("="*70)

    return {
        'lgbm_cv_auc': lgbm_auc,
        'xgb_cv_auc': xgb_auc,
        'ensemble_cv_auc': ensemble_auc,
        'submission_path': output_path
    }

if __name__ == '__main__':
    results = main()
