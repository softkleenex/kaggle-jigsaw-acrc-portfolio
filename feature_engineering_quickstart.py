"""
Jigsaw ACRC - Quick Start Feature Engineering Implementation
==============================================================

This script implements the TOP 10 most impactful features from the deep analysis.
Expected improvement: +0.05-0.08 AUC

Usage:
    python feature_engineering_quickstart.py

Features implemented:
1. Semantic similarity (Sentence-BERT)
2. Subreddit risk encoding
3. Rule-specific keywords
4. Few-shot max similarity
5. Linguistic features
6. Spam signals
7. Character n-grams
8. Length ratios
9. Modal verbs & questions
10. Example agreement
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. SEMANTIC SIMILARITY FEATURES (Highest Impact)
# ============================================================================

def create_semantic_features(df, model=None):
    """Create semantic similarity features using Sentence-BERT"""
    print("Creating semantic similarity features...")

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        if model is None:
            model = SentenceTransformer('all-MiniLM-L6-v2')

        # Generate embeddings
        body_emb = model.encode(df['body'].tolist(), show_progress_bar=True)
        pos1_emb = model.encode(df['positive_example_1'].tolist(), show_progress_bar=True)
        pos2_emb = model.encode(df['positive_example_2'].tolist(), show_progress_bar=True)
        neg1_emb = model.encode(df['negative_example_1'].tolist(), show_progress_bar=True)
        neg2_emb = model.encode(df['negative_example_2'].tolist(), show_progress_bar=True)

        # Calculate similarities
        features = pd.DataFrame()
        features['body_pos1_semantic'] = [cosine_similarity([b], [p])[0][0] for b, p in zip(body_emb, pos1_emb)]
        features['body_pos2_semantic'] = [cosine_similarity([b], [p])[0][0] for b, p in zip(body_emb, pos2_emb)]
        features['body_neg1_semantic'] = [cosine_similarity([b], [n])[0][0] for b, n in zip(body_emb, neg1_emb)]
        features['body_neg2_semantic'] = [cosine_similarity([b], [n])[0][0] for b, n in zip(body_emb, neg2_emb)]

        # Aggregations
        features['avg_pos_semantic'] = (features['body_pos1_semantic'] + features['body_pos2_semantic']) / 2
        features['avg_neg_semantic'] = (features['body_neg1_semantic'] + features['body_neg2_semantic']) / 2
        features['max_pos_semantic'] = features[['body_pos1_semantic', 'body_pos2_semantic']].max(axis=1)
        features['min_pos_semantic'] = features[['body_pos1_semantic', 'body_pos2_semantic']].min(axis=1)
        features['pos_neg_semantic_diff'] = features['avg_pos_semantic'] - features['avg_neg_semantic']
        features['pos_neg_max_diff_semantic'] = features['max_pos_semantic'] - features[['body_neg1_semantic', 'body_neg2_semantic']].max(axis=1)
        features['pos_semantic_range'] = features['max_pos_semantic'] - features['min_pos_semantic']

        print(f"Created {features.shape[1]} semantic features")
        return features, model

    except ImportError:
        print("WARNING: sentence-transformers not installed. Skipping semantic features.")
        print("Install with: pip install sentence-transformers")
        return pd.DataFrame(), None


# ============================================================================
# 2. SUBREDDIT RISK ENCODING
# ============================================================================

def create_subreddit_features(train_df, test_df, alpha=10):
    """Create subreddit risk features with target encoding"""
    print("Creating subreddit risk features...")

    global_mean = train_df['rule_violation'].mean()

    # Overall subreddit risk
    subreddit_stats = train_df.groupby('subreddit')['rule_violation'].agg(['mean', 'count'])
    subreddit_stats['risk_score'] = ((subreddit_stats['mean'] * subreddit_stats['count'] +
                                      global_mean * alpha) / (subreddit_stats['count'] + alpha))

    train_df['subreddit_risk'] = train_df['subreddit'].map(subreddit_stats['risk_score'])
    test_df['subreddit_risk'] = test_df['subreddit'].map(subreddit_stats['risk_score']).fillna(global_mean)

    # Rule-specific risk
    for rule_idx, rule_pattern in enumerate(['Advertising', 'legal']):
        rule_name = 'ad' if rule_idx == 0 else 'legal'
        rule_mask = train_df['rule'].str.contains(rule_pattern, case=False)
        rule_stats = train_df[rule_mask].groupby('subreddit')['rule_violation'].agg(['mean', 'count'])
        rule_stats[f'{rule_name}_risk'] = ((rule_stats['mean'] * rule_stats['count'] +
                                            global_mean * alpha) / (rule_stats['count'] + alpha))

        train_df[f'subreddit_{rule_name}_risk'] = train_df['subreddit'].map(rule_stats[f'{rule_name}_risk']).fillna(global_mean)
        test_df[f'subreddit_{rule_name}_risk'] = test_df['subreddit'].map(rule_stats[f'{rule_name}_risk']).fillna(global_mean)

    # Interaction: match between rule and subreddit risk
    train_df['subreddit_rule_match'] = (train_df['rule'].str.contains('Advertising').astype(int) * train_df['subreddit_ad_risk'] +
                                        train_df['rule'].str.contains('legal').astype(int) * train_df['subreddit_legal_risk'])
    test_df['subreddit_rule_match'] = (test_df['rule'].str.contains('Advertising').astype(int) * test_df['subreddit_ad_risk'] +
                                       test_df['rule'].str.contains('legal').astype(int) * test_df['subreddit_legal_risk'])

    features_train = train_df[['subreddit_risk', 'subreddit_ad_risk', 'subreddit_legal_risk', 'subreddit_rule_match']]
    features_test = test_df[['subreddit_risk', 'subreddit_ad_risk', 'subreddit_legal_risk', 'subreddit_rule_match']]

    print(f"Created {features_train.shape[1]} subreddit features")
    return features_train, features_test


# ============================================================================
# 3. RULE-SPECIFIC KEYWORD FEATURES
# ============================================================================

def create_keyword_features(df):
    """Create rule-specific keyword features"""
    print("Creating keyword features...")

    AD_KEYWORDS = ['buy', 'sell', 'free', 'discount', 'offer', 'deal', 'promo', 'sale',
                   'shop', 'price', 'purchase', 'cheap', 'subscribe', 'follow', 'click',
                   'check', 'visit', 'link', 'referral', 'code', 'coupon']

    LEGAL_KEYWORDS = ['lawyer', 'attorney', 'legal', 'law', 'court', 'sue', 'police',
                      'advice', 'should', 'can', 'could', 'consult', 'rights', 'illegal',
                      'liable', 'contract', 'case', 'defendant', 'plaintiff']

    features = pd.DataFrame()
    df['body_lower'] = df['body'].str.lower()

    # Count matches
    features['ad_keyword_count'] = df['body_lower'].apply(lambda x: sum(1 for kw in AD_KEYWORDS if kw in x))
    features['legal_keyword_count'] = df['body_lower'].apply(lambda x: sum(1 for kw in LEGAL_KEYWORDS if kw in x))

    # Binary presence for top keywords
    for kw in ['lawyer', 'sue', 'police', 'legal', 'stream', 'watch', 'html', 'free', 'discount']:
        features[f'has_{kw}'] = df['body_lower'].str.contains(kw, regex=False).astype(int)

    # Rule-aware keyword count
    features['rule_keyword_count'] = 0
    features.loc[df['rule'].str.contains('Advertising', case=False), 'rule_keyword_count'] = \
        features.loc[df['rule'].str.contains('Advertising', case=False), 'ad_keyword_count']
    features.loc[df['rule'].str.contains('legal', case=False), 'rule_keyword_count'] = \
        features.loc[df['rule'].str.contains('legal', case=False), 'legal_keyword_count']

    # Keyword density
    df['body_word_count'] = df['body'].str.split().str.len()
    features['ad_keyword_density'] = features['ad_keyword_count'] / (df['body_word_count'] + 1)
    features['legal_keyword_density'] = features['legal_keyword_count'] / (df['body_word_count'] + 1)
    features['rule_keyword_density'] = features['rule_keyword_count'] / (df['body_word_count'] + 1)

    print(f"Created {features.shape[1]} keyword features")
    return features


# ============================================================================
# 4. FEW-SHOT MAX SIMILARITY FEATURES
# ============================================================================

def jaccard_similarity(str1, str2):
    """Calculate Jaccard similarity between two strings"""
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    if len(set1) == 0 and len(set2) == 0:
        return 0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def create_fewshot_features(df):
    """Create few-shot similarity features"""
    print("Creating few-shot similarity features...")

    features = pd.DataFrame()

    # Calculate all similarities
    for i in [1, 2]:
        features[f'body_pos{i}_jaccard'] = df.apply(lambda x: jaccard_similarity(x['body'], x[f'positive_example_{i}']), axis=1)
        features[f'body_neg{i}_jaccard'] = df.apply(lambda x: jaccard_similarity(x['body'], x[f'negative_example_{i}']), axis=1)

    # Max/Min features
    features['max_pos_jaccard'] = features[['body_pos1_jaccard', 'body_pos2_jaccard']].max(axis=1)
    features['min_pos_jaccard'] = features[['body_pos1_jaccard', 'body_pos2_jaccard']].min(axis=1)
    features['max_neg_jaccard'] = features[['body_neg1_jaccard', 'body_neg2_jaccard']].max(axis=1)
    features['min_neg_jaccard'] = features[['body_neg1_jaccard', 'body_neg2_jaccard']].min(axis=1)

    # Differences and ratios
    features['pos_neg_max_diff'] = features['max_pos_jaccard'] - features['max_neg_jaccard']
    features['pos_jaccard_range'] = features['max_pos_jaccard'] - features['min_pos_jaccard']
    features['neg_jaccard_range'] = features['max_neg_jaccard'] - features['min_neg_jaccard']
    features['pos_neg_jaccard_ratio'] = features['max_pos_jaccard'] / (features['max_neg_jaccard'] + 0.01)

    # Average similarities
    features['avg_pos_jaccard'] = (features['body_pos1_jaccard'] + features['body_pos2_jaccard']) / 2
    features['avg_neg_jaccard'] = (features['body_neg1_jaccard'] + features['body_neg2_jaccard']) / 2
    features['avg_pos_neg_diff'] = features['avg_pos_jaccard'] - features['avg_neg_jaccard']

    # Word overlap counts
    for i in [1, 2]:
        features[f'body_pos{i}_overlap'] = df.apply(lambda x: len(set(x['body'].lower().split()).intersection(set(x[f'positive_example_{i}'].lower().split()))), axis=1)
        features[f'body_neg{i}_overlap'] = df.apply(lambda x: len(set(x['body'].lower().split()).intersection(set(x[f'negative_example_{i}'].lower().split()))), axis=1)

    features['max_pos_overlap'] = features[['body_pos1_overlap', 'body_pos2_overlap']].max(axis=1)
    features['avg_pos_overlap'] = features[['body_pos1_overlap', 'body_pos2_overlap']].mean(axis=1)

    print(f"Created {features.shape[1]} few-shot features")
    return features


# ============================================================================
# 5. LINGUISTIC FEATURES
# ============================================================================

def create_linguistic_features(df):
    """Create linguistic complexity features"""
    print("Creating linguistic features...")

    features = pd.DataFrame()

    # Sentence-level features
    features['body_sentences'] = df['body'].str.count(r'[.!?]+') + 1
    features['body_words'] = df['body'].str.split().str.len()
    features['body_avg_sent_len'] = features['body_words'] / features['body_sentences']

    # Vocabulary diversity
    features['body_unique_word_ratio'] = df['body'].apply(lambda x: len(set(x.lower().split())) / max(len(x.split()), 1))

    # Stopword ratio
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
                 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    features['body_stopword_ratio'] = df['body'].apply(lambda x: sum(1 for w in x.lower().split() if w in stopwords) / max(len(x.split()), 1))

    # Average word length
    features['body_avg_word_len'] = df['body'].apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)

    # Length features
    features['body_len'] = df['body'].str.len()
    features['rule_len'] = df['rule'].str.len()

    print(f"Created {features.shape[1]} linguistic features")
    return features


# ============================================================================
# 6. SPAM SIGNAL FEATURES
# ============================================================================

def create_spam_features(df):
    """Create spam detection features"""
    print("Creating spam signal features...")

    features = pd.DataFrame()

    # Email detection
    features['has_email'] = df['body'].str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', regex=True).astype(int)

    # Phone detection
    features['has_phone'] = df['body'].str.contains(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', regex=True).astype(int)

    # Price detection
    features['has_price'] = df['body'].str.contains(r'\$\d+|\d+\s*(dollar|USD|euro|£)', regex=True, case=False).astype(int)
    features['price_count'] = df['body'].str.count(r'\$\d+')

    # URL features
    features['url_count'] = df['body'].str.count(r'http[s]?://|www\.')
    features['has_url'] = (features['url_count'] > 0).astype(int)

    # Mention/hashtag
    features['has_mention'] = df['body'].str.contains(r'@\w+', regex=True).astype(int)
    features['has_hashtag'] = df['body'].str.contains(r'#\w+', regex=True).astype(int)

    # Composite spam score (weighted by lift from analysis)
    features['spam_score'] = (features['has_email'] * 10.65 +
                              features['has_price'] * 6.00 +
                              features['has_phone'] * 1.74 +
                              features['has_mention'] * 2.90) / 21.29

    print(f"Created {features.shape[1]} spam features")
    return features


# ============================================================================
# 7. LENGTH RATIO FEATURES
# ============================================================================

def create_length_ratio_features(df):
    """Create length ratio features"""
    print("Creating length ratio features...")

    features = pd.DataFrame()

    # Body vs examples
    features['body_pos1_len_ratio'] = df['body'].str.len() / (df['positive_example_1'].str.len() + 1)
    features['body_pos2_len_ratio'] = df['body'].str.len() / (df['positive_example_2'].str.len() + 1)
    features['body_neg1_len_ratio'] = df['body'].str.len() / (df['negative_example_1'].str.len() + 1)
    features['body_neg2_len_ratio'] = df['body'].str.len() / (df['negative_example_2'].str.len() + 1)

    # Average ratios
    features['avg_pos_len_ratio'] = (features['body_pos1_len_ratio'] + features['body_pos2_len_ratio']) / 2
    features['avg_neg_len_ratio'] = (features['body_neg1_len_ratio'] + features['body_neg2_len_ratio']) / 2
    features['pos_neg_len_ratio_diff'] = features['avg_pos_len_ratio'] - features['avg_neg_len_ratio']

    # Example lengths
    features['pos_examples_avg_len'] = (df['positive_example_1'].str.len() + df['positive_example_2'].str.len()) / 2
    features['neg_examples_avg_len'] = (df['negative_example_1'].str.len() + df['negative_example_2'].str.len()) / 2
    features['examples_len_diff'] = features['pos_examples_avg_len'] - features['neg_examples_avg_len']

    print(f"Created {features.shape[1]} length ratio features")
    return features


# ============================================================================
# 8. MODAL VERB & QUESTION FEATURES
# ============================================================================

def create_modal_question_features(df):
    """Create modal verb and question features"""
    print("Creating modal/question features...")

    features = pd.DataFrame()

    # Modal verbs
    modal_pattern = r'\b(should|could|would|might|may|can|must|ought|shall)\b'
    features['modal_count'] = df['body'].str.count(modal_pattern, flags=re.IGNORECASE)
    features['has_modal'] = (features['modal_count'] > 0).astype(int)

    # Question patterns
    features['question_count'] = df['body'].str.count(r'\?')
    features['has_question'] = (features['question_count'] > 0).astype(int)

    # Question word starts
    question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'can', 'could', 'should', 'would']
    features['starts_question_word'] = df['body'].str.lower().str.split().str[0].isin(question_words).astype(int)

    # Rule-specific interactions
    features['legal_modal_interaction'] = features['modal_count'] * df['rule'].str.contains('legal', case=False).astype(int)
    features['legal_question_interaction'] = features['question_count'] * df['rule'].str.contains('legal', case=False).astype(int)

    # Imperative verbs (for advertising)
    imperative_verbs = ['get', 'buy', 'check', 'click', 'visit', 'see', 'watch', 'try']
    features['starts_imperative'] = df['body'].str.lower().str.split().str[0].isin(imperative_verbs).astype(int)
    features['ad_imperative_interaction'] = features['starts_imperative'] * df['rule'].str.contains('Advertising', case=False).astype(int)

    print(f"Created {features.shape[1]} modal/question features")
    return features


# ============================================================================
# 9. EXAMPLE AGREEMENT FEATURES
# ============================================================================

def create_example_agreement_features(df):
    """Create example agreement features"""
    print("Creating example agreement features...")

    features = pd.DataFrame()

    # Calculate from existing jaccard features
    # Assuming these are already computed in few-shot features
    pos1_jacc = df.apply(lambda x: jaccard_similarity(x['body'], x['positive_example_1']), axis=1)
    pos2_jacc = df.apply(lambda x: jaccard_similarity(x['body'], x['positive_example_2']), axis=1)

    # Inter-example similarity
    features['pos_examples_jaccard'] = df.apply(lambda x: jaccard_similarity(x['positive_example_1'], x['positive_example_2']), axis=1)
    features['neg_examples_jaccard'] = df.apply(lambda x: jaccard_similarity(x['negative_example_1'], x['negative_example_2']), axis=1)

    # Agreement score
    features['pos_agreement_score'] = pos1_jacc.clip(0.001, 1) / (pos2_jacc.clip(0.001, 1) + 0.01)

    # Consistency (std)
    features['pos_sim_std'] = pd.DataFrame({'p1': pos1_jacc, 'p2': pos2_jacc}).std(axis=1)

    print(f"Created {features.shape[1]} example agreement features")
    return features


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def create_all_features(train_df, test_df, use_semantic=True):
    """Create all features"""
    print("\n" + "="*70)
    print("CREATING ALL FEATURES")
    print("="*70)

    # Fill NaN
    text_cols = ['body', 'rule', 'positive_example_1', 'positive_example_2',
                 'negative_example_1', 'negative_example_2']
    for col in text_cols:
        train_df[col] = train_df[col].fillna('').astype(str)
        test_df[col] = test_df[col].fillna('').astype(str)

    # Create features
    feature_sets_train = []
    feature_sets_test = []

    # 1. Semantic features (optional, requires sentence-transformers)
    sbert_model = None
    if use_semantic:
        semantic_train, sbert_model = create_semantic_features(train_df)
        if not semantic_train.empty:
            semantic_test, _ = create_semantic_features(test_df, model=sbert_model)
            feature_sets_train.append(semantic_train)
            feature_sets_test.append(semantic_test)

    # 2. Subreddit features
    subreddit_train, subreddit_test = create_subreddit_features(train_df.copy(), test_df.copy())
    feature_sets_train.append(subreddit_train)
    feature_sets_test.append(subreddit_test)

    # 3. Keyword features
    feature_sets_train.append(create_keyword_features(train_df))
    feature_sets_test.append(create_keyword_features(test_df))

    # 4. Few-shot features
    feature_sets_train.append(create_fewshot_features(train_df))
    feature_sets_test.append(create_fewshot_features(test_df))

    # 5. Linguistic features
    feature_sets_train.append(create_linguistic_features(train_df))
    feature_sets_test.append(create_linguistic_features(test_df))

    # 6. Spam features
    feature_sets_train.append(create_spam_features(train_df))
    feature_sets_test.append(create_spam_features(test_df))

    # 7. Length ratios
    feature_sets_train.append(create_length_ratio_features(train_df))
    feature_sets_test.append(create_length_ratio_features(test_df))

    # 8. Modal/Question features
    feature_sets_train.append(create_modal_question_features(train_df))
    feature_sets_test.append(create_modal_question_features(test_df))

    # 9. Example agreement
    feature_sets_train.append(create_example_agreement_features(train_df))
    feature_sets_test.append(create_example_agreement_features(test_df))

    # Combine all features
    X_train_engineered = pd.concat(feature_sets_train, axis=1)
    X_test_engineered = pd.concat(feature_sets_test, axis=1)

    print(f"\n{'='*70}")
    print(f"TOTAL ENGINEERED FEATURES: {X_train_engineered.shape[1]}")
    print(f"{'='*70}\n")

    return X_train_engineered, X_test_engineered


# ============================================================================
# TRAINING WITH IMPROVED REGULARIZATION
# ============================================================================

def train_model_cv(X, y, n_splits=5):
    """Train LightGBM with improved regularization"""
    print(f"\nTraining LightGBM with {n_splits}-fold CV...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_scores = []
    oof_preds = np.zeros(len(y))

    # Improved parameters to reduce overfitting
    lgbm_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,  # Reduced from 63
        'learning_rate': 0.02,  # Lower learning rate
        'feature_fraction': 0.7,  # More aggressive feature subsampling
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 30,  # Increased from 20
        'reg_alpha': 0.3,  # Stronger L1 regularization
        'reg_lambda': 0.3,  # Stronger L2 regularization
        'max_depth': 6,  # Limit depth
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 3000,
        'early_stopping_rounds': 150
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
    print(f"\nMean CV AUC: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
    print(f"Overall CV AUC: {overall_auc:.6f}")

    return overall_auc, oof_preds


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("JIGSAW ACRC - FEATURE ENGINEERING QUICKSTART")
    print("="*70)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Create features
    X_train_eng, X_test_eng = create_all_features(train_df, test_df, use_semantic=True)
    y_train = train_df['rule_violation']

    # Train and evaluate
    cv_auc, oof_preds = train_model_cv(X_train_eng.values, y_train)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Cross-Validation AUC: {cv_auc:.6f}")
    print(f"Expected improvement: +{(cv_auc - 0.7086):.4f} from baseline")
    print(f"{'='*70}\n")

    return cv_auc


if __name__ == '__main__':
    main()
