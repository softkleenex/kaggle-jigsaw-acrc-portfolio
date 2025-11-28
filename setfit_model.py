"""
SetFit Model for Jigsaw ACRC
Few-shot learning approach using sentence transformers
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 SetFit Model - Few-Shot Learning Approach")
print("="*70)

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers available")
    SBERT_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers not available - installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'sentence-transformers'])
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True

from sklearn.metrics.pairwise import cosine_similarity

# Load data
print("\n📂 Loading data...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_sub = pd.read_csv('data/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Initialize Sentence Transformer
print("\n🤖 Loading sentence transformer model...")
model_name = 'all-MiniLM-L6-v2'  # Fast and effective
sbert_model = SentenceTransformer(model_name)
print(f"Model loaded: {model_name}")

def create_text_input(row, use_rule=True, use_body=True):
    """Create formatted text input"""
    parts = []
    if use_rule:
        parts.append(f"Rule: {row['rule']}")
    if use_body:
        parts.append(f"Comment: {row['body']}")
    return " ".join(parts)

def get_embeddings(texts, model, batch_size=32):
    """Get sentence embeddings"""
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)

print("\n🔄 Creating text inputs...")
# Create combined text for body + rule
train['text_input'] = train.apply(lambda row: create_text_input(row), axis=1)
test['text_input'] = test.apply(lambda row: create_text_input(row), axis=1)

# Create example texts
train['pos_ex1_text'] = train.apply(lambda row: f"Rule: {row['rule']} Comment: {row['positive_example_1']}", axis=1)
train['pos_ex2_text'] = train.apply(lambda row: f"Rule: {row['rule']} Comment: {row['positive_example_2']}", axis=1)
train['neg_ex1_text'] = train.apply(lambda row: f"Rule: {row['rule']} Comment: {row['negative_example_1']}", axis=1)
train['neg_ex2_text'] = train.apply(lambda row: f"Rule: {row['rule']} Comment: {row['negative_example_2']}", axis=1)

test['pos_ex1_text'] = test.apply(lambda row: f"Rule: {row['rule']} Comment: {row['positive_example_1']}", axis=1)
test['pos_ex2_text'] = test.apply(lambda row: f"Rule: {row['rule']} Comment: {row['positive_example_2']}", axis=1)
test['neg_ex1_text'] = test.apply(lambda row: f"Rule: {row['rule']} Comment: {row['negative_example_1']}", axis=1)
test['neg_ex2_text'] = test.apply(lambda row: f"Rule: {row['rule']} Comment: {row['negative_example_2']}", axis=1)

print("\n💫 Generating embeddings...")
# Get embeddings for main texts
print("  - Body + Rule embeddings...")
train_embeddings = get_embeddings(train['text_input'].tolist(), sbert_model)
test_embeddings = get_embeddings(test['text_input'].tolist(), sbert_model)

# Get embeddings for examples
print("  - Positive examples embeddings...")
train_pos1_emb = get_embeddings(train['pos_ex1_text'].tolist(), sbert_model)
train_pos2_emb = get_embeddings(train['pos_ex2_text'].tolist(), sbert_model)
test_pos1_emb = get_embeddings(test['pos_ex1_text'].tolist(), sbert_model)
test_pos2_emb = get_embeddings(test['pos_ex2_text'].tolist(), sbert_model)

print("  - Negative examples embeddings...")
train_neg1_emb = get_embeddings(train['neg_ex1_text'].tolist(), sbert_model)
train_neg2_emb = get_embeddings(train['neg_ex2_text'].tolist(), sbert_model)
test_neg1_emb = get_embeddings(test['neg_ex1_text'].tolist(), sbert_model)
test_neg2_emb = get_embeddings(test['neg_ex2_text'].tolist(), sbert_model)

print("\n🎯 Computing similarity features...")

def compute_similarity_features(body_emb, pos1_emb, pos2_emb, neg1_emb, neg2_emb):
    """Compute similarity features between body and examples"""
    n_samples = body_emb.shape[0]

    # Reshape for cosine_similarity
    features = []

    for i in range(n_samples):
        body_vec = body_emb[i].reshape(1, -1)

        # Similarity with positive examples (should be high if violates rule)
        sim_pos1 = cosine_similarity(body_vec, pos1_emb[i].reshape(1, -1))[0][0]
        sim_pos2 = cosine_similarity(body_vec, pos2_emb[i].reshape(1, -1))[0][0]

        # Similarity with negative examples (should be low if violates rule)
        sim_neg1 = cosine_similarity(body_vec, neg1_emb[i].reshape(1, -1))[0][0]
        sim_neg2 = cosine_similarity(body_vec, neg2_emb[i].reshape(1, -1))[0][0]

        # Aggregate features
        avg_pos_sim = (sim_pos1 + sim_pos2) / 2
        avg_neg_sim = (sim_neg1 + sim_neg2) / 2
        max_pos_sim = max(sim_pos1, sim_pos2)
        min_neg_sim = min(sim_neg1, sim_neg2)
        diff_sim = avg_pos_sim - avg_neg_sim  # Positive means closer to violations

        features.append([
            sim_pos1, sim_pos2, sim_neg1, sim_neg2,
            avg_pos_sim, avg_neg_sim, max_pos_sim, min_neg_sim, diff_sim
        ])

    return np.array(features)

# Compute features
X_train_sim = compute_similarity_features(
    train_embeddings, train_pos1_emb, train_pos2_emb,
    train_neg1_emb, train_neg2_emb
)

X_test_sim = compute_similarity_features(
    test_embeddings, test_pos1_emb, test_pos2_emb,
    test_neg1_emb, test_neg2_emb
)

# Combine: embeddings + similarity features
X_train_combined = np.hstack([train_embeddings, X_train_sim])
X_test_combined = np.hstack([test_embeddings, X_test_sim])

y_train = train['rule_violation'].values

print(f"\nFeature shape: {X_train_combined.shape}")
print(f"  - Embedding features: {train_embeddings.shape[1]}")
print(f"  - Similarity features: {X_train_sim.shape[1]}")

# Cross-validation with Logistic Regression
print("\n🔬 Cross-validation...")
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train_combined))
test_preds = np.zeros(len(X_test_combined))
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train), 1):
    print(f"\nFold {fold}/{n_folds}")

    X_tr = X_train_combined[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train_combined[val_idx]
    y_val = y_train[val_idx]

    # Train classifier
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr)

    # Predict
    oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
    test_preds += clf.predict_proba(X_test_combined)[:, 1] / n_folds

    # Score
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    cv_scores.append(fold_auc)
    print(f"  Fold {fold} AUC: {fold_auc:.6f}")

# Overall score
overall_auc = roc_auc_score(y_train, oof_preds)
mean_cv = np.mean(cv_scores)
std_cv = np.std(cv_scores)

print(f"\n{'='*70}")
print(f"📊 Final Results")
print(f"{'='*70}")
print(f"Overall CV AUC: {overall_auc:.6f}")
print(f"Mean CV AUC: {mean_cv:.6f} (+/- {std_cv:.6f})")
print(f"\nComparison with Baseline:")
print(f"  Baseline: 0.614")
print(f"  SetFit: {overall_auc:.6f}")
print(f"  Improvement: {(overall_auc - 0.614):.6f} ({(overall_auc - 0.614)/0.614*100:.2f}%)")
print(f"{'='*70}")

# Generate submission
print("\n📤 Generating submission...")
submission = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': test_preds
})

submission.to_csv('submissions/setfit_v1.csv', index=False)
print("✅ Submission saved: submissions/setfit_v1.csv")

print(f"\nPrediction statistics:")
print(f"  Min: {test_preds.min():.6f}")
print(f"  Max: {test_preds.max():.6f}")
print(f"  Mean: {test_preds.mean():.6f}")
print(f"  Median: {np.median(test_preds):.6f}")

print(f"\n{'='*70}")
print("✅ SetFit Model Complete!")
print(f"{'='*70}")
