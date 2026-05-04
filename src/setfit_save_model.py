"""
SetFit Model - Save for Kaggle Dataset Upload
Train locally and save all model files
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 SetFit Model - Train & Save")
print("="*70)

# Install if needed
try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers available")
except ImportError:
    print("⚠️  Installing sentence-transformers...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'sentence-transformers'])
    from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

# Create output directory
os.makedirs('models/setfit_v1', exist_ok=True)

# Load data
print("\n📂 Loading data...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"Train: {train.shape}, Test: {test.shape}")

# Initialize model
print("\n🤖 Loading sentence transformer...")
model_name = 'all-MiniLM-L6-v2'
sbert_model = SentenceTransformer(model_name)
print(f"Model loaded: {model_name}")

# Save the SBERT model
print("\n💾 Saving SBERT model...")
sbert_model.save('models/setfit_v1/sbert_model')
print("✅ SBERT model saved")

def create_text_input(row):
    return f"Rule: {row['rule']} Comment: {row['body']}"

def get_embeddings(texts, model, batch_size=32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)

print("\n🔄 Creating text inputs...")
train['text_input'] = train.apply(create_text_input, axis=1)
test['text_input'] = test.apply(create_text_input, axis=1)

# Create example texts
for col, prefix in [('pos_ex1', 'positive_example_1'), ('pos_ex2', 'positive_example_2'),
                     ('neg_ex1', 'negative_example_1'), ('neg_ex2', 'negative_example_2')]:
    train[f'{col}_text'] = train.apply(lambda row: f"Rule: {row['rule']} Comment: {row[prefix]}", axis=1)
    test[f'{col}_text'] = test.apply(lambda row: f"Rule: {row['rule']} Comment: {row[prefix]}", axis=1)

print("\n💫 Generating embeddings...")
print("  - Body + Rule...")
train_embeddings = get_embeddings(train['text_input'].tolist(), sbert_model)
test_embeddings = get_embeddings(test['text_input'].tolist(), sbert_model)

print("  - Positive examples...")
train_pos1_emb = get_embeddings(train['pos_ex1_text'].tolist(), sbert_model)
train_pos2_emb = get_embeddings(train['pos_ex2_text'].tolist(), sbert_model)
test_pos1_emb = get_embeddings(test['pos_ex1_text'].tolist(), sbert_model)
test_pos2_emb = get_embeddings(test['pos_ex2_text'].tolist(), sbert_model)

print("  - Negative examples...")
train_neg1_emb = get_embeddings(train['neg_ex1_text'].tolist(), sbert_model)
train_neg2_emb = get_embeddings(train['neg_ex2_text'].tolist(), sbert_model)
test_neg1_emb = get_embeddings(test['neg_ex1_text'].tolist(), sbert_model)
test_neg2_emb = get_embeddings(test['neg_ex2_text'].tolist(), sbert_model)

print("\n🎯 Computing similarity features...")

def compute_similarity_features(body_emb, pos1_emb, pos2_emb, neg1_emb, neg2_emb):
    n_samples = body_emb.shape[0]
    features = []

    for i in range(n_samples):
        body_vec = body_emb[i].reshape(1, -1)

        sim_pos1 = cosine_similarity(body_vec, pos1_emb[i].reshape(1, -1))[0][0]
        sim_pos2 = cosine_similarity(body_vec, pos2_emb[i].reshape(1, -1))[0][0]
        sim_neg1 = cosine_similarity(body_vec, neg1_emb[i].reshape(1, -1))[0][0]
        sim_neg2 = cosine_similarity(body_vec, neg2_emb[i].reshape(1, -1))[0][0]

        avg_pos_sim = (sim_pos1 + sim_pos2) / 2
        avg_neg_sim = (sim_neg1 + sim_neg2) / 2
        max_pos_sim = max(sim_pos1, sim_pos2)
        min_neg_sim = min(sim_neg1, sim_neg2)
        diff_sim = avg_pos_sim - avg_neg_sim

        features.append([
            sim_pos1, sim_pos2, sim_neg1, sim_neg2,
            avg_pos_sim, avg_neg_sim, max_pos_sim, min_neg_sim, diff_sim
        ])

    return np.array(features)

X_train_sim = compute_similarity_features(
    train_embeddings, train_pos1_emb, train_pos2_emb,
    train_neg1_emb, train_neg2_emb
)

X_test_sim = compute_similarity_features(
    test_embeddings, test_pos1_emb, test_pos2_emb,
    test_neg1_emb, test_neg2_emb
)

# Combine
X_train_combined = np.hstack([train_embeddings, X_train_sim])
X_test_combined = np.hstack([test_embeddings, X_test_sim])
y_train = train['rule_violation'].values

print(f"\nFeature shape: {X_train_combined.shape}")

# Train final model on all data
print("\n🔬 Training final model on all data...")
clf = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train_combined, y_train)

# Save the classifier
print("\n💾 Saving classifier...")
with open('models/setfit_v1/classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("✅ Classifier saved")

# Cross-validation for scoring
print("\n🔬 Cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train_combined))
test_preds = np.zeros(len(X_test_combined))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train), 1):
    X_tr = X_train_combined[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train_combined[val_idx]
    y_val = y_train[val_idx]

    clf_cv = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42)
    clf_cv.fit(X_tr, y_tr)

    oof_preds[val_idx] = clf_cv.predict_proba(X_val)[:, 1]
    test_preds += clf_cv.predict_proba(X_test_combined)[:, 1] / 5

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"  Fold {fold} AUC: {fold_auc:.6f}")

overall_auc = roc_auc_score(y_train, oof_preds)

print(f"\n{'='*70}")
print(f"📊 Final Results")
print(f"{'='*70}")
print(f"Overall CV AUC: {overall_auc:.6f}")
print(f"Baseline: 0.566")
print(f"Improvement: +{(overall_auc - 0.566):.6f} ({(overall_auc - 0.566)/0.566*100:.2f}%)")
print(f"{'='*70}")

# Generate submission
submission = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': test_preds
})
submission.to_csv('models/setfit_v1/submission.csv', index=False)

# Save metadata
metadata = {
    'model_name': model_name,
    'cv_auc': float(overall_auc),
    'n_embedding_features': train_embeddings.shape[1],
    'n_similarity_features': X_train_sim.shape[1],
    'n_total_features': X_train_combined.shape[1]
}

with open('models/setfit_v1/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n✅ All files saved to models/setfit_v1/")
print("Files:")
print("  - sbert_model/ (Sentence Transformer)")
print("  - classifier.pkl (Logistic Regression)")
print("  - metadata.pkl (Model info)")
print("  - submission.csv (Predictions)")
print(f"\n{'='*70}")
print("🎉 Ready for Kaggle Dataset Upload!")
print(f"{'='*70}")
