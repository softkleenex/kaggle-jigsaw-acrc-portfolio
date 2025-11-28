# IMPLEMENTATION GUIDE - QUICK START
## Priority Experiments with Code Templates

**Goal**: Get transformers running ASAP
**Timeline**: Day 1-2 implementations

---

## EXPERIMENT 1: DeBERTa-v3-base Fine-tuning

### Overview
- **Time**: 4-5 hours (2h code + 3h training)
- **Expected Impact**: +0.08-0.12 AUC
- **Hardware**: Kaggle GPU (P100/T4)

### Step-by-Step Implementation

#### 1. Data Preparation (30 min)

```python
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset

# Load data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

def create_input_text(row):
    """
    Format: [CLS] rule [SEP] body [SEP] positive_examples [SEP] negative_examples
    """
    text = f"""Rule: {row['rule']}
Comment: {row['body']}
Positive Examples: {row['positive_example_1']} | {row['positive_example_2']}
Negative Examples: {row['negative_example_1']} | {row['negative_example_2']}"""
    return text

# Create formatted text
train_df['text'] = train_df.apply(create_input_text, axis=1)
test_df['text'] = test_df.apply(create_input_text, axis=1)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512  # Adjust based on token analysis
    )

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df[['text', 'rule_violation']])
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.rename_column('rule_violation', 'labels')
```

#### 2. Model Training (1h code + 3h training)

```python
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Training configuration
training_args = TrainingArguments(
    output_dir='./deberta_results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type='linear',
    fp16=True,  # Mixed precision (2x faster)
    gradient_accumulation_steps=2,  # Effective batch size = 16
    evaluation_strategy='steps',
    eval_steps=100,
    save_strategy='steps',
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    logging_steps=50,
    report_to='none',
    seed=42,
)

# Custom metric for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    auc = roc_auc_score(labels, predictions)
    return {'auc': auc}

# 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
oof_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['rule_violation']), 1):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold}/5")
    print(f"{'='*50}")

    # Split data
    train_fold = train_dataset.select(train_idx.tolist())
    val_fold = train_dataset.select(val_idx.tolist())

    # Initialize model (fresh for each fold)
    model = AutoModelForSequenceClassification.from_pretrained(
        'microsoft/deberta-v3-base',
        num_labels=2,
        problem_type='single_label_classification'
    )
    model.to(device)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_fold,
        eval_dataset=val_fold,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Validation predictions
    val_outputs = trainer.predict(val_fold)
    val_probs = torch.softmax(torch.tensor(val_outputs.predictions), dim=-1)[:, 1].numpy()
    oof_predictions[val_idx] = val_probs

    # Test predictions
    test_dataset = Dataset.from_pandas(test_df[['text']])
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_outputs = trainer.predict(test_dataset)
    test_probs = torch.softmax(torch.tensor(test_outputs.predictions), dim=-1)[:, 1].numpy()
    test_predictions += test_probs / 5  # Average across folds

    # Fold score
    fold_auc = roc_auc_score(train_df.iloc[val_idx]['rule_violation'], val_probs)
    cv_scores.append(fold_auc)
    print(f"Fold {fold} AUC: {fold_auc:.6f}")

    # Clear memory
    del model, trainer
    torch.cuda.empty_cache()

# Overall CV score
overall_auc = roc_auc_score(train_df['rule_violation'], oof_predictions)
print(f"\n{'='*50}")
print(f"CV Results")
print(f"{'='*50}")
print(f"Fold scores: {[f'{s:.6f}' for s in cv_scores]}")
print(f"Mean CV AUC: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
print(f"Overall CV AUC: {overall_auc:.6f}")
```

#### 3. Submission (30 min)

```python
# Create submission
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': test_predictions
})

submission.to_csv('submission_deberta_v1.csv', index=False)
print(f"\nSubmission saved!")
print(f"Prediction stats:")
print(f"  Min: {test_predictions.min():.6f}")
print(f"  Max: {test_predictions.max():.6f}")
print(f"  Mean: {test_predictions.mean():.6f}")
print(f"  Median: {np.median(test_predictions):.6f}")
```

### Optimization Tips

1. **Sequence Length Analysis**:
   ```python
   # Check token lengths
   lengths = [len(tokenizer.encode(text)) for text in train_df['text']]
   print(f"Token length percentiles:")
   print(f"  50%: {np.percentile(lengths, 50):.0f}")
   print(f"  95%: {np.percentile(lengths, 95):.0f}")
   print(f"  99%: {np.percentile(lengths, 99):.0f}")

   # If 95% < 512, use 512. If most are shorter, use 384 or 256 for speed
   ```

2. **Memory Management**:
   ```python
   # If out of memory, reduce batch size
   per_device_train_batch_size=4  # Instead of 8
   gradient_accumulation_steps=4  # Instead of 2 (keep effective batch size = 16)
   ```

3. **Speed Up Training**:
   ```python
   # Use smaller validation set for faster evaluation
   val_fold = val_fold.select(range(min(len(val_fold), 200)))
   ```

---

## EXPERIMENT 2: Cross-Encoder Similarity

### Overview
- **Time**: 3 hours (1h code + 2h training)
- **Expected Impact**: +0.05-0.08 AUC
- **Hardware**: Kaggle CPU/GPU

### Implementation

```python
from sentence_transformers import CrossEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

# Load data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

# Initialize cross-encoder
print("Loading cross-encoder model...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

def compute_cross_encoder_features(df, cross_encoder):
    """
    Compute similarity scores between body and examples using cross-encoder
    """
    features = []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{len(df)}")

        body = row['body']

        # Compute scores with positive examples
        score_pos1 = cross_encoder.predict([(body, row['positive_example_1'])])[0]
        score_pos2 = cross_encoder.predict([(body, row['positive_example_2'])])[0]

        # Compute scores with negative examples
        score_neg1 = cross_encoder.predict([(body, row['negative_example_1'])])[0]
        score_neg2 = cross_encoder.predict([(body, row['negative_example_2'])])[0]

        # Aggregate features
        features.append([
            score_pos1, score_pos2, score_neg1, score_neg2,
            (score_pos1 + score_pos2) / 2,  # avg_pos
            (score_neg1 + score_neg2) / 2,  # avg_neg
            max(score_pos1, score_pos2),    # max_pos
            min(score_neg1, score_neg2),    # min_neg
            (score_pos1 + score_pos2 - score_neg1 - score_neg2) / 2,  # diff
        ])

    return np.array(features)

# Compute features
print("\nComputing cross-encoder features for train...")
X_train = compute_cross_encoder_features(train_df, cross_encoder)

print("\nComputing cross-encoder features for test...")
X_test = compute_cross_encoder_features(test_df, cross_encoder)

y_train = train_df['rule_violation'].values

# Optional: Add text statistics
from sklearn.feature_extraction.text import TfidfVectorizer

print("\nAdding TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
train_text = train_df['body'] + ' ' + train_df['rule']
test_text = test_df['body'] + ' ' + test_df['rule']
train_tfidf = tfidf.fit_transform(train_text).toarray()
test_tfidf = tfidf.transform(test_text).toarray()

X_train = np.hstack([X_train, train_tfidf])
X_test = np.hstack([X_test, test_tfidf])

print(f"\nFeature shape: {X_train.shape}")

# Cross-validation with LightGBM
print("\n5-Fold Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
oof_predictions = np.zeros(len(X_train))
test_predictions = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\nFold {fold}/5")

    # Split
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Train
    model = LGBMClassifier(
        objective='binary',
        metric='auc',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000,
        random_state=42,
        verbose=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # Predict
    oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
    test_predictions += model.predict_proba(X_test)[:, 1] / 5

    # Score
    fold_auc = roc_auc_score(y_val, oof_predictions[val_idx])
    cv_scores.append(fold_auc)
    print(f"Fold {fold} AUC: {fold_auc:.6f}")

overall_auc = roc_auc_score(y_train, oof_predictions)
print(f"\n{'='*50}")
print(f"Cross-Encoder + LightGBM Results")
print(f"{'='*50}")
print(f"Mean CV AUC: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
print(f"Overall CV AUC: {overall_auc:.6f}")

# Submission
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': test_predictions
})
submission.to_csv('submission_crossencoder_v1.csv', index=False)
print("\nSubmission saved!")
```

### Optimization: Batch Processing

```python
# Faster version using batch prediction
def compute_cross_encoder_features_batch(df, cross_encoder, batch_size=32):
    """
    Compute features in batches (faster)
    """
    n_samples = len(df)
    features = np.zeros((n_samples, 9))

    # Prepare all pairs
    pairs_pos1 = [(row['body'], row['positive_example_1']) for _, row in df.iterrows()]
    pairs_pos2 = [(row['body'], row['positive_example_2']) for _, row in df.iterrows()]
    pairs_neg1 = [(row['body'], row['negative_example_1']) for _, row in df.iterrows()]
    pairs_neg2 = [(row['body'], row['negative_example_2']) for _, row in df.iterrows()]

    # Batch predict
    print("Predicting positive example 1...")
    scores_pos1 = cross_encoder.predict(pairs_pos1, batch_size=batch_size, show_progress_bar=True)
    print("Predicting positive example 2...")
    scores_pos2 = cross_encoder.predict(pairs_pos2, batch_size=batch_size, show_progress_bar=True)
    print("Predicting negative example 1...")
    scores_neg1 = cross_encoder.predict(pairs_neg1, batch_size=batch_size, show_progress_bar=True)
    print("Predicting negative example 2...")
    scores_neg2 = cross_encoder.predict(pairs_neg2, batch_size=batch_size, show_progress_bar=True)

    # Compute aggregate features
    features[:, 0] = scores_pos1
    features[:, 1] = scores_pos2
    features[:, 2] = scores_neg1
    features[:, 3] = scores_neg2
    features[:, 4] = (scores_pos1 + scores_pos2) / 2  # avg_pos
    features[:, 5] = (scores_neg1 + scores_neg2) / 2  # avg_neg
    features[:, 6] = np.maximum(scores_pos1, scores_pos2)  # max_pos
    features[:, 7] = np.minimum(scores_neg1, scores_neg2)  # min_neg
    features[:, 8] = features[:, 4] - features[:, 5]  # diff

    return features
```

---

## EXPERIMENT 3: SetFit Contrastive Learning

### Overview
- **Time**: 3-4 hours
- **Expected Impact**: +0.06-0.10 AUC
- **Hardware**: Kaggle GPU

### Implementation

```python
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

def create_contrastive_pairs(df):
    """
    Create positive and negative pairs for contrastive learning
    """
    pairs = []
    labels = []

    for _, row in df.iterrows():
        body = row['body']
        rule = row['rule']
        text_with_rule = f"Rule: {rule} Comment: {body}"

        # Positive pairs (same class)
        if row['rule_violation'] == 1:
            # Violation - pair with positive examples
            pairs.append((text_with_rule, f"Rule: {rule} Comment: {row['positive_example_1']}"))
            labels.append(1)
            pairs.append((text_with_rule, f"Rule: {rule} Comment: {row['positive_example_2']}"))
            labels.append(1)
        else:
            # No violation - pair with negative examples
            pairs.append((text_with_rule, f"Rule: {rule} Comment: {row['negative_example_1']}"))
            labels.append(1)
            pairs.append((text_with_rule, f"Rule: {rule} Comment: {row['negative_example_2']}"))
            labels.append(1)

        # Negative pairs (different class)
        if row['rule_violation'] == 1:
            # Violation - pair with negative examples (dissimilar)
            pairs.append((text_with_rule, f"Rule: {rule} Comment: {row['negative_example_1']}"))
            labels.append(0)
        else:
            # No violation - pair with positive examples (dissimilar)
            pairs.append((text_with_rule, f"Rule: {rule} Comment: {row['positive_example_1']}"))
            labels.append(0)

    return pairs, labels

# 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
oof_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['rule_violation']), 1):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold}/5")
    print(f"{'='*50}")

    # Split data
    train_fold = train_df.iloc[train_idx].copy()
    val_fold = train_df.iloc[val_idx].copy()

    # Create contrastive pairs for training
    print("Creating contrastive pairs...")
    train_pairs, train_pair_labels = create_contrastive_pairs(train_fold)

    # Create datasets
    train_texts = [f"Rule: {row['rule']} Comment: {row['body']}" for _, row in train_fold.iterrows()]
    val_texts = [f"Rule: {row['rule']} Comment: {row['body']}" for _, row in val_fold.iterrows()]
    test_texts = [f"Rule: {row['rule']} Comment: {row['body']}" for _, row in test_df.iterrows()]

    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_fold['rule_violation'].values
    })

    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_fold['rule_violation'].values
    })

    # Initialize SetFit model
    print("Initializing SetFit model...")
    model = SetFitModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    # Train with contrastive learning
    print("Training with contrastive learning...")
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss_class=CosineSimilarityLoss,
        metric='accuracy',
        batch_size=16,
        num_iterations=20,  # Contrastive learning iterations
        num_epochs=1,  # Classification head epochs
        column_mapping={'text': 'text', 'label': 'label'}
    )

    trainer.train()

    # Validation predictions
    val_probs = model.predict_proba(val_texts)[:, 1]
    oof_predictions[val_idx] = val_probs

    # Test predictions
    test_probs = model.predict_proba(test_texts)[:, 1]
    test_predictions += test_probs / 5

    # Fold score
    fold_auc = roc_auc_score(val_fold['rule_violation'], val_probs)
    cv_scores.append(fold_auc)
    print(f"Fold {fold} AUC: {fold_auc:.6f}")

# Overall CV score
overall_auc = roc_auc_score(train_df['rule_violation'], oof_predictions)
print(f"\n{'='*50}")
print(f"SetFit Contrastive Learning Results")
print(f"{'='*50}")
print(f"Mean CV AUC: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
print(f"Overall CV AUC: {overall_auc:.6f}")

# Submission
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': test_predictions
})
submission.to_csv('submission_setfit_contrastive_v1.csv', index=False)
print("\nSubmission saved!")
```

---

## DEBUGGING CHECKLIST

### If DeBERTa Fails to Train:

1. **Out of Memory**:
   - Reduce batch size to 4
   - Increase gradient accumulation to 4
   - Use smaller model: `microsoft/deberta-v3-small`

2. **Training Too Slow**:
   - Ensure `fp16=True`
   - Reduce max_length to 384 or 256
   - Use fewer epochs (2 instead of 3)

3. **Poor CV Score**:
   - Check tokenization (print examples)
   - Verify labels are correct (0/1)
   - Try different learning rates (1e-5, 3e-5, 5e-5)

### If Cross-Encoder Fails:

1. **Too Slow**:
   - Use batch prediction (see optimization section)
   - Use smaller cross-encoder model

2. **Poor Performance**:
   - Add TF-IDF features
   - Try different cross-encoder models
   - Combine with SetFit features

### If SetFit Fails:

1. **Poor Contrastive Learning**:
   - Increase num_iterations to 50
   - Try different loss functions
   - Use larger model: `all-mpnet-base-v2`

2. **Slow Training**:
   - Reduce num_iterations to 10
   - Use smaller model: `all-MiniLM-L6-v2`

---

## NEXT STEPS AFTER IMPLEMENTATION

1. **Monitor Results**:
   - Track CV scores in spreadsheet
   - Compare CV vs LB
   - Identify best model

2. **Build Ensemble** (Day 3):
   - Combine DeBERTa + Cross-encoder + SetFit
   - Use stacking or weighted average
   - Expected +0.03-0.05 boost

3. **Iterate**:
   - If DeBERTa works → Try RoBERTa-large
   - If cross-encoder works → Try different models
   - If SetFit works → Fine-tune hyperparameters

---

**Good luck with implementation! Focus on getting DeBERTa working first - it's the highest priority.**
