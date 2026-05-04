# Jigsaw ACRC - Analysis Summary & Quick Reference

## Quick Stats

**Dataset:**
- Training: 2,029 samples (50.8% violations)
- Test: 10 samples (all subreddits in training data)
- 2 rules: Advertising (43.3% viol), Legal Advice (58.3% viol)
- 100 unique subreddits

**Current Performance:**
- CV: 0.7086
- LB: 0.670
- Gap to target (0.93): 0.263

---

## Top 5 Insights

### 1. Few-Shot Signal is Strongest (Impact: HIGH)
- Violations are **59% MORE similar** to positive examples
- Max similarity (0.0742 vs 0.0458) beats average similarity
- **Action:** Use max similarity + semantic embeddings

### 2. Subreddit Context is Critical (Impact: HIGH)
- Violation rates: 2.9% (soccerstreams) to 90.5% (churning)
- All test subreddits present in training data
- **Action:** Target encode subreddit risk, create rule-specific risks

### 3. Rule-Specific Patterns (Impact: HIGH)
- Legal keywords: 23x lift for "lawyer", 11x for "police"
- Questions indicate NON-violations in legal advice
- **Action:** Separate feature engineering per rule type

### 4. Overfitting Problem (Impact: HIGH)
- CV-LB gap: 0.0386
- **Action:** Increase regularization, feature selection

### 5. Keywords Show Clear Separation (Impact: MEDIUM-HIGH)
- Violation: sue (30x), lawyer (11.6x), legal (5.7x)
- Non-violation: stream (0.04x), html (0.06x), watch (0.13x)
- **Action:** Create keyword match features

---

## Feature Impact Rankings

| Rank | Feature | Expected Impact | Complexity | Priority |
|------|---------|----------------|------------|----------|
| 1 | Semantic similarity (Sentence-BERT) | +0.02-0.03 | Medium | **MUST** |
| 2 | Subreddit risk encoding | +0.015-0.025 | Low | **MUST** |
| 3 | Rule-specific keywords | +0.01-0.02 | Low | **MUST** |
| 4 | Few-shot max similarity | +0.015-0.02 | Low | **MUST** |
| 5 | Linguistic features | +0.01-0.015 | Low | **MUST** |
| 6 | Spam signals | +0.008-0.012 | Low | High |
| 7 | Character n-grams | +0.008-0.01 | Medium | High |
| 8 | Length ratios | +0.005-0.01 | Low | High |
| 9 | Modal verbs & questions | +0.005-0.008 | Low | High |
| 10 | Imperative mood | +0.005-0.008 | Low | Medium |

**Total Expected Improvement (Top 5):** +0.07-0.11 AUC

---

## Key Statistics

### Text Patterns

| Metric | Violation | Non-Violation | Difference |
|--------|-----------|---------------|------------|
| Body Length | 195.5 chars | 157.6 chars | **+24%** |
| Sentence Length | 11.1 words | 8.4 words | **+31%** |
| Stopword Ratio | 0.324 | 0.259 | **+25%** |
| URL Presence | 36.6% | 50.0% | **-27% (inverted!)** |

### Few-Shot Similarity

| Metric | Violation | Non-Violation | Lift |
|--------|-----------|---------------|------|
| Body-Positive Avg | 0.0542 | 0.0340 | **+59%** |
| Body-Negative Avg | 0.0379 | 0.0361 | +5% |
| Max Pos Similarity | 0.0742 | 0.0458 | **+62%** |
| Pos-Neg Difference | +0.0163 | -0.0020 | **inverted** |

### Discriminative Keywords

**Top Violation Indicators:**
- sue (30.4x), lawyer (11.6x), police (11.0x), legal (5.7x)

**Top Non-Violation Indicators:**
- stream (0.04x), html (0.06x), live (0.10x), watch (0.13x)

### Spam Signals

| Signal | Violation | Non-Violation | Lift |
|--------|-----------|---------------|------|
| Email | 1.07% | 0.10% | **10.65x** |
| Price | 6.01% | 1.00% | **6.00x** |
| Phone | 1.75% | 1.00% | 1.74x |

---

## Implementation Checklist

### Phase 1: Immediate (Expected +0.05-0.08 AUC)
- [ ] Install sentence-transformers: `pip install sentence-transformers`
- [ ] Implement semantic similarity features (Sentence-BERT)
- [ ] Add subreddit risk encoding with target encoding
- [ ] Create rule-specific keyword features
- [ ] Implement few-shot max similarity (not just average)
- [ ] Add linguistic features (sentence length, stopword ratio)
- [ ] Increase regularization:
  - [ ] `num_leaves`: 63 → 31
  - [ ] `min_child_samples`: 20 → 30
  - [ ] `reg_alpha`, `reg_lambda`: 0.1 → 0.3
  - [ ] `learning_rate`: 0.03 → 0.02

### Phase 2: High-Value (Expected +0.02-0.04 AUC)
- [ ] Add spam signal features (email, price, phone)
- [ ] Character 3-gram similarity
- [ ] Length ratio features (body/examples)
- [ ] Modal verb & question features
- [ ] Imperative mood detection

### Phase 3: Polish (Expected +0.01-0.02 AUC)
- [ ] Feature selection (top 1000 by importance)
- [ ] Ensemble: LightGBM + XGBoost + Sentence-BERT
- [ ] Optimize ensemble weights
- [ ] Pseudo-labeling (high confidence only)

---

## Quick Start

### Option 1: Use Provided Script
```bash
cd /mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC
python feature_engineering_quickstart.py
```

### Option 2: Manual Implementation

**1. Semantic Similarity:**
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
body_emb = model.encode(df['body'].tolist())
pos1_emb = model.encode(df['positive_example_1'].tolist())

df['semantic_sim'] = [cosine_similarity([b], [p])[0][0]
                      for b, p in zip(body_emb, pos1_emb)]
```

**2. Subreddit Risk:**
```python
global_mean = train_df['rule_violation'].mean()
subreddit_stats = train_df.groupby('subreddit')['rule_violation'].agg(['mean', 'count'])
subreddit_stats['risk'] = ((subreddit_stats['mean'] * subreddit_stats['count'] +
                            global_mean * 10) / (subreddit_stats['count'] + 10))
train_df['subreddit_risk'] = train_df['subreddit'].map(subreddit_stats['risk'])
```

**3. Keywords:**
```python
legal_keywords = ['lawyer', 'sue', 'police', 'legal', 'attorney']
df['legal_keyword_count'] = df['body'].str.lower().apply(
    lambda x: sum(1 for kw in legal_keywords if kw in x)
)
```

---

## Model Parameters

### Current (Overfitting)
```python
lgbm_params = {
    'num_leaves': 63,
    'learning_rate': 0.03,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}
```

### Recommended (Better Generalization)
```python
lgbm_params = {
    'num_leaves': 31,           # ↓ Reduced
    'learning_rate': 0.02,      # ↓ Slower learning
    'min_child_samples': 30,    # ↑ More regularization
    'reg_alpha': 0.3,           # ↑ Stronger L1
    'reg_lambda': 0.3,          # ↑ Stronger L2
    'feature_fraction': 0.7,    # ↓ More subsampling
    'max_depth': 6,             # NEW: Limit depth
}
```

---

## Expected Performance Trajectory

| Phase | CV AUC | LB AUC | Improvement |
|-------|--------|--------|-------------|
| Current | 0.7086 | 0.670 | Baseline |
| After Phase 1 | 0.75-0.78 | 0.72-0.75 | +0.05-0.08 |
| After Phase 2 | 0.77-0.80 | 0.75-0.78 | +0.02-0.03 |
| After Phase 3 | 0.78-0.82 | 0.77-0.81 | +0.02-0.03 |

**Realistic Target:** 0.77-0.81 LB
**Stretch Goal:** 0.82-0.85 LB
**Gap to 0.93:** Still significant (0.08-0.16)

---

## Common Pitfalls to Avoid

1. **Don't assume URL = violation**
   - Non-violations actually have MORE URLs (50% vs 37%)
   - Context matters: reference links vs promotional links

2. **Don't ignore rule type**
   - Legal advice has 35% higher violation rate
   - Different keywords and patterns per rule

3. **Don't use only average similarity**
   - Max similarity is more discriminative
   - Similarity variance matters

4. **Don't augment training data**
   - Current gap suggests overfitting, not underfitting
   - Focus on regularization instead

5. **Don't overtune on CV**
   - Test set is only 10 samples (high variance)
   - Prioritize robust features

---

## Key Files

- **Full Analysis:** `/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/DEEP_DATA_ANALYSIS_REPORT.md`
- **Implementation:** `/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/feature_engineering_quickstart.py`
- **This Summary:** `/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/ANALYSIS_SUMMARY.md`

---

## Contact & Questions

For detailed explanations, see the full report (`DEEP_DATA_ANALYSIS_REPORT.md`).

**Key Sections:**
- Section 3: Few-Shot Example Analysis
- Section 4: Rule-Specific Deep Dive
- Section 6: Top 20 Feature Ideas
- Section 11: Concrete Next Steps
