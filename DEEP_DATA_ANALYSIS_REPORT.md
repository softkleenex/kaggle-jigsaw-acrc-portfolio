# Jigsaw ACRC - Deep Data Analysis & Feature Engineering Strategy

**Date:** 2025-10-17
**Current Performance:** CV 0.7086, LB 0.670 (Gap: 0.263 to target 0.93)
**Dataset:** 2,029 training samples, 10 test samples

---

## EXECUTIVE SUMMARY: Top 5 Critical Insights

### 1. **Few-Shot Signal is Strong and Underutilized** (Impact: HIGH)
- Violation samples are **59% MORE similar** to positive examples (0.0542 vs 0.0340 Jaccard)
- Non-violation samples show **NEGATIVE correlation** with positive examples (-0.0020 difference)
- **Max similarity** feature (0.0742 vs 0.0458) is more discriminative than average
- **Current gap:** Simple Jaccard similarity is used, but semantic embeddings and edit distance provide stronger signal

### 2. **Rule-Specific Patterns are Dramatically Different** (Impact: HIGH)
- **Advertising violations:** 82.9% contain URLs (vs 84.3% non-violations) - URL presence is NOT enough!
- **Legal advice violations:** 1.55 legal keywords vs 0.61 (2.5x difference)
- Legal advice violations have 23x lift for "lawyer", 10.9x for "police", 5.7x for "legal"
- **Critical:** Need separate feature sets and models for each rule type

### 3. **Subreddit Context is Extremely Predictive** (Impact: HIGH)
- Violation rates range from 2.9% (soccerstreams) to 90.5% (churning) - **31x variance!**
- Top risky: churning (90.5%), sex (81%), legaladvice (78.9%), personalfinance (69.6%)
- **Test set** has all 9 subreddits in training data with known risk scores
- **Opportunity:** Subreddit risk encoding + subreddit-rule interaction features

### 4. **Discriminative Keywords Show Clear Separation** (Impact: MEDIUM-HIGH)
- Top violation indicators: sue (30x), lawyer (11.6x), police (11x), legal (5.7x)
- Top non-violation indicators: stream (0.04x), html (0.06x), live (0.10x), watch (0.13x)
- Legal keywords in violations: 15.8% "legal", 6.1% "sue", 4.7% "lawyer"
- **Gap:** Currently using TF-IDF on all text, but targeted keyword features are missing

### 5. **Body Length + Linguistic Features are Underexploited** (Impact: MEDIUM)
- Violations are **24% longer** (195 vs 158 chars)
- Violations have **31% higher stopword ratio** (0.324 vs 0.259) - more conversational
- Violations have **31% longer sentences** (11.1 vs 8.4 words/sentence)
- Email presence: **10.7x lift** for violations (1.07% vs 0.10%)
- Price mentions: **6x lift** for violations (6.01% vs 1.00%)

---

## 1. DATASET CHARACTERISTICS

### 1.1 Basic Statistics
```
Training Data: 2,029 samples
- Violations (1): 1,031 (50.8%)
- Non-violations (0): 998 (49.2%)
- Perfect balance!

Test Data: 10 samples
- All 9 test subreddits present in training
- Both rule types present
- Slightly shorter bodies: 141 chars (train: 177)

Text Fields:
- body: Main text to classify (51-499 chars)
- rule: 2 unique rules (54-103 chars)
- positive_example_1/2: Violation examples
- negative_example_1/2: Non-violation examples
- subreddit: 100 unique subreddits
```

### 1.2 Rule Distribution
```
Rule 1: "No Advertising: Spam, referral links, unsolicited advertising..."
- Samples: 1,012 (49.9%)
- Violation rate: 43.3%

Rule 2: "No legal advice: Do not offer or request legal advice."
- Samples: 1,017 (50.1%)
- Violation rate: 58.3%

KEY INSIGHT: Legal advice rule has 35% higher violation rate!
```

### 1.3 Subreddit Distribution
**Top 10 by Volume:**
1. legaladvice (213 samples, 78.9% violation rate)
2. AskReddit (152 samples, 48.0% violation rate)
3. soccerstreams (139 samples, 2.9% violation rate)
4. personalfinance (125 samples, 69.6% violation rate)
5. relationships (106 samples, 61.3% violation rate)
6. The_Donald (94 samples, 55.3% violation rate)
7. TwoXChromosomes (87 samples, 46.0% violation rate)
8. news (65 samples, 58.5% violation rate)
9. movies (56 samples, 37.5% violation rate)
10. videos (50 samples, 40.0% violation rate)

**Extreme Risk Subreddits:**
- churning: 90.5% violation rate (21 samples)
- sex: 81.0% violation rate (42 samples)
- legaladvice: 78.9% violation rate (213 samples)

**Safe Subreddits:**
- soccerstreams: 2.9% violation rate (139 samples)
- jailbreak: 15.4% violation rate (13 samples)
- whatisthisthing: 20.0% violation rate (10 samples)

### 1.4 Data Quality Issues
- **160 duplicate bodies** (7.9% of dataset) - potential data leakage
- **138 very long bodies** (>400 chars) - may need truncation handling
- **No missing values** - clean dataset
- **No obvious label noise** - positive/negative examples have low overlap

---

## 2. DEEP TEXT PATTERN ANALYSIS

### 2.1 Length Features (HIGH IMPACT)

| Feature | Violation | Non-Violation | Difference | % Diff |
|---------|-----------|---------------|------------|--------|
| **Body Length** | 195.5 | 157.6 | +37.9 | **+24.0%** |
| Rule Length | 74.8 | 82.2 | -7.4 | -9.0% |
| Pos1 Length | 200.1 | 186.7 | +13.4 | +7.2% |
| Pos2 Length | 194.7 | 189.0 | +5.8 | +3.1% |
| Neg1 Length | 152.9 | 148.2 | +4.6 | +3.1% |
| Neg2 Length | 146.2 | 149.0 | -2.8 | -1.9% |

**Correlation with target:** body_len (0.1668), rule_len (0.1503)

### 2.2 URL and Link Patterns (COUNTERINTUITIVE!)

```
URL Presence in Body:
- Violations: 36.6% have URLs
- Non-violations: 50.0% have URLs
- INVERTED SIGNAL: Non-violations have MORE URLs!

URL Count:
- Violations: 0.51 URLs on average
- Non-violations: 0.90 URLs on average

EXPLANATION:
- Advertising rule: Both violations and non-violations contain URLs (82-84%)
- Legal advice rule: Non-violations more likely to include reference links
- URL presence alone is NOT enough - need context!
```

### 2.3 Special Characters & Formatting

| Feature | Violation | Non-Violation | Lift |
|---------|-----------|---------------|------|
| Exclamation marks | 0.28 | 0.23 | 1.22x |
| Question marks | 0.27 | 0.29 | 0.93x |
| CAPS ratio | 0.0418 | 0.0548 | 0.76x |
| All-caps words | 0.79 | 0.94 | 0.84x |
| Emoji count | 0.01 | 0.01 | 1.00x |
| Newlines | 1.07 | 1.20 | 0.89x |

**Insight:** Special characters have weak signal - not priority features.

### 2.4 Discriminative Keywords (VERY HIGH IMPACT)

**Top 20 Violation Indicators (sorted by lift):**
1. sue (30.42x) - appears 41 times in violations, 1 in non-violations
2. lawyer (11.63x) - 47 vs 3
3. police (10.95x) - 59 vs 4
4. code (8.90x) - 36 vs 3
5. legal (5.67x) - 84 vs 11
6. law (4.14x) - 67 vs 12
7. girl (3.19x) - 43 vs 10
8. case (2.89x) - 39 vs 10
9. even (2.43x) - 59 vs 18
10. illegal (2.19x) - 59 vs 20
11. first (2.17x) - 41 vs 14
12. they (2.11x) - 247 vs 87
13. because (2.06x) - 64 vs 23
14. find (2.04x) - 44 vs 16
15. get (1.91x) - 206 vs 80
16. child (1.81x) - 39 vs 16
17. their (1.75x) - 85 vs 36
18. could (1.69x) - 73 vs 32
19. pay (1.69x) - 41 vs 18
20. don (1.69x) - 123 vs 54

**Top 20 Non-Violation Indicators:**
1. stream (0.04x) - 10 vs 209
2. html (0.06x) - 11 vs 130
3. live (0.10x) - 14 vs 99
4. watch (0.13x) - 21 vs 120
5. great (0.20x) - 11 vs 41
6. php (0.20x) - 14 vs 52
7. quality (0.26x) - 14 vs 40
8. pokemon (0.28x) - 12 vs 32
9. video (0.36x) - 15 vs 31
10. www (0.37x) - 151 vs 303
11. yes (0.37x) - 32 vs 64
12. check (0.41x) - 33 vs 60
13. https (0.43x) - 111 vs 193
14. more (0.44x) - 55 vs 93
15. new (0.47x) - 31 vs 49
16. post (0.48x) - 20 vs 31
17. than (0.48x) - 26 vs 40
18. online (0.48x) - 26 vs 40
19. http (0.50x) - 267 vs 399
20. com (0.59x) - 340 vs 429

### 2.5 Linguistic Features

| Feature | Violation | Non-Violation | Interpretation |
|---------|-----------|---------------|----------------|
| **Avg Sentence Length** | 11.1 words | 8.4 words | Violations are more verbose |
| **Unique Word Ratio** | 0.9005 | 0.9177 | Violations less diverse vocabulary |
| **Stopword Ratio** | 0.3243 | 0.2585 | Violations more conversational |
| **Starts with Capital** | 75.7% | 64.9% | Violations more formal start |

### 2.6 Spam & Promotion Signals

| Signal | Violation | Non-Violation | Lift |
|--------|-----------|---------------|------|
| **Email address** | 1.07% | 0.10% | **10.65x** |
| **Price mention** | 6.01% | 1.00% | **6.00x** |
| **Phone number** | 1.75% | 1.00% | 1.74x |
| **@ mention** | 1.45% | 0.50% | 2.90x |
| Hashtag | 1.65% | 2.61% | 0.63x |

---

## 3. FEW-SHOT EXAMPLE ANALYSIS (CRITICAL!)

### 3.1 Similarity Metrics (Jaccard)

**Body-Example Similarity:**
```
                        Violation   Non-Violation   Difference   % Diff
Body-Positive Avg:      0.0542      0.0340          +0.0201      +59.1%
Body-Negative Avg:      0.0379      0.0361          +0.0018      +5.1%
Pos-Neg Difference:     +0.0163     -0.0020         +0.0183      HUGE!

KEY INSIGHT:
- Violations are MORE similar to positive examples
- Non-violations are slightly MORE similar to negative examples (inverted signal!)
- The difference between pos/neg similarity is highly discriminative
```

**Max/Min Similarity Patterns:**
```
                        Violation   Non-Violation   Lift
Max Pos Similarity:     0.0742      0.0458          1.62x
Min Pos Similarity:     0.0341      0.0223          1.53x
Max Neg Similarity:     0.0578      0.0440          1.31x
Pos-Neg Max Diff:       +0.0221     -0.0079         -2.80x (inverted)

INSIGHT: Max similarity is more discriminative than average!
```

**Similarity Variance:**
```
Pos Similarity Std:     0.0283      0.0166          1.70x
Neg Similarity Std:     0.0195      0.0165          1.18x

INSIGHT: Violations have more variance in positive similarity
         → one example matches well, the other doesn't
```

### 3.2 Word Overlap Counts

```
                        Violation   Non-Violation
Body-Pos (avg):         3.13 words  1.99 words     (+57.3%)
Body-Neg (avg):         2.14 words  1.57 words     (+36.3%)

Difference:             +0.99       +0.42          (+135.7%)
```

### 3.3 Example Consistency

```
Pos Examples Similarity:    0.0522 (violations) vs 0.0491 (non-violations)
Neg Examples Similarity:    0.0383 (violations) vs 0.0316 (non-violations)

INSIGHT: Positive examples are more consistent than negative examples
         → Violations have clearer patterns than non-violations
```

### 3.4 Advanced Similarity Metrics (Sample Analysis)

**Edit Distance Similarity:**
```
Body-Pos1:              0.1416 (viol) vs 0.1515 (non-viol)  [INVERTED]
Body-Neg1:              0.1670 (viol) vs 0.1868 (non-viol)  [INVERTED]

INSIGHT: Edit distance shows inverted pattern - needs investigation
```

**Character 3-gram Overlap:**
```
Body-Pos1:              0.0846 (viol) vs 0.0618 (non-viol)  [+36.9%]

INSIGHT: Character-level similarity is more discriminative than word-level
```

**Bigram Overlap:**
```
Body-Pos1 bigram:       0.0014 (viol) vs 0.0013 (non-viol)  [negligible]

INSIGHT: Bigram overlap is too sparse, not useful
```

### 3.5 Length Ratio Features

```
Body/Pos1 ratio:        1.39 (viol) vs 1.26 (non-viol)  [+10.3%]
Body/Neg1 ratio:        1.78 (viol) vs 1.44 (non-viol)  [+23.6%]

INSIGHT: Violations are longer relative to examples
```

### 3.6 URL Alignment with Examples

```
Positive examples have URL:     43.4% (pos1), 43.0% (pos2)
Negative examples have URL:     45.9% (neg1), 45.1% (neg2)

URL alignment score:
- Violations: 0.9263
- Non-violations: 0.9028

INSIGHT: URL patterns in examples provide weak signal
```

---

## 4. RULE-SPECIFIC DEEP DIVE

### 4.1 Advertising Rule (43.3% violation rate)

**Key Characteristics:**
- URL presence is NOT discriminative: 82.9% (viol) vs 84.3% (non-viol)
- Ad keyword count: 0.77 (viol) vs 0.44 (non-viol) [+75%]
- Imperative verbs: 7.1% (viol) vs 5.1% (non-viol) [+39%]

**Discriminative Keywords:**
- buy (1.48x), sell (2.46x), free (1.66x), discount (2.26x), click (1.45x)

**Top First Words (violations):**
- "www.freekarma.com" (19 times)
- "findsextoday" (15 times)
- "get" (13 times)

**Pattern:** Self-promotional content with direct calls-to-action

### 4.2 Legal Advice Rule (58.3% violation rate)

**Key Characteristics:**
- Legal keyword count: 1.55 (viol) vs 0.61 (non-viol) [+154%]
- Modal verbs: 0.81 (viol) vs 0.65 (non-viol) [+25%]
- Question presence: 20.9% (viol) vs 30.2% (non-viol) [INVERTED!]

**Discriminative Keywords:**
- lawyer (23.2x), attorney (inf), legal (5.3x), sue (4.7x), court (8.4x)

**Pattern:**
- Violations: Prescriptive advice with legal terminology
- Non-violations: Questions seeking information (often with URLs to legal resources)

**CRITICAL INSIGHT:** Questions are LESS likely to be violations!
- Violations are statements/advice
- Non-violations are questions or factual information

---

## 5. TEST SET ANALYSIS

### 5.1 Test Distribution
```
Test samples: 10
Subreddits (all in training):
- AskReddit (2 samples) - 48.0% train violation rate
- hiphopheads (1) - 20.0%
- gonewild (1) - 36.8%
- personalfinance (1) - 69.6%
- Showerthoughts (1) - 62.1%
- leagueoflegends (1) - 35.7%
- BlackPeopleTwitter (1) - 50.0%
- movies (1) - 37.5%
- pics (1) - 56.7%

Rules: Both types present
Body length: 140.9 chars (vs 176.8 train) - shorter
```

### 5.2 Train vs Test Distribution Shift
```
Body Length: Train 176.8 ± 113.6, Test 140.9 ± 114.1
- Test bodies are ~20% shorter on average
- Similar variance - good sign

Subreddit Coverage: 100% (all test subreddits in train)
- Can leverage subreddit-specific patterns
- No domain shift issue

Rule Coverage: 100% (both rules present)
```

---

## 6. TOP 20 FEATURE IDEAS (Ranked by Expected Impact)

### Tier 1: MUST-HAVE (Expected Impact: HIGH - Individual AUC lift 0.01-0.03)

#### 1. **Semantic Similarity Features (Sentence-BERT embeddings)**
**Implementation:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
body_emb = model.encode(df['body'].tolist())
pos1_emb = model.encode(df['positive_example_1'].tolist())
pos2_emb = model.encode(df['positive_example_2'].tolist())
neg1_emb = model.encode(df['negative_example_1'].tolist())
neg2_emb = model.encode(df['negative_example_2'].tolist())

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
df['body_pos1_semantic'] = [cosine_similarity([b], [p])[0][0]
                             for b, p in zip(body_emb, pos1_emb)]
df['body_pos2_semantic'] = [cosine_similarity([b], [p])[0][0]
                             for b, p in zip(body_emb, pos2_emb)]
df['body_neg1_semantic'] = [cosine_similarity([b], [n])[0][0]
                             for b, n in zip(body_emb, neg1_emb)]
df['body_neg2_semantic'] = [cosine_similarity([b], [n])[0][0]
                             for b, n in zip(body_emb, neg2_emb)]

# Aggregations
df['avg_pos_semantic'] = (df['body_pos1_semantic'] + df['body_pos2_semantic']) / 2
df['avg_neg_semantic'] = (df['body_neg1_semantic'] + df['body_neg2_semantic']) / 2
df['max_pos_semantic'] = df[['body_pos1_semantic', 'body_pos2_semantic']].max(axis=1)
df['min_pos_semantic'] = df[['body_pos1_semantic', 'body_pos2_semantic']].min(axis=1)
df['pos_neg_semantic_diff'] = df['avg_pos_semantic'] - df['avg_neg_semantic']
df['pos_neg_max_diff'] = df['max_pos_semantic'] - df[['body_neg1_semantic', 'body_neg2_semantic']].max(axis=1)
```
**Expected Impact:** 0.02-0.03 AUC improvement
**Complexity:** Medium (requires sentence-transformers library, ~2-3 min compute time)
**Why:** Semantic similarity captures intent better than word overlap

#### 2. **Subreddit Risk Encoding + Interaction**
**Implementation:**
```python
# Target encoding with smoothing
def target_encode_subreddit(train_df, test_df, alpha=10):
    global_mean = train_df['rule_violation'].mean()

    # Calculate subreddit statistics
    subreddit_stats = train_df.groupby('subreddit')['rule_violation'].agg(['mean', 'count'])
    subreddit_stats['risk_score'] = (subreddit_stats['mean'] * subreddit_stats['count'] +
                                     global_mean * alpha) / (subreddit_stats['count'] + alpha)

    # Map to datasets
    train_df['subreddit_risk'] = train_df['subreddit'].map(subreddit_stats['risk_score'])
    test_df['subreddit_risk'] = test_df['subreddit'].map(subreddit_stats['risk_score']).fillna(global_mean)

    # Rule-specific risk
    for rule_idx, rule_name in enumerate(['advertising', 'legal']):
        rule_mask = train_df['rule'].str.contains('Advertising' if rule_idx == 0 else 'legal')
        rule_stats = train_df[rule_mask].groupby('subreddit')['rule_violation'].agg(['mean', 'count'])
        rule_stats[f'{rule_name}_risk'] = (rule_stats['mean'] * rule_stats['count'] +
                                           global_mean * alpha) / (rule_stats['count'] + alpha)

        train_df[f'subreddit_{rule_name}_risk'] = train_df['subreddit'].map(rule_stats[f'{rule_name}_risk'])
        test_df[f'subreddit_{rule_name}_risk'] = test_df['subreddit'].map(rule_stats[f'{rule_name}_risk']).fillna(global_mean)

    # Interaction: is_rule_match
    train_df['subreddit_rule_match'] = ((train_df['rule'].str.contains('Advertising')) *
                                        train_df['subreddit_advertising_risk'] +
                                        (train_df['rule'].str.contains('legal')) *
                                        train_df['subreddit_legal_risk'])

    return train_df, test_df

train_df, test_df = target_encode_subreddit(train_df, test_df)
```
**Expected Impact:** 0.015-0.025 AUC improvement
**Complexity:** Low (simple aggregation)
**Why:** Subreddit context is highly predictive (2.9% to 90.5% violation rate range)

#### 3. **Rule-Specific Keyword Match Features**
**Implementation:**
```python
# Define rule-specific keywords
AD_KEYWORDS = ['buy', 'sell', 'free', 'discount', 'offer', 'deal', 'promo', 'sale',
               'shop', 'price', 'purchase', 'cheap', 'subscribe', 'follow', 'click',
               'check', 'visit', 'link', 'referral', 'code', 'coupon']

LEGAL_KEYWORDS = ['lawyer', 'attorney', 'legal', 'law', 'court', 'sue', 'police',
                  'advice', 'should', 'can', 'could', 'consult', 'rights', 'illegal',
                  'liable', 'contract', 'case', 'defendant', 'plaintiff']

def keyword_features(df):
    df['body_lower'] = df['body'].str.lower()

    # Count matches
    df['ad_keyword_count'] = df['body_lower'].apply(lambda x: sum(1 for kw in AD_KEYWORDS if kw in x))
    df['legal_keyword_count'] = df['body_lower'].apply(lambda x: sum(1 for kw in LEGAL_KEYWORDS if kw in x))

    # Binary presence
    for kw in ['lawyer', 'sue', 'police', 'legal', 'stream', 'watch', 'html']:
        df[f'has_{kw}'] = df['body_lower'].str.contains(kw, regex=False).astype(int)

    # Rule-aware keyword count
    df['rule_keyword_count'] = 0
    df.loc[df['rule'].str.contains('Advertising'), 'rule_keyword_count'] = df.loc[df['rule'].str.contains('Advertising'), 'ad_keyword_count']
    df.loc[df['rule'].str.contains('legal'), 'rule_keyword_count'] = df.loc[df['rule'].str.contains('legal'), 'legal_keyword_count']

    # Keyword density
    df['body_word_count'] = df['body'].str.split().str.len()
    df['ad_keyword_density'] = df['ad_keyword_count'] / (df['body_word_count'] + 1)
    df['legal_keyword_density'] = df['legal_keyword_count'] / (df['body_word_count'] + 1)

    return df
```
**Expected Impact:** 0.01-0.02 AUC improvement
**Complexity:** Low
**Why:** Legal keywords have 5-30x lift, advertising keywords have 1.5-2.5x lift

#### 4. **Few-Shot Max Similarity Features**
**Implementation:**
```python
def jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    if len(set1) == 0 and len(set2) == 0:
        return 0
    return len(set1.intersection(set2)) / len(set1.union(set2))

# Calculate all similarities
for i in [1, 2]:
    df[f'body_pos{i}_jaccard'] = df.apply(lambda x: jaccard_similarity(x['body'], x[f'positive_example_{i}']), axis=1)
    df[f'body_neg{i}_jaccard'] = df.apply(lambda x: jaccard_similarity(x['body'], x[f'negative_example_{i}']), axis=1)

# Max/Min features (more discriminative than average)
df['max_pos_jaccard'] = df[['body_pos1_jaccard', 'body_pos2_jaccard']].max(axis=1)
df['min_pos_jaccard'] = df[['body_pos1_jaccard', 'body_pos2_jaccard']].min(axis=1)
df['max_neg_jaccard'] = df[['body_neg1_jaccard', 'body_neg2_jaccard']].max(axis=1)
df['min_neg_jaccard'] = df[['body_neg1_jaccard', 'body_neg2_jaccard']].min(axis=1)

# Differences and ratios
df['pos_neg_max_diff'] = df['max_pos_jaccard'] - df['max_neg_jaccard']
df['pos_jaccard_range'] = df['max_pos_jaccard'] - df['min_pos_jaccard']
df['neg_jaccard_range'] = df['max_neg_jaccard'] - df['min_neg_jaccard']
df['pos_neg_jaccard_ratio'] = df['max_pos_jaccard'] / (df['max_neg_jaccard'] + 0.01)
```
**Expected Impact:** 0.015-0.02 AUC improvement
**Complexity:** Low
**Why:** Max similarity shows 1.62x lift vs 1.59x for average

#### 5. **Linguistic Complexity Features**
**Implementation:**
```python
def linguistic_features(df):
    # Sentence-level features
    df['body_sentences'] = df['body'].str.count(r'[.!?]+') + 1
    df['body_words'] = df['body'].str.split().str.len()
    df['body_avg_sent_len'] = df['body_words'] / df['body_sentences']

    # Vocabulary diversity
    df['body_unique_word_ratio'] = df['body'].apply(lambda x: len(set(x.lower().split())) / max(len(x.split()), 1))

    # Stopword ratio
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
                 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    df['body_stopword_ratio'] = df['body'].apply(lambda x: sum(1 for w in x.lower().split() if w in stopwords) / max(len(x.split()), 1))

    # Average word length
    df['body_avg_word_len'] = df['body'].apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)

    return df
```
**Expected Impact:** 0.01-0.015 AUC improvement
**Complexity:** Low
**Why:** Violations have 31% longer sentences, 25% higher stopword ratio

### Tier 2: HIGH-VALUE (Expected Impact: MEDIUM - Individual AUC lift 0.005-0.01)

#### 6. **Spam Signal Features**
**Implementation:**
```python
import re

def spam_features(df):
    # Email detection
    df['has_email'] = df['body'].str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', regex=True).astype(int)

    # Phone detection
    df['has_phone'] = df['body'].str.contains(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', regex=True).astype(int)

    # Price detection
    df['has_price'] = df['body'].str.contains(r'\$\d+|\d+\s*(dollar|USD|euro|£)', regex=True, case=False).astype(int)
    df['price_count'] = df['body'].str.count(r'\$\d+')

    # Mention/hashtag
    df['has_mention'] = df['body'].str.contains(r'@\w+', regex=True).astype(int)
    df['has_hashtag'] = df['body'].str.contains(r'#\w+', regex=True).astype(int)

    # URL features
    df['url_count'] = df['body'].str.count(r'http[s]?://|www\.')
    df['has_url'] = (df['url_count'] > 0).astype(int)

    # Composite spam score
    df['spam_score'] = (df['has_email'] * 10.65 +
                        df['has_price'] * 6.00 +
                        df['has_phone'] * 1.74 +
                        df['has_mention'] * 2.90) / 21.29  # Normalized by sum of lifts

    return df
```
**Expected Impact:** 0.008-0.012 AUC improvement
**Complexity:** Low
**Why:** Email (10.7x lift), price (6x lift) are strong signals

#### 7. **Character-Level N-gram Features**
**Implementation:**
```python
def char_ngram_similarity(str1, str2, n=3):
    ngrams1 = set([str1[i:i+n] for i in range(len(str1)-n+1)])
    ngrams2 = set([str2[i:i+n] for i in range(len(str2)-n+1)])
    if len(ngrams1.union(ngrams2)) == 0:
        return 0
    return len(ngrams1.intersection(ngrams2)) / len(ngrams1.union(ngrams2))

# Apply to body-example pairs
for i in [1, 2]:
    df[f'body_pos{i}_char3'] = df.apply(lambda x: char_ngram_similarity(x['body'].lower(), x[f'positive_example_{i}'].lower(), 3), axis=1)
    df[f'body_neg{i}_char3'] = df.apply(lambda x: char_ngram_similarity(x['body'].lower(), x[f'negative_example_{i}'].lower(), 3), axis=1)

df['max_pos_char3'] = df[['body_pos1_char3', 'body_pos2_char3']].max(axis=1)
df['pos_neg_char3_diff'] = df['max_pos_char3'] - df[['body_neg1_char3', 'body_neg2_char3']].max(axis=1)
```
**Expected Impact:** 0.008-0.01 AUC improvement
**Complexity:** Medium (slower computation)
**Why:** Character 3-gram overlap shows 36.9% higher for violations

#### 8. **Length Ratio Features**
**Implementation:**
```python
def length_ratio_features(df):
    # Body vs examples
    df['body_pos1_len_ratio'] = df['body'].str.len() / (df['positive_example_1'].str.len() + 1)
    df['body_pos2_len_ratio'] = df['body'].str.len() / (df['positive_example_2'].str.len() + 1)
    df['body_neg1_len_ratio'] = df['body'].str.len() / (df['negative_example_1'].str.len() + 1)
    df['body_neg2_len_ratio'] = df['body'].str.len() / (df['negative_example_2'].str.len() + 1)

    # Average ratios
    df['avg_pos_len_ratio'] = (df['body_pos1_len_ratio'] + df['body_pos2_len_ratio']) / 2
    df['avg_neg_len_ratio'] = (df['body_neg1_len_ratio'] + df['body_neg2_len_ratio']) / 2
    df['pos_neg_len_ratio_diff'] = df['avg_pos_len_ratio'] - df['avg_neg_len_ratio']

    # Example lengths
    df['pos_examples_avg_len'] = (df['positive_example_1'].str.len() + df['positive_example_2'].str.len()) / 2
    df['neg_examples_avg_len'] = (df['negative_example_1'].str.len() + df['negative_example_2'].str.len()) / 2
    df['examples_len_diff'] = df['pos_examples_avg_len'] - df['neg_examples_avg_len']

    return df
```
**Expected Impact:** 0.005-0.01 AUC improvement
**Complexity:** Low
**Why:** Violations have 23.6% higher body/neg1 ratio

#### 9. **Modal Verb & Question Features**
**Implementation:**
```python
import re

def modal_question_features(df):
    # Modal verbs (for legal advice)
    modal_pattern = r'\b(should|could|would|might|may|can|must|ought|shall)\b'
    df['modal_count'] = df['body'].str.count(modal_pattern, flags=re.IGNORECASE)
    df['has_modal'] = (df['modal_count'] > 0).astype(int)

    # Question patterns
    df['question_count'] = df['body'].str.count(r'\?')
    df['has_question'] = (df['question_count'] > 0).astype(int)

    # Question word starts
    question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'can', 'could', 'should', 'would']
    df['starts_question_word'] = df['body'].str.lower().str.split().str[0].isin(question_words).astype(int)

    # Rule-specific features
    df['legal_modal_interaction'] = df['modal_count'] * df['rule'].str.contains('legal').astype(int)
    df['legal_question_interaction'] = df['question_count'] * df['rule'].str.contains('legal').astype(int)

    return df
```
**Expected Impact:** 0.005-0.008 AUC improvement
**Complexity:** Low
**Why:** Legal violations have 25% more modals, but 30% FEWER questions (inverted signal!)

#### 10. **Imperative Mood Detection**
**Implementation:**
```python
def imperative_features(df):
    # Imperative verbs
    imperative_verbs = ['get', 'buy', 'check', 'click', 'visit', 'see', 'watch', 'try',
                        'call', 'contact', 'follow', 'subscribe', 'join', 'sign', 'download',
                        'install', 'use', 'make', 'take', 'learn', 'discover', 'find']

    df['starts_imperative'] = df['body'].str.lower().str.split().str[0].isin(imperative_verbs).astype(int)
    df['imperative_count'] = df['body'].apply(lambda x: sum(1 for verb in imperative_verbs if verb in x.lower().split()[:5]))

    # Rule-specific
    df['ad_imperative_interaction'] = df['starts_imperative'] * df['rule'].str.contains('Advertising').astype(int)

    return df
```
**Expected Impact:** 0.005-0.008 AUC improvement
**Complexity:** Low
**Why:** 39% higher for advertising violations

### Tier 3: NICE-TO-HAVE (Expected Impact: LOW-MEDIUM - Individual AUC lift 0.003-0.008)

#### 11. **Example Agreement Features**
**Implementation:**
```python
def example_agreement_features(df):
    # Calculate inter-example similarity
    df['pos_examples_jaccard'] = df.apply(lambda x: jaccard_similarity(x['positive_example_1'], x['positive_example_2']), axis=1)
    df['neg_examples_jaccard'] = df.apply(lambda x: jaccard_similarity(x['negative_example_1'], x['negative_example_2']), axis=1)

    # Agreement score (do both positive examples agree on body?)
    df['pos_agreement_score'] = df[['body_pos1_jaccard', 'body_pos2_jaccard']].min(axis=1) / (df[['body_pos1_jaccard', 'body_pos2_jaccard']].max(axis=1) + 0.01)
    df['neg_agreement_score'] = df[['body_neg1_jaccard', 'body_neg2_jaccard']].min(axis=1) / (df[['body_neg1_jaccard', 'body_neg2_jaccard']].max(axis=1) + 0.01)

    # Consistency (std of similarities)
    df['pos_sim_std'] = df[['body_pos1_jaccard', 'body_pos2_jaccard']].std(axis=1)
    df['neg_sim_std'] = df[['body_neg1_jaccard', 'body_neg2_jaccard']].std(axis=1)

    return df
```
**Expected Impact:** 0.005-0.008 AUC improvement
**Complexity:** Low
**Why:** Violations have 70% higher positive similarity std (0.0283 vs 0.0166)

#### 12. **Capitalization & Formatting Features**
**Implementation:**
```python
def capitalization_features(df):
    # Capitalization patterns
    df['body_upper_ratio'] = df['body'].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    df['body_starts_caps'] = df['body'].str[0].str.isupper().astype(int)
    df['body_all_caps_words'] = df['body'].apply(lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1))

    # Special characters
    df['body_exclaim_count'] = df['body'].str.count('!')
    df['body_question_count'] = df['body'].str.count(r'\?')
    df['body_punct_ratio'] = df['body'].apply(lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()) / max(len(x), 1))

    # Newlines and paragraphs
    df['body_newline_count'] = df['body'].str.count('\n')
    df['body_paragraph_count'] = df['body'].str.count('\n\n') + 1

    return df
```
**Expected Impact:** 0.003-0.005 AUC improvement
**Complexity:** Low
**Why:** Weak signal but easy to compute

#### 13. **First/Last Word Features**
**Implementation:**
```python
def position_features(df):
    # Extract first/last words
    df['body_first_word'] = df['body'].str.strip().str.split().str[0].str.lower()
    df['body_last_word'] = df['body'].str.strip().str.split().str[-1].str.lower()

    # Check if first word is in top violation words
    violation_first_words = ['i', 'you', 'if', 'this', 'get', 'he', 'she', 'they', 'my']
    df['first_word_violation_flag'] = df['body_first_word'].isin(violation_first_words).astype(int)

    # Check if first word is in top non-violation words
    non_violation_first_words = ['sd', '**hd**', 'oh', "we're", 'must']
    df['first_word_nonviolation_flag'] = df['body_first_word'].isin(non_violation_first_words).astype(int)

    return df
```
**Expected Impact:** 0.003-0.006 AUC improvement
**Complexity:** Low
**Why:** Some clear patterns but limited coverage

#### 14. **TF-IDF on Rule-Body Concatenation**
**Implementation:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def rule_body_tfidf(train_df, test_df, max_features=5000):
    # Create rule-aware text
    train_df['rule_body'] = train_df['rule'] + ' [SEP] ' + train_df['body']
    test_df['rule_body'] = test_df['rule'] + ' [SEP] ' + test_df['body']

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                           min_df=2, max_df=0.9, sublinear_tf=True)

    train_tfidf = tfidf.fit_transform(train_df['rule_body'])
    test_tfidf = tfidf.transform(test_df['rule_body'])

    return train_tfidf, test_tfidf
```
**Expected Impact:** 0.005-0.008 AUC improvement
**Complexity:** Low
**Why:** Rule context may help disambiguate borderline cases

#### 15. **Word Overlap Weighted by IDF**
**Implementation:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def idf_weighted_overlap(df, idf_dict):
    def weighted_jaccard(str1, str2, idf_dict):
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())

        intersection_weight = sum(idf_dict.get(w, 1.0) for w in words1.intersection(words2))
        union_weight = sum(idf_dict.get(w, 1.0) for w in words1.union(words2))

        return intersection_weight / union_weight if union_weight > 0 else 0

    # Calculate IDF from body texts
    tfidf = TfidfVectorizer()
    tfidf.fit(df['body'])
    idf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

    # Apply
    df['body_pos1_idf_jaccard'] = df.apply(lambda x: weighted_jaccard(x['body'], x['positive_example_1'], idf_dict), axis=1)
    df['body_pos2_idf_jaccard'] = df.apply(lambda x: weighted_jaccard(x['body'], x['positive_example_2'], idf_dict), axis=1)

    df['max_pos_idf_jaccard'] = df[['body_pos1_idf_jaccard', 'body_pos2_idf_jaccard']].max(axis=1)

    return df
```
**Expected Impact:** 0.005-0.008 AUC improvement
**Complexity:** Medium
**Why:** Weights important words more heavily

#### 16. **Duplicate Body Flag**
**Implementation:**
```python
def duplicate_features(df):
    # Mark duplicates
    df['body_is_duplicate'] = df['body'].duplicated(keep=False).astype(int)

    # Duplicate cluster size
    duplicate_counts = df['body'].value_counts()
    df['body_duplicate_count'] = df['body'].map(duplicate_counts)

    return df
```
**Expected Impact:** 0.002-0.004 AUC improvement
**Complexity:** Low
**Why:** 160 duplicate bodies - may indicate spam patterns

#### 17. **Rule Text Overlap Features**
**Implementation:**
```python
def rule_body_overlap_features(df):
    # Word overlap with rule
    df['body_rule_overlap'] = df.apply(lambda x: len(set(x['body'].lower().split()).intersection(set(x['rule'].lower().split()))), axis=1)
    df['body_rule_jaccard'] = df.apply(lambda x: jaccard_similarity(x['body'], x['rule']), axis=1)

    # Specific rule keyword presence
    df['body_has_spam'] = df['body'].str.lower().str.contains('spam').astype(int)
    df['body_has_advertising'] = df['body'].str.lower().str.contains('advertis').astype(int)
    df['body_has_referral'] = df['body'].str.lower().str.contains('referral').astype(int)

    return df
```
**Expected Impact:** 0.003-0.005 AUC improvement
**Complexity:** Low
**Why:** Violations have slightly more overlap with rule text (0.65 vs 0.42 words)

#### 18. **URL Domain Features**
**Implementation:**
```python
import re
from urllib.parse import urlparse

def url_domain_features(df):
    def extract_domains(text):
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        domains = [urlparse(url).netloc for url in urls]
        return domains

    df['url_domains'] = df['body'].apply(extract_domains)
    df['unique_domain_count'] = df['url_domains'].apply(len)

    # Check for common domains
    common_domains = ['youtube.com', 'reddit.com', 'imgur.com', 'twitter.com']
    df['has_common_domain'] = df['url_domains'].apply(lambda x: any(d in str(x) for d in common_domains)).astype(int)

    return df
```
**Expected Impact:** 0.003-0.005 AUC improvement
**Complexity:** Medium
**Why:** URL presence is counterintuitive - domain analysis may help

#### 19. **Positional N-gram Features**
**Implementation:**
```python
def positional_ngram_features(df):
    # First 3 words
    df['body_first_3words'] = df['body'].str.split().str[:3].str.join(' ').str.lower()

    # Last 3 words
    df['body_last_3words'] = df['body'].str.split().str[-3:].str.join(' ').str.lower()

    # TF-IDF on positional n-grams (can be applied separately)
    # This would be a separate TfidfVectorizer call

    return df
```
**Expected Impact:** 0.002-0.004 AUC improvement
**Complexity:** Low
**Why:** Some clear first-word patterns but limited coverage

#### 20. **Cross-Example Similarity Features**
**Implementation:**
```python
def cross_example_features(df):
    # Positive example 1 vs negative examples
    df['pos1_neg1_jaccard'] = df.apply(lambda x: jaccard_similarity(x['positive_example_1'], x['negative_example_1']), axis=1)
    df['pos1_neg2_jaccard'] = df.apply(lambda x: jaccard_similarity(x['positive_example_1'], x['negative_example_2']), axis=1)
    df['pos2_neg1_jaccard'] = df.apply(lambda x: jaccard_similarity(x['positive_example_2'], x['negative_example_1']), axis=1)
    df['pos2_neg2_jaccard'] = df.apply(lambda x: jaccard_similarity(x['positive_example_2'], x['negative_example_2']), axis=1)

    # Average cross-similarity (how similar are pos/neg examples?)
    df['avg_cross_similarity'] = (df['pos1_neg1_jaccard'] + df['pos1_neg2_jaccard'] +
                                   df['pos2_neg1_jaccard'] + df['pos2_neg2_jaccard']) / 4

    # Separation score (are examples well-separated?)
    df['example_separation'] = df['pos_examples_jaccard'] - df['avg_cross_similarity']

    return df
```
**Expected Impact:** 0.003-0.005 AUC improvement
**Complexity:** Low
**Why:** Example quality and separation may provide meta-information

---

## 7. FEATURE ENGINEERING STRATEGY

### 7.1 Implementation Priority

**Phase 1 (Immediate - Expected +0.05-0.08 AUC):**
1. Semantic similarity with Sentence-BERT (Feature #1)
2. Subreddit risk encoding (Feature #2)
3. Rule-specific keyword features (Feature #3)
4. Few-shot max similarity (Feature #4)
5. Linguistic features (Feature #5)

**Phase 2 (Short-term - Expected +0.02-0.04 AUC):**
6. Spam signal features (Feature #6)
7. Character n-gram similarity (Feature #7)
8. Length ratio features (Feature #8)
9. Modal verb & question features (Feature #9)
10. Imperative mood detection (Feature #10)

**Phase 3 (If time permits - Expected +0.01-0.02 AUC):**
11-20. Remaining features

### 7.2 Computational Budget

**Current Setup:**
- TF-IDF: ~15,000 features, ~2 min compute
- Basic features: 17 features, <1 min compute
- Model training (5-fold CV): ~5-10 min

**With New Features:**
- Sentence-BERT embeddings: ~2-3 min (one-time)
- Semantic similarity: ~1 min
- All other features: ~2-3 min
- Total feature engineering: ~8-10 min
- Model training: ~10-15 min (more features)
- **Total pipeline: ~20-25 min** (well within 9hr limit)

### 7.3 Feature Selection Strategy

**Option 1: Feature Importance Filtering (Recommended)**
```python
from sklearn.ensemble import RandomForestClassifier

# Train RF on all features
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Get feature importance
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Keep top N features
top_features = importance.head(1000)['feature'].tolist()
X_train_selected = X_train[:, top_features]
```

**Option 2: Correlation-Based Selection**
```python
# Remove highly correlated features
corr_matrix = pd.DataFrame(X_train).corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
```

**Option 3: Recursive Feature Elimination (RFE)**
```python
from sklearn.feature_selection import RFECV

selector = RFECV(estimator=LGBMClassifier(), step=100, cv=5, scoring='roc_auc')
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
```

**Recommendation:** Use Feature Importance Filtering (fastest, most reliable)

### 7.4 Interaction Features

**High-Value Interactions:**
```python
# Subreddit × Rule
df['subreddit_rule_interaction'] = df['subreddit_risk'] * df['rule_keyword_count']

# Similarity × Length
df['pos_sim_times_len'] = df['max_pos_jaccard'] * df['body_len']

# Keyword × Rule type
df['keyword_rule_match'] = ((df['rule'].str.contains('Advertising')) * df['ad_keyword_count'] +
                            (df['rule'].str.contains('legal')) * df['legal_keyword_count'])

# Semantic similarity × Lexical similarity
df['semantic_lexical_product'] = df['max_pos_semantic'] * df['max_pos_jaccard']
df['semantic_lexical_ratio'] = df['max_pos_semantic'] / (df['max_pos_jaccard'] + 0.01)
```

---

## 8. MODEL-SPECIFIC RECOMMENDATIONS

### 8.1 Tree-Based Models (LightGBM, XGBoost)

**Best Features:**
- Subreddit risk scores (categorical feature)
- Keyword counts (non-linear relationship)
- Max/Min similarity (capturing extremes)
- Length ratios (non-linear)
- Spam signals (binary features)

**Optimal Parameters:**
```python
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,  # Reduce from 63 to prevent overfitting
    'learning_rate': 0.02,  # Lower learning rate
    'feature_fraction': 0.7,  # More aggressive subsampling
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 30,  # Increase from 20
    'reg_alpha': 0.3,  # Stronger L1 regularization
    'reg_lambda': 0.3,  # Stronger L2 regularization
    'max_depth': 6,  # Limit depth
    'verbose': -1,
    'random_state': 42,
    'n_estimators': 3000,
    'early_stopping_rounds': 150
}
```

**Why:** Current gap (CV 0.7086 vs LB 0.670) suggests overfitting. Need stronger regularization.

### 8.2 Neural Models (BERT, SetFit)

**Best Features:**
- Semantic embeddings (natural for transformers)
- Concatenated text: `[CLS] body [SEP] rule [SEP] pos1 [SEP] pos2 [SEP] neg1 [SEP] neg2`
- Token-level attention (not word-level)

**Optimal Architecture:**
```python
from sentence_transformers import SentenceTransformer, losses
from setfit import SetFitModel, SetFitTrainer

# SetFit with contrastive learning
model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# Create training pairs
train_examples = []
for _, row in train_df.iterrows():
    body = row['body']
    if row['rule_violation'] == 1:
        # Positive pairs
        train_examples.append(InputExample(texts=[body, row['positive_example_1']], label=1.0))
        train_examples.append(InputExample(texts=[body, row['positive_example_2']], label=1.0))
        # Negative pairs
        train_examples.append(InputExample(texts=[body, row['negative_example_1']], label=0.0))
        train_examples.append(InputExample(texts=[body, row['negative_example_2']], label=0.0))
    else:
        # Inverted for non-violations
        train_examples.append(InputExample(texts=[body, row['positive_example_1']], label=0.0))
        train_examples.append(InputExample(texts=[body, row['positive_example_2']], label=0.0))
        train_examples.append(InputExample(texts=[body, row['negative_example_1']], label=1.0))
        train_examples.append(InputExample(texts=[body, row['negative_example_2']], label=1.0))

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_examples,
    num_iterations=20,
    column_mapping={"text": "text", "label": "label"}
)
trainer.train()
```

**Why:** SetFit leverages few-shot examples explicitly via contrastive learning.

### 8.3 Hybrid Approach (RECOMMENDED)

**Strategy:**
1. **Tree-based model** on engineered features (Phase 1 + Phase 2)
2. **Semantic model** (Sentence-BERT + similarity features)
3. **Ensemble:** Weighted average or stacking

**Implementation:**
```python
# Model 1: LightGBM with engineered features
lgbm_preds = train_lgbm(X_engineered, y)

# Model 2: Sentence-BERT similarity
sbert_preds = train_sbert_classifier(texts, y)

# Model 3: SetFit few-shot
setfit_preds = train_setfit(texts, examples, y)

# Weighted ensemble
ensemble_preds = 0.5 * lgbm_preds + 0.3 * sbert_preds + 0.2 * setfit_preds

# Or stacking
from sklearn.linear_model import LogisticRegression
meta_features = np.column_stack([lgbm_preds, sbert_preds, setfit_preds])
stacker = LogisticRegression()
stacker.fit(meta_features, y)
final_preds = stacker.predict_proba(meta_features_test)[:, 1]
```

---

## 9. DATA AUGMENTATION ANALYSIS

### 9.1 Why V10 Augmentation Failed

**Likely Issues:**
1. **Distribution shift:** Synthetic examples don't match real data distribution
2. **Label noise:** Generated examples may have incorrect labels
3. **Overfitting:** Model memorizes synthetic patterns that don't generalize
4. **Small test set:** With only 10 test samples, variance is high

### 9.2 Better Augmentation Strategies

#### Strategy 1: Back-Translation (Conservative)
```python
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, src='en', pivot='fr'):
    # Translate en -> fr
    model_name = f'Helsinki-NLP/opus-mt-{src}-{pivot}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    pivot_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    # Translate fr -> en
    model_name = f'Helsinki-NLP/opus-mt-{pivot}-{src}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    back_translated = model.generate(**tokenizer(pivot_text, return_tensors="pt", padding=True))
    return tokenizer.decode(back_translated[0], skip_special_tokens=True)

# Apply conservatively
augmented_df = train_df.copy()
augmented_df['body'] = augmented_df['body'].apply(back_translate)
combined_df = pd.concat([train_df, augmented_df])
```

**Pros:** Preserves semantics, adds natural variation
**Cons:** Slow, may lose domain-specific terminology
**Recommendation:** Use sparingly (10-20% augmentation)

#### Strategy 2: Example Swapping (Rule-Specific)
```python
def swap_examples(df):
    augmented = []

    for rule_type in df['rule'].unique():
        rule_subset = df[df['rule'] == rule_type]

        # Swap positive examples within same rule
        pos_examples_1 = rule_subset['positive_example_1'].values
        pos_examples_2 = rule_subset['positive_example_2'].values

        # Create new samples by swapping
        for i in range(len(rule_subset)):
            new_row = rule_subset.iloc[i].copy()
            new_row['positive_example_1'] = np.random.choice(pos_examples_1)
            new_row['positive_example_2'] = np.random.choice(pos_examples_2)
            augmented.append(new_row)

    return pd.concat([df, pd.DataFrame(augmented)])
```

**Pros:** Fast, maintains data distribution
**Cons:** May introduce label noise if examples are context-dependent
**Recommendation:** Use with caution, validate on CV

#### Strategy 3: Mixup on Embeddings
```python
def mixup_embeddings(X_emb, y, alpha=0.2):
    # Mixup for embeddings
    lam = np.random.beta(alpha, alpha, size=len(X_emb))
    indices = np.random.permutation(len(X_emb))

    X_mixed = lam[:, None] * X_emb + (1 - lam[:, None]) * X_emb[indices]
    y_mixed = lam * y + (1 - lam) * y[indices]

    return X_mixed, y_mixed
```

**Pros:** Smooth interpolation, proven effective
**Cons:** Only works with embeddings, not text
**Recommendation:** Use for Sentence-BERT features

#### Strategy 4: NO AUGMENTATION (RECOMMENDED)
**Reasoning:**
- Only 2,029 samples, but high-dimensional features (15k TF-IDF)
- Test set is only 10 samples - augmentation may increase variance
- Current CV-LB gap suggests overfitting, not underfitting
- **Focus on regularization, not more data**

**Alternative:** Pseudo-labeling on test set (see Section 10.2)

---

## 10. ADVANCED STRATEGIES

### 10.1 Cross-Validation Strategy Improvements

**Current:** 5-fold StratifiedKFold

**Improvements:**

#### Option 1: Subreddit-Aware CV
```python
from sklearn.model_selection import GroupKFold

# Ensure same subreddit doesn't appear in train/val
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=df['subreddit'])):
    # Train on train_idx, validate on val_idx
    pass
```

**Pros:** More robust validation (test set has known subreddits)
**Cons:** Imbalanced folds
**Recommendation:** Use as secondary validation

#### Option 2: Repeated CV with Different Seeds
```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
all_scores = []

for train_idx, val_idx in rskf.split(X, y):
    # Train and validate
    score = evaluate_model(train_idx, val_idx)
    all_scores.append(score)

print(f"Mean CV: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
```

**Pros:** More stable estimates
**Cons:** 3x slower
**Recommendation:** Use for final model selection

#### Option 3: Leave-One-Subreddit-Out (LOSO)
```python
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
scores = []

for train_idx, val_idx in logo.split(X, y, groups=df['subreddit']):
    if len(val_idx) < 10:  # Skip small subreddits
        continue
    score = evaluate_model(train_idx, val_idx)
    scores.append(score)
```

**Pros:** Tests generalization to unseen subreddits
**Cons:** Not applicable (all test subreddits in train)
**Recommendation:** Skip

### 10.2 Pseudo-Labeling on Test Set

**Strategy:**
```python
# Step 1: Train on full training set
model.fit(X_train, y_train)

# Step 2: Predict on test set
test_preds = model.predict_proba(X_test)[:, 1]

# Step 3: Select high-confidence predictions
confident_mask = (test_preds > 0.9) | (test_preds < 0.1)
pseudo_labels = (test_preds > 0.5).astype(int)

# Step 4: Retrain with pseudo-labeled test samples
X_combined = np.vstack([X_train, X_test[confident_mask]])
y_combined = np.hstack([y_train, pseudo_labels[confident_mask]])

model.fit(X_combined, y_combined)

# Step 5: Final prediction
final_preds = model.predict_proba(X_test)[:, 1]
```

**Pros:** Leverages test data distribution
**Cons:** Risk of error propagation with wrong pseudo-labels
**Recommendation:** Use conservatively (only >0.95 or <0.05 confidence)

### 10.3 Threshold Optimization

**Current:** Default 0.5 threshold

**Optimization:**
```python
from sklearn.metrics import roc_auc_score, roc_curve

# Find optimal threshold on validation set
fpr, tpr, thresholds = roc_curve(y_val, val_preds)

# Optimize for AUC (no threshold needed) or F1
# For this competition, AUC is the metric, so no threshold optimization needed

# BUT: If ensembling, optimize weights
from scipy.optimize import minimize

def objective(weights):
    ensemble = weights[0] * lgbm_preds + weights[1] * sbert_preds + weights[2] * setfit_preds
    return -roc_auc_score(y_val, ensemble)

result = minimize(objective, x0=[0.33, 0.33, 0.34], bounds=[(0, 1)] * 3,
                  constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

optimal_weights = result.x
```

### 10.4 Adversarial Validation

**Check Train-Test Distribution Shift:**
```python
from sklearn.ensemble import RandomForestClassifier

# Label train=0, test=1
train_df['is_test'] = 0
test_df['is_test'] = 1
combined = pd.concat([train_df, test_df])

# Train classifier to distinguish train/test
X_combined = create_features(combined)
y_is_test = combined['is_test']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf, X_combined, y_is_test, cv=5, scoring='roc_auc')

print(f"Adversarial Validation AUC: {np.mean(cv_scores):.4f}")
# If AUC > 0.6, there's a distribution shift
# If AUC ~ 0.5, train/test are similar (good!)
```

**Current Analysis:**
- Test bodies are 20% shorter (minor shift)
- All test subreddits in train (no shift)
- Both rules present (no shift)
- **Expected AV AUC: ~0.52-0.55 (minimal shift)**

---

## 11. CONCRETE NEXT STEPS

### Phase 1: Quick Wins (1-2 hours, Expected +0.05-0.08 AUC)

1. **Implement Semantic Similarity**
   ```bash
   pip install sentence-transformers
   ```
   - Use `all-MiniLM-L6-v2` model
   - Calculate body-example similarities
   - Add to feature set

2. **Add Subreddit Risk Encoding**
   - Target encoding with smoothing
   - Rule-specific risk scores
   - Subreddit-rule interaction

3. **Implement Rule-Specific Keyword Features**
   - Legal keywords: lawyer, sue, police, legal (5-30x lift)
   - Ad keywords: buy, sell, free, discount (1.5-2.5x lift)
   - Keyword density features

4. **Improve Few-Shot Features**
   - Max/Min similarity (currently using average)
   - Similarity variance
   - Pos-neg difference features

5. **Add Linguistic Features**
   - Avg sentence length
   - Stopword ratio
   - Unique word ratio

6. **Strengthen Regularization**
   - Increase `min_child_samples` from 20 to 30
   - Increase `reg_alpha` and `reg_lambda` to 0.3
   - Reduce `num_leaves` from 63 to 31
   - Lower `learning_rate` to 0.02

### Phase 2: Medium-Term (2-4 hours, Expected +0.02-0.04 AUC)

7. **Add Spam Signal Features**
   - Email detection (10.7x lift)
   - Price detection (6x lift)
   - Composite spam score

8. **Character N-gram Similarity**
   - 3-gram overlap (36.9% higher for violations)
   - Max character similarity

9. **Length Ratio Features**
   - Body/example ratios
   - Example length differences

10. **Modal Verb & Question Features**
    - Modal verb counts (legal advice specific)
    - Question detection (inverted signal!)
    - Rule-specific interactions

11. **Implement Ensemble**
    - LightGBM on engineered features
    - XGBoost on same features
    - Sentence-BERT semantic model
    - Weighted ensemble (optimize weights)

### Phase 3: Advanced (4-6 hours, Expected +0.01-0.02 AUC)

12. **Feature Selection**
    - Feature importance from RF
    - Keep top 1000 features
    - Remove highly correlated features

13. **Pseudo-Labeling**
    - High-confidence test predictions (>0.95 or <0.05)
    - Retrain with pseudo-labeled test data

14. **Advanced Ensemble**
    - Stacking with meta-learner
    - Multiple CV folds for diversity

### Debugging if CV-LB Gap Persists

15. **Adversarial Validation**
    - Verify train-test similarity
    - Focus features on stable patterns

16. **Repeated CV**
    - 3x repeated 5-fold CV
    - Check consistency across folds

17. **Subreddit-Aware CV**
    - Validate generalization to unseen subreddit patterns

---

## 12. EXPECTED PERFORMANCE TRAJECTORY

**Current:** CV 0.7086, LB 0.670

**After Phase 1:** CV 0.75-0.78, LB 0.72-0.75 (+0.05-0.08)
**After Phase 2:** CV 0.77-0.80, LB 0.75-0.78 (+0.03-0.03)
**After Phase 3:** CV 0.78-0.82, LB 0.77-0.81 (+0.02-0.03)

**Realistic Target:** CV 0.80-0.82, LB 0.77-0.81

**Gap to 0.93:** 0.12-0.16 (still significant)

### Why 0.93 May Be Difficult

1. **Small dataset:** 2,029 samples is limiting
2. **Inherent ambiguity:** Some cases are genuinely borderline
3. **Label noise:** Human annotations may have inconsistencies
4. **Test set variance:** Only 10 samples - high variance

### Alternative Targets

- **Conservative:** 0.77-0.80 (achievable with Phase 1-2)
- **Optimistic:** 0.82-0.85 (requires Phase 3 + novel ideas)
- **Moonshot:** 0.90+ (may require external data or ensemble of 20+ models)

---

## 13. RISK MITIGATION

### Risk 1: Overfitting (Current Gap: 0.0386)

**Mitigation:**
- Stronger regularization (done)
- Feature selection (reduce noise)
- Ensemble diversity (uncorrelated models)
- Cross-validation with different strategies

### Risk 2: Computational Limits (9hr Code Competition)

**Mitigation:**
- Pre-compute embeddings (save to disk)
- Optimize hyperparameters offline
- Use fast models (LightGBM > XGBoost > NN)
- Limit max_features in TF-IDF to 10k

### Risk 3: Test Set Distribution Shift

**Mitigation:**
- Adversarial validation
- Robust features (semantic > lexical)
- Subreddit-aware features (all test subreddits in train)

### Risk 4: Feature Engineering Bugs

**Mitigation:**
- Validate each feature individually
- Check correlation with target
- Ensure no data leakage
- Unit tests for feature functions

---

## 14. VISUALIZATION RECOMMENDATIONS

### 14.1 Feature Importance Plot
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importance from LightGBM
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': lgbm_model.feature_importances_
}).sort_values('importance', ascending=False).head(30)

plt.figure(figsize=(10, 12))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Top 30 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

### 14.2 Similarity Distribution Plot
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Jaccard similarities
axes[0, 0].hist(train_df[train_df['rule_violation']==1]['avg_pos_jaccard'], bins=50, alpha=0.5, label='Violation')
axes[0, 0].hist(train_df[train_df['rule_violation']==0]['avg_pos_jaccard'], bins=50, alpha=0.5, label='Non-violation')
axes[0, 0].set_title('Body-Positive Example Similarity')
axes[0, 0].legend()

axes[0, 1].hist(train_df[train_df['rule_violation']==1]['avg_neg_jaccard'], bins=50, alpha=0.5, label='Violation')
axes[0, 1].hist(train_df[train_df['rule_violation']==0]['avg_neg_jaccard'], bins=50, alpha=0.5, label='Non-violation')
axes[0, 1].set_title('Body-Negative Example Similarity')
axes[0, 1].legend()

# Semantic similarities (after implementation)
axes[1, 0].hist(train_df[train_df['rule_violation']==1]['avg_pos_semantic'], bins=50, alpha=0.5, label='Violation')
axes[1, 0].hist(train_df[train_df['rule_violation']==0]['avg_pos_semantic'], bins=50, alpha=0.5, label='Non-violation')
axes[1, 0].set_title('Body-Positive Example Semantic Similarity')
axes[1, 0].legend()

# Difference
axes[1, 1].hist(train_df[train_df['rule_violation']==1]['pos_neg_jaccard_diff'], bins=50, alpha=0.5, label='Violation')
axes[1, 1].hist(train_df[train_df['rule_violation']==0]['pos_neg_jaccard_diff'], bins=50, alpha=0.5, label='Non-violation')
axes[1, 1].set_title('Pos-Neg Similarity Difference')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('similarity_distributions.png')
```

### 14.3 Subreddit Risk Heatmap
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate risk by subreddit and rule
risk_matrix = train_df.groupby(['subreddit', 'rule'])['rule_violation'].mean().unstack()

plt.figure(figsize=(12, 20))
sns.heatmap(risk_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0.5)
plt.title('Violation Rate by Subreddit and Rule')
plt.tight_layout()
plt.savefig('subreddit_risk_heatmap.png')
```

### 14.4 Error Analysis
```python
# Analyze misclassified samples
val_preds = model.predict_proba(X_val)[:, 1]
val_pred_labels = (val_preds > 0.5).astype(int)

false_positives = val_df[(val_df['rule_violation'] == 0) & (val_pred_labels == 1)]
false_negatives = val_df[(val_df['rule_violation'] == 1) & (val_pred_labels == 0)]

print("Top 10 False Positives:")
print(false_positives[['body', 'rule', 'subreddit']].head(10))

print("\nTop 10 False Negatives:")
print(false_negatives[['body', 'rule', 'subreddit']].head(10))
```

---

## 15. CONCLUSION

### Key Takeaways

1. **Few-shot signal is the strongest discriminator** (59% difference in similarity)
2. **Subreddit context is highly predictive** (2.9% to 90.5% violation rate)
3. **Rule-specific patterns require separate feature engineering**
4. **Current model is overfitting** (CV-LB gap of 0.0386)
5. **Semantic embeddings are underutilized** (only using TF-IDF)

### Recommended Approach

**Prioritize:**
1. Semantic similarity (Sentence-BERT)
2. Subreddit risk encoding
3. Rule-specific keywords
4. Few-shot max similarity
5. Stronger regularization

**Avoid:**
- Data augmentation (may increase overfitting)
- Too many engineered features (risk of noise)
- Complex ensembles (diminishing returns)

### Realistic Target

**Conservative:** 0.77-0.80 LB (achievable with Phase 1-2)
**Optimistic:** 0.82-0.85 LB (requires Phase 3)
**Stretch:** 0.90+ LB (may require novel approaches)

### Final Note

The 0.263 gap to 0.93 is substantial. While the strategies outlined here should yield significant improvements (+0.10-0.15), reaching 0.93 may require:
- Larger model ensembles (10-20 models)
- External data (more training samples)
- Novel architectures (custom few-shot learning)
- Manual feature engineering for specific error cases

Focus on achieving 0.80+ first, then iterate if time permits.

---

**End of Report**
