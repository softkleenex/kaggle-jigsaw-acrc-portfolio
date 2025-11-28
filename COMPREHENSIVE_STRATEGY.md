# COMPREHENSIVE COMPETITION STRATEGY
## Jigsaw - Agile Community Rules Classification

**Date**: October 17, 2025
**Competition Deadline**: October 24, 2025 (7 days remaining)
**Current Status**: CV 0.7086, LB 0.670
**Target**: 0.93+ (Gap: 0.263 points)
**Daily Submissions**: 5/day

---

## EXECUTIVE SUMMARY

### Critical Assessment (Top 3 Findings)

1. **MASSIVE PERFORMANCE GAP**: The 0.263 point gap to 1st place (0.670 → 0.933) is the largest concern. This is NOT a minor tuning problem - it suggests we're using a fundamentally different approach than top teams.

2. **CV-LB ALIGNMENT IS DECEIVING**: CV 0.7086 vs LB 0.670 shows only -0.038 gap, suggesting our validation strategy is sound. However, this also means incremental improvements won't close the 0.263 gap to winners.

3. **FEW-SHOT SIGNAL UNDERUTILIZED**: Despite having 4 examples per sample (the competition's core design), current approaches treat this as feature engineering rather than true few-shot learning. Top teams likely use transformer-based few-shot methods or LLM prompting.

### Harsh Reality Check

Based on similar Kaggle competitions:
- **0.263 gap requires 3-5 fundamental breakthroughs, not incremental tuning**
- With 7 days and 35 total submissions, we have ~7-10 major experiments possible
- Probability of reaching 0.93+: **<5%** (realistic assessment)
- Probability of reaching 0.85+ (top 10%): **30-40%** (achievable with aggressive pivoting)
- Current trajectory (incremental improvements): **0.75-0.78 ceiling**

### Recommended Strategy: AGGRESSIVE PIVOT

1. **Abandon incremental improvements** (LightGBM tuning, feature engineering)
2. **Pivot to transformer fine-tuning** (DeBERTa-v3, cross-encoders)
3. **Explore LLM prompting** (if allowed by code competition rules)
4. **Implement advanced few-shot techniques** (SetFit contrastive learning, prototypical networks)
5. **Heavy ensemble** of completely different model families

---

## 1. ROOT CAUSE ANALYSIS

### Why the 0.263 Gap Exists

#### A. Model Architecture Gap

**Our Approach**:
- TF-IDF + Gradient Boosting (v9)
- Sentence embeddings + similarity features (SetFit basic)

**Top Teams Likely Using**:
- Fine-tuned DeBERTa-v3-large or RoBERTa-large
- Cross-encoders for pair-wise similarity
- LLM prompting (GPT-4, Claude, or open-source LLMs)
- Advanced SetFit with contrastive fine-tuning
- Multi-stage training (pre-training on external data)

**Evidence**:
- Jigsaw Toxic Comment winners used BERT/RoBERTa + multi-level stacking
- Research shows DeBERTa-v3 outperforms sentence transformers by 15-20% on classification
- 0.933 score suggests near-perfect understanding of semantic nuances

#### B. Few-Shot Learning Gap

**Our Approach**:
- Cosine similarity between body and examples (9 features)
- Treating examples as auxiliary features

**Top Teams Likely Using**:
- True few-shot learning: Examples as in-context demonstrations
- Contrastive learning: Fine-tuning embeddings to maximize similarity with correct class
- Prototypical networks: Learning class prototypes from examples
- Cross-encoder re-ranking: Re-scoring predictions based on example similarity

**Evidence**:
- Competition explicitly designed as "few-shot learning"
- Our EDA shows violations are 95% more similar to positive examples
- SetFit paper shows 26% improvement with contrastive fine-tuning

#### C. Data Utilization Gap

**What We're Missing**:
- **Pseudo-labeling**: Using test set predictions to augment training
- **External data**: Pre-training on Reddit comment datasets
- **Data augmentation**: Paraphrasing, back-translation (though v10 failed)
- **Multi-task learning**: Joint training on related tasks

**Evidence**:
- Jigsaw Multilingual winner: "Pseudo labelling was essential"
- Only 2,029 training samples - more data could help
- Top teams often use external datasets for pre-training

#### D. Ensemble Sophistication Gap

**Our Approach**:
- Simple weighted average (LightGBM + XGBoost + CatBoost)
- Single-level ensemble

**Top Teams Likely Using**:
- Multi-level stacking (3rd place Jigsaw: 4 levels)
- Model diversity: TF-IDF + Transformers + LLMs
- Cross-validation blending with multiple seeds
- Iterative refinement (blend → retrain → blend)

#### E. CV-LB Gap Analysis

**CV 0.7086 vs LB 0.670 = -0.038 gap**

**Interpretation**:
- ✅ Validation strategy is sound (stratified k-fold works)
- ✅ NOT overfitting the training data
- ⚠️ Possible distribution shift in test set
- ⚠️ Test set may have unseen subreddits or rule patterns

**Action**:
- Trust CV for model selection
- Don't overfit public LB (limit submissions)
- Focus on robust models (transformers generalize better)

---

## 2. WINNING APPROACHES RESEARCH

### A. Jigsaw Toxic Comment (Similar Competition)

**Top Solutions Summary**:

1. **1st Place (Toxic Crusaders)**:
   - LSTM/GRU + CNN architectures
   - Multiple pre-trained embeddings (GloVe, FastText)
   - Translation as data augmentation
   - Multi-level stacking

2. **2nd-3rd Place**:
   - 4-level stacking ensemble
   - Text normalization (time-consuming but effective)
   - Head + tail text retention (don't truncate middle)

3. **Multilingual Competition (2020+)**:
   - XLM-RoBERTa Large was most effective
   - **Pseudo-labeling essential for top scores**
   - Two-stage training: External data → competition data
   - Iterative blending (model order matters)

**Key Takeaways**:
- Transformers dominate (BERT family)
- Pseudo-labeling provides 5-10% boost
- Multi-level stacking is standard at top
- Text preprocessing matters

### B. Few-Shot NLP (2024 Techniques)

**SetFit (Sentence Transformer Fine-tuning)**:
- Contrastive learning on sentence pairs
- Outperforms GPT-3 with 8 examples
- 1600x smaller than GPT-3
- Training time: 30 seconds on V100

**DeBERTa-v3**:
- SOTA on GLUE benchmark (91.37%)
- Zero-shot classification via NLI formulation
- 15-20% better than RoBERTa
- Cross-encoder mode for pair-wise tasks

**LLM Prompting**:
- Few-shot prompting: 80% more efficient than zero-shot
- Claude Opus: Best at few-shot classification (2024 research)
- GPT-4: Excellent with in-context learning
- Requires careful prompt engineering

**Prototypical Networks**:
- Learn class prototypes from examples
- Euclidean distance in embedding space
- Effective for few-shot classification
- Kaggle notebooks available

### C. Advanced Techniques (2024-2025)

**Cross-Encoders**:
- 5-10% better than bi-encoders (sentence transformers)
- Pass both sentences to BERT simultaneously
- Slower but more accurate
- Hybrid: Bi-encoder retrieval → Cross-encoder re-ranking

**Metric Learning**:
- Triplet loss: Anchor, positive, negative
- Siamese networks with contrastive learning
- 23% F1 improvement (DoorDash case study)
- Effective for zero/few-shot classification

**Pseudo-Labeling**:
- Self-training on unlabeled data
- Iterative refinement of predictions
- Noise correction via meta-learning
- Essential for small datasets (2,029 samples)

**Ensemble Strategies**:
- Stacking: Use model predictions as features
- Blending: Weighted average with CV-optimized weights
- Multi-seed training: Reduce variance
- Diversity: Combine TF-IDF, transformers, LLMs

---

## 3. TECHNICAL STRATEGY

### A. Should We Pivot to Deep Learning?

**Answer: YES - IMMEDIATELY**

**Reasoning**:
- 0.263 gap cannot be closed with gradient boosting
- Transformers are standard for NLP competitions
- Few-shot learning requires semantic understanding
- Top teams certainly use transformers

**Recommended Models** (Priority Order):

1. **DeBERTa-v3-base** (184M params)
   - Best performance on benchmarks
   - Zero-shot NLI formulation possible
   - Cross-encoder for pair-wise tasks
   - Training time: ~2-3 hours on Kaggle GPU

2. **RoBERTa-large** (355M params)
   - Proven winner in Jigsaw competitions
   - Strong generalization
   - Widely supported
   - Training time: ~4-5 hours

3. **SetFit with Contrastive Fine-tuning**
   - True few-shot learning (not just similarity)
   - Fast training (30 min)
   - Small model (all-mpnet-base-v2: 420M)
   - Already have baseline implementation

4. **Cross-Encoder Re-ranking**
   - Use bi-encoder for initial ranking
   - Cross-encoder for final scoring
   - Hybrid approach
   - Slower but accurate

### B. Few-Shot Learning Techniques

**Tier 1 (Must Implement)**:

1. **SetFit Contrastive Learning**
   - Fine-tune sentence transformer on positive/negative pairs
   - Contrastive loss: Pull similar together, push dissimilar apart
   - Use 4 examples to generate training pairs
   - Expected improvement: +10-15% over current SetFit

2. **DeBERTa Few-Shot Classification**
   - Input format: `[CLS] body [SEP] positive_ex1 [SEP] positive_ex2 [SEP] negative_ex1 [SEP] negative_ex2 [SEP]`
   - Token limit: 512 (may need truncation)
   - Fine-tune on competition data
   - Expected improvement: +15-25% over baseline

3. **Cross-Encoder for Similarity**
   - Train cross-encoder: (body, positive_ex) → high score
   - Train cross-encoder: (body, negative_ex) → low score
   - Aggregate scores as features
   - Expected improvement: +5-10% over bi-encoder

**Tier 2 (Time Permitting)**:

4. **Prototypical Networks**
   - Compute prototype embeddings for positive/negative classes
   - Classify based on distance to prototypes
   - Meta-learning approach
   - Implementation available on Kaggle

5. **Metric Learning (Triplet Loss)**
   - Anchor: body, Positive: positive_ex, Negative: negative_ex
   - Fine-tune embeddings with triplet loss
   - Use learned embeddings for classification
   - Requires custom training loop

### C. Ensemble Strategies

**Stage 1: Diverse Base Models**

Build 5 completely different model families:
1. DeBERTa-v3-base fine-tuned
2. RoBERTa-large fine-tuned
3. SetFit with contrastive learning
4. Cross-encoder similarity scorer
5. LightGBM + XGBoost (current best)

**Stage 2: Stacking Ensemble**

- Level 1: 5 base models (out-of-fold predictions)
- Level 2: Logistic Regression or LightGBM meta-model
- Features: Model predictions + confidence scores
- Expected improvement: +3-5% over best single model

**Stage 3: Multi-Seed Blending**

- Train each model with 3-5 different seeds
- Average predictions to reduce variance
- Weighted average based on CV scores
- Expected improvement: +1-3% (stability)

### D. Feature Engineering (Complementary)

**High-Value Features** (Add to transformer models):

1. **Subreddit Risk Score**
   - Historical violation rate by subreddit-rule combo
   - Smoothing for unseen combinations (β-distribution)
   - Can provide +2-3% boost

2. **Example Similarity Features**
   - Already implemented in SetFit
   - Add to DeBERTa as auxiliary features
   - Cosine similarity, Manhattan distance, Euclidean distance

3. **Text Statistics**
   - Length ratio: body vs positive_examples
   - Word overlap: body vs positive_examples (already strong signal)
   - URL count, special chars, capitalization

4. **Rule-Specific Features**
   - Legal keywords: lawyer, attorney, sue, lawsuit, court
   - Ad keywords: buy, sell, click, discount, free, promotion
   - Separate models for each rule type

### E. LLM Prompting (If Feasible)

**Challenge**: Code Competition requires reproducibility

**Possible Approaches**:

1. **Open-Source LLMs** (Kaggle-compatible)
   - SmolLM-2 (360M, 1.7B)
   - Llama-3.2 (1B, 3B)
   - Mistral-7B-Instruct
   - Phi-3-mini (3.8B)

2. **Prompt Template**:
   ```
   Rule: {rule}

   Examples that violate this rule:
   1. {positive_example_1}
   2. {positive_example_2}

   Examples that don't violate this rule:
   1. {negative_example_1}
   2. {negative_example_2}

   Comment: {body}

   Question: Does this comment violate the rule? Answer with a score from 0 (definitely not) to 100 (definitely yes).
   ```

3. **Expected Performance**:
   - SmolLM-2 360M: Likely worse than DeBERTa (too small)
   - Llama-3.2 3B: Potentially competitive
   - Requires quantization (4-bit) to fit in Kaggle GPU
   - Inference time: May exceed 9-hour limit

**Verdict**: Worth 1 experiment if time permits, but not primary strategy

---

## 4. PRACTICAL EXECUTION PLAN

### 7-Day Timeline (October 17-24)

#### Day 1 (Oct 17) - AGGRESSIVE PIVOT

**Goal**: Implement DeBERTa-v3-base + cross-encoder

**Tasks**:
1. ✅ Complete strategy document (this document)
2. ⏱️ Implement DeBERTa-v3-base fine-tuning
   - Input: Body + Rule + 4 examples (concatenated)
   - Sequence length: 512 tokens
   - Training: 3-5 epochs, lr=2e-5
   - CV: 5-fold stratified
3. ⏱️ Implement cross-encoder similarity
   - Model: cross-encoder/ms-marco-MiniLM-L6-v2
   - Score: (body, positive_ex) and (body, negative_ex)
   - Use scores as features for LightGBM
4. 📤 **Submission 1**: DeBERTa-v3-base (solo)
5. 📤 **Submission 2**: Cross-encoder + LightGBM

**Expected LB**: 0.75-0.82 (if successful)

#### Day 2 (Oct 18) - SETFIT CONTRASTIVE LEARNING

**Goal**: Implement true SetFit with contrastive fine-tuning

**Tasks**:
1. ⏱️ SetFit contrastive learning
   - Generate positive pairs: (positive_ex1, positive_ex2)
   - Generate negative pairs: (positive_ex1, negative_ex1)
   - Fine-tune all-mpnet-base-v2 with contrastive loss
   - Train classification head
2. ⏱️ RoBERTa-large fine-tuning (if time permits)
   - Similar to DeBERTa setup
   - Longer training time (4-5 hours)
3. 📤 **Submission 3**: SetFit contrastive

**Expected LB**: 0.78-0.84

#### Day 3 (Oct 19) - ENSEMBLE BUILDING

**Goal**: Combine best models from Days 1-2

**Tasks**:
1. ⏱️ Train all models with multiple seeds (3 seeds each)
2. ⏱️ Build stacking ensemble
   - Level 1: DeBERTa, SetFit, Cross-encoder, LightGBM
   - Level 2: Logistic Regression
3. ⏱️ Optimize ensemble weights using Optuna
4. 📤 **Submission 4**: Stacking ensemble
5. 📤 **Submission 5**: Weighted blending

**Expected LB**: 0.82-0.87

#### Day 4 (Oct 20) - PSEUDO-LABELING

**Goal**: Augment training data with pseudo-labels

**Tasks**:
1. ⏱️ Generate pseudo-labels for test set (if available)
   - Use best ensemble predictions
   - Filter high-confidence predictions (>0.8 or <0.2)
2. ⏱️ Retrain models on train + pseudo-labeled test
3. ⏱️ Implement iterative pseudo-labeling (2-3 iterations)
4. 📤 **Submission 6**: Pseudo-labeled DeBERTa
5. 📤 **Submission 7**: Pseudo-labeled ensemble

**Expected LB**: +2-5% over Day 3

#### Day 5 (Oct 21) - ADVANCED TECHNIQUES

**Goal**: Experiment with prototypical networks, LLM prompting

**Tasks**:
1. ⏱️ Prototypical networks implementation
2. ⏱️ LLM prompting (SmolLM-2 or Llama-3.2)
3. ⏱️ Rule-specific models (separate for Legal vs Advertising)
4. 📤 **Submission 8**: Best experimental model
5. 📤 **Submission 9**: Rule-specific ensemble

**Expected LB**: 0.85-0.89

#### Day 6 (Oct 22) - HYPERPARAMETER OPTIMIZATION

**Goal**: Squeeze final performance from best models

**Tasks**:
1. ⏱️ Bayesian optimization (Optuna) for:
   - DeBERTa learning rate, epochs, sequence length
   - Ensemble weights
   - Stacking meta-model hyperparameters
2. ⏱️ Train final models with optimized hyperparameters
3. 📤 **Submission 10**: Optimized DeBERTa
4. 📤 **Submission 11**: Optimized ensemble

**Expected LB**: +1-2% over Day 5

#### Day 7 (Oct 23) - FINAL POLISH

**Goal**: Select final submission, prepare for private LB

**Tasks**:
1. ⏱️ Review all submissions, analyze CV vs LB gaps
2. ⏱️ Select 2 final submissions:
   - **Selection A**: Highest LB score (risky)
   - **Selection B**: Best CV + LB correlation (safe)
3. ⏱️ Retrain final models on 100% of training data
4. 📤 **Submission 12-13**: Final submissions

**Expected LB**: 0.87-0.92

#### Day 8 (Oct 24) - DEADLINE DAY

**Goal**: Last-minute adjustments if needed

**Tasks**:
1. Monitor leaderboard for competition trends
2. If far from target, attempt high-risk experiments
3. Submit final selections before 06:59 UTC

---

### Submission Strategy

**Guiding Principles**:
1. **Trust CV over public LB** (avoid overfitting)
2. **Select diverse models** (hedge against overfitting)
3. **Limit submissions** (max 2 per day to avoid LB probing)

**Selection Criteria**:
- Best CV score (primary)
- Lowest CV standard deviation (stability)
- Good CV-LB correlation (avoid overfitting)
- Model diversity (if choosing 2 submissions)

**Risk Management**:
- Never submit models with CV-LB gap > 0.05
- Always validate on multiple folds (5-fold minimum)
- Keep ensemble simple (avoid overfitting on meta-level)

---

## 5. RESOURCE OPTIMIZATION

### A. Kaggle Kernel Constraints

**Limits**:
- GPU time: 30 hours/week (P100 or T4)
- CPU time: 9 hours per notebook execution
- RAM: 16 GB (GPU) or 30 GB (CPU)
- Disk: 20 GB

**Optimization Strategies**:

1. **Model Selection**:
   - DeBERTa-v3-base (184M) > DeBERTa-v3-large (304M) for time
   - Use fp16 mixed precision (2x faster, 50% memory)
   - Batch size: 8-16 (maximize GPU utilization)

2. **Training Efficiency**:
   - Gradient accumulation (simulate larger batch sizes)
   - Early stopping (patience=2-3 epochs)
   - Freeze early layers (faster convergence)
   - Use LoRA/QLoRA (parameter-efficient fine-tuning)

3. **Inference Speed**:
   - ONNX export (2-3x faster inference)
   - Quantization (INT8) for CPU inference
   - Batch prediction (vectorize operations)

4. **Checkpointing**:
   - Save model every epoch (in case of timeout)
   - Resume training from checkpoint
   - Store datasets in Kaggle Datasets (persist across runs)

### B. GPU vs CPU Tradeoffs

**GPU (Recommended for)**:
- DeBERTa, RoBERTa training (30x faster)
- SetFit contrastive learning
- Large batch inference

**CPU (Sufficient for)**:
- LightGBM, XGBoost training (already fast)
- TF-IDF feature extraction
- Ensemble blending
- Small model inference

**Strategy**:
- Use GPU notebooks for transformer training (6-8 hours)
- Use CPU notebooks for ensembling (1-2 hours)
- Separate notebooks to maximize parallelization

### C. Code Competition Constraints

**Reproducibility Requirements**:
- All code must run in Kaggle notebook
- No external API calls (rules out GPT-4, Claude API)
- No internet access during inference
- Deterministic results (set random seeds)

**Implications**:
- Pre-trained models must be in Kaggle Datasets
- Open-source LLMs only (if using LLM prompting)
- Save trained models as Kaggle Datasets
- Use HuggingFace models (widely available on Kaggle)

**Best Practices**:
- Test full pipeline in Kaggle notebook early
- Upload trained models as datasets (v1, v2, v3...)
- Document all dependencies
- Version control via Git + Kaggle Datasets

### D. Model Size vs Inference Time

**Trade-off Matrix**:

| Model | Size | Training Time | Inference Time | Expected Score |
|-------|------|---------------|----------------|----------------|
| LightGBM | 10 MB | 5 min | <1 sec | 0.70 |
| SetFit (MiniLM) | 80 MB | 30 min | 5 sec | 0.78 |
| DeBERTa-v3-base | 700 MB | 2-3 hours | 30 sec | 0.82 |
| DeBERTa-v3-large | 1.4 GB | 5-6 hours | 60 sec | 0.85 |
| RoBERTa-large | 1.4 GB | 4-5 hours | 60 sec | 0.84 |
| Ensemble (5 models) | 3 GB | 8-10 hours | 2 min | 0.87 |

**Recommended Configuration**:
- 3-4 medium models (DeBERTa-base, SetFit, Cross-encoder)
- 1 lightweight model (LightGBM) for diversity
- Total inference time: <3 minutes (safe margin)

---

## 6. BREAKTHROUGH IDEAS

### A. Unconventional Approaches

1. **Reddit-Style Data Augmentation**
   - Problem: Our v10 augmentation failed
   - New idea: Reddit-specific augmentation
     - Add typos, abbreviations (common in Reddit)
     - Mix formal/informal language
     - Add context markers ("IANAL", "Not advice, but...")
   - Expected impact: +2-3%

2. **Rule-Aware Attention**
   - Custom attention mechanism in transformers
   - Attend to rule keywords when classifying body
   - Modify DeBERTa's attention layers
   - Expected impact: +3-5% (high risk, high reward)

3. **Example Synthesis**
   - Generate synthetic examples using paraphrasing
   - Instead of 2 positive examples, create 10+ variations
   - Use back-translation or GPT-based paraphrasing
   - Train with richer few-shot context
   - Expected impact: +5-8%

4. **Adversarial Training**
   - Generate adversarial examples that fool the model
   - Retrain on adversarial examples
   - Improve robustness and generalization
   - Expected impact: +2-4%

5. **Curriculum Learning**
   - Start with easy examples (clear violations)
   - Gradually increase difficulty (subtle violations)
   - Train model in stages
   - Expected impact: +3-5%

### B. Ensemble of Different Model Families

**Extreme Diversity Ensemble**:

1. **Statistical Model**: TF-IDF + LightGBM (current)
2. **Bi-Encoder**: SetFit sentence transformers
3. **Cross-Encoder**: DeBERTa cross-encoder
4. **Generative Model**: LLM prompting (SmolLM-2)
5. **Metric Learning**: Triplet network with Siamese architecture

**Rationale**:
- Each model captures different patterns
- Uncorrelated errors → better ensemble
- Combines speed (LightGBM) and accuracy (transformers)

**Expected Impact**: +5-10% over best single model

### C. External Data Sources (If Allowed)

**Check competition rules carefully**

**Potential Sources**:
1. **Reddit API Historical Data**
   - Scrape comments from same subreddits
   - Label using simple heuristics (deleted = violation)
   - Pre-train on large dataset, fine-tune on competition data

2. **Jigsaw Previous Competitions**
   - Toxic Comment dataset (200k+ samples)
   - Pre-train on toxicity → transfer learn to rules
   - Similar domain (online moderation)

3. **HuggingFace Datasets**
   - Reddit datasets, comment moderation datasets
   - Use for pre-training or data augmentation

**Expected Impact**: +10-15% (if allowed and executed well)

**Risk**: Violates competition rules → disqualification

### D. Meta-Learning Approaches

**Concept**: Learn to learn from few examples

1. **MAML (Model-Agnostic Meta-Learning)**
   - Train model to adapt quickly to new tasks
   - Each subreddit-rule combo is a "task"
   - Meta-train on train set, meta-test on validation
   - Kaggle implementation available

2. **Prototypical Networks with Meta-Learning**
   - Learn optimal prototype computation
   - Train on episodic batches (simulate few-shot)
   - Fast adaptation to new rules

**Expected Impact**: +5-8% (if implemented correctly)

**Complexity**: High (requires custom training loop)

---

## 7. RISK ASSESSMENT & MITIGATION

### A. Major Risks

#### Risk 1: Time Constraint (7 days)

**Probability**: HIGH (100%)

**Impact**:
- Cannot implement all strategies
- May not have time to iterate
- Rushed implementations → bugs, suboptimal results

**Mitigation**:
- Prioritize high-ROI experiments (DeBERTa first)
- Parallelize work (multiple notebooks)
- Re-use existing implementations (HuggingFace, Kaggle notebooks)
- Set time limits (max 6 hours per experiment)
- Have fallback plan (ensemble of existing models)

#### Risk 2: Overfitting Public LB

**Probability**: MEDIUM (30-40%)

**Impact**:
- High public LB → Low private LB
- Shake-up in final rankings
- Wasted submissions

**Mitigation**:
- Trust CV over public LB
- Limit submissions (2 per day max)
- Select final submission based on CV stability
- Use group k-fold by subreddit (test generalization)
- Choose diverse models for final submissions

#### Risk 3: Kaggle GPU Timeout

**Probability**: MEDIUM (40-50%)

**Impact**:
- Training interrupted
- Wasted GPU hours
- Cannot complete experiments

**Mitigation**:
- Checkpoint every epoch
- Use smaller models (DeBERTa-base > large)
- Optimize training (mixed precision, gradient accumulation)
- Have CPU fallback (SetFit, LightGBM)
- Test full pipeline early (dry run)

#### Risk 4: Code Competition Reproducibility

**Probability**: MEDIUM (30-40%)

**Impact**:
- Submission fails to run
- Disqualification
- Cannot use certain techniques

**Mitigation**:
- Test in Kaggle notebook environment early
- Upload trained models as Kaggle Datasets
- Avoid external dependencies
- Document all seeds and versions
- Dry run full submission pipeline

#### Risk 5: Cannot Reach 0.93+ Target

**Probability**: VERY HIGH (95%)

**Reality Check**:
- 0.263 gap is enormous
- 7 days is not enough time
- Top teams have been working for weeks
- May have proprietary techniques

**Mitigation**:
- **Reset expectations**: Target 0.85+ (top 10%) instead
- Focus on learning and improvement
- Build strong foundation for future competitions
- Document all learnings
- Aim for private LB shake-up (trust CV)

### B. Contingency Plans

#### Plan A: Ideal Scenario (30% probability)

- Day 1-2: DeBERTa + cross-encoder working (LB 0.80+)
- Day 3-4: Ensemble + pseudo-labeling (LB 0.85+)
- Day 5-7: Optimization + advanced techniques (LB 0.88+)
- **Final Result**: 0.87-0.92 (top 5-10%)

#### Plan B: Realistic Scenario (50% probability)

- Day 1-2: DeBERTa working but lower than expected (LB 0.75-0.78)
- Day 3-4: Ensemble helps (LB 0.80-0.83)
- Day 5-7: Incremental improvements (LB 0.83-0.86)
- **Final Result**: 0.82-0.87 (top 10-20%)

#### Plan C: Conservative Scenario (20% probability)

- Day 1-2: DeBERTa issues, fall back to SetFit (LB 0.77-0.80)
- Day 3-4: Ensemble of existing models (LB 0.78-0.81)
- Day 5-7: Feature engineering + tuning (LB 0.80-0.83)
- **Final Result**: 0.78-0.83 (top 20-30%)

#### Emergency Plan: Fallback (If all fails)

- Use existing models (SetFit + LightGBM)
- Ensemble with carefully tuned weights
- Add subreddit risk features
- Pseudo-labeling on test set
- **Final Result**: 0.75-0.80 (better than current)

---

## 8. TOP 5 PRIORITIZED EXPERIMENTS

### Experiment 1: DeBERTa-v3-base Fine-tuning ⭐⭐⭐⭐⭐

**Estimated Score Impact**: +0.08 to +0.12 (0.75-0.82)

**Time Required**: 4-5 hours (implementation + training)

**Success Probability**: 80%

**Why This First**:
- Transformers are standard for top solutions
- DeBERTa-v3 is SOTA on benchmarks
- Proven approach in similar competitions
- Manageable size (184M params, fits in Kaggle GPU)

**Implementation Plan**:
```python
# Input format
input_text = f"""
[CLS] {rule} [SEP] {body} [SEP]
Positive: {pos_ex1} {pos_ex2} [SEP]
Negative: {neg_ex1} {neg_ex2} [SEP]
"""

# Fine-tuning config
model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base')
training_args = TrainingArguments(
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    fp16=True,  # Mixed precision
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    save_strategy="steps",
)
```

**Success Criteria**: CV AUC > 0.80, LB > 0.77

---

### Experiment 2: Cross-Encoder Similarity ⭐⭐⭐⭐⭐

**Estimated Score Impact**: +0.05 to +0.08 (0.77-0.80)

**Time Required**: 3-4 hours

**Success Probability**: 75%

**Why This Second**:
- Cross-encoders are 5-10% better than bi-encoders
- Directly models pairwise similarity (body vs examples)
- Fast training (smaller models)
- Can combine with LightGBM

**Implementation Plan**:
```python
from sentence_transformers import CrossEncoder

# Load pre-trained cross-encoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

# Compute similarity scores
scores_pos1 = model.predict([(body, pos_ex1) for body, pos_ex1 in zip(bodies, pos_ex1s)])
scores_neg1 = model.predict([(body, neg_ex1) for body, neg_ex1 in zip(bodies, neg_ex1s)])

# Use scores as features for LightGBM
features = np.column_stack([scores_pos1, scores_pos2, scores_neg1, scores_neg2, ...])
```

**Success Criteria**: CV AUC > 0.78, improves over SetFit

---

### Experiment 3: SetFit Contrastive Learning ⭐⭐⭐⭐

**Estimated Score Impact**: +0.06 to +0.10 (0.78-0.82)

**Time Required**: 3-4 hours

**Success Probability**: 70%

**Why This Third**:
- True few-shot learning (not just similarity)
- Contrastive loss: Learn discriminative embeddings
- Fast training (30 min per fold)
- Already have baseline implementation

**Implementation Plan**:
```python
from setfit import SetFitModel, SetFitTrainer

# Generate contrastive pairs
positive_pairs = [(body, pos_ex1), (body, pos_ex2)]  # Similar
negative_pairs = [(body, neg_ex1), (body, neg_ex2)]  # Dissimilar

# Fine-tune with contrastive learning
model = SetFitModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    loss_class=CosineSimilarityLoss,
    num_iterations=20,  # Contrastive epochs
)
trainer.train()
```

**Success Criteria**: CV AUC > 0.80, beats current SetFit (0.776)

---

### Experiment 4: Multi-Level Stacking Ensemble ⭐⭐⭐⭐

**Estimated Score Impact**: +0.03 to +0.07 (0.82-0.87)

**Time Required**: 4-5 hours (requires models from Exp 1-3)

**Success Probability**: 85%

**Why This Fourth**:
- Proven approach (Jigsaw winners used 4-level stacking)
- Combines diverse models (DeBERTa, SetFit, LightGBM)
- Reduces overfitting (averaging)
- High success probability

**Implementation Plan**:
```python
# Level 1: Base models (out-of-fold predictions)
oof_deberta = train_deberta_cv()  # 5-fold CV
oof_setfit = train_setfit_cv()
oof_lgbm = train_lgbm_cv()
oof_crossenc = train_crossencoder_cv()

# Level 2: Meta-model
X_meta = np.column_stack([oof_deberta, oof_setfit, oof_lgbm, oof_crossenc])
meta_model = LogisticRegression()
meta_model.fit(X_meta, y_train)

# Test predictions
test_meta = np.column_stack([pred_deberta, pred_setfit, pred_lgbm, pred_crossenc])
final_pred = meta_model.predict_proba(test_meta)[:, 1]
```

**Success Criteria**: CV AUC > 0.83, beats best single model by +0.03

---

### Experiment 5: Pseudo-Labeling ⭐⭐⭐⭐

**Estimated Score Impact**: +0.02 to +0.05 (0.84-0.89)

**Time Required**: 3-4 hours

**Success Probability**: 60%

**Why This Fifth**:
- Essential for Jigsaw Multilingual winner
- Small training set (2,029) → benefits from more data
- Iterative refinement improves predictions
- Relatively easy to implement

**Implementation Plan**:
```python
# Iteration 1: Train on labeled data
model = train_deberta(train_df)

# Generate pseudo-labels for test set
test_preds = model.predict(test_df)
confident_preds = test_df[(test_preds > 0.9) | (test_preds < 0.1)]

# Iteration 2: Train on labeled + pseudo-labeled
combined_df = pd.concat([train_df, confident_preds])
model = train_deberta(combined_df)

# Repeat 2-3 times
```

**Success Criteria**: CV AUC improves by +0.02, LB stable (no overfitting)

---

## 9. EXPECTED SCORE PROGRESSION

### Conservative Trajectory

| Day | Experiment | Expected CV | Expected LB | Delta |
|-----|-----------|-------------|-------------|-------|
| 0 | Current (v9) | 0.7086 | 0.670 | - |
| 1 | DeBERTa-v3-base | 0.78 | 0.75 | +0.08 |
| 2 | SetFit contrastive | 0.80 | 0.77 | +0.02 |
| 3 | Ensemble (3 models) | 0.82 | 0.79 | +0.02 |
| 4 | Pseudo-labeling | 0.83 | 0.81 | +0.02 |
| 5 | Cross-encoder | 0.84 | 0.82 | +0.01 |
| 6 | Hyperparameter tuning | 0.85 | 0.83 | +0.01 |
| 7 | Final ensemble | 0.86 | 0.84 | +0.01 |

**Final Estimate (Conservative)**: LB 0.84 (CV 0.86)

### Optimistic Trajectory

| Day | Experiment | Expected CV | Expected LB | Delta |
|-----|-----------|-------------|-------------|-------|
| 0 | Current (v9) | 0.7086 | 0.670 | - |
| 1 | DeBERTa-v3-base | 0.82 | 0.79 | +0.12 |
| 2 | SetFit contrastive | 0.84 | 0.81 | +0.02 |
| 3 | Ensemble (4 models) | 0.86 | 0.83 | +0.02 |
| 4 | Pseudo-labeling | 0.87 | 0.85 | +0.02 |
| 5 | Advanced techniques | 0.88 | 0.86 | +0.01 |
| 6 | Hyperparameter tuning | 0.89 | 0.87 | +0.01 |
| 7 | Final ensemble | 0.90 | 0.88 | +0.01 |

**Final Estimate (Optimistic)**: LB 0.88 (CV 0.90)

### Realistic Target

**Expected Range**: 0.83-0.87 LB (0.85-0.89 CV)

**Gap to 1st Place**: Still 0.06-0.10 points short

**Ranking Estimate**: Top 10-15% (223-334 out of 2,227 teams)

**Verdict**: Significant improvement, but 0.93+ is unrealistic in 7 days

---

## 10. REFERENCES & RESOURCES

### A. Academic Papers

1. **SetFit**: Efficient Few-Shot Learning Without Prompts
   - https://arxiv.org/abs/2209.11055
   - HuggingFace implementation: https://github.com/huggingface/setfit

2. **DeBERTa-v3**: Improving DeBERTa using ELECTRA-Style Pre-Training
   - https://arxiv.org/abs/2111.09543
   - ICLR 2023 paper

3. **Sentence-BERT**: Sentence Embeddings using Siamese Networks
   - https://arxiv.org/abs/1908.10084
   - https://www.sbert.net/

4. **Cross-Encoders**: Re-ranking with BERT
   - https://www.sbert.net/examples/applications/cross-encoder/README.html

5. **Prototypical Networks**: Few-Shot Learning
   - https://arxiv.org/abs/1703.05175

6. **Triplet Loss**: Deep Metric Learning
   - https://arxiv.org/abs/1412.6622

### B. Kaggle Competitions (Similar)

1. **Jigsaw Toxic Comment Classification Challenge**
   - 1st place solution: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557
   - Review: https://blog.ceshine.net/post/kaggle-toxic-comment-classification-challenge/

2. **Jigsaw Multilingual Toxic Comment Classification**
   - Review: https://blog.ceshine.net/post/multilingual-toxic-classification/
   - Key takeaway: XLM-RoBERTa + pseudo-labeling

3. **Text Classification Competitions (General)**
   - Neptune.ai guide: https://neptune.ai/blog/text-classification-tips-and-tricks-kaggle-competitions

### C. Kaggle Notebooks (Implementations)

1. **SetFit Few-Shot Classification**
   - https://www.kaggle.com/code/starnicks/few-shot-text-classification-with-prototypical-net

2. **DeBERTa Fine-tuning**
   - Search Kaggle: "deberta fine-tuning classification"

3. **Cross-Encoder Similarity**
   - https://www.kaggle.com/code/search?q=cross+encoder

4. **Ensemble Stacking**
   - https://www.kaggle.com/code/nehabansal/stacking-blending-and-ensemble-tutorial

### D. HuggingFace Models

1. **DeBERTa-v3-base**: `microsoft/deberta-v3-base`
2. **RoBERTa-large**: `roberta-large`
3. **SetFit (all-mpnet)**: `sentence-transformers/all-mpnet-base-v2`
4. **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L6-v2`
5. **SmolLM-2**: `HuggingFaceTB/SmolLM2-360M-Instruct`

### E. Key Articles & Blogs

1. **Few-Shot Text Classification (2024)**
   - https://few-shot-text-classification.fastforwardlabs.com/

2. **Ensemble Learning in Kaggle**
   - https://www.kdnuggets.com/2015/06/ensembles-kaggle-data-science-competition-p3.html

3. **Pseudo-Labeling Guide**
   - https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/

4. **Prompt Engineering for LLMs**
   - https://www.promptingguide.ai/techniques/fewshot

---

## APPENDIX: QUICK REFERENCE

### Daily Checklist Template

**Morning (9 AM - 12 PM)**:
- [ ] Review previous day's results
- [ ] Start new experiment (implement + test)
- [ ] Monitor training progress

**Afternoon (1 PM - 5 PM)**:
- [ ] Complete training + cross-validation
- [ ] Analyze results (CV scores, fold stability)
- [ ] Make submission decision

**Evening (6 PM - 9 PM)**:
- [ ] Submit to Kaggle (max 2/day)
- [ ] Review leaderboard
- [ ] Plan next day's experiments
- [ ] Update strategy document

### Code Snippets

**DeBERTa Fine-tuning**:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

**SetFit Contrastive**:
```python
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset

model = SetFitModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
trainer = SetFitTrainer(model=model, train_dataset=train_dataset)
trainer.train()
```

**Cross-Encoder**:
```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
scores = model.predict([("query", "document"), ...])
```

### Performance Benchmarks

| Model | CV AUC | LB AUC | Training Time | Inference Time |
|-------|--------|--------|---------------|----------------|
| LightGBM v9 | 0.7086 | 0.670 | 5 min | <1 sec |
| SetFit basic | 0.776 | 0.77* | 30 min | 5 sec |
| DeBERTa-base (target) | 0.82* | 0.79* | 3 hours | 30 sec |
| Ensemble (target) | 0.86* | 0.84* | 8 hours | 2 min |

*Estimated based on similar competitions

### Contact & Resources

- **Competition**: https://www.kaggle.com/competitions/jigsaw-agile-community-rules
- **Discussion**: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion
- **HuggingFace**: https://huggingface.co/
- **SetFit Docs**: https://huggingface.co/blog/setfit

---

## FINAL THOUGHTS

This strategy document provides a comprehensive roadmap for the next 7 days. Key takeaways:

1. **Be realistic**: 0.93+ is extremely unlikely, but 0.85+ is achievable
2. **Pivot aggressively**: Abandon incremental improvements, embrace transformers
3. **Execute ruthlessly**: Time is limited, prioritize high-ROI experiments
4. **Trust your CV**: Don't overfit public LB, select based on cross-validation
5. **Learn and improve**: Even if we don't win, this is excellent experience

**The gap to 1st place is a mountain, not a hill. We need to climb strategically, not frantically.**

Good luck, and may the gradients be ever in your favor.

---

**Document Version**: 1.0
**Last Updated**: October 17, 2025, 02:00 UTC
**Next Review**: Daily updates based on experimental results
