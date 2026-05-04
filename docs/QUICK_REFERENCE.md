# QUICK REFERENCE CARD
## Jigsaw ACRC - Essential Info at a Glance

**Print this page and keep it visible during the competition**

---

## CURRENT STATUS
- **LB**: 0.670 (CV: 0.7086)
- **1st Place**: 0.933
- **Gap**: 0.263 points
- **Days Left**: 7
- **Submissions Left**: 35 (5/day)

---

## TODAY'S PRIORITIES (Day 1)

### 1. DeBERTa-v3-base ⭐⭐⭐⭐⭐
- **Time**: 4-5 hours
- **Expected**: +0.08-0.12 → LB 0.75-0.82
- **Model**: `microsoft/deberta-v3-base`
- **Status**: START IMMEDIATELY

### 2. Cross-Encoder ⭐⭐⭐⭐⭐
- **Time**: 3 hours
- **Expected**: +0.05-0.08 → LB 0.77-0.80
- **Model**: `cross-encoder/ms-marco-MiniLM-L6-v2`
- **Status**: After DeBERTa

### 3. Submit (End of Day)
- **Target**: 2 submissions
- **Expected LB**: 0.75-0.80

---

## REALISTIC TARGETS

| Timeline | CV Target | LB Target | Method |
|----------|-----------|-----------|--------|
| Day 1 | 0.78-0.80 | 0.75-0.77 | DeBERTa + Cross-encoder |
| Day 3 | 0.82-0.84 | 0.79-0.81 | Ensemble (3-4 models) |
| Day 5 | 0.85-0.87 | 0.82-0.85 | + Pseudo-labeling |
| Day 7 | 0.86-0.90 | 0.83-0.88 | Final optimization |

**Final Expected Ranking**: Top 10-15% (223-334 / 2,227 teams)

---

## KEY MODELS TO IMPLEMENT

### Tier 1 (Must Have)
1. ✅ **DeBERTa-v3-base** - 184M params, 3h training
2. ✅ **Cross-Encoder** - 23M params, 2h training
3. ✅ **SetFit Contrastive** - 420M params, 1h training

### Tier 2 (If Time Permits)
4. **RoBERTa-large** - 355M params, 5h training
5. **Prototypical Networks** - Custom, 2h training
6. **LLM Prompting** - SmolLM-2, experimental

---

## ENSEMBLE STRATEGY

### Level 1: Base Models
```
DeBERTa-v3-base   (weight: 0.40)
Cross-Encoder      (weight: 0.25)
SetFit Contrastive (weight: 0.20)
LightGBM v9        (weight: 0.15)
```

### Level 2: Stacking
```python
meta_model = LogisticRegression()
meta_model.fit(oof_predictions, y_train)
```

**Expected Boost**: +0.03-0.07 over best single model

---

## CRITICAL SUCCESS FACTORS

### DO
- ✅ Trust CV over LB (avoid overfitting)
- ✅ Implement transformers ASAP
- ✅ Use fp16 mixed precision (2x faster)
- ✅ Checkpoint every epoch
- ✅ Parallelize experiments (multiple notebooks)
- ✅ Limit submissions (2/day max)

### DON'T
- ❌ Waste time on incremental tuning
- ❌ Overfit public leaderboard
- ❌ Try too many experiments (focus on top 5)
- ❌ Forget to save models to Kaggle Datasets
- ❌ Submit without CV validation
- ❌ Panic if first attempts fail

---

## RESOURCE LIMITS

### Kaggle Constraints
- **GPU**: 30 hours/week
- **CPU**: 9 hours/notebook
- **RAM**: 16 GB (GPU), 30 GB (CPU)
- **Disk**: 20 GB

### Time Budget (Per Model)
- DeBERTa: 5 hours (2h code + 3h train)
- Cross-Encoder: 3 hours
- SetFit: 3 hours
- Ensemble: 2 hours
- Submission prep: 1 hour

---

## TROUBLESHOOTING

### Out of Memory
```python
per_device_train_batch_size=4  # Reduce
gradient_accumulation_steps=4  # Increase
fp16=True  # Enable mixed precision
```

### Training Too Slow
```python
max_length=256  # Reduce from 512
num_train_epochs=2  # Reduce from 3
eval_steps=200  # Increase from 100
```

### Poor CV Score
- Check input format (print examples)
- Try different learning rates (2e-5, 3e-5)
- Verify tokenization is correct
- Increase training epochs

---

## SUBMISSION CHECKLIST

Before Each Submission:
- [ ] CV score > previous best
- [ ] CV std < 0.02 (stable)
- [ ] CV-LB gap checked (if previous submission)
- [ ] Tested in Kaggle notebook
- [ ] Model saved to Kaggle Dataset
- [ ] Code runs end-to-end (<9h)
- [ ] Submission file format correct

---

## DAILY GOALS

### Day 1 (Oct 17) - PIVOT
- [ ] DeBERTa-v3-base working
- [ ] Cross-encoder working
- [ ] 2 submissions made
- [ ] LB: 0.75-0.80

### Day 2 (Oct 18) - FEW-SHOT
- [ ] SetFit contrastive working
- [ ] RoBERTa-large (optional)
- [ ] 1-2 submissions
- [ ] LB: 0.78-0.82

### Day 3 (Oct 19) - ENSEMBLE
- [ ] Multi-seed training
- [ ] Stacking ensemble
- [ ] 2 submissions
- [ ] LB: 0.82-0.85

### Day 4 (Oct 20) - AUGMENT
- [ ] Pseudo-labeling
- [ ] Iterative refinement
- [ ] 2 submissions
- [ ] LB: 0.83-0.87

### Day 5 (Oct 21) - ADVANCED
- [ ] Experimental models
- [ ] Rule-specific models
- [ ] 2 submissions
- [ ] LB: 0.85-0.88

### Day 6 (Oct 22) - OPTIMIZE
- [ ] Hyperparameter tuning
- [ ] Ensemble optimization
- [ ] 2 submissions
- [ ] LB: 0.86-0.89

### Day 7 (Oct 23) - FINALIZE
- [ ] Select 2 final submissions
- [ ] Retrain on 100% data
- [ ] Submit before deadline
- [ ] LB: 0.83-0.88

---

## EXPECTED SCORE PROGRESSION

```
Current:  ████████████████░░░░░░░░░░░░░░░░░░░░ 0.670
Day 1:    ████████████████████░░░░░░░░░░░░░░░░ 0.75-0.77
Day 3:    ████████████████████████░░░░░░░░░░░░ 0.79-0.81
Day 5:    ███████████████████████████░░░░░░░░░ 0.82-0.85
Day 7:    ████████████████████████████░░░░░░░░ 0.83-0.88
1st:      ██████████████████████████████████░░ 0.933

Gap Remaining: 0.05-0.10 points
```

---

## CRITICAL REMINDERS

### 1. Time is Your Enemy
- 7 days = 168 hours
- Sleep: 56 hours (8h/day)
- Work: 112 hours available
- Effective: ~70 hours (accounting for breaks)
- **Each hour counts!**

### 2. 0.93+ is Unrealistic
- Top teams have been working for weeks
- 0.263 gap requires 3-5 breakthroughs
- **Realistic target: 0.85+ (top 10-15%)**

### 3. Trust the Process
- CV is your compass
- LB is just one data point
- Model diversity > chasing LB
- Private LB may differ significantly

### 4. Have a Backup Plan
If transformers fail:
- Ensemble existing models (SetFit + LightGBM)
- Add features (subreddit risk, text stats)
- Pseudo-labeling
- **Target: 0.78-0.82 (still better than current)**

---

## MOTIVATION

### You've Got This Because:
1. ✅ Strategy is solid (based on winning solutions)
2. ✅ Models are proven (DeBERTa, cross-encoders work)
3. ✅ Resources available (Kaggle GPU, HuggingFace models)
4. ✅ Time is tight but manageable (7 days is enough)

### Remember:
- Every 0.01 improvement matters
- Top solutions combine 5-10 models
- Persistence beats perfection
- Learning > winning

---

## EMERGENCY CONTACTS

- **Competition**: https://www.kaggle.com/competitions/jigsaw-agile-community-rules
- **Discussion**: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion
- **HuggingFace Models**: https://huggingface.co/models
- **Documentation**:
  - Full Strategy: `COMPREHENSIVE_STRATEGY.md`
  - Executive Summary: `EXECUTIVE_SUMMARY.md`
  - Implementation: `IMPLEMENTATION_GUIDE.md`

---

## ONE-LINE COMMAND REFERENCE

```bash
# Start Kaggle notebook
kaggle kernels push -p /path/to/notebook

# Check GPU status
nvidia-smi

# Install dependencies
pip install transformers datasets setfit sentence-transformers

# Download model
from transformers import AutoModel
model = AutoModel.from_pretrained('microsoft/deberta-v3-base')

# Train DeBERTa (see IMPLEMENTATION_GUIDE.md)
# Train Cross-Encoder (see IMPLEMENTATION_GUIDE.md)
# Create ensemble (see COMPREHENSIVE_STRATEGY.md)
```

---

## FINAL MESSAGE

**You have 7 days to climb from 0.670 to 0.85+.**

**The path is clear. The tools are ready. Now execute.**

**Focus → Implement → Test → Submit → Iterate**

**Let's make it happen! 🚀**

---

*Last Updated: Oct 17, 2025*
*Next Review: Daily*
*Good luck!*
