# LoRA Adapter Configuration Comparison

> **Side-by-side analysis of mahmoudmohamed vs. seojinpark adapters**

## Summary Table

| Aspect | mahmoudmohamed | seojinpark fold3 | Winner |
|--------|----------------|------------------|--------|
| **Base Model** | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4` | ✅ seojinpark (explicit GPTQ) |
| **LoRA Rank (r)** | 16 | 8 | ⚖️ Depends (16 = more capacity, 8 = less overfitting) |
| **LoRA Alpha** | 32 | 16 | ⚖️ Depends (scaling factor) |
| **Effective Scaling** | 32/16 = 2.0 | 16/8 = 2.0 | Tie (same scaling) |
| **Target Modules** | q_proj, v_proj | q_proj, v_proj, k_proj, o_proj | ✅ seojinpark (more comprehensive) |
| **LoRA Dropout** | 0.05 | 0.1 | ✅ seojinpark (more regularization) |
| **Training Artifacts** | ❌ None | ✅ train.pkl (5.9MB), val.pkl (726KB) | ✅ seojinpark |
| **Methodology Evidence** | ❌ None | ✅ K-fold (fold0-4) | ✅ seojinpark |
| **Dataset Name Consistency** | 🚩 "reddit-**4b**-think" vs. config "**1.5B**" | ✅ No conflicts | ✅ seojinpark |
| **Adapter Size** | ~50MB | ~25MB | ✅ seojinpark (matches r=8 expectation) |
| **Result on Inference** | ❌ All predictions 0.0 | ⏱️ Not tested (competition ended) | N/A |

---

## Detailed Comparison

### 1. Base Model Specification

**mahmoudmohamed:**
```json
{
  "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"
}
```

**seojinpark:**
```json
{
  "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
}
```

**Analysis:**

| Aspect | mahmoudmohamed | seojinpark |
|--------|----------------|------------|
| Model family | ✅ Qwen 2.5 | ✅ Qwen 2.5 |
| Model size | ✅ 1.5B | ✅ 1.5B |
| Variant | Generic Instruct | **GPTQ-Int4** quantized |
| Specificity | Low | **High** (explicit quantization) |

**Why specificity matters:**

Qwen 2.5 1.5B comes in multiple variants:
- `Qwen2.5-1.5B-Instruct` (FP16, ~3GB)
- `Qwen2.5-1.5B-Instruct-GPTQ-Int4` (INT4, ~1GB)
- `Qwen2.5-1.5B-Instruct-AWQ` (AWQ quantization)

**If you train LoRA on GPTQ-Int4 and load onto FP16:** Dimension mismatch or degenerate behavior.

**seojinpark's explicit specification** eliminates ambiguity.

---

### 2. LoRA Hyperparameters

#### Rank (r)

**mahmoudmohamed:** r=16
**seojinpark:** r=8

**What is rank?**

LoRA decomposes weight updates as:
```
ΔW = A × B
where A: [d, r], B: [r, d]
```

Higher `r` = more parameters = more capacity but risk of overfitting.

**Comparison:**

| Rank | Parameters (Qwen 1.5B) | Pros | Cons |
|------|------------------------|------|------|
| r=8 | ~8M trainable params | Less overfitting, faster | May underfit complex tasks |
| r=16 | ~16M trainable params | More capacity | Risk of overfitting small datasets |

**For Jigsaw ACRC:**
- Training set: ~9,000 samples
- r=8 is safer (less overfitting risk)
- r=16 might overfit but capture more nuance

---

#### LoRA Alpha

**mahmoudmohamed:** lora_alpha=32
**seojinpark:** lora_alpha=16

**What is alpha?**

Scaling factor for LoRA updates:
```
effective_scaling = lora_alpha / r
```

**Comparison:**

| Adapter | r | alpha | Scaling |
|---------|---|-------|---------|
| mahmoudmohamed | 16 | 32 | 32/16 = 2.0 |
| seojinpark | 8 | 16 | 16/8 = 2.0 |

**Interesting:** Both use the **same effective scaling** despite different r/alpha values.

This suggests both adapters give LoRA updates equal weight relative to base model weights.

---

#### Target Modules

**mahmoudmohamed:**
```json
{
  "target_modules": ["q_proj", "v_proj"]
}
```

**seojinpark:**
```json
{
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
}
```

**What are these modules?**

In transformer attention:
```
Q = query projection (q_proj)
K = key projection (k_proj)
V = value projection (v_proj)
O = output projection (o_proj)
```

**Comparison:**

| Adapter | Modules | Coverage | Parameters |
|---------|---------|----------|------------|
| mahmoudmohamed | Q, V | 50% of attention | ~8M (r=16) |
| seojinpark | Q, K, V, O | 100% of attention | ~8M (r=8, but 4 modules) |

**Trade-off:**

- mahmoudmohamed: Higher rank (r=16) on fewer modules → deep but narrow
- seojinpark: Lower rank (r=8) on all modules → shallow but wide

**For instruction-following tasks:** Adapting all attention modules (Q, K, V, O) is generally better.

---

#### LoRA Dropout

**mahmoudmohamed:** 0.05 (5% dropout)
**seojinpark:** 0.1 (10% dropout)

**What is LoRA dropout?**

During training, randomly zero out LoRA weights to prevent overfitting.

**Comparison:**

| Dropout | Regularization | Risk of Overfitting |
|---------|----------------|---------------------|
| 0.05 | Weak | Higher |
| 0.1 | Moderate | Lower |

**For small datasets (9K samples):** Higher dropout (0.1) is safer.

---

### 3. Training Evidence

**mahmoudmohamed:**

Files:
```
adapter_config.json
adapter_model.bin
README.md (generic)
```

**No training artifacts.** Can't verify:
- What data was used
- Training/validation split
- Training logs or metrics
- Convergence behavior

---

**seojinpark:**

Files:
```
adapter_config.json
adapter_model.bin
train.pkl (5.9 MB)
val.pkl (726 KB)
```

**Has training artifacts!** This shows:
- ✅ Actual training data used (~9,000 samples in train.pkl)
- ✅ Validation split (~1,000 samples in val.pkl)
- ✅ Proper train/val methodology
- ✅ K-fold strategy (fold0-4 available)

**Credibility:** seojinpark's adapters are **provably trained**, not just uploaded configs.

---

### 4. K-Fold Strategy

**mahmoudmohamed:** Single adapter

**seojinpark:** 5 adapters (fold0, fold1, fold2, fold3, fold4)

**Why K-fold matters:**

1. **Reduces overfitting:** Each fold trained on different train/val split
2. **Enables ensembling:** Can average predictions from all 5 folds
3. **Robust validation:** 5-fold CV is more reliable than single split

**Expected LB improvement:**

- Single fold: 0.91 (hypothetical)
- 5-fold ensemble: 0.92-0.925 (typical +0.01-0.015 boost)

---

### 5. Dataset Name Analysis

**mahmoudmohamed:**

Dataset slug: `mahmoudmohamed/reddit-4b-think`

**Red flags:**
- "**4b**" in name but config says "**1.5B**"
- Was this trained on Qwen 4B and config edited for compatibility?
- Or is "4b" referring to something else (4 billion tokens trained on)?

**Ambiguity creates risk.**

---

**seojinpark:**

Dataset slug: `seojinpark/jigsaw-qwen-fold3`

**No conflicts:**
- "jigsaw" matches competition
- "qwen" matches model family
- "fold3" matches K-fold methodology
- No size claims in name

**Clear, unambiguous naming.**

---

## Failure Analysis: Why mahmoudmohamed Failed

### Hypothesis: Base Model Mismatch

**Evidence:**

1. 🚩 Dataset name: "reddit-**4b**-think"
2. ✅ Config claims: "Qwen2.5-**1.5B**-Instruct"
3. ❌ No training artifacts to verify
4. ❌ All inference outputs: 0.0 (degenerate behavior)

**Likely scenario:**

1. Adapter was trained on Qwen 4B (or different base model)
2. Config was edited to "1.5B-Instruct" for sharing/compatibility
3. When loaded onto actual 1.5B model, dimensions don't align
4. PEFT library handles gracefully (no crash) but produces degenerate outputs

**Why 0.0 specifically?**

When LoRA weights are misaligned:
- Model falls back to conservative behavior
- 0.0 = "not a violation" = safe default
- Alternatively, corrupted logits heavily favor first class

---

## What Would Have Happened with seojinpark?

**Hypothesis:** Would have worked correctly.

**Evidence supporting this:**

1. ✅ Explicit base model match (GPTQ-Int4)
2. ✅ Training artifacts prove real training
3. ✅ K-fold methodology (professional approach)
4. ✅ Consistent naming (no conflicts)
5. ✅ More comprehensive LoRA (all 4 attention modules)
6. ✅ Higher regularization (dropout=0.1)

**Expected results:**

- Non-degenerate outputs (diverse probabilities)
- Correlation with violation examples
- LB score: 0.91-0.92 (single fold)
- With 5-fold ensemble: 0.92-0.925

**Why I didn't test it:**

- ⏱️ Competition ended before I could run Tier 1 v3
- 🎯 Spent 4 hours debugging mahmoudmohamed approach first
- 📊 Should have validated both adapters in parallel

---

## Lessons: Adapter Selection Criteria

### Red Flags (Avoid)

- ❌ Dataset name conflicts with config
- ❌ No training artifacts (train.pkl, logs)
- ❌ Generic base model specification (no quantization variant)
- ❌ Single adapter (no K-fold or ensemble strategy)
- ❌ Suspiciously small file size (<10MB for r>8)

### Green Flags (Prefer)

- ✅ Explicit base model variant (GPTQ-Int4, AWQ, etc.)
- ✅ Training artifacts present (pkl files, logs)
- ✅ K-fold methodology (multiple folds available)
- ✅ Comprehensive target modules (Q, K, V, O)
- ✅ Appropriate regularization (dropout 0.1+)
- ✅ Reasonable file sizes (20-50MB for r=8-16)

---

## Conclusion

**Winner:** seojinpark adapters

**Why:**
1. Explicit base model specification
2. Training artifacts prove credibility
3. K-fold methodology for robustness
4. More comprehensive LoRA coverage
5. Better regularization
6. No naming conflicts

**mahmoudmohamed failure taught us:**

> "Always validate external resources before depending on them in production (or competitions)."

This is a **production engineering skill**, not just a competition trick.

---

**Last updated:** Oct 25, 2024
**Author:** LSJ
