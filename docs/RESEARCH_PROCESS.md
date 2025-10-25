# Research Process: Finding Public LoRA Adapters

> **How I discovered public Qwen LoRA weights through Kaggle Discussion analysis**

## Context

After establishing a DeBERTa baseline (0.904), I wanted to explore larger models for potential score improvement. However, training large models like Qwen 7B was impractical due to:

1. **Time constraints:** Competition ending in 2 days
2. **GPU quota:** Kaggle's 9 hours/week limit
3. **Environment issues:** NumPy compatibility blocking training libraries

**Strategy:** Find and leverage public LoRA adapters for inference-only approach.

---

## Research Methodology

### Step 1: Kaggle Discussion Analysis

**Source:** [Jigsaw ACRC Discussion Forum](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion)

**Search keywords:**
- "Qwen"
- "LoRA"
- "test-time training"
- "public notebook"
- "0.92" (medal score range)

**Key findings:**

1. **Multiple participants mentioned Qwen models** in discussions
2. **Some referenced "public datasets"** without explicit links (keeping competitive advantage)
3. **Test-time training was confirmed legal** by competition hosts

---

### Step 2: Kaggle Dataset Search

**Search strategy:**

```
Search query: "qwen jigsaw"
Filters: Public datasets only
Sort by: Recently updated
```

**Discovered datasets:**

| Dataset | Author | Date Added | Model | Notes |
|---------|--------|------------|-------|-------|
| reddit-4b-think | mahmoudmohamed | Oct 21, 2024 | Qwen (4B?) | ⚠️ Name suggests 4B model |
| seojinpark-fold0 | seojinpark | Oct 19, 2024 | Qwen 1.5B-GPTQ | ✅ Explicit 1.5B match |
| seojinpark-fold1 | seojinpark | Oct 20, 2024 | Qwen 1.5B-GPTQ | ✅ K-fold strategy |
| seojinpark-fold2 | seojinpark | Oct 20, 2024 | Qwen 1.5B-GPTQ | ✅ K-fold strategy |
| seojinpark-fold3 | seojinpark | Oct 21, 2024 | Qwen 1.5B-GPTQ | ✅ K-fold strategy |
| seojinpark-fold4 | seojinpark | Oct 21, 2024 | Qwen 1.5B-GPTQ | ✅ K-fold strategy |
| qwen-lora-v1 | (various) | Oct 18-22 | Various | 10+ other datasets |

**Total found:** 15+ public LoRA adapter datasets

---

### Step 3: Dataset Validation

**Validation checklist** (developed after failures):

```python
def validate_public_adapter(dataset_slug):
    """Validate a public LoRA adapter before use"""
    checks = []

    # 1. Download and inspect adapter_config.json
    config = load_adapter_config(dataset_slug)
    checks.append({
        "base_model": config.get("base_model_name_or_path"),
        "lora_r": config.get("r"),
        "lora_alpha": config.get("lora_alpha")
    })

    # 2. Check dataset metadata
    dataset_info = kaggle.api.dataset_metadata(dataset_slug)
    checks.append({
        "dataset_name": dataset_info["title"],
        "size": dataset_info["totalBytes"],
        "last_updated": dataset_info["lastUpdated"]
    })

    # 3. Look for training artifacts
    files = kaggle.api.dataset_list_files(dataset_slug)
    has_training_artifacts = any(
        f in files for f in ["train.pkl", "val.pkl", "training_log.txt"]
    )
    checks.append({"has_artifacts": has_training_artifacts})

    # 4. Cross-reference dataset name with config
    name_model_match = check_name_config_consistency(
        dataset_info["title"],
        config.get("base_model_name_or_path")
    )
    checks.append({"name_config_match": name_model_match})

    return checks
```

**What I should have checked (but didn't):**

- ❌ Dataset name vs. config base model match
- ❌ Presence of training artifacts (pkl files)
- ❌ File sizes (small = suspicious, large = likely real)
- ❌ Test on 3 samples before full inference

**What I actually checked:**

- ✅ adapter_config.json exists
- ✅ Config has base model path
- ⚠️ Assumed config was accurate → **Critical mistake**

---

## Key Discoveries

### Discovery 1: mahmoudmohamed Adapter

**Dataset:** `mahmoudmohamed/reddit-4b-think`

**First impression:** ✅ Large dataset (~50MB), recently updated (Oct 21)

**adapter_config.json:**
```json
{
  "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
  "lora_alpha": 32,
  "r": 16
}
```

**Initial assessment:** Looks good! Base model matches my inference model.

**Red flag I missed:** Dataset name says "**4b**-think" but config says "**1.5B**-Instruct"

**Lesson:** Cross-reference dataset metadata with config contents.

---

### Discovery 2: seojinpark K-Fold Adapters

**Dataset:** `seojinpark/jigsaw-qwen-fold[0-4]`

**First impression:** ✅ Professional K-fold strategy, recent updates (Oct 19-21)

**adapter_config.json (fold3):**
```json
{
  "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
  "lora_alpha": 16,
  "r": 8
}
```

**Key differences from mahmoudmohamed:**

1. ✅ **Explicit GPTQ quantization** in base model path
2. ✅ **Training artifacts present** (train.pkl: 5.9MB, val.pkl: 726KB)
3. ✅ **K-fold methodology** (5 folds = robust ensemble)
4. ✅ **Smaller LoRA rank** (r=8 vs. r=16, suggests careful tuning)

**Assessment:** This looks much more credible.

**Why I didn't use it first:** Discovered mahmoudmohamed earlier, wanted to test quickly. **Time pressure mistake.**

---

## Timeline of Discovery

**Oct 21, 16:00 KST:** Discussion analysis → identified "Qwen" as promising approach

**Oct 21, 17:00 KST:** Kaggle dataset search → found mahmoudmohamed adapter

**Oct 21, 18:00 KST:** Downloaded adapter, created Tier 1 v1 kernel

**Oct 22, 00:00 KST:** Tier 1 v1 results → all 0.0 (failed)

**Oct 22, 02:00 KST:** Created Tier 1 v2 with better prompts

**Oct 22, 06:00 KST:** Tier 1 v2 results → still all 0.0 (failed)

**Oct 22, 10:00 KST:** Discovered seojinpark adapters (better validated)

**Oct 22, 14:00 KST:** Downloaded seojinpark fold3 for testing

**Oct 24, 06:59 KST:** Competition ended before testing seojinpark

---

## Lessons Learned

### 1. Validate Public Resources Early

**Time spent:**
- Tier 1 v1: 2 hours
- Tier 1 v2: 2 hours
- **Total:** 4 hours on incompatible adapter

**Time saved if I had validated:**
- 3-sample test: 10 minutes
- Would have caught issue immediately
- **Net savings:** 3.5 hours

**Validation script (what I should have run):**

```python
def quick_adapter_test(base_model, adapter_path, test_samples=3):
    """Test adapter on 3 samples before full inference"""
    model = load_model(base_model, adapter_path)

    # Test on diverse samples
    samples = [
        "Clear violation example",
        "Clear non-violation example",
        "Ambiguous gray-area example"
    ]

    outputs = [predict(s) for s in samples]

    # Validate diversity
    assert np.std(outputs) > 0.05, f"Degenerate: {outputs}"
    assert max(outputs) > 0.3, f"All low: {outputs}"
    assert min(outputs) < 0.7, f"All high: {outputs}"

    print(f"✅ Adapter passed validation: {outputs}")
    return True
```

---

### 2. Look for Training Artifacts

**Good signs:**

- ✅ `train.pkl` or `training_data.pkl` (shows actual training)
- ✅ `val.pkl` or `validation_data.pkl` (shows validation split)
- ✅ `training_log.txt` or wandb logs (shows training process)
- ✅ Multiple checkpoints (e.g., fold0, fold1, ..., fold4)

**Bad signs:**

- ❌ Only `adapter_model.bin` and `adapter_config.json`
- ❌ No training data or logs
- ❌ Suspiciously small file size (<10MB for LoRA)

**mahmoudmohamed:** ❌ No training artifacts
**seojinpark:** ✅ Has train.pkl (5.9MB), val.pkl (726KB)

---

### 3. Cross-Reference Dataset Metadata

**What to check:**

1. **Dataset name** vs. **adapter config base model**
   - Example: "reddit-**4b**-think" but config says "**1.5B**-Instruct" → 🚩

2. **File sizes** vs. **expected sizes**
   - LoRA r=16: ~30-50MB per adapter
   - LoRA r=8: ~15-25MB per adapter
   - Full model: 1-3GB (too large for LoRA)

3. **Update dates** vs. **competition timeline**
   - Updated during competition: ✅ Likely trained on competition data
   - Updated before competition: ⚠️ May be for different task

---

## Public Resource Validation Checklist

Use this checklist for future competitions:

```
[ ] Downloaded adapter_config.json and inspected base_model_name_or_path
[ ] Checked dataset name matches config (no size mismatches)
[ ] Verified training artifacts exist (pkl files, logs)
[ ] Checked file sizes are reasonable for LoRA rank
[ ] Tested on 3-5 samples before full inference
[ ] Validated output diversity (std > 0.05)
[ ] Checked update date aligns with competition timeline
[ ] Read dataset description for any warnings/notes
[ ] Cross-referenced with discussion forum mentions
[ ] Verified author has other credible datasets/kernels
```

---

## Alternative Research Methods

### Method 1: Public Notebook Search

```
Search: "jigsaw qwen"
Filter: Public notebooks with code
Sort: Best score
```

**Pros:**
- Can see full implementation
- Can copy exact prompts/hyperparameters
- Can check if notebook actually runs

**Cons:**
- Top solutions rarely share code during competition
- May not have public LoRA weights (just methodology)

---

### Method 2: Discussion Forum Deep Dive

**Strategy:**

1. Read ALL discussion threads (even low-vote ones)
2. Look for code snippets mentioning specific datasets
3. Follow user profiles to see their published datasets
4. Check replies for "thank you" messages (indicates shared resource)

**Time investment:** 2-3 hours

**Payoff:** Often finds hidden gems

---

### Method 3: Kaggle API Search

```python
import kaggle

# Search datasets
datasets = kaggle.api.dataset_list(search="qwen")

for ds in datasets:
    print(f"{ds.ref}: {ds.title} ({ds.totalBytes} bytes)")
    # Download and validate
```

**Pros:**
- Programmatic, can batch process
- Can filter by size, update date, etc.

**Cons:**
- Still need manual validation
- API rate limits

---

## Conclusion

Finding public resources requires:

1. **Discussion analysis** - Where community shares hints
2. **Dataset search** - Where resources are published
3. **Validation** - DON'T trust blindly
4. **Testing** - Always test on small samples first

**Time saved by proper validation:** 3-4 hours per failed attempt

**Skill demonstrated:** Ability to find and evaluate external resources (critical for production ML).

---

**Date:** Oct 25, 2024
**Author:** LSJ
