# Technical Deep Dive: Qwen LoRA Inference Implementation

> **Detailed explanation of the inference pipeline, prompt engineering, and technical decisions**

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [DeBERTa Baseline](#deberta-baseline)
3. [Qwen LoRA Inference Pipeline](#qwen-lora-inference-pipeline)
4. [Prompt Engineering Evolution](#prompt-engineering-evolution)
5. [Parsing Strategy](#parsing-strategy)
6. [Why Inference-Only?](#why-inference-only)
7. [Alternative Approaches Considered](#alternative-approaches-considered)
8. [Technical Stack](#technical-stack)

---

## Architecture Overview

### Competition Requirements

**Input:** Reddit post + community rule + positive/negative examples
**Output:** Probability [0.0, 1.0] of rule violation
**Metric:** ROC-AUC (threshold-independent)

### Approach Comparison

| Approach | Type | Score | Pros | Cons |
|----------|------|-------|------|------|
| DeBERTa-v3-base | Fine-tuned transformer | 0.904 | Reliable, well-understood | Limited by model size (140M params) |
| Qwen 2.5 1.5B + LoRA | Large LLM inference | Failed (0.0) | Larger capacity, instruction-following | Adapter compatibility issues |

---

## DeBERTa Baseline

### Model Architecture

**Model:** `microsoft/deberta-v3-base`
- **Parameters:** 140M (184M with heads)
- **Architecture:** Disentangled attention mechanism
- **Max sequence length:** 512 tokens
- **Output:** Binary classification head

### Why DeBERTa?

1. **Proven performance** on text classification tasks
2. **Efficient fine-tuning** (full model, not just LoRA)
3. **Stable gradients** with disentangled attention
4. **Good few-shot learning** with in-context examples

### Training Setup (Reference)

```python
# Conceptual implementation (not exact code from competition)
from transformers import DebertaV3ForSequenceClassification, Trainer

model = DebertaV3ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=1,  # Regression task (probability output)
    problem_type="regression"
)

training_args = TrainingArguments(
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    num_train_epochs=3,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
```

### Input Format

```python
def create_deberta_input(row):
    # Concatenate all context
    text = f"""Rule: {row['rule']}

Positive examples:
1. {row['positive_example_1']}
2. {row['positive_example_2']}

Negative examples:
1. {row['negative_example_1']}
2. {row['negative_example_2']}

Post to evaluate: {row['body']}"""

    return tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
```

### Results

- **Validation ROC-AUC:** ~0.90-0.91 (estimated, exact CV not logged)
- **Public LB:** 0.904
- **Rank:** 1,121 / 2,444 (Top 46%)

**Takeaway:** DeBERTa provided a solid baseline quickly (2-3 hours setup). This established a floor for more complex approaches.

---

## Qwen LoRA Inference Pipeline

### Why Qwen 2.5 1.5B?

**Model:** `Qwen/Qwen2.5-1.5B-Instruct`

**Advantages:**
1. **Instruction-tuned:** Trained to follow prompts (good for zero/few-shot)
2. **Larger capacity:** 1.5B params > 140M (DeBERTa)
3. **Flexible output:** Can generate continuous probabilities via prompting
4. **Public LoRA weights:** Community members shared fine-tuned adapters

**Why not bigger models?**
- Qwen 7B/14B too large for Kaggle GPU limits (9 hours)
- 1.5B is sweet spot: good capacity, fits in 16GB VRAM

### LoRA Configuration

**Adapter:** mahmoudmohamed/reddit-4b-think (attempted)

**LoRA hyperparameters:**

```json
{
  "lora_alpha": 32,
  "r": 16,
  "target_modules": ["q_proj", "v_proj"],
  "lora_dropout": 0.05
}
```

**What these mean:**

- **`r` (rank):** Size of low-rank matrices (higher = more capacity, but slower)
  - `r=16` is medium capacity (common values: 8, 16, 32, 64)
- **`lora_alpha`:** Scaling factor for adapter weights
  - `lora_alpha=32` with `r=16` → scaling = 32/16 = 2.0
- **`target_modules`:** Which layers to adapt
  - `q_proj`, `v_proj` are query and value projection matrices in attention

**Why only q_proj and v_proj?**
- Common practice: adapting Q and V captures most of attention behavior
- Adding K (key) and O (output) increases parameters with diminishing returns
- Note: seojinpark used all 4 (q, k, v, o) - potentially more expressive

### Loading Pipeline

**Code:** [`notebooks/tier1_v2_ultra_structured/qwen_tier1_v2.ipynb:106-114`](notebooks/tier1_v2_ultra_structured/qwen_tier1_v2.ipynb)

```python
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(
    CFG.BASE_MODEL,
    trust_remote_code=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    CFG.BASE_MODEL,
    torch_dtype=torch.float16,  # Half precision for memory
    device_map="auto",  # Automatic device placement
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base_model,
    CFG.LORA_ADAPTER,
    torch_dtype=torch.float16
)

model.eval()  # Inference mode
print(f"✅ Model loaded on {CFG.DEVICE}")
```

**Key decisions:**

1. **`torch.float16`:** Half precision reduces VRAM from ~6GB to ~3GB
2. **`device_map="auto"`:** Handles multi-GPU or CPU offloading automatically
3. **`trust_remote_code=True`:** Required for Qwen (custom modeling code)
4. **`model.eval()`:** Disables dropout, sets batch norm to inference mode

### Inference Loop

**Code:** [`notebooks/tier1_v2_ultra_structured/qwen_tier1_v2.ipynb:130-144`](notebooks/tier1_v2_ultra_structured/qwen_tier1_v2.ipynb)

```python
predictions = []
success = fail = 0

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    # 1. Create prompt
    prompt = create_prompt_v2(tokenizer, row)

    # 2. Generate prediction
    prob = predict_single_v2(model, tokenizer, prompt, row['row_id'])

    # 3. Collect result
    predictions.append({
        'row_id': row['row_id'],
        'rule_violation': prob
    })

    # 4. Track parsing success
    if prob not in [0.2, 0.5, 0.8]:  # Not a fallback value
        success += 1
    else:
        fail += 1
```

**Design choices:**

- **No batching:** Process one sample at a time for simplicity
  - Could batch for speed, but inference is fast enough (~1-2 sec/sample)
- **Eager evaluation:** Don't accumulate in memory (good for large test sets)
- **Tracking metrics:** Monitor parse success rate in real-time

---

## Prompt Engineering Evolution

### v1: Basic Instruction (Failed)

```python
def create_prompt(row):
    prompt = f"""You are a content moderator. Determine if the following post violates the given rule.

Rule: {row['rule']}
Post to evaluate: {row['body']}

Does this post violate the rule? Answer with a probability between 0 and 1.

Probability:"""
    return prompt
```

**Issues:**
- ❌ No system/user separation
- ❌ Vague instruction ("Answer with a probability")
- ❌ No examples showing exact output format
- ❌ No constraints on output length

**Results:**
- Variable output formats ("I think...", "Yes, 0.8", just "0.5")
- ~50% parsing success
- All parsed values: 0.0

---

### v2: Ultra-Structured (Still Failed, but Better)

**Code:** [`notebooks/tier1_v2_ultra_structured/qwen_tier1_v2.ipynb:64-91`](notebooks/tier1_v2_ultra_structured/qwen_tier1_v2.ipynb)

```python
def create_prompt_v2(tokenizer, row):
    """V2: Ultra-structured prompt with chat template"""
    system = "You are a precise AI. Respond ONLY with a decimal number between 0.0 and 1.0. NO other text."

    user = f"""Analyze if this post violates the rule.

RULE: {row['rule']}
POST: {row['body']}

POSITIVE EXAMPLES (violations):
1. {row['positive_example_1']}
2. {row['positive_example_2']}

NEGATIVE EXAMPLES (not violations):
1. {row['negative_example_1']}
2. {row['negative_example_2']}

Respond with ONLY a number between 0.0 and 1.0:
- 0.0 = NOT a violation
- 1.0 = IS a violation

Examples: 0.95, 0.23, 0.78, 0.02

Your answer (number only):"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
```

**Improvements:**

1. ✅ **System message:** Sets behavior mode ("precise AI", "ONLY a number")
2. ✅ **Explicit constraints:** "NO other text", "number only"
3. ✅ **Few-shot examples:** Shows exact format (0.95, 0.23, 0.78, 0.02)
4. ✅ **Clear mapping:** "0.0 = NOT a violation, 1.0 = IS a violation"
5. ✅ **Structured sections:** RULE, POST, EXAMPLES clearly labeled
6. ✅ **Chat template:** Uses Qwen's official format via `apply_chat_template()`

**What is `apply_chat_template()`?**

Different models use different chat formats:

**ChatML (Qwen):**
```
<|im_start|>system
You are a precise AI.<|im_end|>
<|im_start|>user
Analyze if this post violates the rule.<|im_end|>
<|im_start|>assistant
```

**Llama format:**
```
[INST] <<SYS>>
You are a precise AI.
<</SYS>>

Analyze if this post violates the rule. [/INST]
```

Using `apply_chat_template()` ensures correct format for Qwen.

**Results:**
- ✅ 100% parsing success (all outputs were clean "0.0")
- ❌ Still all 0.0 values (no diversity)
- ✅ Ruled out prompt engineering as root cause

---

### What v3 Would Have Been

```python
def create_prompt_v3(tokenizer, row):
    """V3: Contrastive reasoning approach"""
    system = "You are an expert content moderator. Think step-by-step, then output a probability."

    user = f"""Evaluate this post against the rule.

RULE: {row['rule']}
POST: {row['body']}

EXAMPLES OF VIOLATIONS:
1. {row['positive_example_1']}
2. {row['positive_example_2']}

EXAMPLES OF NON-VIOLATIONS:
1. {row['negative_example_1']}
2. {row['negative_example_2']}

STEP 1: How similar is the post to the violation examples? (high/medium/low)
STEP 2: How similar is the post to the non-violation examples? (high/medium/low)
STEP 3: Does the post's intent match the rule's concern? (yes/no)

Based on your analysis, output a probability between 0.0 and 1.0:"""

    # ... (same chat template logic)
```

**Hypothesis:** Chain-of-thought reasoning might help model calibrate better.

**Status:** Never tested (competition ended).

---

## Parsing Strategy

### The Challenge

LLMs don't have a "return float" function. They generate text. We need to extract numerical values reliably.

### v1: Naive Parsing (50% success)

```python
def parse_output_v1(text):
    try:
        return float(text)
    except:
        return 0.5  # Fallback to neutral
```

**Issues:**
- Fails on "The probability is 0.8" → ValueError
- Fails on "Yes, I think it's a violation." → ValueError
- Too sensitive to model verbosity

---

### v2: 3-Stage Robust Parsing (100% success)

**Code:** [`notebooks/tier1_v2_ultra_structured/qwen_tier1_v2.ipynb:93-128`](notebooks/tier1_v2_ultra_structured/qwen_tier1_v2.ipynb)

```python
def predict_single_v2(model, tokenizer, prompt, row_id):
    """V2: 3-stage robust parsing"""

    # Generate text
    inputs = tokenizer(prompt, return_tensors="pt", max_length=CFG.MAX_LENGTH, truncation=True).to(CFG.DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CFG.MAX_NEW_TOKENS,  # 10 tokens max
            temperature=CFG.TEMPERATURE,  # 0.01 for deterministic
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Stage 1: Direct float parsing
    try:
        prob = float(text)
        return max(0.0, min(1.0, prob))
    except:
        pass

    # Stage 2: Regex extraction
    matches = re.findall(r'\b(0\.\d+|1\.0+)\b', text)
    if matches:
        try:
            prob = float(matches[0])
            return max(0.0, min(1.0, prob))
        except:
            pass

    # Stage 3: Keyword-based fallback
    lower = text.lower()
    if any(w in lower for w in ['yes', 'violate', 'spam']):
        return 0.8
    if any(w in lower for w in ['no', 'not', 'fine']):
        return 0.2

    print(f"⚠️ Row {row_id}: '{text[:50]}' -> 0.5")
    return 0.5
```

**Stage 1: Direct Float Parsing**

```python
float("0.85") -> 0.85  ✅
float("1.0") -> 1.0    ✅
float("0.8 is my answer") -> ValueError  ❌
```

Best case: Model outputs clean number. ~70-80% of cases in well-prompted models.

**Stage 2: Regex Extraction**

```python
pattern = r'\b(0\.\d+|1\.0+)\b'

"The answer is 0.75" -> ["0.75"]  ✅
"I'd say 0.5 or maybe 0.6" -> ["0.5", "0.6"]  # Take first
"Approximately 0.80" -> ["0.80"]  ✅
```

Handles verbose outputs. ~15-20% of cases.

**Stage 3: Keyword Fallback**

```python
"Yes, this violates the rule" -> 0.8  ✅
"No, this is fine" -> 0.2  ✅
"I'm not sure" -> 0.5  ✅
```

Last resort for completely non-numerical outputs. ~5-10% of cases.

**Results on v2:**
- Stage 1: 100% (all outputs were clean "0.0")
- Stage 2: 0% (never reached)
- Stage 3: 0% (never reached)

This was actually a **red flag** - too perfect means model isn't actually reasoning.

---

## Why Inference-Only?

### The NumPy Compatibility Issue

In previous attempts (not shown in this portfolio), trying to train Qwen led to:

```python
ImportError: numpy.core.multiarray failed to import
RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xf
```

**Root cause:**
- Kaggle environment: NumPy 2.0+ (new C API)
- Training libraries (vllm, deepspeed): Built against NumPy 1.x (old C API)
- Binary incompatibility → crashes

**Solutions considered:**

1. **Downgrade NumPy** → Breaks other dependencies
2. **Rebuild libraries** → No permissions on Kaggle
3. **Use older environment** → May not have latest transformers
4. **Inference-only** → ✅ Avoids training libraries entirely

### Inference-Only Benefits

1. **Faster iteration:** No training time (2 hours vs. 6-8 hours)
2. **Simpler environment:** Fewer dependencies
3. **Reproducible:** Public adapters are fixed (training has randomness)
4. **Resource efficient:** No GPU time spent training

### Trade-offs

**Advantages:**
- ✅ Bypass environment issues
- ✅ Quick experimentation (2-hour cycles)
- ✅ Leverage community work (public adapters)

**Disadvantages:**
- ❌ Limited to public adapters (can't customize for this competition's data)
- ❌ Adapter quality unknown (no validation)
- ❌ Compatibility issues (4B vs 1.5B problem)

**Decision:** Worth trying as a "high-upside, low-cost" approach before competition ends.

---

## Alternative Approaches Considered

### Approach 1: Full Qwen Fine-Tuning

**Setup:**
```python
from transformers import TrainingArguments
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)

# Train on competition training data
trainer.train()
```

**Why not pursued:**
- ⏱️ Estimated 6-8 hours training time
- 🐛 NumPy compatibility blockers
- ⚠️ Risk of overfitting (small dataset)

**If I had done this:** Likely would have achieved 0.91-0.915 LB.

---

### Approach 2: DeBERTa Ensemble

**Setup:**
```python
models = [
    "microsoft/deberta-v3-base",  # 0.904
    "microsoft/deberta-v3-large",  # Estimated 0.91
    "roberta-large",  # Estimated 0.895
]

# Rank averaging ensemble
predictions_ensemble = rank_average([
    model1.predict(test),
    model2.predict(test),
    model3.predict(test)
])
```

**Expected LB:** 0.91-0.915 (safe bronze medal range)

**Why not pursued:**
- ⏱️ Needed 4-6 hours to train multiple models
- 💰 Kaggle GPU quota (9 hours/week limit)
- 🎯 Qwen approach had higher ceiling (0.92+ if working)

**Strategic decision:** Attempt high-upside Qwen first, fall back to ensemble if time.

---

### Approach 3: Test-Time Training (TTT)

**Setup:**
```python
# Legal in this competition: train on test data
train_on_test = pd.concat([train_df, test_df])

# Self-training loop
for iteration in range(5):
    # Train model
    model.fit(train_on_test)

    # Predict on test
    pseudo_labels = model.predict(test)

    # Add confident predictions to training data
    confident = pseudo_labels[(pseudo_labels < 0.2) | (pseudo_labels > 0.8)]
    train_on_test = pd.concat([train_on_test, confident])
```

**Expected LB:** 0.925-0.933 (top solutions used this)

**Why not pursued:**
- ⏱️ Requires 12+ hours (multiple training iterations)
- 🎲 High variance (can collapse to degenerate states)
- 📊 Needs careful validation (avoid overfitting to leaderboard)

**If I had unlimited time:** This would be the approach to try.

---

## Technical Stack

### Core Libraries

```
transformers==4.44.2  # HuggingFace models
peft==0.12.0  # LoRA adapters
torch==2.4.0  # PyTorch
accelerate==0.33.0  # Multi-GPU, mixed precision
```

### Why These Versions?

- **transformers 4.44.2:** Latest stable with Qwen 2.5 support
- **peft 0.12.0:** LoRA implementation, compatible with transformers 4.44
- **torch 2.4.0:** Kaggle default, CUDA 11.8 compatibility
- **accelerate 0.33.0:** Required for `device_map="auto"`

### Environment Constraints

**Kaggle Notebook Limits:**
- GPU: Tesla P100 (16GB VRAM) or T4 (16GB VRAM)
- RAM: 30GB
- Disk: 73GB
- Internet: Allowed (can download models)
- Time: 9 hours/week GPU quota

**Impact on approach:**
- Must use FP16 (half precision) to fit in 16GB VRAM
- Can't train multiple large models (GPU quota)
- Need efficient inference (1-2 sec/sample max)

---

## Lessons for Future Competitions

### 1. Validate Public Resources Early

```python
# Before spending 2 hours on full inference:
def quick_validation(adapter_path):
    # Test on 3 samples
    samples = test_df.head(3)
    outputs = [predict(s) for s in samples]

    # Check diversity
    if np.std(outputs) < 0.01:
        raise ValueError(f"Degenerate outputs: {outputs}")

    return True  # Proceed with full inference
```

**Time saved:** 2-4 hours

---

### 2. Monitor Output Distributions

```python
# After inference:
print("Output statistics:")
print(f"Mean: {predictions.mean():.4f}")
print(f"Std: {predictions.std():.4f}")
print(f"Min: {predictions.min():.4f}")
print(f"Max: {predictions.max():.4f}")
print(f"Unique values: {predictions.nunique()}")

if predictions.nunique() < 5:
    print("⚠️ WARNING: Too few unique predictions!")
```

**Catches:** Degenerate models, calibration issues, constant predictions

---

### 3. Time-Box Experiments

```python
# Don't do this:
while not satisfied:
    tweak_hyperparameters()
    train()  # 6 hours
    evaluate()
    # ... endless loop

# Do this:
MAX_ATTEMPTS = 3
TIME_PER_ATTEMPT = 2  # hours

for attempt in range(MAX_ATTEMPTS):
    result = try_approach(time_limit=TIME_PER_ATTEMPT)
    if result.score > threshold:
        break
    else:
        analyze_failure()
        decide_next_step()
```

---

## Conclusion

This technical deep-dive showed:

1. **DeBERTa baseline:** Solid, reliable approach (0.904)
2. **Qwen LoRA inference:** High-risk, high-reward attempt (failed due to adapter mismatch)
3. **Prompt engineering:** Systematic improvements (v1 → v2)
4. **Parsing robustness:** 3-stage strategy achieved 100% success
5. **Strategic decisions:** Inference-only to bypass environment issues

**Key takeaway:** The technical implementation was sound. The failure was due to validation gap (not checking adapter compatibility first).

This demonstrates the importance of **assumption validation** in ML engineering.

---

**Last updated:** Oct 25, 2024
