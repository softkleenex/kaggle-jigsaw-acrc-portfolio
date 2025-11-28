# Notebooks Directory

This directory contains all implementation attempts from the competition.

## Structure

```
notebooks/
└── tier1_v2_ultra_structured/
    ├── qwen_tier1_v2.ipynb          # Main inference notebook (v2)
    ├── kernel-metadata.json          # Kaggle kernel configuration
    └── tier1_v2_output/
        └── submission.csv            # Output file (all predictions = 0.0)
```

## Notebooks Overview

### Tier 1 v2: Ultra-Structured Prompt

**File:** `tier1_v2_ultra_structured/qwen_tier1_v2.ipynb`

**Approach:**
- Base model: Qwen 2.5 1.5B-Instruct
- Adapter: mahmoudmohamed/reddit-4b-think (public LoRA)
- Prompt: Ultra-structured with chat template
- Parsing: 3-stage robust strategy

**Key Features:**
1. System/user message separation
2. Few-shot numerical examples (0.95, 0.23, 0.78...)
3. Temperature: 0.01 (deterministic)
4. 3-stage parsing (direct float → regex → keyword fallback)

**Results:**
- Parse success: **100%**
- Predictions: All 0.0 ❌
- Root cause: Base model mismatch (see [FAILURE_ANALYSIS.md](../FAILURE_ANALYSIS.md))

**Learning:**
- 100% parsing success ≠ correct model behavior
- Always validate adapters on small test set first
- Cross-reference adapter config with dataset name

---

## Tier 1 v1 (Not Preserved)

**Status:** Notebook not saved to permanent location (was in `/tmp/`)

**Approach:**
- Same base model and adapter as v2
- Simpler prompt (no chat template)
- Basic parsing

**Results:**
- Parse success: ~50%
- Predictions: All 0.0 ❌

**Why not preserved:**
- v2 was strict improvement over v1
- `/tmp/` directory cleared between sessions
- v2 demonstrates same failure with better engineering

---

## How to Run

### Prerequisites

```bash
pip install -r ../PORTFOLIO_REQUIREMENTS.txt
```

### Kaggle Environment

1. **Upload notebook to Kaggle:**
   ```bash
   kaggle kernels push -p tier1_v2_ultra_structured/
   ```

2. **Add datasets:**
   - Competition: `jigsaw-agile-community-rules`
   - Adapter: `softkleenex/mahmoudmohamed-lora-adapter`

3. **Configure kernel:**
   - GPU: ON (Tesla P100 or T4)
   - Internet: ON (to download base model)

4. **Run and wait:** ~15-20 minutes for full inference

### Local Environment

**Note:** Requires GPU with 12GB+ VRAM for FP16 inference.

```bash
cd tier1_v2_ultra_structured/
jupyter notebook qwen_tier1_v2.ipynb
```

**Limitations:**
- Need to download Qwen base model (~3GB)
- Need to download mahmoudmohamed adapter (~50MB)
- Results will match Kaggle (all 0.0) due to adapter incompatibility

---

## Expected vs. Actual Outputs

### Expected (If Adapter Was Compatible)

```csv
row_id,rule_violation
2029,0.23
2030,0.85
2031,0.05
2032,0.92
2033,0.31
...
```

**Characteristics:**
- Diverse values
- Some high (0.8-1.0) for violations
- Some low (0.0-0.2) for non-violations
- Continuous distribution

---

### Actual (With Adapter Mismatch)

```csv
row_id,rule_violation
2029,0.0
2030,0.0
2031,0.0
2032,0.0
2033,0.0
...
```

**Characteristics:**
- All identical (0.0)
- No correlation with input
- Degenerate distribution

**See:** [`tier1_v2_ultra_structured/tier1_v2_output/submission.csv`](tier1_v2_ultra_structured/tier1_v2_output/submission.csv)

---

## Code Highlights

### Prompt Engineering (v2)

**Location:** `qwen_tier1_v2.ipynb` cell 4

```python
def create_prompt_v2(tokenizer, row):
    system = "You are a precise AI. Respond ONLY with a decimal number between 0.0 and 1.0."

    user = f"""Analyze if this post violates the rule.

RULE: {row['rule']}
POST: {row['body']}

Examples: 0.95, 0.23, 0.78, 0.02

Your answer (number only):"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

**Key decisions:**
- Clear system role: "ONLY a number"
- Few-shot examples showing exact format
- Uses Qwen's chat template for proper formatting

---

### 3-Stage Parsing

**Location:** `qwen_tier1_v2.ipynb` cell 5

```python
def predict_single_v2(model, tokenizer, prompt, row_id):
    # ... generation code ...

    # Stage 1: Direct float
    try:
        prob = float(text)
        return max(0.0, min(1.0, prob))
    except:
        pass

    # Stage 2: Regex
    matches = re.findall(r'\b(0\.\d+|1\.0+)\b', text)
    if matches:
        return max(0.0, min(1.0, float(matches[0])))

    # Stage 3: Keywords
    if any(w in text.lower() for w in ['yes', 'violate']):
        return 0.8
    return 0.5
```

**Result:** 100% parsing success (all Stage 1)

---

## Further Reading

- **[FAILURE_ANALYSIS.md](../FAILURE_ANALYSIS.md)** - Deep dive into why this approach failed
- **[TECHNICAL_DEEP_DIVE.md](../TECHNICAL_DEEP_DIVE.md)** - Detailed implementation explanations
- **[../configs/](../configs/)** - Adapter configuration comparisons

---

**Last updated:** Oct 25, 2024
**Author:** LSJ
