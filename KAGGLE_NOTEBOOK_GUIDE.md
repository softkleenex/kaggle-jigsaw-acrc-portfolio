# Kaggle SetFit Submission Notebook Guide

## Overview

This guide explains how to use the `kaggle_setfit_submission.ipynb` notebook for the Jigsaw ACRC competition.

## File Information

- **Filename**: `kaggle_setfit_submission.ipynb`
- **Location**: `/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/`
- **Size**: ~23KB
- **Format**: Jupyter Notebook compatible with Kaggle

## Expected Performance

Based on local validation:
- **CV AUC**: 0.776110 (±0.014379)
- **Runtime**: ~5 minutes on Kaggle (with GPU acceleration)
- **Improvement over baseline**: +26.4%

## How to Use on Kaggle

### Step 1: Upload to Kaggle

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Click "File" → "Upload Notebook"
4. Select `kaggle_setfit_submission.ipynb`
5. Wait for upload to complete

### Step 2: Add Competition Data

1. In the Kaggle notebook, click "Add data" (right sidebar)
2. Search for "Jigsaw Agile Community Rules Classification"
3. Add the competition dataset
4. The data will be available at `/kaggle/input/jigsaw-agile-community-rules-classification/`

### Step 3: Configure Settings

**Important Settings**:
- **Accelerator**: GPU (optional, but recommended for faster embedding generation)
- **Internet**: ON (required for installing sentence-transformers)
- **Persistence**: OFF (not needed for this notebook)

### Step 4: Run the Notebook

1. Click "Run All" or execute cells sequentially
2. Monitor progress in each cell
3. Total runtime: ~5 minutes

### Step 5: Submit

1. After successful run, find `submission.csv` in the output
2. Click "Submit to Competition"
3. Check your score on the leaderboard

## Notebook Structure

### Section 1: Setup & Installation
- Installs `sentence-transformers` package
- ~30 seconds

### Section 2: Import Libraries
- Imports all required libraries
- Validates installations

### Section 3: Load Data
- Loads train.csv, test.csv, sample_submission.csv
- Displays basic statistics

### Section 4: Initialize Model
- Loads `all-MiniLM-L6-v2` sentence transformer
- 384-dimensional embeddings

### Section 5: Text Preprocessing
- Creates formatted text inputs
- Combines rule + body/examples
- ~5 seconds

### Section 6: Generate Embeddings
- Most time-consuming step (~3-4 minutes)
- Generates embeddings for:
  - Main texts (body + rule)
  - Positive examples (2 per sample)
  - Negative examples (2 per sample)

### Section 7: Compute Similarity Features
- Creates 9 similarity features
- Measures body similarity with examples
- ~30 seconds

### Section 8: Combine Features
- Combines 384 embeddings + 9 similarity features
- Total: 393 features

### Section 9: Cross-Validation
- Stratified 5-Fold CV
- Logistic Regression classifier
- Displays fold-wise and overall AUC
- ~1 minute

### Section 10: Generate Submission
- Creates submission.csv
- Validates output format

### Section 11: Summary
- Final statistics and recommendations

## Key Features

### 1. Few-Shot Learning Approach
The model leverages positive and negative examples through similarity features:
- **Positive examples**: Comments that violate the rule
- **Negative examples**: Comments that don't violate the rule
- **Similarity computation**: Cosine similarity between body and examples

### 2. Nine Similarity Features

1. `sim_pos1`: Similarity with positive example 1
2. `sim_pos2`: Similarity with positive example 2
3. `sim_neg1`: Similarity with negative example 1
4. `sim_neg2`: Similarity with negative example 2
5. `avg_pos_sim`: Average positive similarity
6. `avg_neg_sim`: Average negative similarity
7. `max_pos_sim`: Maximum positive similarity
8. `min_neg_sim`: Minimum negative similarity
9. `diff_sim`: Difference (avg_pos - avg_neg)

### 3. Model Architecture

```
Input Text → Sentence Transformer → 384-dim Embedding
                                           ↓
                                    [Embedding + 9 Similarity Features]
                                           ↓
                                    Logistic Regression
                                           ↓
                                    Probability [0, 1]
```

## Troubleshooting

### Issue 1: Internet Access Required
**Error**: Cannot install sentence-transformers
**Solution**: Enable "Internet" in notebook settings

### Issue 2: Memory Error
**Error**: Out of memory during embedding generation
**Solution**:
- Enable GPU accelerator
- Or reduce batch_size in `get_embeddings()` function

### Issue 3: Data Path Error
**Error**: FileNotFoundError for data files
**Solution**: Ensure competition data is added (see Step 2)

### Issue 4: Slow Execution
**Optimization**:
- Enable GPU accelerator (Settings → Accelerator → GPU)
- Expected speedup: 2-3x faster embedding generation

## Expected Output

### Cross-Validation Results
```
Fold 1/5 AUC: ~0.794
Fold 2/5 AUC: ~0.757
Fold 3/5 AUC: ~0.778
Fold 4/5 AUC: ~0.789
Fold 5/5 AUC: ~0.762

Overall CV AUC: ~0.776
Mean CV AUC: ~0.776 (± 0.014)
```

### Submission Statistics
```
Shape: (9000, 2)
Min prediction: ~0.05
Max prediction: ~0.95
Mean prediction: ~0.30 (class imbalance)
```

## Next Steps After First Submission

### Immediate Improvements (Easy)
1. **Tune LogisticRegression hyperparameters**:
   - Try different C values: [0.1, 0.5, 1.0, 2.0, 5.0]
   - Experiment with different solvers: ['lbfgs', 'saga', 'liblinear']

2. **Add text features**:
   - Text length (character count, word count)
   - Punctuation features
   - Capital letter ratio

3. **Feature engineering**:
   - Subreddit-rule interaction features
   - Historical violation rates per subreddit

### Advanced Improvements (Medium)
1. **Better embeddings**:
   - Use larger model: `all-mpnet-base-v2` (768-dim)
   - Use domain-specific model: `all-roberta-large-v1`

2. **Ensemble methods**:
   - Combine SetFit with TF-IDF + LightGBM
   - Stack multiple sentence transformer models

3. **Model upgrades**:
   - Replace LogisticRegression with XGBoost/LightGBM
   - Try neural network classifier on top of embeddings

### Expert Improvements (Hard)
1. **Fine-tune sentence transformer**:
   - Use contrastive learning on competition data
   - Create triplets: (body, positive_example, negative_example)

2. **BERT fine-tuning**:
   - Fine-tune RoBERTa or DeBERTa on the task
   - Multi-task learning with rule classification

3. **Advanced ensembling**:
   - Weighted average based on CV performance
   - Stacking with meta-model

## Model Comparison

| Model | CV AUC | Runtime | Difficulty |
|-------|--------|---------|------------|
| **SetFit (This)** | **0.776** | 5 min | Easy |
| Baseline (TF-IDF) | 0.614 | 20 sec | Very Easy |
| BERT Fine-tuned | 0.80+ | 30 min | Hard |

## Files Generated

After running the notebook:
1. `submission.csv` - Main submission file (required)
2. Console output - CV scores and statistics

## Code Competition Compatibility

This notebook is designed for **Code Competition** format:
- All dependencies installable via pip
- No external data sources (except competition data)
- Runs within Kaggle time limits (~9 hours)
- Reproducible with random seed

## Additional Resources

### Documentation
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Competition Discussion Forum](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion)

### Related Papers
- SetFit: Efficient Few-Shot Learning Without Prompts (2022)
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (2019)

## Support

For issues or questions:
1. Check Kaggle notebook comments
2. Post in competition discussion forum
3. Review local validation results in `MODEL_RESULTS.md`

## License

This code is provided for the Jigsaw ACRC competition on Kaggle.

---

**Last Updated**: 2025-10-13
**Version**: 1.0
**Author**: Competition participant
**Status**: Ready for submission
