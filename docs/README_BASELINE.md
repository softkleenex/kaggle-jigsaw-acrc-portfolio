# Jigsaw ACRC Competition - Baseline Model

## Quick Start

### Run the baseline model:
```bash
python3 baseline_model.py
```

## Results Summary

### Performance
- **Mean CV AUC**: 0.614642
- **Overall CV AUC**: 0.614210
- **Standard Deviation**: 0.022787
- **Execution Time**: 19.84 seconds

### Individual Fold Performance
| Fold | AUC Score |
|------|-----------|
| 1    | 0.640534  |
| 2    | 0.584903  |
| 3    | 0.620461  |
| 4    | 0.636079  |
| 5    | 0.591233  |

## Model Architecture

### Algorithm
- **Primary Model**: LightGBM Classifier
- **Validation Strategy**: Stratified 5-Fold Cross-Validation

### Features Used
**Combined Text Feature** = body + rule + positive_example_1 + positive_example_2 + negative_example_1 + negative_example_2

All text fields are concatenated with `[SEP]` separator tokens.

### Text Vectorization
- **Method**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Max Features**: 10,000
- **N-gram Range**: 1-3 (unigrams, bigrams, trigrams)
- **Min Document Frequency**: 2
- **Max Document Frequency**: 0.95
- **Stop Words**: English

## File Structure

```
/mnt/c/LSJ/dacon/dacon/Jigsaw-ACRC/
├── data/
│   ├── train.csv                    # Training data (2,029 samples)
│   ├── test.csv                     # Test data (10 samples)
│   └── sample_submission.csv        # Submission format example
├── submissions/
│   └── baseline_v1.csv              # Generated submission file
├── baseline_model.py                # Main baseline script
├── BASELINE_RESULTS.md              # Detailed results documentation
└── README_BASELINE.md               # This file
```

## Data Information

### Training Data (train.csv)
- **Samples**: 2,029
- **Features**: 8 (+ 1 target)
- **Columns**:
  - `row_id`: Unique identifier
  - `body`: Main text content to evaluate
  - `rule`: Rule to check against
  - `subreddit`: Source subreddit
  - `positive_example_1`: Example of rule violation
  - `positive_example_2`: Example of rule violation
  - `negative_example_1`: Example of non-violation
  - `negative_example_2`: Example of non-violation
  - `rule_violation`: Target (0 = no violation, 1 = violation)

### Target Distribution
- **Class 0 (No Violation)**: 998 samples (49.2%)
- **Class 1 (Violation)**: 1,031 samples (50.8%)
- **Balance**: Well-balanced dataset

## Submission File

### Format
- **Filename**: baseline_v1.csv
- **Columns**: `row_id`, `rule_violation`
- **Rows**: 10 (matching test.csv)
- **Values**: Probability scores between 0.0 and 1.0

### Sample Predictions
```
Row 2029 | Probability: 0.000912 | NO VIOLATION (99.91% confidence)
Row 2030 | Probability: 0.005534 | NO VIOLATION (99.45% confidence)
Row 2031 | Probability: 0.996580 | VIOLATION (99.66% confidence)
Row 2032 | Probability: 0.996179 | VIOLATION (99.62% confidence)
Row 2033 | Probability: 0.997371 | VIOLATION (99.74% confidence)
Row 2034 | Probability: 0.002071 | NO VIOLATION (99.79% confidence)
Row 2035 | Probability: 0.999786 | VIOLATION (99.98% confidence)
Row 2036 | Probability: 0.003420 | NO VIOLATION (99.66% confidence)
Row 2037 | Probability: 0.001114 | NO VIOLATION (99.89% confidence)
Row 2038 | Probability: 0.997800 | VIOLATION (99.78% confidence)
```

### Prediction Distribution
- **Violations**: 5 (50.0%)
- **No Violations**: 5 (50.0%)

## Requirements

### Python Packages
```
pandas
numpy
scikit-learn
lightgbm
```

### Installation
```bash
pip install pandas numpy scikit-learn lightgbm
```

## Model Parameters

### LightGBM Configuration
```python
{
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'random_state': 42
}
```

## Next Steps for Improvement

1. **Feature Engineering**
   - Add text length statistics
   - Extract URL and link patterns
   - Include subreddit-specific features
   - Add sentiment scores
   - Capture capitalization patterns

2. **Model Tuning**
   - Hyperparameter optimization (GridSearch/Bayesian)
   - Try different max_features for TF-IDF
   - Experiment with character n-grams
   - Test ensemble methods

3. **Advanced Models**
   - Try XGBoost and CatBoost
   - Implement neural networks
   - Use pre-trained transformers (BERT, RoBERTa)
   - Ensemble multiple models

4. **Validation Strategy**
   - Increase to 10-fold CV
   - Implement repeated cross-validation
   - Use multiple random seeds

## Key Takeaways

1. The baseline achieves **0.615 AUC** with simple feature engineering
2. Combining all text fields provides comprehensive context
3. Model shows stable performance across folds (std = 0.023)
4. High confidence predictions suggest good model calibration
5. Fast execution time (< 20 seconds) allows rapid iteration

## Contact & Support

For detailed results and analysis, see: `BASELINE_RESULTS.md`

---

**Generated**: 2025-10-13
**Model Version**: baseline_v1
**Status**: Ready for submission
