# 🔍 Kaggle Jigsaw ACRC: Production ML & Debugging Case Study

<div align="center">

> **"Beyond the Leaderboard: A Deep Dive into LoRA Compatibility & Systemic Debugging"**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.44-FFD21E?style=for-the-badge)](https://huggingface.co/docs/transformers)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)

**🏆 Final Rank: 1,121 / 2,444 (Top 46%)** | **📊 Score: 0.904 ROC-AUC**

[📖 Read the Full Failure Analysis](docs/FAILURE_ANALYSIS.md) • [🇰🇷 한국어 버전](README_KR.md)

</div>

---

## 🎯 Executive Summary
This repository contains the code and post-mortem analysis for the **Jigsaw Agile Community Rules Classification** competition. Instead of merely showcasing a final model, this portfolio emphasizes **systematic problem-solving** and **production ML debugging**—specifically, diagnosing a silent failure caused by a LoRA adapter base-model mismatch.

### 📊 Competition Results
* **Final Rank:** 1,121 out of 2,444 teams (Top 46%)
* **Final Score:** 0.904 ROC-AUC
* **Gap to Bronze Medal:** +0.016 ROC-AUC (Bronze cut-off: ~0.920)
* **Gap to 1st Place:** +0.029 ROC-AUC (1st Place: 0.933)

## 🧩 The Challenge
The competition required building an NLP model to perform **Few-Shot Binary Classification**.
Given a Reddit comment and a specific subreddit rule, the model had to determine if the comment violated the rule. Each sample provided two positive and two negative examples.
* **Target Distribution:** 50.8% Violations, 49.2% No Violations (Highly Balanced).

## 💡 Core Achievement: Systemic Debugging
While experimenting with Qwen 2.5 (1.5B-Instruct), a public LoRA adapter caused the model to uniformly predict `0.0` for all instances. 

Instead of blind trial-and-error, I implemented a **Time-boxed, Hypothesis-driven Debugging Strategy**:
1. **Hypothesis A (Prompt Mismatch):** Tested by injecting ultra-structured chat templates. Result: 100% parsing success, but outputs remained `0.0`. (Hypothesis rejected).
2. **Hypothesis B (Data Imbalance):** Analyzed training distribution. Result: Data was perfectly balanced. (Hypothesis rejected).
3. **Hypothesis C (Base Model Mismatch):** Investigated the adapter config. Discovered that the adapter was trained on a **4B parameter model**, but applied to a **1.5B parameter inference kernel**.
   * *The Learning:* The `PEFT` library loaded the weights without crashing (graceful degradation), but the matrix dimensions were misaligned, leading to degenerate outputs. 

**Production Takeaway:** "Config files can lie." Automated compatibility verification is essential when loading external model weights in a CI/CD pipeline.

## 🔬 Modeling Approaches

| Model | Strategy | Result |
|-------|----------|--------|
| **LightGBM + TF-IDF** | Baseline structural combination of `[SEP]` tokens across rule and examples. | `0.614 CV` |
| **SetFit & DeBERTa-v3** | Advanced Sentence Transformer fine-tuning leveraging Few-Shot contrastive learning. | `0.904 ROC-AUC` |
| **Qwen 2.5 + LoRA** | Instruction-tuned generation. (Case study in failure analysis). | `Degenerate` |

## 📁 Repository Structure
```text
├── docs/          # Deep-dive analyses, logs, and strategy documents
├── src/           # Python training and inference scripts
├── notebooks/     # Kaggle execution notebooks
├── logs/          # Execution outputs and debugging traces
├── submissions/   # Output CSVs for Kaggle scoring
└── README.md      # This file
```

## 🚀 How to Navigate
1. **For Technical Depth:** Start with [`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md) to review the debugging methodology.
2. **For Engineering:** Review [`src/baseline_model.py`](src/baseline_model.py) for the structured TF-IDF implementation.
3. **For Infrastructure:** See [`PORTFOLIO_REQUIREMENTS.txt`](PORTFOLIO_REQUIREMENTS.txt) for the exact CUDA/GPU environment specs.
