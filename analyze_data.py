"""
Jigsaw ACRC - Data Analysis and Visualization Script
=====================================================

This script generates comprehensive statistics and visualizations
from the deep data analysis.

Usage:
    python analyze_data.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def jaccard_similarity(str1, str2):
    """Calculate Jaccard similarity"""
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    if len(set1) == 0 and len(set2) == 0:
        return 0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def generate_statistics():
    """Generate key statistics"""
    print("\n" + "="*70)
    print("JIGSAW ACRC - KEY STATISTICS")
    print("="*70)

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Basic stats
    print("\n### DATASET OVERVIEW ###")
    print(f"Training samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    print(f"Violation rate: {train['rule_violation'].mean():.2%}")
    print(f"Unique subreddits: {train['subreddit'].nunique()}")
    print(f"Unique rules: {train['rule'].nunique()}")

    # Length statistics
    print("\n### TEXT LENGTH STATISTICS ###")
    print(f"\nBody Length:")
    print(f"  Violation:     {train[train['rule_violation']==1]['body'].str.len().mean():.1f} chars")
    print(f"  Non-violation: {train[train['rule_violation']==0]['body'].str.len().mean():.1f} chars")
    print(f"  Difference:    {train[train['rule_violation']==1]['body'].str.len().mean() - train[train['rule_violation']==0]['body'].str.len().mean():.1f} chars (+{(train[train['rule_violation']==1]['body'].str.len().mean() / train[train['rule_violation']==0]['body'].str.len().mean() - 1) * 100:.1f}%)")

    # Few-shot similarity
    print("\n### FEW-SHOT SIMILARITY (Jaccard) ###")
    train['avg_pos_sim'] = train.apply(lambda x: (jaccard_similarity(x['body'], x['positive_example_1']) + jaccard_similarity(x['body'], x['positive_example_2'])) / 2, axis=1)
    train['avg_neg_sim'] = train.apply(lambda x: (jaccard_similarity(x['body'], x['negative_example_1']) + jaccard_similarity(x['body'], x['negative_example_2'])) / 2, axis=1)

    print(f"\nBody-Positive Similarity:")
    print(f"  Violation:     {train[train['rule_violation']==1]['avg_pos_sim'].mean():.4f}")
    print(f"  Non-violation: {train[train['rule_violation']==0]['avg_pos_sim'].mean():.4f}")
    print(f"  Lift:          {train[train['rule_violation']==1]['avg_pos_sim'].mean() / train[train['rule_violation']==0]['avg_pos_sim'].mean():.2f}x (+{(train[train['rule_violation']==1]['avg_pos_sim'].mean() / train[train['rule_violation']==0]['avg_pos_sim'].mean() - 1) * 100:.1f}%)")

    # Top discriminative words
    print("\n### TOP DISCRIMINATIVE KEYWORDS ###")

    violation_words = []
    non_violation_words = []

    for text in train[train['rule_violation']==1]['body']:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        violation_words.extend(words)

    for text in train[train['rule_violation']==0]['body']:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        non_violation_words.extend(words)

    viol_counter = Counter(violation_words)
    non_viol_counter = Counter(non_violation_words)

    word_lift = []
    for word in set(list(viol_counter.keys()) + list(non_viol_counter.keys())):
        if viol_counter[word] >= 10:
            viol_freq = viol_counter[word] / len(violation_words)
            non_viol_freq = non_viol_counter[word] / len(non_violation_words)
            if non_viol_freq > 0:
                lift = viol_freq / non_viol_freq
                word_lift.append((word, lift, viol_counter[word], non_viol_counter[word]))

    word_lift_sorted = sorted(word_lift, key=lambda x: x[1], reverse=True)

    print("\nTop 10 Violation Indicators:")
    for word, lift, v_count, nv_count in word_lift_sorted[:10]:
        print(f"  {word:15s} - {lift:5.2f}x (V:{v_count:4d}, NV:{nv_count:4d})")

    print("\nTop 10 Non-Violation Indicators:")
    word_lift_sorted_reverse = sorted(word_lift, key=lambda x: x[1])
    for word, lift, v_count, nv_count in word_lift_sorted_reverse[:10]:
        print(f"  {word:15s} - {lift:5.2f}x (V:{v_count:4d}, NV:{nv_count:4d})")

    # Subreddit risk
    print("\n### SUBREDDIT RISK ###")
    subreddit_stats = train.groupby('subreddit')['rule_violation'].agg(['mean', 'count'])
    subreddit_stats = subreddit_stats[subreddit_stats['count'] >= 10].sort_values('mean', ascending=False)

    print("\nTop 10 Riskiest Subreddits (>=10 samples):")
    for subreddit, row in subreddit_stats.head(10).iterrows():
        print(f"  {subreddit:25s} - {row['mean']:.1%} ({int(row['count'])} samples)")

    print("\nTop 10 Safest Subreddits (>=10 samples):")
    for subreddit, row in subreddit_stats.tail(10).iterrows():
        print(f"  {subreddit:25s} - {row['mean']:.1%} ({int(row['count'])} samples)")

    print("\n" + "="*70 + "\n")


def plot_similarity_distributions():
    """Plot similarity distributions"""
    print("Generating similarity distribution plots...")

    train = pd.read_csv('data/train.csv')

    # Calculate similarities
    train['avg_pos_sim'] = train.apply(lambda x: (jaccard_similarity(x['body'], x['positive_example_1']) + jaccard_similarity(x['body'], x['positive_example_2'])) / 2, axis=1)
    train['avg_neg_sim'] = train.apply(lambda x: (jaccard_similarity(x['body'], x['negative_example_1']) + jaccard_similarity(x['body'], x['negative_example_2'])) / 2, axis=1)
    train['pos_neg_diff'] = train['avg_pos_sim'] - train['avg_neg_sim']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Positive similarity
    axes[0, 0].hist(train[train['rule_violation']==1]['avg_pos_sim'], bins=50, alpha=0.6, label='Violation', color='red')
    axes[0, 0].hist(train[train['rule_violation']==0]['avg_pos_sim'], bins=50, alpha=0.6, label='Non-violation', color='green')
    axes[0, 0].set_xlabel('Jaccard Similarity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Body-Positive Example Similarity')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Negative similarity
    axes[0, 1].hist(train[train['rule_violation']==1]['avg_neg_sim'], bins=50, alpha=0.6, label='Violation', color='red')
    axes[0, 1].hist(train[train['rule_violation']==0]['avg_neg_sim'], bins=50, alpha=0.6, label='Non-violation', color='green')
    axes[0, 1].set_xlabel('Jaccard Similarity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Body-Negative Example Similarity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Difference
    axes[1, 0].hist(train[train['rule_violation']==1]['pos_neg_diff'], bins=50, alpha=0.6, label='Violation', color='red')
    axes[1, 0].hist(train[train['rule_violation']==0]['pos_neg_diff'], bins=50, alpha=0.6, label='Non-violation', color='green')
    axes[1, 0].set_xlabel('Similarity Difference (Pos - Neg)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Positive-Negative Similarity Difference')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Box plot comparison
    data_to_plot = [
        train[train['rule_violation']==1]['avg_pos_sim'].values,
        train[train['rule_violation']==0]['avg_pos_sim'].values,
        train[train['rule_violation']==1]['avg_neg_sim'].values,
        train[train['rule_violation']==0]['avg_neg_sim'].values
    ]
    axes[1, 1].boxplot(data_to_plot, labels=['Viol-Pos', 'Non-Pos', 'Viol-Neg', 'Non-Neg'])
    axes[1, 1].set_ylabel('Jaccard Similarity')
    axes[1, 1].set_title('Similarity Comparison')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('similarity_distributions.png', dpi=150, bbox_inches='tight')
    print(f"Saved: similarity_distributions.png")


def plot_subreddit_risk():
    """Plot subreddit risk heatmap"""
    print("Generating subreddit risk visualization...")

    train = pd.read_csv('data/train.csv')

    # Calculate risk by subreddit and rule
    subreddit_rule_stats = train.groupby(['subreddit', 'rule'])['rule_violation'].agg(['mean', 'count']).reset_index()
    subreddit_rule_stats = subreddit_rule_stats[subreddit_rule_stats['count'] >= 5]  # Filter for clarity

    # Pivot for heatmap
    risk_matrix = subreddit_rule_stats.pivot_table(values='mean', index='subreddit', columns='rule', fill_value=0)

    # Get top 30 subreddits by count
    top_subreddits = train['subreddit'].value_counts().head(30).index
    risk_matrix = risk_matrix.loc[risk_matrix.index.isin(top_subreddits)]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(risk_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0.5,
                cbar_kws={'label': 'Violation Rate'}, linewidths=0.5, ax=ax)
    ax.set_title('Violation Rate by Subreddit and Rule Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Rule', fontsize=12)
    ax.set_ylabel('Subreddit', fontsize=12)

    plt.tight_layout()
    plt.savefig('subreddit_risk_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Saved: subreddit_risk_heatmap.png")


def plot_length_distributions():
    """Plot text length distributions"""
    print("Generating length distribution plots...")

    train = pd.read_csv('data/train.csv')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Body length
    axes[0, 0].hist(train[train['rule_violation']==1]['body'].str.len(), bins=50, alpha=0.6, label='Violation', color='red')
    axes[0, 0].hist(train[train['rule_violation']==0]['body'].str.len(), bins=50, alpha=0.6, label='Non-violation', color='green')
    axes[0, 0].set_xlabel('Body Length (chars)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Body Length Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Word count
    train['body_words'] = train['body'].str.split().str.len()
    axes[0, 1].hist(train[train['rule_violation']==1]['body_words'], bins=50, alpha=0.6, label='Violation', color='red')
    axes[0, 1].hist(train[train['rule_violation']==0]['body_words'], bins=50, alpha=0.6, label='Non-violation', color='green')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Body Word Count Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Subreddit distribution (top 20)
    subreddit_counts = train['subreddit'].value_counts().head(20)
    axes[1, 0].barh(range(len(subreddit_counts)), subreddit_counts.values)
    axes[1, 0].set_yticks(range(len(subreddit_counts)))
    axes[1, 0].set_yticklabels(subreddit_counts.index)
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_title('Top 20 Subreddits by Sample Count')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # Rule distribution
    rule_stats = train.groupby('rule')['rule_violation'].agg(['count', 'mean'])
    axes[1, 1].bar(range(len(rule_stats)), rule_stats['count'], alpha=0.7, label='Total Samples')
    ax2 = axes[1, 1].twinx()
    ax2.plot(range(len(rule_stats)), rule_stats['mean'], 'ro-', linewidth=2, markersize=10, label='Violation Rate')
    axes[1, 1].set_xticks(range(len(rule_stats)))
    axes[1, 1].set_xticklabels(['Advertising', 'Legal Advice'], rotation=0)
    axes[1, 1].set_ylabel('Sample Count')
    ax2.set_ylabel('Violation Rate')
    axes[1, 1].set_title('Rule Distribution and Violation Rates')
    axes[1, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('length_distributions.png', dpi=150, bbox_inches='tight')
    print(f"Saved: length_distributions.png")


def plot_keyword_analysis():
    """Plot keyword lift analysis"""
    print("Generating keyword analysis plot...")

    train = pd.read_csv('data/train.csv')

    # Get word frequencies
    violation_words = []
    non_violation_words = []

    for text in train[train['rule_violation']==1]['body']:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        violation_words.extend(words)

    for text in train[train['rule_violation']==0]['body']:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        non_violation_words.extend(words)

    viol_counter = Counter(violation_words)
    non_viol_counter = Counter(non_violation_words)

    # Calculate lift for keywords
    keywords = ['lawyer', 'sue', 'police', 'legal', 'stream', 'watch', 'html', 'free', 'click', 'buy']
    lifts = []
    for kw in keywords:
        viol_freq = viol_counter[kw] / len(violation_words)
        non_viol_freq = non_viol_counter[kw] / len(non_violation_words)
        if non_viol_freq > 0:
            lift = viol_freq / non_viol_freq
            lifts.append((kw, lift))
        else:
            lifts.append((kw, 0))

    lifts_sorted = sorted(lifts, key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red' if lift > 1 else 'green' for _, lift in lifts_sorted]
    ax.barh(range(len(lifts_sorted)), [lift for _, lift in lifts_sorted], color=colors, alpha=0.7)
    ax.set_yticks(range(len(lifts_sorted)))
    ax.set_yticklabels([kw for kw, _ in lifts_sorted])
    ax.set_xlabel('Lift (Viol / Non-Viol)', fontsize=12)
    ax.set_title('Keyword Lift Analysis', fontsize=14, fontweight='bold')
    ax.axvline(x=1, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Neutral (1x)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()

    for i, (kw, lift) in enumerate(lifts_sorted):
        ax.text(lift + 0.3, i, f'{lift:.2f}x', va='center')

    plt.tight_layout()
    plt.savefig('keyword_lift_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: keyword_lift_analysis.png")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("JIGSAW ACRC - DATA ANALYSIS & VISUALIZATION")
    print("="*70)

    # Generate statistics
    generate_statistics()

    # Generate plots
    print("\nGenerating visualizations...")
    plot_similarity_distributions()
    plot_subreddit_risk()
    plot_length_distributions()
    plot_keyword_analysis()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - similarity_distributions.png")
    print("  - subreddit_risk_heatmap.png")
    print("  - length_distributions.png")
    print("  - keyword_lift_analysis.png")
    print("\nFor detailed analysis, see: DEEP_DATA_ANALYSIS_REPORT.md")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
