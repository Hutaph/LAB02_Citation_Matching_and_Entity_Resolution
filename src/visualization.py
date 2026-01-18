"""
Visualization Utilities for Parsing and Matching Pipelines

This module provides visualization functions for displaying parsing statistics,
matching results, feature distributions, and data split information in a 
professional format suitable for papers and reports.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

# Set style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_distribution(
    counts: Counter,
    title: str,
    xlabel: str,
    ylabel: str = "Number of Papers",
    log_scale: bool = True,
    max_bins: int = 50
) -> None:
    """
    Plot distribution histogram of counts.
    
    Args:
        counts: Counter object with counts per paper
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Whether to use log scale for y-axis
        max_bins: Maximum number of bins
    """
    if not counts:
        print(f"No data to plot for {title}")
        return
    
    values = list(counts.values())
    bins = min(max_bins, len(set(values)))
    
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    if log_scale:
        plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_top_papers(
    counts: Counter,
    title: str,
    top_n: int = 10,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot bar chart of top N papers.
    
    Args:
        counts: Counter object with counts per paper
        title: Plot title
        top_n: Number of top papers to show
        figsize: Figure size (width, height)
    """
    if not counts:
        print(f"No data to plot for {title}")
        return
    
    top_items = counts.most_common(top_n)
    papers = [item[0] for item in top_items]
    values = [item[1] for item in top_items]
    
    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(papers)), values, color='steelblue', edgecolor='black')
    plt.yticks(range(len(papers)), papers)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Paper ID', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(val, i, f' {val:,}', va='center', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_missing_data(
    total_papers: int,
    zero_bib: int,
    zero_ref: int,
    title: str = "Missing Data Distribution"
) -> None:
    """
    Plot pie chart showing papers with/without bibitems and references.
    
    Args:
        total_papers: Total number of papers
        zero_bib: Number of papers with no bibitems
        zero_ref: Number of papers with no references
        title: Plot title
    """
    if total_papers == 0:
        print("No papers to visualize")
        return
    
    # Calculate papers with both
    has_bib = total_papers - zero_bib
    has_ref = total_papers - zero_ref
    has_both = total_papers - zero_bib - zero_ref + max(0, zero_bib + zero_ref - total_papers)
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bibitems pie chart
    bib_labels = ['Has Bibitems', 'No Bibitems']
    bib_sizes = [has_bib, zero_bib]
    bib_colors = ['#2ecc71', '#e74c3c']
    ax1.pie(bib_sizes, labels=bib_labels, autopct='%1.1f%%', 
            colors=bib_colors, startangle=90, textprops={'fontsize': 11})
    ax1.set_title('Bibitems Distribution', fontsize=13, fontweight='bold')
    
    # References pie chart
    ref_labels = ['Has References', 'No References']
    ref_sizes = [has_ref, zero_ref]
    ref_colors = ['#3498db', '#e74c3c']
    ax2.pie(ref_sizes, labels=ref_labels, autopct='%1.1f%%',
            colors=ref_colors, startangle=90, textprops={'fontsize': 11})
    ax2.set_title('References Distribution', fontsize=13, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_processing_summary(
    total: int,
    processed: int,
    skipped: int,
    title: str = "Processing Summary"
) -> None:
    """
    Plot bar chart showing processing success vs skipped.
    
    Args:
        total: Total number of papers
        processed: Number of successfully processed papers
        skipped: Number of skipped papers
        title: Plot title
    """
    if total == 0:
        print("No papers to visualize")
        return
    
    categories = ['Total', 'Processed', 'Skipped']
    values = [total, processed, skipped]
    colors = ['#95a5a6', '#2ecc71', '#e74c3c']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Number of Papers', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_comparison(
    bib_counts: Counter,
    ref_counts: Counter,
    title: str = "Bibitems vs References Comparison"
) -> None:
    """
    Plot side-by-side comparison of bibitems and references distributions.
    
    Args:
        bib_counts: Counter of bibitems per paper
        ref_counts: Counter of references per paper
        title: Plot title
    """
    if not bib_counts and not ref_counts:
        print("No data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bibitems distribution
    if bib_counts:
        bib_values = list(bib_counts.values())
        ax1.hist(bib_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Number of Bibitems', fontsize=11)
        ax1.set_ylabel('Number of Papers', fontsize=11)
        ax1.set_title('Bibitems Distribution', fontsize=12, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
    
    # References distribution
    if ref_counts:
        ref_values = list(ref_counts.values())
        ax2.hist(ref_values, bins=50, edgecolor='black', alpha=0.7, color='coral')
        ax2.set_xlabel('Number of References', fontsize=11)
        ax2.set_ylabel('Number of Papers', fontsize=11)
        ax2.set_title('References Distribution', fontsize=12, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_parsing_statistics(
    stats: Dict,
    processing_summary: Optional[Dict] = None,
    top_n: int = 10
) -> None:
    """
    Create comprehensive visualization dashboard for parsing statistics.
    
    Args:
        stats: Statistics dictionary from compute_statistics()
        processing_summary: Optional dict with 'total', 'processed', 'skipped'
        top_n: Number of top papers to show
    """
    if processing_summary:
        plot_processing_summary(
            total=processing_summary.get('total', 0),
            processed=processing_summary.get('processed', 0),
            skipped=processing_summary.get('skipped', 0)
        )
    
    plot_missing_data(
        total_papers=stats['total_papers'],
        zero_bib=len(stats['zero_bib']),
        zero_ref=len(stats['zero_ref'])
    )
    
    plot_comparison(
        bib_counts=stats['bib_counts'],
        ref_counts=stats['ref_counts']
    )
    
    plot_top_papers(
        counts=stats['bib_counts'],
        title=f"Top {top_n} Papers by Bibitems Count",
        top_n=top_n
    )
    
    plot_top_papers(
        counts=stats['ref_counts'],
        title=f"Top {top_n} Papers by References Count",
        top_n=top_n
    )


# ============================================================================
# Matching and Feature Extraction Visualizations
# ============================================================================


def plot_label_distribution(matches_path: Path) -> None:
    """
    Plot label distribution (positive vs negative samples).
    
    Args:
        matches_path: Path to matches_fe.jsonl
    """
    labels = Counter()
    if not matches_path.exists():
        print(f"File not found: {matches_path}")
        return
    
    with matches_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                labels[obj.get("label", 0)] += 1
            except json.JSONDecodeError:
                continue
    
    if not labels:
        print("No data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    label_names = {0: "Negative", 1: "Positive"}
    colors = ['#e74c3c', '#2ecc71']
    sizes = [labels[0], labels[1]]
    labels_list = [label_names.get(k, f"Label {k}") for k in sorted(labels.keys())]
    
    ax1.pie(sizes, labels=labels_list, autopct='%1.1f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 11})
    ax1.set_title('Label Distribution', fontsize=13, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(labels_list, sizes, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Label Counts', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    
    # Add value labels on bars
    for bar, val in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Label Distribution Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(matches_path: Path, features: Optional[List[str]] = None) -> None:
    """
    Plot distributions of matching features.
    
    Args:
        matches_path: Path to matches_fe.jsonl
        features: List of feature names to plot (default: ['score', 'levenshtein', 'jaccard', 'year_diff'])
    """
    if features is None:
        features = ['score', 'levenshtein', 'jaccard', 'year_diff']
    
    data = {feat: [] for feat in features}
    
    if not matches_path.exists():
        print(f"File not found: {matches_path}")
        return
    
    with matches_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                for feat in features:
                    val = obj.get(feat)
                    if val is not None:
                        data[feat].append(float(val))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    
    if not any(data.values()):
        print("No data to plot")
        return
    
    n_features = len([f for f in features if data[f]])
    if n_features == 0:
        return
    
    cols = min(2, n_features)
    rows = (n_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    if n_features == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, feat in enumerate(features):
        if not data[feat]:
            continue
        
        ax = axes[idx] if n_features > 1 else axes[0]
        values = np.array(data[feat])
        
        # Filter out invalid values
        if feat == 'year_diff':
            values = values[values < 100]  # Remove missing year markers
        
        if len(values) == 0:
            continue
        
        ax.hist(values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel(feat.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{feat.replace("_", " ").title()} Distribution', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_split_distribution(split_counts: Dict[str, int], target_pct: Dict[str, float] = None) -> None:
    """
    Plot train/val/test split distribution.
    
    Args:
        split_counts: Dictionary mapping split name to count
        target_pct: Optional dictionary with target percentages
    """
    if not split_counts:
        print("No split data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    splits = ['train', 'val', 'test']
    colors = ['#3498db', '#f39c12', '#e74c3c']
    sizes = [split_counts.get(s, 0) for s in splits]
    labels = [s.title() for s in splits]
    
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 11})
    ax1.set_title('Split Distribution', fontsize=13, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(labels, sizes, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Number of Rows', fontsize=12)
    ax2.set_title('Split Counts', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    
    # Add value labels
    for bar, val in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add target percentages if provided
    if target_pct:
        total = sum(sizes)
        target_text = "\n".join([f"{k.title()}: {v*100:.1f}%" for k, v in target_pct.items()])
        ax2.text(0.02, 0.98, f"Target:\n{target_text}", 
                transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Data Split Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_score_distribution_by_label(matches_path: Path) -> None:
    """
    Plot TF-IDF score distribution separated by label (positive vs negative).
    
    Args:
        matches_path: Path to matches_fe.jsonl
    """
    positive_scores = []
    negative_scores = []
    
    if not matches_path.exists():
        print(f"File not found: {matches_path}")
        return
    
    with matches_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                score = obj.get("score")
                label = obj.get("label", 0)
                if score is not None:
                    score = float(score)
                    if label == 1:
                        positive_scores.append(score)
                    else:
                        negative_scores.append(score)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    
    if not positive_scores and not negative_scores:
        print("No data to plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    if positive_scores:
        plt.hist(positive_scores, bins=50, alpha=0.7, label='Positive', 
                color='#2ecc71', edgecolor='black')
    if negative_scores:
        plt.hist(negative_scores, bins=50, alpha=0.7, label='Negative',
                color='#e74c3c', edgecolor='black')
    
    plt.xlabel('TF-IDF Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Score Distribution by Label', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_correlation(matches_path: Path, max_samples: int = 10000) -> None:
    """
    Plot correlation heatmap between features.
    
    Args:
        matches_path: Path to matches_fe.jsonl
        max_samples: Maximum number of samples to use for correlation
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required for correlation plot. Install with: pip install pandas")
        return
    
    features = ['score', 'levenshtein', 'jaccard', 'year_diff', 'year_match']
    data = {feat: [] for feat in features}
    
    if not matches_path.exists():
        print(f"File not found: {matches_path}")
        return
    
    count = 0
    with matches_path.open(encoding="utf-8") as f:
        for line in f:
            if count >= max_samples:
                break
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                row = {}
                for feat in features:
                    val = obj.get(feat)
                    if val is not None:
                        try:
                            row[feat] = float(val)
                        except (ValueError, TypeError):
                            row[feat] = None
                    else:
                        row[feat] = None
                
                # Only add row if we have at least some values
                if any(v is not None for v in row.values()):
                    for feat in features:
                        data[feat].append(row[feat])
                    count += 1
            except json.JSONDecodeError:
                continue
    
    if count == 0:
        print("No data to plot")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Compute correlation
    corr = df.corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def visualize_matching_statistics(
    matches_path: Path,
    split_counts: Optional[Dict[str, int]] = None,
    target_pct: Optional[Dict[str, float]] = None
) -> None:
    """
    Create comprehensive visualization dashboard for matching statistics.
    
    Args:
        matches_path: Path to matches_fe.jsonl
        split_counts: Optional dictionary with split counts
        target_pct: Optional dictionary with target percentages
    """
    plot_label_distribution(matches_path)
    plot_feature_distributions(matches_path)
    plot_score_distribution_by_label(matches_path)
    
    if split_counts:
        plot_split_distribution(split_counts, target_pct)
    
    plot_feature_correlation(matches_path)


# ============================================================================
# Modeling Visualizations
# ============================================================================


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    labels: List[str] = None
) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix array
        title: Plot title
        labels: Optional list of class labels
    """
    if labels is None:
        labels = ['Negative', 'Positive']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"shrink": 0.8})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_classification_report(
    report: Dict,
    title: str = "Classification Report"
) -> None:
    """
    Plot classification report as a heatmap.
    
    Args:
        report: Classification report dictionary from sklearn
        title: Plot title
    """
    # Extract metrics for each class
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics = ['precision', 'recall', 'f1-score', 'support']
    
    data = []
    for cls in classes:
        row = [report[cls].get(metric, 0) for metric in metrics]
        data.append(row)
    
    # Add macro avg and weighted avg
    if 'macro avg' in report:
        data.append([report['macro avg'].get(metric, 0) for metric in metrics])
        classes.append('macro avg')
    if 'weighted avg' in report:
        data.append([report['weighted avg'].get(metric, 0) for metric in metrics])
        classes.append('weighted avg')
    
    # Create DataFrame
    try:
        import pandas as pd
        df = pd.DataFrame(data, index=classes, columns=metrics)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', 
                    cbar_kws={"shrink": 0.8}, vmin=0, vmax=1)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("pandas is required for classification report visualization")


def plot_feature_importance(
    feature_imp_df,
    title: str = "Feature Importance",
    top_n: Optional[int] = None
) -> None:
    """
    Plot feature importance bar chart.
    
    Args:
        feature_imp_df: DataFrame with 'Feature' and 'Importance' columns
        title: Plot title
        top_n: Optional number of top features to show
    """
    plt.figure(figsize=(10, 6))
    # Always show all features for completeness
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_imp_df,
        color='#4C72B0'
    )
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_mrr_distribution(
    mrr_scores: List[float],
    title: str = "MRR Score Distribution"
) -> None:
    """
    Plot distribution of MRR scores.
    
    Args:
        mrr_scores: List of MRR scores
        title: Plot title
    """
    if not mrr_scores:
        print("No MRR scores to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    ax1.hist(mrr_scores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(mrr_scores), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(mrr_scores):.4f}')
    ax1.set_xlabel('MRR Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('MRR Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(mrr_scores, vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
    ax2.set_ylabel('MRR Score', fontsize=12)
    ax2.set_title('MRR Box Plot', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_model_performance(
    eval_results: Dict,
    feature_imp_df=None,
    mrr_scores: Optional[List[float]] = None
) -> None:
    """
    Create comprehensive visualization dashboard for model performance.
    
    Args:
        eval_results: Dictionary with evaluation results from evaluate_model()
        feature_imp_df: Optional DataFrame with feature importance
        mrr_scores: Optional list of MRR scores
    """
    # Confusion Matrix
    if 'test' in eval_results:
        plot_confusion_matrix(
            eval_results['test']['confusion_matrix'],
            title="Confusion Matrix (Test Set)"
        )
        
        # Classification Report
        plot_classification_report(
            eval_results['test']['report'],
            title="Classification Report (Test Set)"
        )
    
    # Feature Importance
    if feature_imp_df is not None:
        plot_feature_importance(feature_imp_df)
    
    # MRR Distribution
    if mrr_scores:
        plot_mrr_distribution(mrr_scores)

