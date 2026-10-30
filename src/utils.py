"""
Utility functions for visualization and analysis in the retrieval comparison project.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi


def top_k_indices(scores, k):
    """Return indices of the k highest scores per row, sorted best-first.

    Uses argpartition so cost is linear in corpus size instead of a full sort.
    Accepts a 1-D score vector or a 2-D (queries x documents) matrix.
    """
    scores = np.atleast_2d(scores)
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1)


def create_comparison_visualization(metrics_dict, title="Performance Comparison", 
                                  metrics_to_compare=None, figsize=(16, 6), 
                                  show_values=True, show_radar=True):
    """
    Create comprehensive comparison visualization for retrieval metrics.
    
    Args:
        metrics_dict (dict): Dictionary with method names as keys and metric dictionaries as values
        title (str): Main title for the visualization
        metrics_to_compare (list): List of metrics to compare. If None, uses common IR metrics
        figsize (tuple): Figure size for the plots
        show_values (bool): Whether to show values on bar chart
        show_radar (bool): Whether to show radar chart alongside bar chart
        
    Returns:
        None (displays the plots)
    """
    
    if metrics_to_compare is None:
        metrics_to_compare = ['Recall@1', 'Recall@5', 'Recall@10', 'Precision@5', 'MRR']
    
    # Filter metrics that exist in all methods
    available_metrics = []
    for metric in metrics_to_compare:
        if all(metric in method_metrics for method_metrics in metrics_dict.values()):
            available_metrics.append(metric)
    
    if not available_metrics:
        print("❌ No common metrics found across all methods")
        return
    
    method_names = list(metrics_dict.keys())
    colors = plt.get_cmap('tab10').colors
    
    # Create subplots - bar chart only or bar + radar
    if show_radar and len(method_names) <= 4:  # Radar chart works best with few methods
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2 + 2, figsize[1]))
        ax2 = None
    
    # Bar Chart
    x = np.arange(len(available_metrics))
    width = 0.8 / len(method_names)  # Dynamic width based on number of methods
    
    bars_list = []
    for i, method_name in enumerate(method_names):
        scores = [metrics_dict[method_name][metric] for metric in available_metrics]
        bars = ax1.bar(x + i*width - width*(len(method_names)-1)/2, scores, width, 
                      label=method_name, alpha=0.8, color=colors[i % len(colors)])
        bars_list.append(bars)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title(title)
    ax1.set_xticks(x)
    ax1.set_xticklabels(available_metrics, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars if requested
    if show_values:
        for bars in bars_list:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
    
    # Radar Chart (if requested and suitable)
    if show_radar and ax2 is not None:
        N = len(available_metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        for i, method_name in enumerate(method_names):
            values = [metrics_dict[method_name][metric] for metric in available_metrics]
            values += values[:1]  # Complete the circle
            
            ax2.plot(angles, values, 'o-', linewidth=2, label=method_name, 
                    color=colors[i % len(colors)])
            ax2.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(available_metrics)
        ax2.set_ylim(0, 1)
        ax2.set_title('Radar View')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def print_comparison_table(metrics_dict, metrics_to_compare=None, title="Performance Comparison"):
    """
    Print a formatted comparison table for retrieval metrics.
    
    Args:
        metrics_dict (dict): Dictionary with method names as keys and metric dictionaries as values
        metrics_to_compare (list): List of metrics to compare. If None, uses common IR metrics
        title (str): Title for the comparison table
    """
    
    if metrics_to_compare is None:
        metrics_to_compare = ['Recall@1', 'Recall@5', 'Recall@10', 'Precision@5', 'MRR']
    
    print(f"📊 {title}")
    print("=" * 70)
    
    # Filter metrics that exist in all methods
    available_metrics = []
    for metric in metrics_to_compare:
        if all(metric in method_metrics for method_metrics in metrics_dict.values()):
            available_metrics.append(metric)
    
    if not available_metrics:
        print("❌ No common metrics found across all methods")
        return
    
    method_names = list(metrics_dict.keys())
    
    # Print comparison table
    for i, metric in enumerate(available_metrics):
        scores_str = " | ".join([f"{method}={metrics_dict[method][metric]:.4f}" 
                                for method in method_names])
        print(f"{metric:12}: {scores_str}")
    
    print("=" * 70)


def reciprocal_rank_fusion(rankings, k=20, rrf_k=60):
    """Fuse several ranked lists with Reciprocal Rank Fusion (Cormack et al., 2009).

    Each document scores sum(1 / (rrf_k + rank)) over the lists it appears in, so
    documents ranked well by several retrievers rise to the top. Only ranks are used,
    which avoids calibrating BM25 scores against cosine similarities.

    Args:
        rankings: List of ranked document-id lists (best first), one per retriever
        k: Number of fused results to return
        rrf_k: Smoothing constant; 60 is the standard default

    Returns:
        List of the top-k document ids after fusion
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]


def random_param_combinations(param_grid, n_samples=10):
    """Generate random parameter combinations"""
    import random
    import itertools

    # Get all possible combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))
    
    # Randomly sample combinations
    random.seed(42)
    sampled = random.sample(all_combinations, min(n_samples, len(all_combinations)))
    
    # Convert back to dictionaries
    return [dict(zip(keys, combo)) for combo in sampled]


"""Analysis and comparison utilities for retrieval results."""

import random


def clean_text(text):
    """Clean text for display."""
    return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')


def compare_retrieval_methods(query_texts, corpus_texts, qrels_dict, 
                              results_dict, n_examples=20, top_k=None, 
                              seed=42, show_ground_truth=True):
    """
    Compare multiple retrieval methods with examples and statistics.
    
    Args:
        query_texts: List of query strings
        corpus_texts: List of corpus documents
        qrels_dict: Ground truth relevance judgments {query_idx: {doc_idx: relevance}}
        results_dict: Dict of {method_name: {query_idx: [doc_indices]}}
        n_examples: Number of query examples to show
        top_k: Number of results to show per method (None = num_relevant)
        seed: Random seed for reproducibility
        show_ground_truth: Whether to display ground truth documents
    
    Returns:
        dict: Statistics for each method
    """
    print("Retrieval Method Comparison\n" + "=" * 60)
    
    # Select queries with ground truth
    random.seed(seed)
    available = [i for i in range(len(query_texts)) if qrels_dict.get(i)]
    selected = random.sample(available, min(n_examples, len(available)))
    
    # Track statistics
    stats = {method: {'correct': 0, 'total': 0} for method in results_dict.keys()}
    
    # Show examples
    for idx, q_idx in enumerate(selected, 1):
        query = query_texts[q_idx]
        relevant_docs = set(qrels_dict[q_idx].keys())
        num_relevant = len(relevant_docs)
        
        print(f"\n{'='*60}")
        print(f"Example {idx}: '{query}'")
        print(f"Ground Truth: {num_relevant} relevant document(s)")
        
        # Show ground truth documents
        if show_ground_truth:
            print(f"\nRelevant Documents:")
            for i, doc_idx in enumerate(sorted(relevant_docs), 1):
                print(f"   {i}. {clean_text(corpus_texts[doc_idx][:120])}...")
        
        # Determine k
        k = top_k if top_k is not None else num_relevant
        
        # Show results for each method
        for method_name, method_results in results_dict.items():
            top_results = method_results[q_idx][:k]
            
            print(f"\n{method_name} (Top {k}):")
            for i, doc_idx in enumerate(top_results, 1):
                is_correct = "✓" if doc_idx in relevant_docs else "✗"
                print(f"   {i}. {is_correct} {clean_text(corpus_texts[doc_idx][:120])}...")
            
            # Update statistics
            hits = len(set(top_results) & relevant_docs)
            stats[method_name]['correct'] += hits
            stats[method_name]['total'] += num_relevant
    
    # Print overall statistics
    print(f"\n{'='*60}")
    print("Overall Statistics:")
    for method_name, method_stats in stats.items():
        correct = method_stats['correct']
        total = method_stats['total']
        pct = (correct / total * 100) if total > 0 else 0
        print(f"   {method_name}: {correct}/{total} documents ({pct:.1f}%)")
    
    return stats