#!/usr/bin/env python3
"""
Visualize benchmark results showing POMA's superiority.

Generates:
1. Accuracy curve (line chart) - accuracy vs token budget (LINEAR scale)
2. Per-question heatmap - tokens needed per method/question
3. Box plot - distribution of token requirements
4. Bar chart - MAX and SUM comparisons
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results_from_benchmark():
    """Load results from latest benchmark JSON file."""
    results_dir = Path(__file__).parent / "results" / "bench_results"
    json_files = sorted(results_dir.glob("bench_idx_*.json"))
    if not json_files:
        raise FileNotFoundError("No benchmark results found. Run benchmark.py first.")

    latest = json_files[-1]
    print(f"Loading results from: {latest.name}")

    with open(latest) as f:
        data = json.load(f)

    # Convert to format expected by visualization functions
    results = {}
    method_map = {
        "poma_openai_chunksets": "poma_chunksets",
        "poma_openai_mixed": "poma_mixed",
        "databricks_rcs_openai": "databricks_rcs",
        "unstructured_openai": "unstructured"
    }

    for method, short_name in method_map.items():
        if method in data.get("results", {}):
            questions = data["results"][method].get("questions", {})
            results[short_name] = {
                uid: q["min_budget"]
                for uid, q in questions.items()
                if q.get("min_budget") is not None
            }

    return results


# Load results dynamically from benchmark output
RESULTS = load_results_from_benchmark()

# Colors and styles - POMA Chunksets is primary, Mixed is secondary/dashed
COLORS = {
    "poma_chunksets": "#2ecc71",     # Green - POMA main method
    "poma_mixed": "#27ae60",          # Darker green - optimization variant (dashed)
    "databricks_rcs": "#e74c3c",      # Red - naive chunking baseline
    "unstructured": "#9b59b6"         # Purple - unstructured baseline
}

LABELS = {
    "poma_chunksets": "POMA Chunksets",
    "poma_mixed": "POMA Mixed (optimized)",
    "databricks_rcs": "Naive Chunking (500/100)",
    "unstructured": "Unstructured.io"
}

# Line styles - solid for main methods, dashed for optimization variant
LINE_STYLES = {
    "poma_chunksets": "-",
    "poma_mixed": "--",
    "databricks_rcs": "-",
    "unstructured": "-"
}

LINE_WIDTHS = {
    "poma_chunksets": 3.0,
    "poma_mixed": 2.0,
    "databricks_rcs": 2.5,
    "unstructured": 2.5
}


def compute_accuracy_curve(results, budgets):
    """Compute accuracy at each budget level."""
    n_questions = len(results)
    accuracies = []
    for budget in budgets:
        n_found = sum(1 for v in results.values() if v <= budget)
        accuracies.append(n_found / n_questions)
    return accuracies


def plot_accuracy_curves(output_path):
    """Plot accuracy vs token budget for all methods - LINEAR scale."""
    # Fine-grained budgets for smooth curves (every 25K up to 2M)
    budgets = list(range(0, 2_050_000, 25_000))
    budgets[0] = 5000  # Start at 5K not 0

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot order: baselines first, then POMA (so POMA is on top)
    plot_order = ["unstructured", "databricks_rcs", "poma_mixed", "poma_chunksets"]

    for method in plot_order:
        if method not in RESULTS:
            continue
        results = RESULTS[method]
        accuracies = compute_accuracy_curve(results, budgets)

        ax.plot(budgets, accuracies,
                linestyle=LINE_STYLES[method],
                color=COLORS[method],
                label=LABELS[method],
                linewidth=LINE_WIDTHS[method],
                marker='o' if method != "poma_mixed" else '',
                markersize=4,
                markevery=4)  # Show markers every 100K

    # LINEAR scale - no log!
    ax.set_xlabel('Token Budget', fontsize=14)
    ax.set_ylabel('Recall (fraction of questions answered)', fontsize=14)
    ax.set_title('Context Recall: POMA vs Baselines\n(20 questions, 14 Treasury Bulletins)',
                 fontsize=16, fontweight='bold')

    # Format x-axis with K/M suffixes
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)

    # Calculate 100% thresholds and percentages
    poma_max = max(RESULTS["poma_chunksets"].values())
    db_max = max(RESULTS["databricks_rcs"].values())
    us_max = max(RESULTS["unstructured"].values())

    # Dynamic x-axis: 10% padding beyond highest method
    data_max = max(db_max, us_max)
    ax.set_xlim(0, data_max * 1.1)

    pct_less_db = (1 - poma_max / db_max) * 100
    pct_less_us = (1 - poma_max / us_max) * 100

    # Add vertical line only for POMA (cleaner)
    ax.axvline(x=poma_max, color=COLORS["poma_chunksets"], linestyle=':', alpha=0.6, linewidth=1.5)

    # Single clean annotation at bottom right
    ax.annotate(f'POMA: 100% @ {poma_max/1e3:.0f}K tokens',
                xy=(poma_max, 0.05), xytext=(poma_max + 50000, 0.15),
                fontsize=10, color=COLORS["poma_chunksets"], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS["poma_chunksets"], alpha=0.6))

    # Key message at top left
    ax.text(0.02, 0.98,
            f'POMA uses {pct_less_db:.0f}% less tokens than naive chunking\n'
            f'and {pct_less_us:.0f}% less than Unstructured.io',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', color=COLORS["poma_chunksets"],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_curves_zoomed(output_path):
    """Plot accuracy curves zoomed to 400K, showing both POMA methods reaching 100%."""
    # Fine-grained budgets up to 400K (to show POMA Chunksets reaching 100% at ~360K)
    budgets = list(range(0, 410_000, 5_000))
    budgets[0] = 2500

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot order: baselines first, then POMA chunksets, then mixed (so mixed is last in legend)
    plot_order = ["unstructured", "databricks_rcs", "poma_chunksets", "poma_mixed"]

    for method in plot_order:
        if method not in RESULTS:
            continue
        results = RESULTS[method]
        accuracies = compute_accuracy_curve(results, budgets)

        ax.plot(budgets, accuracies,
                linestyle=LINE_STYLES[method],
                color=COLORS[method],
                label=LABELS[method],
                linewidth=LINE_WIDTHS[method],
                marker='o' if method != "poma_mixed" else '',
                markersize=5,
                markevery=5)

    ax.set_xlabel('Token Budget', fontsize=14)
    ax.set_ylabel('Recall (%)', fontsize=14)
    ax.set_title('Context Recall: POMA vs Baselines\n(20 questions, 14 documents, ~2,150 pages)',
                 fontsize=16, fontweight='bold')

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y*100:.0f}%'))
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 400_000)
    ax.set_ylim(-0.02, 1.05)

    # Get 100% thresholds
    poma_max = max(RESULTS["poma_chunksets"].values())
    db_max = max(RESULTS["databricks_rcs"].values())
    us_max = max(RESULTS["unstructured"].values())

    # Get recall at 400K for baselines
    db_recall_400k = sum(1 for v in RESULTS["databricks_rcs"].values() if v <= 400000) / len(RESULTS["databricks_rcs"])
    us_recall_400k = sum(1 for v in RESULTS["unstructured"].values() if v <= 400000) / len(RESULTS["unstructured"])

    # Vertical line for POMA Chunksets 100%
    ax.axvline(x=poma_max, color=COLORS["poma_chunksets"], linestyle=':', alpha=0.6, linewidth=1.5)

    # Label positions are manually tuned to sit in gaps between curves.
    # Revisit if dataset or methods change significantly.
    ax.annotate(f'POMA: 100% @ {poma_max/1e3:.0f}K',
                xy=(poma_max, 1.0), xytext=(350000, 0.88),
                fontsize=10, color=COLORS["poma_chunksets"], fontweight='bold',
                ha='right',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2),
                arrowprops=dict(arrowstyle='->', color=COLORS["poma_chunksets"], alpha=0.6))

    ax.annotate(f'Naive: {db_recall_400k:.0%} @ 400K (100% @ {db_max/1e6:.1f}M) \u2192',
                xy=(395000, db_recall_400k), xytext=(250000, 0.73),
                fontsize=9, color=COLORS["databricks_rcs"], ha='left',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2),
                arrowprops=dict(arrowstyle='->', color=COLORS["databricks_rcs"], alpha=0.5))

    ax.annotate(f'Unstructured: {us_recall_400k:.0%} @ 400K (100% @ {us_max/1e6:.1f}M) \u2192',
                xy=(395000, us_recall_400k), xytext=(265000, 0.52),
                fontsize=9, color=COLORS["unstructured"], ha='left',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2),
                arrowprops=dict(arrowstyle='->', color=COLORS["unstructured"], alpha=0.5))

    # Key message - top left
    pct_less_db = (1 - poma_max / db_max) * 100
    ax.text(0.02, 0.98,
            f'POMA reaches 100% at {poma_max/1e3:.0f}K\n'
            f'Baselines need {db_max/1e6:.1f}M - {us_max/1e6:.1f}M\n'
            f'({pct_less_db:.0f}% less tokens)',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', color=COLORS["poma_chunksets"],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_token_comparison_bars(output_path):
    """Bar chart comparing MAX tokens across methods."""
    # Only show main 3 methods (not mixed variant)
    methods = ["poma_chunksets", "databricks_rcs", "unstructured"]
    maxes = [max(RESULTS[m].values()) for m in methods]

    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(methods))
    bars = ax.bar(x, [m/1000 for m in maxes],
                  color=[COLORS[m] for m in methods],
                  width=0.6)

    ax.set_ylabel('Context Window Size (thousands of tokens)', fontsize=12)
    ax.set_title('Context Window Needed to Answer EACH Question\n(worst-case across 20 questions, lower is better)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], fontsize=11)

    # Add value labels on bars
    for bar, val in zip(bars, maxes):
        height = bar.get_height()
        label = f'{val/1e6:.1f}M' if val >= 1e6 else f'{val/1e3:.0f}K'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add improvement percentages (POMA uses X% less)
    poma_max = maxes[0]
    for i, (method, val) in enumerate(zip(methods[1:], maxes[1:]), 1):
        pct_less = (1 - poma_max / val) * 100
        ax.annotate(f'POMA uses\n{pct_less:.0f}% less',
                    xy=(i, val/1000 * 0.4),
                    ha='center', fontsize=9, color='white', fontweight='bold')

    ax.set_ylim(0, max(maxes)/1000 * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_question_heatmap(output_path):
    """Heatmap showing token requirements per question/method."""
    # Main 3 methods only
    methods = ["poma_chunksets", "databricks_rcs", "unstructured"]
    uids = sorted(RESULTS["poma_chunksets"].keys())

    # Build matrix (log scale for visibility in heatmap)
    data = np.zeros((len(methods), len(uids)))
    for i, method in enumerate(methods):
        for j, uid in enumerate(uids):
            data[i, j] = np.log10(RESULTS[method][uid])

    fig, ax = plt.subplots(figsize=(16, 5))

    im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r')

    ax.set_xticks(range(len(uids)))
    ax.set_xticklabels([u.replace('UID', '') for u in uids], rotation=45, ha='right')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([LABELS[m] for m in methods])

    ax.set_xlabel('Question ID', fontsize=12)
    ax.set_title('Token Requirements per Question (log scale)\nGreen=fewer tokens needed, Red=more tokens needed',
                 fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(tokens)', fontsize=12)

    # Add text annotations for extreme values
    for i, method in enumerate(methods):
        for j, uid in enumerate(uids):
            val = RESULTS[method][uid]
            if val > 500000:
                ax.text(j, i, f'{val/1e6:.1f}M', ha='center', va='center',
                       fontsize=7, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_box_distribution(output_path):
    """Box plot showing distribution of token requirements."""
    # Main 3 methods only
    methods = ["poma_chunksets", "databricks_rcs", "unstructured"]

    fig, ax = plt.subplots(figsize=(10, 7))

    data = [list(RESULTS[m].values()) for m in methods]
    labels = [LABELS[m] for m in methods]
    colors = [COLORS[m] for m in methods]

    bp = ax.boxplot(data, patch_artist=True, labels=labels)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # LINEAR scale for this one too
    ax.set_ylabel('Tokens Required', fontsize=12)
    ax.set_title('Distribution of Token Requirements\nPer Method (20 questions)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

    # Add horizontal line at POMA's max
    poma_max = max(RESULTS["poma_chunksets"].values())
    ax.axhline(y=poma_max, color=COLORS["poma_chunksets"], linestyle='--', alpha=0.7,
               label=f'POMA 100% threshold ({poma_max/1e3:.0f}K)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table():
    """Print summary statistics table."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    headers = ["Method", "MAX (100%)", "Median", "Min", "vs POMA"]
    print(f"{headers[0]:<30} {headers[1]:>12} {headers[2]:>10} {headers[3]:>10} {headers[4]:>15}")
    print("-"*80)

    poma_max = max(RESULTS["poma_chunksets"].values())

    # Show main methods first
    for method in ["poma_chunksets", "databricks_rcs", "unstructured"]:
        values = list(RESULTS[method].values())
        max_v = max(values)
        median_v = np.median(values)
        min_v = min(values)

        if method == "poma_chunksets":
            vs_str = "baseline"
        else:
            pct_more = (max_v / poma_max - 1) * 100
            vs_str = f"+{pct_more:.0f}% tokens"
        print(f"{LABELS[method]:<30} {max_v:>12,} {median_v:>10,.0f} {min_v:>10,} {vs_str:>15}")

    print("-"*80)

    # Show mixed as optimization variant
    if "poma_mixed" in RESULTS:
        values = list(RESULTS["poma_mixed"].values())
        max_v = max(values)
        median_v = np.median(values)
        min_v = min(values)
        pct_less = (1 - max_v / poma_max) * 100
        print(f"{'POMA Mixed (optimized)':<30} {max_v:>12,} {median_v:>10,.0f} {min_v:>10,} {pct_less:>+.0f}% less")

    print("="*80)


def plot_stacked_context_boxes(output_path):
    """
    Stacked box visualization showing tokens needed per question for each method.

    Each method gets a column of stacked boxes, where each box represents one question.
    Box height = tokens needed for that question. Same colors across methods for comparison.
    Total column height = total tokens needed for 100% recall.
    """
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap

    # Methods to compare (order: best to worst)
    methods = ["poma_mixed", "poma_chunksets", "databricks_rcs", "unstructured"]
    method_labels = {
        "poma_mixed": "POMA\nMixed",
        "poma_chunksets": "POMA\nChunksets",
        "databricks_rcs": "Naive\nChunking",
        "unstructured": "Unstructured.io"
    }

    # Get all UIDs (sorted for consistent ordering)
    all_uids = sorted(RESULTS["poma_chunksets"].keys())
    n_questions = len(all_uids)

    # Create a color palette for questions (20 distinct colors)
    # Use a perceptually uniform colormap
    cmap = plt.colormaps.get_cmap('tab20')
    uid_colors = {uid: cmap(i / n_questions) for i, uid in enumerate(all_uids)}

    fig, ax = plt.subplots(figsize=(14, 10))

    bar_width = 0.6
    x_positions = np.arange(len(methods))

    # For each method, sort questions by token requirement and stack them
    for method_idx, method in enumerate(methods):
        if method not in RESULTS:
            continue

        # Get tokens for each question, sorted by size (largest at bottom for stability)
        uid_tokens = [(uid, RESULTS[method].get(uid, 0)) for uid in all_uids]
        uid_tokens_sorted = sorted(uid_tokens, key=lambda x: -x[1])  # Largest first (bottom)

        # Stack boxes
        bottom = 0
        for uid, tokens in uid_tokens_sorted:
            ax.bar(x_positions[method_idx], tokens, bar_width,
                   bottom=bottom, color=uid_colors[uid],
                   edgecolor='white', linewidth=0.5)

            # Add UID label for large boxes (>5% of total)
            total = sum(t for _, t in uid_tokens)
            if tokens / total > 0.08:
                ax.text(x_positions[method_idx], bottom + tokens/2,
                       uid.replace('UID', ''),
                       ha='center', va='center', fontsize=7,
                       color='white', fontweight='bold')

            bottom += tokens

        # Add total label on top
        total_tokens = sum(t for _, t in uid_tokens)
        ax.text(x_positions[method_idx], total_tokens + 50000,
               f'{total_tokens/1e6:.2f}M', ha='center', va='bottom',
               fontsize=11, fontweight='bold')

    ax.set_xticks(x_positions)
    ax.set_xticklabels([method_labels[m] for m in methods], fontsize=12)
    ax.set_ylabel('Total Context Tokens for 100% Recall', fontsize=13)
    ax.set_title('Context Budget Composition by Question\n(Each color = one question, stacked largest-to-smallest)',
                fontsize=14, fontweight='bold')

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    ax.set_ylim(0, max(sum(RESULTS[m].values()) for m in methods if m in RESULTS) * 1.15)

    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add annotation showing POMA advantage
    poma_total = sum(RESULTS["poma_chunksets"].values())
    naive_total = sum(RESULTS["databricks_rcs"].values())
    us_total = sum(RESULTS["unstructured"].values())

    ax.text(0.02, 0.98,
           f'POMA Chunksets total: {poma_total/1e6:.2f}M\n'
           f'vs Naive: {naive_total/1e6:.1f}M ({naive_total/poma_total:.1f}x more)\n'
           f'vs Unstructured: {us_total/1e6:.1f}M ({us_total/poma_total:.1f}x more)',
           transform=ax.transAxes, fontsize=10, fontweight='bold',
           verticalalignment='top', color='#2c3e50',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_token_heaps(output_path):
    """
    Waffle/unit chart: Each small square = 5K tokens.
    All boxes are SAME SIZE across all methods - single coordinate system.
    Visual area directly shows token cost difference.
    """
    import matplotlib.patches as patches

    UNIT_SIZE = 5000  # Each box = 5K tokens
    BOX_SIZE = 1.0    # Size of each box in plot units

    methods = ["poma_mixed", "poma_chunksets", "databricks_rcs", "unstructured"]
    method_labels = {
        "poma_mixed": "POMA Mixed",
        "poma_chunksets": "POMA Chunksets",
        "databricks_rcs": "Naive Chunking",
        "unstructured": "Unstructured.io"
    }

    # Get all UIDs sorted consistently
    all_uids = sorted(RESULTS["poma_chunksets"].keys())
    n_questions = len(all_uids)

    # Color palette for questions
    cmap = plt.colormaps.get_cmap('tab20')
    uid_colors = {uid: cmap(i / n_questions) for i, uid in enumerate(all_uids)}

    # Calculate heap data for all methods first
    heap_data = {}
    max_heap_width = 0

    for method in methods:
        if method not in RESULTS:
            continue

        # Calculate boxes per question
        uid_boxes = []
        for uid in all_uids:
            tokens = RESULTS[method].get(uid, 0)
            n_boxes = max(1, int(np.ceil(tokens / UNIT_SIZE)))
            uid_boxes.append((uid, n_boxes, uid_colors[uid]))

        # Sort by box count (largest first)
        uid_boxes_sorted = sorted(uid_boxes, key=lambda x: -x[1])

        total_boxes = sum(b[1] for b in uid_boxes_sorted)
        grid_width = int(np.ceil(np.sqrt(total_boxes)))
        max_heap_width = max(max_heap_width, grid_width)

        heap_data[method] = {
            'uid_boxes': uid_boxes_sorted,
            'total_boxes': total_boxes,
            'grid_width': grid_width,
            'total_tokens': sum(RESULTS[method].values())
        }

    # Single figure with one axes - all heaps in same coordinate system
    fig, ax = plt.subplots(figsize=(18, 12))

    # Position heaps horizontally with spacing
    # Use largest heap width + margin for spacing
    heap_spacing = max_heap_width + 5

    for method_idx, method in enumerate(methods):
        if method not in heap_data:
            continue

        data = heap_data[method]
        x_offset = method_idx * heap_spacing

        # Place boxes in grid
        current_row = 0
        current_col = 0
        grid_width = data['grid_width']

        for uid, n_boxes, color in data['uid_boxes']:
            for _ in range(n_boxes):
                rect = patches.Rectangle(
                    (x_offset + current_col * BOX_SIZE, current_row * BOX_SIZE),
                    BOX_SIZE * 0.9, BOX_SIZE * 0.9,
                    facecolor=color, edgecolor='white', linewidth=0.2
                )
                ax.add_patch(rect)
                current_col += 1
                if current_col >= grid_width:
                    current_col = 0
                    current_row += 1

        # Label below heap
        heap_height = current_row + (1 if current_col > 0 else 0)
        total_tokens = data['total_tokens']
        ax.text(x_offset + grid_width * BOX_SIZE / 2, -3,
               f"{method_labels[method]}\n{total_tokens/1e6:.2f}M tokens",
               ha='center', va='top', fontsize=11, fontweight='bold')

    # Set limits to show all heaps with same scale
    total_width = len(methods) * heap_spacing
    ax.set_xlim(-2, total_width)
    ax.set_ylim(-8, max_heap_width + 2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.set_title('Context Token Heaps: Each Square = 5K Tokens\n(Visual area shows true token cost difference)',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_stacked_context_waterfall(output_path):
    """
    Alternative view: Waterfall/area chart showing cumulative tokens per question.
    X-axis = questions (sorted by POMA tokens), Y-axis = cumulative tokens.
    Each method is a line/area showing how tokens accumulate.
    """
    methods = ["poma_mixed", "poma_chunksets", "databricks_rcs", "unstructured"]

    # Get all UIDs sorted by POMA chunkset tokens (easiest to hardest)
    all_uids = sorted(RESULTS["poma_chunksets"].keys(),
                      key=lambda u: RESULTS["poma_chunksets"].get(u, 0))

    fig, ax = plt.subplots(figsize=(14, 8))

    for method in methods:
        if method not in RESULTS:
            continue

        # Compute cumulative tokens in the same UID order
        tokens = [RESULTS[method].get(uid, 0) for uid in all_uids]
        cumulative = np.cumsum(tokens)

        ax.fill_between(range(len(all_uids)), cumulative, alpha=0.3,
                       color=COLORS[method], label=LABELS[method])
        ax.plot(range(len(all_uids)), cumulative, color=COLORS[method],
               linewidth=2, linestyle=LINE_STYLES[method])

    ax.set_xlabel('Questions (sorted by POMA token requirement)', fontsize=12)
    ax.set_ylabel('Cumulative Tokens', fontsize=12)
    ax.set_title('Cumulative Context Tokens: How Costs Accumulate\n(Questions sorted easiest → hardest for POMA)',
                fontsize=14, fontweight='bold')

    # Format axes
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    ax.set_xlim(0, len(all_uids) - 1)

    # Add question count markers
    ax.set_xticks([0, 4, 9, 14, 19])
    ax.set_xticklabels(['1', '5', '10', '15', '20'])

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.text(0.98, 0.02,
           'Steeper slope = more expensive questions\n'
           'POMA stays flat while baselines spike',
           transform=ax.transAxes, fontsize=9,
           ha='right', va='bottom', style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    output_dir = Path(__file__).parent / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")

    # Only generate zoomed version (non-zoomed was redundant)
    plot_accuracy_curves_zoomed(output_dir / "accuracy_curves.png")
    plot_token_comparison_bars(output_dir / "token_comparison.png")
    plot_per_question_heatmap(output_dir / "question_heatmap.png")
    plot_box_distribution(output_dir / "token_distribution.png")
    plot_stacked_context_boxes(output_dir / "stacked_context_boxes.png")
    plot_stacked_context_waterfall(output_dir / "cumulative_context.png")
    plot_token_heaps(output_dir / "token_heaps.png")

    generate_summary_table()

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
