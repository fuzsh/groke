import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from scipy.stats import gaussian_kde

# Load data
with open('annotations/correctness_hardness.json', 'r') as f:
    data = json.load(f)


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Parse Data
rows = []
for key, value in data.items():
    comp = value['complexity']
    rows.append({
        'instruction_id': key,
        'cognitive': comp['cognitive'],
        'spatial': comp['spatial'],
        'execution': comp['execution']
    })

df = pd.DataFrame(rows)
df['average_complexity'] = df[['cognitive', 'spatial', 'execution']].mean(axis=1)

# Normalize
min_val, max_val = df['average_complexity'].min(), df['average_complexity'].max()
if max_val - min_val == 0:
    df['normalized_complexity'] = 0.5
else:
    df['normalized_complexity'] = (df['average_complexity'] - min_val) / (max_val - min_val)

# Categorize
def categorize(score):
    if score <= 1/3: return 'Easy'
    elif score >= 2/3: return 'Hard'
    return 'Medium'

df['category'] = df['normalized_complexity'].apply(categorize)

# =============================================================================
# OPTION 1: Violin + Strip Plot (shows distribution AND individual points)
# =============================================================================
def plot_violin_strip():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color mapping
    colors = {'Easy': '#2ecc71', 'Medium': '#f39c12', 'Hard': '#e74c3c'}

    # Create violin for each category
    categories = ['Easy', 'Medium', 'Hard']
    positions = [0, 1, 2]

    for i, cat in enumerate(categories):
        subset = df[df['category'] == cat]['normalized_complexity']
        if len(subset) > 1:
            parts = ax.violinplot([subset], positions=[i], showmeans=True, widths=0.7)
            for pc in parts['bodies']:
                pc.set_facecolor(colors[cat])
                pc.set_alpha(0.6)

        # Overlay strip plot (jittered points)
        jitter = np.random.uniform(-0.15, 0.15, len(subset))
        ax.scatter(np.full(len(subset), i) + jitter, subset,
                   alpha=0.5, s=30, c=colors[cat], edgecolor='white', linewidth=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{cat}\n(n={len(df[df["category"]==cat])})' for cat in categories])
    ax.set_ylabel('Normalized Complexity Score')
    ax.set_title('Complexity Distribution by Category')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

# =============================================================================
# OPTION 2: Stacked Component Breakdown (shows what drives complexity)
# =============================================================================
def plot_component_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Box plots for each component
    ax1 = axes[0]
    components = ['cognitive', 'spatial', 'execution']
    box_data = [df[c] for c in components]
    bp = ax1.boxplot(box_data, patch_artist=True, labels=components)
    colors = ['#3498db', '#9b59b6', '#1abc9c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel('Score')
    ax1.set_title('Component Score Distributions')
    ax1.grid(axis='y', alpha=0.3)

    # Right: Stacked bar showing average contribution per category
    ax2 = axes[1]
    category_order = ['Easy', 'Medium', 'Hard']
    x = np.arange(len(category_order))
    width = 0.6

    bottom = np.zeros(len(category_order))
    for comp, color in zip(components, colors):
        means = [df[df['category'] == cat][comp].mean() for cat in category_order]
        ax2.bar(x, means, width, label=comp.capitalize(), bottom=bottom, color=color, alpha=0.8)
        bottom += means

    ax2.set_xticks(x)
    ax2.set_xticklabels(category_order)
    ax2.set_ylabel('Cumulative Score')
    ax2.set_title('Component Contribution by Difficulty')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig

# =============================================================================
# OPTION 3: Heatmap / 2D Density (good for large datasets)
# =============================================================================
def plot_2d_density():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    pairs = [('cognitive', 'spatial'), ('cognitive', 'execution'), ('spatial', 'execution')]

    for ax, (x_col, y_col) in zip(axes, pairs):
        x, y = df[x_col], df[y_col]

        # Hexbin for density
        hb = ax.hexbin(x, y, gridsize=15, cmap='YlOrRd', mincnt=1)
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel(y_col.capitalize())
        ax.set_title(f'{x_col.capitalize()} vs {y_col.capitalize()}')
        plt.colorbar(hb, ax=ax, label='Count')

    plt.suptitle('Component Correlations', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig

# =============================================================================
# OPTION 4: Radar Chart Summary (overall profile)
# =============================================================================
def plot_radar_summary():
    from math import pi

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    categories_list = ['Easy', 'Medium', 'Hard']
    components = ['cognitive', 'spatial', 'execution']
    n_components = len(components)

    angles = [n / float(n_components) * 2 * pi for n in range(n_components)]
    angles += angles[:1]  # Close the polygon

    colors = {'Easy': '#2ecc71', 'Medium': '#f39c12', 'Hard': '#e74c3c'}

    for cat in categories_list:
        subset = df[df['category'] == cat]
        if len(subset) == 0:
            continue
        values = [subset[c].mean() for c in components]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=f'{cat} (n={len(subset)})', color=colors[cat])
        ax.fill(angles, values, alpha=0.15, color=colors[cat])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.capitalize() for c in components], size=12)
    ax.set_title('Average Complexity Profile by Category', size=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    return fig

# =============================================================================
# OPTION 5: Clean histogram with better styling
# =============================================================================
def plot_clean_histogram():
    fig, ax = plt.subplots(figsize=(11, 6))

    t1, t2 = 1/3, 2/3
    colors = {'Easy': '#2ecc71', 'Medium': '#f39c12', 'Hard': '#e74c3c'}

    # Plot separate histograms per category (stacked effect)
    bins = np.linspace(0, 1, 20)

    for cat, color in colors.items():
        subset = df[df['category'] == cat]['normalized_complexity']
        ax.hist(subset, bins=bins, alpha=0.7, label=f'{cat} (n={len(subset)})',
                color=color, edgecolor='white', linewidth=0.5)

    # Add KDE curve
    density = gaussian_kde(df['normalized_complexity'])
    x_vals = np.linspace(0, 1, 200)
    y_vals = density(x_vals)
    scale = ax.get_ylim()[1] / y_vals.max() * 0.9
    ax.plot(x_vals, y_vals * scale, color='#2c3e50', linewidth=2.5,
            linestyle='--', label='Density', alpha=0.8)

    # Zone lines
    ax.axvline(x=t1, color='#7f8c8d', linestyle=':', alpha=0.8, linewidth=1.5)
    ax.axvline(x=t2, color='#7f8c8d', linestyle=':', alpha=0.8, linewidth=1.5)

    # Stats annotation
    stats_text = f'μ={df["normalized_complexity"].mean():.2f}  σ={df["normalized_complexity"].std():.2f}'
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=1))

    # ax.set_xlabel('Normalized Complexity Score', fontsize=11)
    ax.set_ylabel('Count', fontsize=12)
    # ax.set_title('Complexity Score Distribution', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    # ax.legend(loc='upper left')
    ax.legend(
        loc='upper left',
        fontsize=14,
        # fontweight='bold',
        frameon=True,
        framealpha=1.0,
        edgecolor='black',
        handlelength=1.8,
        handleheight=1.2,
        borderpad=0.6,
        labelspacing=0.5,
        prop={'family': 'serif', 'weight': 'bold'}
    )
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('complexity_score_distribution.pdf')
    return fig

# =============================================================================
# Run all visualizations
# =============================================================================
if __name__ == '__main__':
    print(f"Total samples: {len(df)}")
    print(f"Distribution: {df['category'].value_counts().to_dict()}")
    print(f"Avg complexity: {df['average_complexity'].mean():.2f} (raw), {df['normalized_complexity'].mean():.2f} (normalized)")

    # Generate all plots
    # fig1 = plot_violin_strip()
    # fig2 = plot_component_breakdown()
    # fig3 = plot_2d_density()
    # fig4 = plot_radar_summary()
    fig5 = plot_clean_histogram()

    plt.show()