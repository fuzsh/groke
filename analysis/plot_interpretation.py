import pandas as pd
import matplotlib.pyplot as plt
from torch.cpu.amp import autocast

# Data
data = {
    'Method': ['Opti. Repr.', 'json', 'textual', 'graph_vis', 'grid'],
    'NE': [41.3, 68.4, 70.8, 96.7, 175.4],
    'SR': [74.0, 63.0, 61.0, 40.0, 10.0],
    'OSR': [82.0, 74.0, 67.0, 50.0, 12.0],
    'nDTW': [0.769, 0.643, 0.633, 0.483, 0.169]
}
df = pd.DataFrame(data)

# Styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Titles as captions
titles = [
    r'(a) NE (m) $\downarrow$',
    r'(b) SR (%) $\uparrow$',
    r'(c) OSR (%) $\uparrow$',
    r'(d) nDTW $\uparrow$'
]
metrics = ['NE', 'SR', 'OSR', 'nDTW']

colors = ['#A5A5A5', '#5b9bd5', '#ed7d31', '#70ad47', '#ffc000']

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for i, metric in enumerate(metrics):
    ax = axes[i]
    bars = ax.bar(df['Method'], df[metric], color=colors, edgecolor='black', linewidth=0.5)

    # --- HATCHING ADDED HERE ---
    # Apply hatching to the first bar ('Optimized Repr.')
    bars[0].set_hatch('///')

    # Label setup
    ax.set_xlabel(titles[i], fontsize=18, labelpad=15, fontweight='bold')
    ax.set_title("")

    # Standard formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    # Annotate values
    for bar in bars:
        height = bar.get_height()
        val_str = f'{height:.3f}' if metric == 'nDTW' else f'{height:.1f}'
        ax.annotate(val_str,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

    # Rotate x labels for better visibility
    # ax.set_xticklabels(df['Method'], ha='right')

plt.tight_layout()
plt.savefig('plot.png', dpi=300)
