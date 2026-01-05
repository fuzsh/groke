import matplotlib.pyplot as plt
import numpy as np

# Data provided by the user
metrics = [
    'NE\n(per 1m reduction)',
    'SR\n(per 1% increase)',
    'OSR\n(per 1% increase)',
    'nDTW\n(per 0.01 increase)'
]
costs = [1334, 1356, 1356, 1731]

# Calculate Cents costs
# Rate: $12 per 1,000,000 tokens
multiplied_rate_cents = (12 / 1_000_000) * 100
costs_cents = [c * multiplied_rate_cents for c in costs]

# Colors (kept from original data source for semantic consistency)
colors = ['#E15759', '#4E79A7', '#F28E2B', '#76B7B2']

# Styling configuration (Times New Roman, Serif)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, ax = plt.subplots(figsize=(8, 6))

# Create bars with specific edge styling
bars = ax.bar(metrics, costs, color=colors, edgecolor='black', linewidth=0.5, width=0.6)


# Grid and Spine styling
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)  # Puts grid behind the bars
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Title and Axis Labels
# Adopting the "Title as X-axis label" style
# ax.set_xlabel('(a) Marginal Cost (Tokens)', fontsize=18, labelpad=15, fontweight='bold')
ax.set_ylabel('Tokens per Unit Gain', fontsize=14)

# Annotations
for bar, cents_cost in zip(bars, costs_cents):
    height = bar.get_height()

    # Top Label (Token count)
    ax.annotate(f'{int(height):,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    # Middle Label (Cents Cost) - White text for contrast
    ax.text(bar.get_x() + bar.get_width() / 2., height / 2.,
            f'{cents_cost:.2f}¢',
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('marginal_cost_with_cents.pdf')
plt.show()