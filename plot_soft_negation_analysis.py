import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup directories
DATA_DIR = "results/soft_negation"
OUT_DIR = "figures/soft_negation_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# Load combined CSV
csv_path = os.path.join(DATA_DIR, "soft_negation_combined.csv")
if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}")
    exit(1)

print("Loading data...")
df = pd.read_csv(csv_path)

# Drop any potential NaNs in crucial columns
df = df.dropna(subset=['sgr', 'negation_failure'])

# Clean negator names (strip leading space)
df['negator'] = df['negator'].astype(str).str.strip()

# Exclude 'maybe'
df = df[df['negator'] != 'maybe']

# Calculate condensed metrics
print("\n--- Final Condensed Metrics ---")
metrics = df.groupby(['model_name', 'negator']).agg(
    samples=('sgr', 'count'),
    failure_rate=('negation_failure', lambda x: x.mean() * 100),
    median_sgr=('sgr', 'median'),
    mean_sgr=('sgr', 'mean'),
    sgr_gt1_percent=('sgr', lambda x: (x > 1).mean() * 100)
).reset_index()

# Sort by model and then failure rate
metrics = metrics.sort_values(by=['model_name', 'failure_rate'], ascending=[True, False])
print(metrics.to_string(index=False, float_format="%.2f"))

# ---------------------------------------------------------
# Plot 1: Failure Rate Comparison
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
g = sns.barplot(
    data=df.groupby(['model_name', 'negator'])['negation_failure'].mean().reset_index(),
    x='model_name', 
    y='negation_failure', 
    hue='negator',
    palette='Set2'
)
plt.title("Failure Rate Comparison by Model and Negator Type", fontsize=14, fontweight='bold')
plt.ylabel("Failure Rate (%)", fontsize=12)
plt.xlabel("Model", fontsize=12)
# Convert y-axis to percentages
vals = g.get_yticks()
g.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
for container in g.containers:
    g.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3, 
                labels=[f"{x.get_height()*100:.1f}%" for x in container])
plt.legend(title='Negator', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "failure_rate_comparison.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot 2: Log SGR Violin Plot
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
import numpy as np
df['log_sgr'] = np.log10(df['sgr'].clip(lower=0.01)) # Clip to avoid log(0)
sns.violinplot(
    data=df,
    x='model_name',
    y='log_sgr',
    hue='negator',
    split=False,
    inner='quartile',
    palette='Set2'
)
plt.axhline(0, color='red', linestyle='--', alpha=0.7, label='SGR = 1 (Threshold)')
plt.title("Distribution of Log(SGR) by Model and Negator", fontsize=14, fontweight='bold')
plt.ylabel("Log10(SGR)", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.legend(title='Negator', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "log_sgr_distribution.png"), dpi=150)
plt.close()


# ---------------------------------------------------------
# Plot 3: Success vs Failure SGR Overlap (KDE)
# Show how SGR separates success/failure across the different negators
# ---------------------------------------------------------
g = sns.FacetGrid(df, col="negator", row="model_name", margin_titles=True, height=3.5, aspect=1.2)
g.map_dataframe(sns.kdeplot, x="log_sgr", hue="negation_failure", fill=True, common_norm=False, palette={True: "red", False: "blue"}, alpha=0.5)

# Add SGR=1 threshold line to all facets
for ax in g.axes.flat:
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    
g.set_axis_labels("Log10(SGR)", "Density")
g.set_titles(col_template="{col_name}", row_template="{row_name}", size=12, fontweight='bold')
g.add_legend(title="Negation Failure")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('SGR Distribution: Success (Blue) vs Failure (Red) Across Negators', fontsize=16, fontweight='bold')
g.fig.savefig(os.path.join(OUT_DIR, "sgr_success_failure_kde.png"), dpi=150)
plt.close()

print(f"\nGraphs saved to: {OUT_DIR}/")

# ---------------------------------------------------------
# Plot 4: Combined Horizontal (1x2)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Ax1: Bar plot
g1 = sns.barplot(
    data=df.groupby(['model_name', 'negator'])['negation_failure'].mean().reset_index(),
    x='model_name', 
    y='negation_failure', 
    hue='negator',
    palette='Set2',
    ax=ax1
)
ax1.set_title("Failure Rate by Model and Negator", fontsize=14, fontweight='bold')
ax1.set_ylabel("Failure Rate (%)", fontsize=12)
ax1.set_xlabel("Model", fontsize=12)
vals1 = ax1.get_yticks()
ax1.set_yticklabels(['{:,.1%}'.format(x) for x in vals1])
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3, 
                labels=[f"{x.get_height()*100:.1f}%" for x in container])
ax1.legend(title='Negator', loc='upper right')

# Ax2: Violin plot
sns.violinplot(
    data=df,
    x='model_name',
    y='log_sgr',
    hue='negator',
    split=False,
    inner='quartile',
    palette='Set2',
    ax=ax2
)
ax2.axhline(0, color='red', linestyle='--', alpha=0.7, label='SGR = 1 (Threshold)')
ax2.set_title("Distribution of Log(SGR) by Model and Negator", fontsize=14, fontweight='bold')
ax2.set_ylabel("Log10(SGR)", fontsize=12)
ax2.set_xlabel("Model", fontsize=12)
ax2.legend(title='Negator', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "combined_horizontal.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# Plot 5: Combined Vertical (2x1)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Ax1: Bar plot
g2 = sns.barplot(
    data=df.groupby(['model_name', 'negator'])['negation_failure'].mean().reset_index(),
    x='model_name', 
    y='negation_failure', 
    hue='negator',
    palette='Set2',
    ax=ax1
)
ax1.set_title("Failure Rate by Model and Negator", fontsize=14, fontweight='bold')
ax1.set_ylabel("Failure Rate (%)", fontsize=12)
ax1.set_xlabel("Model", fontsize=12)
vals2 = ax1.get_yticks()
ax1.set_yticklabels(['{:,.1%}'.format(x) for x in vals2])
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3, 
                labels=[f"{x.get_height()*100:.1f}%" for x in container])
ax1.legend(title='Negator', loc='upper right')

# Ax2: Violin plot
sns.violinplot(
    data=df,
    x='model_name',
    y='log_sgr',
    hue='negator',
    split=False,
    inner='quartile',
    palette='Set2',
    ax=ax2
)
ax2.axhline(0, color='red', linestyle='--', alpha=0.7, label='SGR = 1 (Threshold)')
ax2.set_title("Distribution of Log(SGR) by Model and Negator", fontsize=14, fontweight='bold')
ax2.set_ylabel("Log10(SGR)", fontsize=12)
ax2.set_xlabel("Model", fontsize=12)
ax2.legend(title='Negator', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "combined_vertical.png"), dpi=150)
plt.close()

print(f"Combined graphs generated successfully!")
