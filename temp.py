import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

datapath = "/home/lilly/phd/Multimodal QA Dashboard  - MDR V1 Onboarding.csv"

# Read the CSV (do not skip rows)
df = pd.read_csv(datapath)

# Strip whitespace and newlines from column names
df.columns = df.columns.str.strip()

# Drop the 'QA Feedback' column if it exists
if 'QA Feedback' in df.columns:
    df = df.drop(columns=['QA Feedback'])

# Ensure 'QC Score' is numeric (in case of any non-numeric values)
if 'QC Score' in df.columns:
    df['QC Score'] = pd.to_numeric(df['QC Score'], errors='coerce')
    # Calculate the average QC Score grouped by Annotator and Round
    avg_scores = df.groupby(['Annotator', 'Round'])['QC Score'].transform('mean')
    df['average_qc_score'] = avg_scores
else:
    print('QC Score column not found. Columns are:', df.columns.tolist())

# Output the resulting DataFrame
print(df)

# Remove rows with NA values in 'QC Score' before analysis
clean_df = df.dropna(subset=['QC Score'])

# Compute mean, min, max, and median for each group using the cleaned DataFrame
agg_df = clean_df.groupby(['Pod', 'Round'])['average_qc_score'].agg(['mean', 'min', 'max', 'median']).reset_index()

print('Aggregated QC Score stats by Pod and Round:')
print(agg_df)

# Plot average_qc_score by Pod and by Round
plt.figure(figsize=(10, 6))

# Use a pastel palette for lighter bars
rounds = sorted(df['Round'].dropna().unique())
palette = sns.color_palette("pastel", n_colors=len(rounds))

ax = sns.barplot(
    data=agg_df, x='Pod', y='mean', hue='Round', dodge=True, palette=palette, edgecolor='black'
)

# Add min-max error bars and value labels using the actual bar positions
for bar, (_, row) in zip(ax.patches, agg_df.iterrows()):
    if not np.isnan(row['mean']):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        yerr_lower = row['mean'] - row['min'] if not np.isnan(row['min']) else 0
        yerr_upper = row['max'] - row['mean'] if not np.isnan(row['max']) else 0
        # Draw error bar
        ax.errorbar(
            x, y,
            yerr=[[yerr_lower], [yerr_upper]],
            fmt='none',
            c='black',
            capsize=5,
            lw=1.5,
            zorder=3
        )
        # Place label above error bar
        label_y = y + yerr_upper + 0.05
        ax.text(x, label_y, f"{y:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Average QC Score by Pod and Round (Min-Max Error Bars)')
plt.ylabel('Average QC Score')
plt.xlabel('Pod')
plt.legend(title='Round')
plt.tight_layout()

plt.savefig('average_qc_score_by_pod_and_round.png')
plt.close()

# Boxplot of QC Score by Pod and Round
plt.figure(figsize=(10, 6))
ax2 = sns.boxplot(data=clean_df, x='Pod', y='QC Score', hue='Round', palette='pastel')
plt.title('QC Score Distribution by Pod and Round')
plt.ylabel('QC Score')
plt.xlabel('Pod')
plt.legend(title='Round')
plt.tight_layout()
plt.savefig('qc_score_boxplot_by_pod_and_round.png')
plt.close()

# Calculate the percentage of rows with 'Yes' in 'Certified' by Pod and Round (using original df)
certified_pct = (
    df.groupby(['Pod', 'Round'])['Certified']
    .apply(lambda x: (x == 'Yes').mean() * 100)
    .reset_index(name='percent_certified')
)

print('\nPercentage of Certified = Yes by Pod and Round:')
print(certified_pct)
