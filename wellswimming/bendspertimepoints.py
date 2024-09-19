import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/maxime/prg/phd/wellswimming/df_swim.csv')

df.head()

from scipy import stats

# Assuming your dataframe is called 'df' and has columns 'minutes', 'bends', and 'conditions'
# If not, replace 'df' with your actual dataframe name

# Calculate average bends and standard error for each time point and condition
avg_bends = df.groupby(['minutes', 'condition'])['bends'].mean().unstack()
sem_bends = df.groupby(['minutes', 'condition'])['bends'].sem().unstack()

# Create a figure with 4 subplots, one for each condition
fig, axs = plt.subplots(4, 1, figsize=(10, 24), sharex=True, sharey=True)
fig.suptitle('Average Bends Over Time by Condition')

# Find global min and max for y-axis
y_min = (avg_bends - sem_bends).min().min()
y_max = (avg_bends + sem_bends).max().max()

# Add some padding to the y-axis limits
y_range = y_max - y_min
y_min -= 0.1 * y_range
y_max += 0.1 * y_range

# Plot data for each condition
for i, condition in enumerate(avg_bends.columns):
    x = avg_bends.index
    y = avg_bends[condition]
    
    # Plot data points with error bars
    axs[i].errorbar(x, y, yerr=sem_bends[condition], 
                    marker='o', capsize=3, capthick=1, elinewidth=1, label='Data')
    
    # Calculate and plot trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * x + intercept
    axs[i].plot(x, line, color='r', linestyle='--', label='Trend Line')
    
    # Add text with slope and R-squared values
    slope_text = f"Slope: {slope:.4f} bends/ 5 minutes"
    r_squared_text = f"R-squared: {r_value**2:.4f}"
    axs[i].text(0.05, 0.95, slope_text + '\n' + r_squared_text, 
                transform=axs[i].transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axs[i].set_title(f'Condition: {condition}')
    axs[i].set_ylabel('Average Bends')
    axs[i].set_ylim(y_min, y_max)
    axs[i].legend()
    
    # Only set x-label for the bottom subplot
    if i == 3:
        axs[i].set_xlabel('Time (minutes)')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
plt.savefig('bendspertimepoints.png')
plt.close()