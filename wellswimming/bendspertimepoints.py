import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/lilly/phd/wellswimming/df_swim.csv')

df.head()

from scipy import stats

# Assuming your dataframe is called 'df' and has columns 'minutes', 'bends', and 'conditions'
# If not, replace 'df' with your actual dataframe name

# Calculate average bends and standard error for each time point and condition
avg_bends = df.groupby(['minutes', 'condition'])['bends'].mean().unstack()
sem_bends = df.groupby(['minutes', 'condition'])['bends'].sem().unstack()


# Create a figure with 4 subplots, one for each condition
fig, axs = plt.subplots(4, 1, figsize=(10, 24), sharex=True, sharey=True)

# Find global min and max for y-axis
y_min = df['bends'].min()
y_max = df['bends'].max()
y_range = y_max - y_min

# Add padding to y-axis limits
y_min -= 0.1 * y_range
y_max += 0.1 * y_range

# Define condition titles
condition_titles = {
    'a': 'Agar ancestry, Agar growth',
    'b': 'Agar ancestry, Scaffold growth',
    'c': 'Scaffold ancestry, Scaffold growth',
    'd': 'Scaffold ancestry, Agar growth'
}

# Plot data for each condition
for i, condition in enumerate(sorted(df['condition'].unique())):
    condition_data = df[df['condition'] == condition]
    
    # Calculate mean and confidence intervals for each time point
    time_points = sorted(condition_data['minutes'].unique())
    means = []
    ci_lower = []
    ci_upper = []
    
    for t in time_points:
        time_data = condition_data[condition_data['minutes'] == t]['bends']
        mean = time_data.mean()
        # Calculate 95% confidence interval
        ci = stats.t.interval(confidence=0.95,  # Changed from alpha=0.95
                     df=len(time_data)-1,
                     loc=mean,
                     scale=stats.sem(time_data))
        means.append(mean)
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])
    
    # Convert to numpy arrays
    time_points = np.array(time_points)
    means = np.array(means)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)
    
    # Plot mean line
    axs[i].plot(time_points, means, 'b-', marker='o', label='Mean', zorder=3)
    
    # Add confidence band
    axs[i].fill_between(time_points, ci_lower, ci_upper, 
                       color='blue', alpha=0.2, 
                       label='95% Confidence Interval',
                       zorder=2)
    
    # Calculate and plot trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, means)
    line = slope * time_points + intercept
    axs[i].plot(time_points, line, color='r', linestyle='--', 
                label='Trend Line', zorder=1)
    
    # Add text with slope and R-squared values
    slope_text = f"Slope: {slope:.4f} bends/ 5 minutes"
    r_squared_text = f"R-squared: {r_value**2:.4f}"
    axs[i].text(0.05, 0.95, slope_text + '\n' + r_squared_text, 
                transform=axs[i].transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize plot
    axs[i].set_title(condition_titles[condition])
    axs[i].set_ylabel('Average bends per 5 seconds')
    axs[i].set_ylim(y_min, y_max)
    axs[i].grid(True, alpha=0.3)
    axs[i].legend()
    
    # Only set x-label for the bottom subplot
    if i == 3:
        axs[i].set_xlabel('Time (minutes)')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
plt.savefig('wellswimming/bendspertimepoints.png')
plt.close()





# First let's look at the structure of your data
print("Number of replicates per timepoint and condition:")
print(df.groupby(['minutes', 'condition'])['bends'].count().unstack())


print("\nExample of SEM values for first few timepoints:")
print(sem_bends.head())

# Let's also look at the coefficient of variation (CV) to see if variability is consistent
cv_bends = df.groupby(['minutes', 'condition'])['bends'].agg(lambda x: x.std()/x.mean()*100)

# Plot CV over time for each condition
plt.figure(figsize=(10, 6))
for condition in cv_bends.unstack().columns:
    plt.plot(cv_bends.unstack().index.to_numpy(), cv_bends.unstack()[condition].to_numpy(), 
             label=condition_titles[condition], marker='o', alpha=0.6)
plt.xlabel('Time (minutes)')
plt.ylabel('Coefficient of Variation (%)')
plt.title('Variability in measurements over time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
plt.savefig('wellswimming/cvvalues.png')
plt.close()

# Let's also look at the distribution of SEM values
plt.figure(figsize=(10, 6))
for condition in sem_bends.columns:
    plt.hist(sem_bends[condition].dropna(), alpha=0.5, 
             label=condition_titles[condition], bins=20)
plt.xlabel('Standard Error of Mean')
plt.ylabel('Frequency')
plt.title('Distribution of SEM values across conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
plt.savefig('wellswimming/semvalues.png')
plt.close()