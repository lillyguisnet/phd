import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('/home/lilly/phd/laeya/worm_metrics_summary.csv')

# Extract condition from img_id (characters before first _)
df['condition'] = df['img_id'].str.extract(r'^([^_]+)')

# Calculate sample size per condition
condition_counts = df.groupby('condition').size()

print(condition_counts)

""" 
Worms per condition
condition
D1    58
D3    63
D5    60
D8    55
L1    30
L2    41
L3    94
L4    37
"""



#Remove 1 row
#df = df[~((df['img_id'] == 'L2_rep3_5') & (df['worm_id'] == 0))]

""" 
# Create a list of tuples containing img_id and worm_id to remove
rows_to_remove = [
    ('L2_rep3_5', 0)  # Remove duplicate L2_rep3_5 image
]

# Remove rows where img_id and worm_id match any tuple in rows_to_remove
for img_id, worm_id in rows_to_remove:
    df = df[~((df['img_id'] == img_id) & (df['worm_id'] == worm_id))]
"""