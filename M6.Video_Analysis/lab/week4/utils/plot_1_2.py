import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plt.style.use('seaborn-deep')

path_results = '/ghome/group03/mcv-m6-2023-team6/week4/Results/Task1_2/'

path_csv1 = path_results + 'results_deqflow.csv'
path_csv2 = path_results + 'results_perceiver.csv'
path_csv3 = path_results + 'results.csv'

# Read the csv files
csv_1 = pd.read_csv(path_csv1)
csv_2 = pd.read_csv(path_csv2)
csv_3 = pd.read_csv(path_csv3)

# Join the csv files
csv = pd.concat([csv_1, csv_2, csv_3], axis=0)

# Remove the row corresponding to the LK method
csv = csv[csv['method'] != 'LK']

# Order the methods according to the msen 
csv = csv.sort_values(by=['msen'])

# Save the csv file with the results
csv.to_csv(path_results + 'results_all.csv', index=False)

names = list(csv["method"])

x = np.arange(len(names))

# Plot the results for the MSEN and PEPN using two y-axes
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

bins = np.arange(len(names))
width = 0.35

ax1.bar(x - width / 2, csv['msen'], width=width, alpha=0.5, color='#1f77b4', label='MSEN')
ax2.bar(x + width / 2, csv['pepn'], width=width, alpha=0.5, color='#ff7f0e', label='PEPN (%)')
# ax1.hist(csv['msen'], bins=bins, width=width, alpha=0.5, color='#1f77b4' ,label='MSEN', align='left')
# ax2.hist(csv['pepn'], bins=bins, width=width, alpha=0.5, color='#ff7f0e' ,label='PEPN', align='right')


ax1.set_xlabel('Method')
ax1.set_ylabel('MSEN')
ax2.set_ylabel('PEPN (%)')

ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=45, ha='right')

# Add text labels inside each bar
for i, (m, p) in enumerate(zip(csv['msen'], csv['pepn'])):
    ax1.text(i - width / 2, m - 0.06, str(round(m, 2)), ha='center', va='bottom', color='black')
    ax2.text(i + width / 2, p - 0.5, str(round(p, 2)), ha='center', va='bottom', color='black')

# Combine the legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')

fig.tight_layout()
fig.savefig(path_results + 'histogram_msen_pepn.png')

# Plot the results for the runtime using a y-axis
fig, ax1 = plt.subplots(figsize=(8, 5))

bins = np.arange(len(names))
width = 0.4

ax1.bar(x, csv['runtime'], width=width, alpha=0.5, color='#2ca02c', label='Runtime')

ax1.set_xlabel('Method')
ax1.set_ylabel('Time (s)')

ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=45, ha='right')

# Add text labels inside each bar
for i, r in enumerate(csv['runtime']):
    ax1.text(i, r - 0.4, str(round(r, 2)), ha='center', va='bottom', color='black')

# Combine the legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')

fig.tight_layout()
fig.savefig(path_results + 'histogram_runtime.png')
