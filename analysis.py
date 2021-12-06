# %% VS Code Notebook
import json
import numpy as np
import pandas as pd

from pathlib import Path

root = Path(__file__).parent

k_val=1
agg = np.mean
# agg = np.std
rows = []
for k in range(1, 7, 2):
    for n in range(1,11):
        with open(root / 'batch2' / f'n-{n}-k-{k}.json', 'r') as f:
            data = json.load(f)
            rows.append({
                'brute_force': agg(data['brute_force_data']),
                'online': agg(data['online_data']),
                'streaming_trained': agg(data['streaming_trained_data']),
                'streaming_theoretical': agg(data['streaming_theoretical_data']),
                'trained_tau': data['trained_tau'],
                'theoretical_tau': data['theoretical_tau'],
                'n': n,
                'k': k,
            })
df = pd.DataFrame(rows)

import matplotlib as mpl
from matplotlib import pyplot as plt
# plt.ax
mpl.rcParams['figure.dpi'] = 300
selected_df = df[
    df['k']==k_val
]
fig, ax = plt.subplots()
ax.set_xticks(selected_df.index)
ax.set_xticklabels(range(1,11))
selected_df[['brute_force','online','streaming_trained','streaming_theoretical']].plot(
    figsize=(5, 3),
    ylim=(0,selected_df.max().max()+1),
    xlabel='n',
    ylabel='profit($)',
    ax=ax,
)
# %%
df.head()
# %%
df.groupby(['k','n']).median()
# %%
k_val = 5
selected_df = df[
    df['k']==k_val
][['trained_tau','theoretical_tau']]
fig, ax = plt.subplots()
ax.set_xticks(selected_df.index)
ax.set_xticklabels(range(1,11))
selected_df.plot(
    figsize=(5, 3),
    ylim=(0,selected_df.max().max()+1),
    xlabel='n',
    ylabel='τ',
    ax=ax,
)
# %%
k_val = 5
selected_df = df[
    df['k']==k_val
]
selected_df['online_ratio'] = selected_df['online']/selected_df['brute_force']
selected_df['streaming_trained_ratio'] = selected_df['streaming_trained']/selected_df['brute_force']
selected_df['streaming_theoretical_ratio'] = selected_df['streaming_theoretical']/selected_df['brute_force']
selected_df[[
    'n',
    'online_ratio',
    'streaming_trained_ratio',
    'streaming_theoretical_ratio',
]].round(2)
# %%
selected_df
# %%
