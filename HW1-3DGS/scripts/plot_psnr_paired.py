#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

CSV0 = 'report/images/3.1/compare_sh0.csv'
CSV1 = 'report/images/3.1/compare_sh1.csv'
OUT_DIR = 'report/images/3.1'
OUT_PATH = os.path.join(OUT_DIR, 'psnr_paired.png')

os.makedirs(OUT_DIR, exist_ok=True)

def read_compare(path):
    df = pd.read_csv(path)
    if 'psnr' in df.columns:
        return df.set_index(df.columns[0])
    df = pd.read_csv(path, header=None)
    df = df.iloc[:, :3]
    df.columns = ['filename','psnr','ssim']
    return df.set_index('filename')

A = read_compare(CSV0)
B = read_compare(CSV1)
ps0 = A['psnr'].astype(float)
ps1 = B['psnr'].astype(float)
delta = ps1 - ps0
n = len(delta)

stat, pval = stats.wilcoxon(ps0, ps1)

rng = np.random.default_rng(1)
boots = rng.choice(delta.values, (2000, n), replace=True).mean(axis=1)
lo, hi = np.percentile(boots, [2.5, 97.5])
mean_delta = float(delta.mean())

# Plotting
x0, x1 = 0.43, 0.6
order = delta.sort_values(ascending=False).index
y0 = ps0.loc[order].values
y1 = ps1.loc[order].values

fig, ax = plt.subplots(figsize=(5,4), dpi=150)
ax.grid(True, linestyle='--', alpha=0.4)
for a, b in zip(y0, y1):
    ax.plot([x0, x1], [a, b], color='#999', zorder=1)
ax.scatter(np.full(n, x0), y0, color='#2C7BB6', edgecolor='k', zorder=3, s=60)
ax.scatter(np.full(n, x1), y1, color='#D95F02', edgecolor='k', zorder=3, s=60)
ax.scatter([x0], [ps0.mean()], marker='D', color='#08519c', edgecolor='k', s=120)
ax.scatter([x1], [ps1.mean()], marker='D', color='#993404', edgecolor='k', s=120)
ax.plot([x0, x1], [ps0.mean(), ps1.mean()], color='#999', zorder=1)

ax.text(x0 - 0.055, ps0.mean(), f"{ps0.mean():.2f}", va='center', fontsize=9, fontweight='bold', color='#08519c', zorder=4)
ax.text(x1 + 0.015, ps1.mean(), f"{ps1.mean():.2f}", va='center', fontsize=9, fontweight='bold', color='#993404', zorder=4)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='sh0', markerfacecolor='#2C7BB6', markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='sh1', markerfacecolor='#D95F02', markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='D', color='w', label='mean-sh0', markerfacecolor='#08519c', markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='D', color='w', label='mean-sh1', markerfacecolor='#993404', markeredgecolor='k', markersize=8),
]
ax.legend(handles=legend_elements, frameon=False, loc='upper right', fontsize=9)

ax.set_xticks([x0, x1])
ax.set_xticklabels(['sh0', 'sh1'])
ax.set_ylabel('PSNR (dB)')
ax.set_title('Paired PSNR')
ax.set_ylim(13, 29)
ax.set_xlim(0.3, 0.75)

ax.text(0.02, 0.03, f"n={n}  p={pval:.3f}  mean Δ={mean_delta:.3f} dB", transform=ax.transAxes, fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches='tight')
plt.close()

print('WROTE', OUT_PATH)
print('summary:', {'n': n, 'p': float(pval), 'mean_delta': mean_delta, 'ci': [float(lo), float(hi)]})
