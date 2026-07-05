import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

submit_dir = './results/test/submission'
gt_dir = './example/ground-truth'
input_dir = './example/test-data'

files = sorted(os.listdir(submit_dir))
for fname in files:
    sub = pd.read_csv(os.path.join(submit_dir, fname), index_col='#')
    gt_full = pd.read_csv(os.path.join(gt_dir, fname), index_col='#')
    inp = pd.read_csv(os.path.join(input_dir, fname), index_col='#')

    # Align ground truth to submission indices
    common = sub.index.intersection(gt_full.index)
    gt = gt_full.loc[common]

    agents = [f'l{i}' for i in range(1,12)] + [f'r{i}' for i in range(1,12)] + ['b']

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    axes = axes.flatten()
    for idx, agent in enumerate(agents):
        if idx >= 23:
            axes[idx].set_visible(False)
            continue
        ax = axes[idx]
        gx, gy = gt[f'{agent}_x'].values, gt[f'{agent}_y'].values
        sx, sy = sub[f'{agent}_x'].values, sub[f'{agent}_y'].values
        ix, iy = inp[f'{agent}_x'].values, inp[f'{agent}_y'].values

        inp_idx = inp.index.values
        sub_idx = sub.index.values

        ax.plot(inp_idx, ix, 'b-', alpha=0.5, label='input')
        ax.plot(inp_idx, iy, 'b-', alpha=0.3)
        ax.plot(sub_idx, gx, 'g-', label='gt', linewidth=2)
        ax.plot(sub_idx, gy, 'g-', alpha=0.5, linewidth=2)
        ax.plot(sub_idx, sx, 'r--', label='pred', linewidth=2)
        ax.plot(sub_idx, sy, 'r--', alpha=0.5, linewidth=2)
        ax.set_title(agent)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.suptitle(fname.replace('.tracking.csv', ''), fontsize=14)
    plt.tight_layout()
    out = f'./results/test/{fname.replace(".tracking.csv", ".png")}'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved {out}')
