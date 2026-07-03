import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, subprocess, sys

# ---- Step 1: generate test_old predictions ----
submit_dir = 'results/test/submission'
gt_dir = 'test_old/gt'
input_dir = 'test_old/input'

subprocess.run([sys.executable, 'main.py', '--model', 'gtpa', '--data', 'robocup2D',
    '--data_dir', 'robocup2d_data', '--batchsize', '16', '--totalTimeSteps', '20',
    '--challenge_data', 'test_old/input', '--cont',
    '--use_perturbation', '--pert_noise_scale', '0.2', '--pert_p_event', '1.0',
    '--pf_alpha', '0.5', '--pf_beta', '0.5', '--pf_gamma', '1.0', '--pf_num_particles', '32',
    '--use_recursive_memory', '--recursive_alpha', '0.3',
    '--use_intercept', '--intercept_beta', '0.5', '--intercept_horizon', '5', '--intercept_weight', '0.5'],
    check=True, capture_output=True)

# ---- Step 2: plot ----
files = sorted(os.listdir(submit_dir))
files = [f for f in files if f.endswith('.tracking.csv') and f[0].isdigit()]
agents = [f'l{i}' for i in range(1,12)] + ['b']

for fname in files:
    sub = pd.read_csv(os.path.join(submit_dir, fname), index_col='#')
    gt_full = pd.read_csv(os.path.join(gt_dir, fname), index_col='#')
    inp = pd.read_csv(os.path.join(input_dir, fname), index_col='#')
    common = sub.index.intersection(gt_full.index)
    gt = gt_full.loc[common]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    all_errs = []
    for idx, agent in enumerate(agents):
        if idx >= 12: break
        ax = axes[idx]
        gx, gy = gt[f'{agent}_x'].values, gt[f'{agent}_y'].values
        sx, sy = sub[f'{agent}_x'].values, sub[f'{agent}_y'].values
        ix, iy = inp[f'{agent}_x'].values, inp[f'{agent}_y'].values
        inp_idx = inp.index.values
        sub_idx = sub.index.values
        last_input = inp_idx[-1]
        err = np.sqrt((sx[-1]-gx[-1])**2 + (sy[-1]-gy[-1])**2)
        all_errs.append(err)
        ax.plot(inp_idx[-10:], ix[-10:], 'b.-', alpha=0.5, label='input')
        ax.plot(inp_idx[-10:], iy[-10:], 'b.-', alpha=0.3)
        ax.plot(sub_idx, sx, 'r.-', label='pred', linewidth=1.5, markersize=3)
        ax.plot(sub_idx, sy, 'r.-', alpha=0.5, linewidth=1.5, markersize=3)
        ax.plot(sub_idx, gx, 'g.-', label='gt', linewidth=1.5, markersize=3, alpha=0.7)
        ax.plot(sub_idx, gy, 'g.-', alpha=0.5, linewidth=1.5, markersize=3)
        ax.axvline(x=last_input, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f'{agent} (end err: {err:.1f}m)')
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('cycle')
        ax.set_ylabel('pos (m)')
    plt.suptitle(f'{fname} | Avg endpoint: {np.mean(all_errs):.1f}m')
    plt.tight_layout()
    out = f'results/test/{fname.replace(".tracking.csv","")}_plot.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved {out}')
