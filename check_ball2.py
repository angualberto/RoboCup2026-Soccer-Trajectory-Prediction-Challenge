import pandas as pd, numpy as np, os
agents = [f'l{i}' for i in range(1,12)] + ['b']
for f in sorted([x for x in os.listdir('results/test/submission') if x[0].isdigit()]):
    sub = pd.read_csv(f'results/test/submission/{f}', index_col='#')
    gt_full = pd.read_csv(f'test_old/gt/{f}', index_col='#')
    gt = gt_full.loc[sub.index.intersection(gt_full.index)]
    print(f'=== {f} ===')
    for a in agents:
        sx,sy = sub[f'{a}_x'].values, sub[f'{a}_y'].values
        gx,gy = gt[f'{a}_x'].values, gt[f'{a}_y'].values
        e = np.sqrt((sx-gx)**2+(sy-gy)**2)
        print(f'  {a}: avg={np.mean(e):.1f}m, end={e[-1]:.1f}m')
    all_e = [np.mean(np.sqrt((sub[f'{a}_x'].values-gt[f'{a}_x'].values)**2+(sub[f'{a}_y'].values-gt[f'{a}_y'].values)**2)) for a in agents]
    print(f'  TOTAL: avg={np.mean(all_e):.1f}m')
    print()
