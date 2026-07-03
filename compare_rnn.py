import pandas as pd, numpy as np, os
agents = ["l1","l2","l3","l4","l5","l6","l7","l8","l9","l10","l11","b"]

rnn_dir = "C:/Users/Andre/Documents/time/desafio/STP-challenge-2025-main/rnn_results"
gtpa_dir = "C:/Users/Andre/Documents/time/desafio/STP-challenge-2025-main/results/test/submission"
gt_dir = "C:/Users/Andre/Documents/time/desafio/STP-challenge-2025-main/test_old/gt"

print(f"{'Scene':>5} {'Agent':>6} {'RNN avg':>8} {'GTPA avg':>8} {'RNN end':>8} {'GTPA end':>8}")
print("-"*55)
tot_r, tot_g = [], []
for f in ["01","02","03"]:
    gt = pd.read_csv(f"{gt_dir}/{f}.tracking.csv", index_col="#")
    for a in agents:
        rnn_f = f"{rnn_dir}/{f}.tracking.csv"
        gtpa_f = f"{gtpa_dir}/{f}.tracking.csv"
        if not os.path.exists(rnn_f):
            print(f"  {f} {a}: RNN file not found at {rnn_f}")
            continue
        sr = pd.read_csv(rnn_f, index_col="#")
        sg = pd.read_csv(gtpa_f, index_col="#")
        common = sr.index.intersection(gt.index)
        gr = gt.loc[common]
        e_r = np.sqrt((sr[f"{a}_x"]-gr[f"{a}_x"])**2+(sr[f"{a}_y"]-gr[f"{a}_y"])**2)
        e_g = np.sqrt((sg[f"{a}_x"]-gr[f"{a}_x"])**2+(sg[f"{a}_y"]-gr[f"{a}_y"])**2)
        print(f'{f:>5} {a:>6} {e_r.mean():>7.1f}m {e_g.mean():>7.1f}m {e_r.values[-1]:>7.1f}m {e_g.values[-1]:>7.1f}m')
        tot_r.append(e_r.mean())
        tot_g.append(e_g.mean())
    print()
print(f'{"TOTAL":>12} {np.mean(tot_r):>7.2f} {np.mean(tot_g):>7.2f}')
