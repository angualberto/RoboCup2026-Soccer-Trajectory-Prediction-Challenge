import pandas as pd, numpy as np, os

agents = [f"l{i}" for i in range(1,12)] + ["b"]
sub_dir = "C:/Users/Andre/Documents/time/desafio/STP-challenge-2025-main/results/test/submission"
gt_dir = "C:/Users/Andre/Documents/time/desafio/STP-challenge-2025-main/test_old/gt"

for f in ["01","02","03"]:
    sub = pd.read_csv(f"{sub_dir}/{f}.tracking.csv", index_col="#")
    gt = pd.read_csv(f"{gt_dir}/{f}.tracking.csv", index_col="#")
    gt = gt.loc[sub.index.intersection(gt.index)]
    print(f"=== {f} ===")
    for a in agents:
        e = np.sqrt((sub[f"{a}_x"]-gt[f"{a}_x"])**2+(sub[f"{a}_y"]-gt[f"{a}_y"])**2)
        print(f"  {a}: avg={e.mean():.1f}m end={e.values[-1]:.1f}m")
    all_e = [np.mean(np.sqrt((sub[f"{a}_x"]-gt[f"{a}_x"])**2+(sub[f"{a}_y"]-gt[f"{a}_y"])**2)) for a in agents]
    print(f"  TOTAL: avg={np.mean(all_e):.2f}")
    print()
