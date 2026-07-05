import pandas as pd, numpy as np
input_dir = 'test_old/input'
gt_dir = 'test_old/gt'
for f in ['01', '02', '03']:
    inp = pd.read_csv(f'{input_dir}/{f}.csv')
    gt_full = pd.read_csv(f'{gt_dir}/{f}.csv')
    last_inp = inp.iloc[-1]
    # GT rows after last input
    gt = gt_full[gt_full['#'] > last_inp['#']]
    print(f'=== Scene {f} ===')
    print(f'Last input ball: pos=({last_inp["b_x"]:.1f},{last_inp["b_y"]:.1f}) vel=({last_inp["b_vx"]:.1f},{last_inp["b_vy"]:.1f})')
    print(f'GT ball frames: {len(gt)} rows after input')
    if len(gt) > 0:
        g0 = gt.iloc[0]
        g1 = gt.iloc[-1]
        print(f'GT first: pos=({g0["b_x"]:.1f},{g0["b_y"]:.1f}) vel=({g0["b_vx"]:.1f},{g0["b_vy"]:.1f})')
        print(f'GT last (f30): pos=({g1["b_x"]:.1f},{g1["b_y"]:.1f}) vel=({g1["b_vx"]:.1f},{g1["b_vy"]:.1f})')
        dt = 0.1
        pred_pos30 = last_inp[['b_x','b_y']].values.astype(float)
        pred_vel = last_inp[['b_vx','b_vy']].values.astype(float)
        for _ in range(30):
            pred_pos30 = pred_pos30 + pred_vel * dt
            pred_vel = pred_vel * 0.98
        err = np.sqrt(np.sum((pred_pos30 - g1[['b_x','b_y']].values.astype(float))**2))
        print(f'Simple decel (0.98^30) endpoint err: {err:.1f}m')
        pred_pos30_2 = last_inp[['b_x','b_y']].values.astype(float)
        pred_vel2 = last_inp[['b_vx','b_vy']].values.astype(float)
        for _ in range(30):
            pred_pos30_2 = pred_pos30_2 + pred_vel2 * dt
            pred_vel2 = pred_vel2 * 1.0
        err2 = np.sqrt(np.sum((pred_pos30_2 - g1[['b_x','b_y']].values.astype(float))**2))
        print(f'Const vel (1.0^30) endpoint err: {err2:.1f}m')
        # check goal location
        print(f'Last GT ball x={g1["b_x"]:.1f} (goal at x=+52.5)')
    print()
