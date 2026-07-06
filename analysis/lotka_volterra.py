"""
Lotka-Volterra N×N Multi-Agent Simulator (standalone analysis tool)

Simulates 30 frames of soccer using only pairwise interaction forces:
- Perseguir (chase): attraction toward opponent with ball
- Desviar (avoid): repulsion from nearby opponents
- Apoiar (support): cooperation toward teammate
- Bola: universal attraction

Usage:
    python analysis/lotka_volterra.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os, sys, glob

# Field dimensions (RoboCup 2D)
FIELD_W, FIELD_H = 105.0, 68.0
HALF_W, HALF_H = FIELD_W / 2, FIELD_H / 2

# Parameters
DT = 1.0
MAX_SPEED = 12.0
DAMPING = 0.92  # velocity multiplier per frame
BALL_GAMMA = 0.8  # fluid ball damping
BALL_SIGMA = 0.02

# Force strengths
K_CHASE = 0.35       # attraction to opponent with ball
K_AVOID = 0.8        # repulsion from nearby marker
K_SUPPORT = 0.12     # cooperation toward teammate
K_BALL = 0.45        # attraction to ball
K_GOAL = 0.08        # goalkeeper attraction to own goal
K_FORMATION = 0.05   # weak attraction to home position

# Interaction ranges
SIGMA_CHASE = 4.0    # m
SIGMA_AVOID = 2.5
SIGMA_SUPPORT = 6.0
D_AVOID_MIN = 2.0    # minimum distance for repulsion


def force_exp(dist, k, sigma):
    """Exponential force: F = k * sign * e^(-d/sigma)."""
    if dist < 0.01:
        return np.array([0.0, 0.0])
    return k * np.exp(-dist / sigma)


def clip_speed(vel, max_speed=MAX_SPEED):
    s = np.linalg.norm(vel)
    if s > max_speed:
        vel = vel / s * max_speed
    return vel


def field_clamp(pos):
    """Soft clamping to keep players on field."""
    margin = 0.5
    pos[0] = np.clip(pos[0], -HALF_W + margin, HALF_W - margin)
    pos[1] = np.clip(pos[1], -HALF_H + margin, HALF_H - margin)
    return pos


class LVPureSimulator:
    def __init__(self, initial_state):
        """
        initial_state: (T_obs, 23, 4) array
        Uses the LAST observed frame as t=0 for the simulation.
        """
        self.state = initial_state.copy()  # (23, 4) [x,y,vx,vy]
        self.T = 0
        self.ball_possession = np.zeros(22, dtype=bool)
        self._update_possession()

    def _update_possession(self):
        """Simple possession heuristic: closest player to ball."""
        ball_pos = self.state[22, :2]
        dists = [np.linalg.norm(self.state[i, :2] - ball_pos) for i in range(22)]
        min_dist = min(dists)
        # Player with ball if within 1m and on same team context
        if min_dist < 1.0:
            self.ball_possession[np.argmin(dists)] = True

    def _is_opponent(self, i, j):
        """Left team = 0-10, Right team = 11-21."""
        return (i < 11) != (j < 11)

    def _is_teammate(self, i, j):
        return (i < 11) == (j < 11) and i != j

    def _force_on_agent(self, i):
        """Compute total force on agent i from all other agents + ball."""
        pos_i = self.state[i, :2]
        vel_i = self.state[i, 2:4]
        force = np.zeros(2)

        # --- Interactions with other players ---
        for j in range(22):
            if i == j:
                continue
            pos_j = self.state[j, :2]
            delta = pos_j - pos_i
            dist = np.linalg.norm(delta)
            if dist < 0.01:
                continue
            direction = delta / dist

            opponent = self._is_opponent(i, j)

            # CHASE: adversary with ball (attraction)
            if opponent and self.ball_possession[j]:
                f = force_exp(dist, K_CHASE, SIGMA_CHASE)
                force += f * direction

            # AVOID: nearby adversary (repulsion)
            elif opponent and dist < D_AVOID_MIN * 2:
                f = force_exp(dist, K_AVOID, SIGMA_AVOID)
                force -= f * direction  # repulsion = negative

            # SUPPORT: teammate (weak cooperation)
            elif self._is_teammate(i, j):
                f = force_exp(dist, K_SUPPORT, SIGMA_SUPPORT)
                force += f * direction * 0.5

        # --- Ball influence ---
        ball_pos = self.state[22, :2]
        delta_ball = ball_pos - pos_i
        dist_ball = np.linalg.norm(delta_ball)
        if dist_ball > 0.01:
            direction_ball = delta_ball / dist_ball
            f_ball = force_exp(dist_ball, K_BALL, SIGMA_CHASE + 2.0)
            force += f_ball * direction_ball

        # --- Goalkeeper: stay near own goal ---
        if i == 0 or i == 11:  # left goalie = 0, right goalie = 11
            goal_pos = np.array([-HALF_W, 0.0]) if i == 0 else np.array([HALF_W, 0.0])
            delta_goal = goal_pos - pos_i
            dist_goal = np.linalg.norm(delta_goal)
            if dist_goal > 0.01:
                direction_goal = delta_goal / dist_goal
                force += K_GOAL * direction_goal

        return force

    def step(self):
        """Advance one frame using Heun integration."""
        dt = DT
        original_state = self.state.copy()

        # --- Ball: fluid ball damping ---
        ball_vel = self.state[22, 2:4]
        ball_drag = 1 - BALL_GAMMA * dt
        noise = np.random.randn(2) * BALL_SIGMA * np.sqrt(dt)
        self.state[22, 2:4] = ball_vel * ball_drag + noise

        # Heun for ball position
        k1 = original_state[22, 2:4]
        k2 = self.state[22, 2:4]
        heun_vel = (k1 + k2) / 2.0
        self.state[22, :2] += heun_vel * dt

        # --- Players: Lotka-Volterra forces ---
        new_vels = np.zeros((22, 2))
        for i in range(22):
            force = self._force_on_agent(i)
            # Euler step for velocity
            v_new = original_state[i, 2:4] + force * dt
            # Damping
            v_new *= DAMPING
            # Clip speed
            v_new = clip_speed(v_new)
            new_vels[i] = v_new

        # Heun for player positions (average k1, k2)
        for i in range(22):
            k1 = original_state[i, 2:4]
            k2 = new_vels[i]
            heun_vel = (k1 + k2) / 2.0
            new_pos = original_state[i, :2] + heun_vel * dt
            self.state[i, 2:4] = new_vels[i]
            self.state[i, :2] = field_clamp(new_pos)

        self.T += 1
        self._update_possession()

    def simulate(self, T=30):
        traj = [self.state.copy()]
        for _ in range(T):
            self.step()
            traj.append(self.state.copy())
        return np.array(traj)  # (T+1, 23, 4)


def load_initial_state(data_dir='test_old/input'):
    """Load last observed frame from first input CSV."""
    files = sorted(glob.glob(os.path.join(data_dir, '*.tracking.csv')))
    if not files:
        raise FileNotFoundError(f'No CSVs found in {data_dir}')
    df = pd.read_csv(files[0], index_col='#')
    state = np.zeros((23, 4), dtype=np.float64)
    # Input has: l1_x,l1_y,l1_vx,l1_vy,...,r11_x,...,b_x,b_y,b_vx,b_vy + other cols
    # Position columns: l1_x,l1_y,...,r11_x,r11_y,b_x,b_y
    for c, name in enumerate(['l', 'r']):
        for a in range(11):
            idx = a if name == 'l' else a + 11
            state[idx, 0] = df.iloc[-1, df.columns.get_loc(f'{name}{a+1}_x')]
            state[idx, 1] = df.iloc[-1, df.columns.get_loc(f'{name}{a+1}_y')]
            state[idx, 2] = df.iloc[-1, df.columns.get_loc(f'{name}{a+1}_vx')]
            state[idx, 3] = df.iloc[-1, df.columns.get_loc(f'{name}{a+1}_vy')]
    state[22, 0] = df.iloc[-1, df.columns.get_loc('b_x')]
    state[22, 1] = df.iloc[-1, df.columns.get_loc('b_y')]
    state[22, 2] = df.iloc[-1, df.columns.get_loc('b_vx')]
    state[22, 3] = df.iloc[-1, df.columns.get_loc('b_vy')]
    return state


def compute_gt_trajectory(gt_file, input_file):
    """Compute GT positions for simulation comparison window."""
    df_gt = pd.read_csv(gt_file, index_col='#')
    df_in = pd.read_csv(input_file, index_col='#')
    T_in = len(df_in)
    gt = df_gt.iloc[T_in:T_in+31]
    if len(gt) < 31:
        return None
    # GT has b_x,b_y then l1..l11_x,y then r1..r11_x,y = 46 cols
    cols_l = [f'l{a}_{c}' for a in range(1,12) for c in ['x','y']]
    cols_r = [f'r{a}_{c}' for a in range(1,12) for c in ['x','y']]
    cols_b = [f'b_{c}' for c in ['x','y']]
    arr = gt[cols_l + cols_r + cols_b].values.reshape(-1, 23, 2)
    out = np.zeros((arr.shape[0], 23, 4))
    out[:, :, :2] = arr
    return out


def plot_trajectory(traj, gt=None, title='LV Pure - Frame {}', save_path=None):
    """Plot one frame of the simulation."""
    fig, ax = plt.subplots(figsize=(10, 7))
    # Field
    ax.add_patch(plt.Rectangle((-HALF_W, -HALF_H), FIELD_W, FIELD_H,
                                fill=False, color='green', lw=2))
    ax.add_patch(plt.Rectangle((-HALF_W, -HALF_H), FIELD_W, FIELD_H,
                                color='green', alpha=0.1))
    # Center line
    ax.axvline(0, color='white', lw=1, alpha=0.5)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, color='white', lw=1, alpha=0.5))

    # Left team (blue)
    for a in range(11):
        ax.plot(traj[:, a, 0], traj[:, a, 1], 'b-', alpha=0.3, lw=0.5)
        ax.scatter(traj[-1, a, 0], traj[-1, a, 1], c='blue', s=30, zorder=5)
    # Right team (red)
    for a in range(11, 22):
        ax.plot(traj[:, a, 0], traj[:, a, 1], 'r-', alpha=0.3, lw=0.5)
        ax.scatter(traj[-1, a, 0], traj[-1, a, 1], c='red', s=30, zorder=5)
    # Ball (black)
    ax.plot(traj[:, 22, 0], traj[:, 22, 1], 'ko-', markersize=4, lw=1)
    ax.scatter(traj[-1, 22, 0], traj[-1, 22, 1], c='black', s=50, zorder=5)

    # GT ball trajectory (if available)
    if gt is not None:
        ax.plot(gt[:, 22, 0], gt[:, 22, 1], 'g--', alpha=0.5, lw=1, label='GT ball')
        # GT players (sample a few)
        for a in [0, 5, 10, 11, 16, 21]:
            ax.plot(gt[:, a, 0], gt[:, a, 1], 'g:', alpha=0.2, lw=0.5)

    ax.set_xlim(-HALF_W - 2, HALF_W + 2)
    ax.set_ylim(-HALF_H - 2, HALF_H + 2)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs('results/test/lv', exist_ok=True)

    # Load
    print('Loading input data...')
    init_state = load_initial_state('test_old/input')
    print(f'Initial state shape: {init_state.shape}')
    print(f'Ball pos: ({init_state[22,0]:.1f}, {init_state[22,1]:.1f})')

    # Load GT for comparison
    gt_files = sorted(glob.glob('test_old/gt/*.tracking.csv'))
    gt = None
    if gt_files:
        gt = compute_gt_trajectory(gt_files[0], sorted(glob.glob('test_old/input/*.tracking.csv'))[0])
        if gt is not None:
            print(f'GT trajectory loaded: {gt.shape}')

    # Simulate
    print('\nRunning LV N×N simulation for 30 frames...')
    sim = LVPureSimulator(init_state)
    traj = sim.simulate(30)
    print(f'Trajectory shape: {traj.shape}')

    # Analyze
    print(f'\nFinal ball pos: ({traj[-1,22,0]:.1f}, {traj[-1,22,1]:.1f})')
    players_on_field = sum(1 for a in range(22)
                           if abs(traj[-1, a, 0]) < HALF_W and abs(traj[-1, a, 1]) < HALF_H)
    print(f'Players on field (final frame): {players_on_field}/22')

    # Speed analysis
    speeds = [np.linalg.norm(traj[:, a, 2:4], axis=1).mean() for a in range(22)]
    print(f'Avg player speed: {np.mean(speeds):.2f} m/frame')
    print(f'Ball speed: {np.linalg.norm(traj[:, 22, 2:4], axis=1).mean():.2f} m/frame')

    # Formation preservation (compare initial vs final positions)
    init_avg_x = np.mean([init_state[a, 0] for a in range(11)])
    final_avg_x = np.mean([traj[-1, a, 0] for a in range(11)])
    print(f'Left team avg X: {init_avg_x:.1f} → {final_avg_x:.1f}')

    # Generate plots
    print('\nGenerating plots...')
    for t in [0, 5, 10, 20, 30]:
        plot_trajectory(traj[:t+1], gt=gt,
                       title=f'LV Pure - Frame {t} / 30',
                       save_path=f'results/test/lv/frame_{t:03d}.png')

    # Full trajectory as GIF frames
    print('Saving all 31 frames...')
    for t in range(0, 31, 2):
        plot_trajectory(traj[:t+1], gt=gt,
                       title=f'LV Pure - Frame {t} / 30',
                       save_path=f'results/test/lv/frame_{t:03d}.png')
    print('Done. Check results/test/lv/')

    # Summary metrics
    print('\n=== LV Pure Summary ===')
    print(f'  Players on field: {players_on_field}/22')
    print(f'  Avg speed: {np.mean(speeds):.2f} m/f')
    print(f'  Team drift: {abs(init_avg_x - final_avg_x):.1f} m')

    # Compare ball with GT
    if gt is not None:
        ball_error = np.mean([np.linalg.norm(traj[t, 22, :2] - gt[t, 22, :2])
                             for t in range(min(31, gt.shape[0]))])
        print(f'  Ball MAE vs GT: {ball_error:.2f} m')

    return traj


if __name__ == '__main__':
    traj = main()
