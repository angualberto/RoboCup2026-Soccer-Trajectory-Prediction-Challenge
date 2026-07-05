import math
import torch
import torch.nn.functional as F

from gtpa.volterra_memory import VolterraMemory
from gtpa.gillespie import GillespiePerturbation


class ParticleFilter:
    """
    Monte-Carlo particle filter following the GTPA+MC+Volterra algorithm.

    Flow
    ----
         History → CausalTransformer → GAT → base initial state
              ↓
         P particles (copies of base state)
              ↓
         Monte Carlo: velocity perturbation biased by Volterra memory
              ↓
         Propagate each particle forward (horizon steps)
              ↓
         Volterra Memory: distance to stored good trajectories
              ↓
         Compute weight = exp(-α·consensus - β·vel_dev - γ·volterra_dist)
              ↓
         Weighted average → final trajectory
              ↓
         EnKF correction (adaptive R) using Volterra as pseudo-measurement
              ↓
         Predictor–Corrector: Heun-style velocity update after EnKF
              ↓
         Update Volterra memory with highest-weight particles

    Parameters
    ----------
    use_perturbation : bool
        Enable Monte Carlo diversification.
    noise_scale : float
        Std of Gaussian perturbation added to initial velocity.
    p_event : float
        Probability that a given agent receives a perturbation.
    alpha, beta, gamma : float
        Coefficients for weight = exp(-α·C - β·V - γ·M).
    use_enkf : bool
        Enable Ensemble Kalman post-hoc correction.
    enkf_r : float
        Base observation noise for the Kalman gain.
    enkf_adaptive : bool
        Scale R with ensemble consensus — larger spread → smaller R → more correction.
    use_volterra_mc : bool
        Bias MC perturbation direction toward stored good trajectories.
    use_predictor_corrector : bool
        After EnKF corrects positions, update velocities for dynamic consistency.
    field_scale : float
        Characteristic soccer field scale for normalising consensus distances.
    """

    def __init__(self, model, num_particles=100, num_actions=10,
                 noise_scale=0.02, p_event=0.3, use_perturbation=False,
                 alpha=1.0, beta=0.5, gamma=2.0, memory_capacity=10,
                 use_enkf=False, enkf_r=1.0, enkf_adaptive=False,
                 use_volterra_mc=False, use_predictor_corrector=False,
                 field_scale=105.0,
                 use_recursive_memory=False, recursive_alpha=0.7,
                 use_interception=False, intercept_lambda=0.2,
                 use_intercept=False, intercept_beta=0.7,
                 intercept_horizon=5, intercept_weight=0.5,
                 use_pn_intercept=False, pn_beta=0.7,
                 pn_N=4.0, pn_k_lateral=2.0,
                 use_fluid_ball=False, fluid_ball_gamma=0.5,
                 fluid_ball_sigma=0.5,
                 fluid_ball_gamma_target=None, fluid_ball_gamma_tau=3.0,
                 use_hybrid_ball=False,
                 hybrid_gamma=0.6, hybrid_linear_speed=0.3,
                 hybrid_fluid_accel=0.5,
                 use_ocsvm_ball=False,
                 ocsvm_model_path='weights/ocsvm_ball.pkl',
                 integrator='euler',
                 use_dynamic_fallback=False,
                 fallback_w_dist=0.5, fallback_w_speed=0.3,
                 fallback_w_horizon=0.2, fallback_w_accel=0.0,
                 fallback_accel_max=30.0):
        self.model = model
        self.num_particles = num_particles
        self.num_actions = num_actions
        self.use_perturbation = use_perturbation
        self.noise_scale = noise_scale
        self.p_event = p_event
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_enkf = use_enkf
        self.enkf_r = enkf_r
        self.enkf_adaptive = enkf_adaptive
        self.use_volterra_mc = use_volterra_mc
        self.use_predictor_corrector = use_predictor_corrector
        self.field_scale = field_scale

        # Volterra trajectory store
        self.memory = VolterraMemory(capacity=memory_capacity)

        # Gillespie perturbation generator
        self.perturb = GillespiePerturbation(noise_scale=noise_scale,
                                             p_event=p_event)

        # Shortcuts
        self.P = num_particles

        # Recursive memory correction (step-wise Volterra query)
        self.use_recursive_memory = use_recursive_memory
        self.recursive_alpha = recursive_alpha  # blend: alpha*network + (1-alpha)*memory

        # Interception vector correction
        self.use_interception = use_interception
        self.intercept_lambda = intercept_lambda  # fraction of delta to apply

        # Intercept-aware dynamics
        self.use_intercept = use_intercept
        self.intercept_beta = intercept_beta  # blend: beta*network + (1-beta)*intercept
        self.intercept_horizon = intercept_horizon  # steps ahead for ball future
        self.intercept_weight = intercept_weight  # weight term in particle scoring

        # Proportional Navigation intercept (replaces simple intercept when enabled)
        self.use_pn_intercept = use_pn_intercept
        self.pn_beta = pn_beta  # blend with network velocity
        self.pn_N = pn_N  # navigation constant (3-5 typical)
        self.pn_k_lateral = pn_k_lateral  # lateral gain multiplier

        # Fluid ball: treat ball as particle in 2D fluid field (Langevin dynamics)
        self.use_fluid_ball = use_fluid_ball
        self.fluid_ball_gamma = fluid_ball_gamma  # drag coefficient (initial/max)
        self.fluid_ball_sigma = fluid_ball_sigma  # noise amplitude
        # Time-dependent gamma: gamma_eff(t) = gamma_target + (gamma - gamma_target) * exp(-t / tau)
        # When gamma_target == gamma, behaves as constant gamma (no time decay)
        if fluid_ball_gamma_target is None:
            self.fluid_ball_gamma_target = fluid_ball_gamma
        else:
            self.fluid_ball_gamma_target = fluid_ball_gamma_target
        self.fluid_ball_gamma_tau = fluid_ball_gamma_tau  # time constant (frames)

        # Hybrid ball: regime-aware dynamics (linear / fluid / chaotic)
        self.use_hybrid_ball = use_hybrid_ball
        self.hybrid_gamma = hybrid_gamma  # drag for FLUID regime
        self.hybrid_linear_speed = hybrid_linear_speed  # speed threshold for LINEAR regime
        self.hybrid_fluid_accel = hybrid_fluid_accel  # accel threshold for FLUID regime
        # Lorenz parameters
        self.lorenz_sigma = 10.0
        self.lorenz_rho = 28.0
        self.lorenz_beta = 8.0 / 3.0

        # OCSVM confidence-based ball dynamics
        self.use_ocsvm_ball = use_ocsvm_ball
        self.ocsvm_model = None
        if self.use_ocsvm_ball:
            import pickle, os
            path = ocsvm_model_path
            for p in [path, os.path.join('..', path), os.path.join(os.path.dirname(__file__), '..', 'weights', 'ocsvm_ball.pkl')]:
                if os.path.exists(p):
                    with open(p, 'rb') as f:
                        self.ocsvm_model = pickle.load(f)
                    print(f'[PF] Loaded OCSVM model from {p}')
                    break
            if self.ocsvm_model is None:
                print(f'[PF] WARNING: OCSVM model not found, disabling')
                self.use_ocsvm_ball = False

        # Integrator: euler, heun, simpson, ab2
        self.integrator = integrator
        self._ball_vel_buffer = []  # stores last 2 ball velocities for multi-step integrators

        # Dynamic fallback: blend baseline (pure network) and Lorentz-corrected
        # ball trajectory using a stability-based weight α ∈ [0,1].
        # When baseline is stable, α ≈ 0 → use more accurate baseline prediction.
        # When ball diverges, α → 1 → Lorentz correction kicks in.
        self.use_dynamic_fallback = use_dynamic_fallback
        self.fallback_w_dist = fallback_w_dist
        self.fallback_w_speed = fallback_w_speed
        self.fallback_w_horizon = fallback_w_horizon
        self.fallback_w_accel = fallback_w_accel
        self.fallback_accel_max = fallback_accel_max
        # Field half-diagonal for normalising ball distance from centre
        self.field_half_diag = math.sqrt((105.0/2)**2 + (68.0/2)**2)  # ~62.5 m

        # Alpha trace: logs blend weight per scene/frame for analysis
        self.alpha_trace = []  # list of dicts: {scene, frame, alpha}

    # ------------------------------------------------------------------
    #  Weight computation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_weights(self, traj: torch.Tensor, base_state: torch.Tensor,
                         batch_size: int, P: int) -> torch.Tensor:
        """
        traj:       (T, B*P, 23, 4)
        base_state: (B, 23, 4)  — deterministic initial state (without perturbation)
        returns:    (B*P,) softmax-normalised weights (sum=1 per batch element)

        Weight = exp( -α * consensus_dist
                      -β * vel_deviation
                      -γ * volterra_dist )
        """
        T = traj.size(0)

        # ----- 1. Consensus distance: how far this particle is from the mean -----
        pos = traj[:, :, :, 0:2]                     # (T, B*P, 23, 2)
        pos_batched = pos.view(T, batch_size, P, 23, 2)

        pos_mean = pos_batched.mean(dim=2, keepdim=True)  # (T, B, 1, 23, 2)
        consensus_dist = torch.norm(pos_batched - pos_mean, dim=-1).mean(dim=(0, 3))  # (B, P)
        consensus_dist = consensus_dist.reshape(batch_size * P)  # (B*P,)

        # ----- 2. Velocity deviation from base -----
        vel = traj[:, :, :22, 2:4]                   # (T, B*P, 22, 2)
        base_vel = base_state[:, :22, 2:4].unsqueeze(0)  # (1, B, 22, 2)
        base_vel = base_vel.repeat_interleave(P, dim=1)  # (1, B*P, 22, 2)
        vel_dev = torch.norm(vel - base_vel, dim=-1).mean(dim=(0, 2))  # (B*P,)

        # ----- 3. Volterra distance (final positions vs stored trajectories) -----
        final_pos = traj[-1, :, :, 0:2]               # (B*P, 23, 2)
        volterra_dist = torch.zeros(batch_size * P, device=traj.device)
        for p in range(batch_size * P):
            volterra_dist[p] = self.memory.query(final_pos[p])

        # ----- Combine -----
        log_weight = -(self.alpha * consensus_dist
                       + self.beta * vel_dev
                       + self.gamma * volterra_dist)
        weight = torch.exp(log_weight)

        # Normalise per batch element
        weight_batched = weight.reshape(batch_size, P)
        weight_batched = weight_batched / (weight_batched.sum(dim=-1, keepdim=True) + 1e-8)
        return weight_batched.reshape(batch_size * P)

    # ------------------------------------------------------------------
    #  Adaptive EnKF correction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _enkf_correct(self, out: torch.Tensor, traj: torch.Tensor,
                       batch_size: int, P: int, burn_in: int) -> torch.Tensor:
        """
        Ensemble Kalman correction with adaptive R.

        R_t = R_base / (1 + consensus_dist / field_scale)

        When particles agree (small consensus_dist), R_t ≈ R_base → K moderate.
        When particles disagree (large consensus_dist), R_t is smaller →
        K larger → Volterra memory has more influence.
        """
        if not self.memory.trajectories:
            return out

        T = out.size(0)
        device = out.device
        out = out.clone()

        for b in range(batch_size):
            lo, hi = b * P, (b + 1) * P
            particle_final = traj[-1, lo:hi, :, 0:2]         # (P, 23, 2)
            weighted_final = out[-1, 0, b].view(23, 4)[:, :2].clone()

            # ---- Adaptive observation noise ----
            # Mean ensemble spread (consensus)
            ensemble_mean = particle_final.mean(dim=0)       # (23, 2)
            consensus = torch.norm(particle_final - ensemble_mean.unsqueeze(0),
                                   dim=-1).mean()            # scalar

            if self.enkf_adaptive:
                # R decreases as consensus grows (more uncertainty → more correction)
                r_eff = self.enkf_r / (1.0 + consensus / self.field_scale)
            else:
                r_eff = self.enkf_r

            # ---- Retrieve closest memory trajectory (quality-weighted) ----
            mem_final = self.memory.query_best(weighted_final)
            if mem_final is None:
                continue
            mem_final = mem_final.to(device)

            # ---- Kalman gain ----
            ensemble_var = particle_final.var(dim=0)          # (23, 2)
            K = ensemble_var / (ensemble_var + r_eff)         # (23, 2)

            # ---- Ramp and apply ----
            for t in range(T):
                if t < burn_in - 1:
                    continue
                alpha = (t - (burn_in - 1)) / max(1, T - burn_in)
                correction = K * (mem_final - weighted_final) * alpha
                out[t, 0, b].view(23, 4)[:, :2] += correction

        return out

    # ------------------------------------------------------------------
    #  Predictor–Corrector (velocity correction after EnKF)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _predictor_corrector(self, out: torch.Tensor) -> torch.Tensor:
        """Heun-style velocity update after EnKF corrected the positions.

        For each timestep:
          v_corrected = (pos_{t+1} - pos_t) / dt

        This ensures the velocity field is dynamically consistent with the
        corrected trajectory, preventing the "jump" that a pure position
        correction would introduce.
        """
        out = out.clone()
        T = out.size(0)
        dt = 1.0

        for t in range(T - 1):
            pos_t = out[t, 0].view(-1, 23, 4)[:, :, 0:2]      # (B, 23, 2)
            pos_t1 = out[t + 1, 0].view(-1, 23, 4)[:, :, 0:2]  # (B, 23, 2)
            vel_consistent = (pos_t1 - pos_t) / dt
            # Write back the corrected velocity (players only)
            out[t, 0].view(-1, 23, 4)[:, :22, 2:4] = vel_consistent[:, :22, :]
        return out

    # ------------------------------------------------------------------
    #  Volterra update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_memory(self, traj: torch.Tensor, weights: torch.Tensor,
                       batch_size: int, P: int,
                       out: torch.Tensor) -> None:
        """Store the highest-weight particle per batch element.

        Also stores a quality score based on how close the weighted-average
        trajectory was to the ground truth (approximated by the inverse of
        the velocity deviation from the base).
        """
        w = weights.reshape(batch_size, P)
        for b in range(batch_size):
            lo, hi = b * P, (b + 1) * P
            best = lo + w[b].argmax()

            # Quality = 1 / (1 + mean(vel_dev)) from the best particle
            particle_final = traj[-1, best, :, 0:2]           # (23, 2)
            weighted_final = out[-1, 0, b].view(23, 4)[:, :2]
            quality = 1.0 / (1.0 + torch.norm(particle_final - weighted_final.to(traj.device),
                                               dim=-1).mean().item())

            self.memory.add(traj[:, best], float(w[b].max().detach().cpu()),
                            quality=quality)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #  OCSVM-based gamma: extract ball features -> score -> gamma
    # ------------------------------------------------------------------
    def _compute_ocsvm_gamma(self, next_state, current, dt):
        """Returns gamma_eff per particle based on OCSVM confidence score."""
        import numpy as np
        ball_vel = next_state[0:1, 22:23, 2:4].squeeze().cpu().numpy()  # (2,)
        ball_pos = next_state[0:1, 22:23, 0:2].squeeze().cpu().numpy()
        prev_vel = current[0:1, 22:23, 2:4].squeeze().cpu().numpy()
        bvx, bvy = float(ball_vel[0]), float(ball_vel[1])
        speed = np.linalg.norm(ball_vel)
        acc = np.linalg.norm(ball_vel - prev_vel) / max(dt, 1e-8)

        # Direction change (angle diff from prev frame)
        angle = np.arctan2(bvy, bvx) if speed > 0.01 else 0.0
        p_speed = np.linalg.norm(prev_vel)
        angle_p = np.arctan2(float(prev_vel[1]), float(prev_vel[0])) if p_speed > 0.01 else 0.0
        ad = abs(angle - angle_p)
        angle_diff = min(ad, 2*np.pi - ad)

        # Player distances
        min_dist = float('inf')
        nearest_speed = 0.0
        for agent_idx in range(22):  # 22 players
            px = float(next_state[0, agent_idx, 0].cpu().item())
            py = float(next_state[0, agent_idx, 1].cpu().item())
            pvx = float(next_state[0, agent_idx, 2].cpu().item())
            pvy = float(next_state[0, agent_idx, 3].cpu().item())
            d = np.sqrt((float(ball_pos[0])-px)**2 + (float(ball_pos[1])-py)**2)
            if d < min_dist:
                min_dist = d
                nearest_speed = np.sqrt(pvx**2 + pvy**2)

        features = np.array([[speed, acc, angle_diff, min_dist, nearest_speed,
                              float(ball_pos[0]), float(ball_pos[1])]])
        mdl = self.ocsvm_model
        X = mdl['scaler'].transform(features)
        Xp = mdl['pca'].transform(X)
        score = float(mdl['ocsvm'].score_samples(Xp)[0])

        # Score to gamma: score high (=normal) -> gamma high; score low (=unusual) -> gamma low
        smin, smax = mdl.get('score_min', 0.0), mdl.get('score_max', 100.0)
        score_norm = np.clip((score - smin) / max(smax - smin, 1e-8), 0.0, 1.0)
        gamma_eff = 0.35 + 0.25 * score_norm  # [0.35, 0.60]
        return gamma_eff

    # ------------------------------------------------------------------
    #  Step-wise simulation with exponential trajectory smoothing
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _stepwise_simulate(self, initial_state, burn_in, horizon, fs=1.0):
        """Per-step propagation with exponential trajectory smoothing.

        At each step t:
          1. Propagate all P particles one step
          2. Compute weighted average (consensus)
          3. Smooth: smoothed[t] = c3*raw[t] + c2*smoothed[t-1] + c1*smoothed[t-2]
          4. Use smoothed[t] as the next input state

        Smoothing prevents any single erroneous step from derailing the
        autoregressive trajectory.
        """
        device = initial_state.device
        B = initial_state.size(2)
        P = self.P
        dt = 1.0 / fs if fs > 0 else 1.0
        T = horizon
        n_agents = 23
        mem_steps = getattr(self.model, 'memory_steps', 3)
        c1, c2, c3 = 0.1, 0.3, 0.6  # smoothing coefficients

        # ---- Allocate ----
        smooth_traj = torch.zeros(B, T, n_agents, 4, device=device)  # smoothed
        raw_traj = torch.zeros(B, T, n_agents, 4, device=device)     # raw network output

        # ---- Burn-in: all particles share the observed state ----
        for t in range(burn_in):
            init = initial_state[t, 0].view(B, -1, 4)
            smooth_traj[:, t] = init
            raw_traj[:, t] = init
            for p in range(P):
                _pad = initial_state[t, 0].view(-1, 23, 4)
                if p == 0:
                    particle_states = _pad
                else:
                    particle_states = torch.cat([particle_states, _pad], dim=0)

        current = particle_states.clone()

        # ---- Pre-fill latent/velocity history ----
        latent_history: list[torch.Tensor] = []
        vel_history: list[torch.Tensor] = []
        ball_vel_history: list[torch.Tensor] = []
        if hasattr(self.model, '_encode_latent'):
            for t in range(burn_in - 1):
                st = initial_state[t, 0].repeat_interleave(P, dim=0).view(-1, n_agents, 4)
                latent = self.model._encode_latent(st)
                latent_history.append(latent.detach())
                vel_history.append(st[:, :22, 2:4].detach())
                ball_vel_history.append(st[0:1, 22:23, 2:4].detach())
            latent = self.model._encode_latent(current)
            latent_history.append(latent.detach())

        # ---- Per-agent perturbations (once at start) ----
        if self.use_perturbation and P > 1:
            players_mask = torch.zeros(1, n_agents, 1, device=device)
            players_mask[:, :22] = 1.0
            event = (torch.rand(B * P, n_agents, 1, device=device) < self.p_event).float()
            noise = torch.randn(B * P, n_agents, 2, device=device) * self.noise_scale
            current[:, :, 2:4] = current[:, :, 2:4] + noise * event * players_mask

        # ---- Step-wise propagation with smoothing ----
        self._ball_vel_buffer = []  # reset Simpson buffer per scene
        for step in range(burn_in - 1, horizon - 1):
            if len(latent_history) >= 2:
                hist = latent_history[-(mem_steps - 1):]
            else:
                hist = None

            # Propagate one step for all P particles
            next_state, latent, _ = self.model.predict_state(
                current, latent_history=hist, vel_history=vel_history,
                ball_vel_history=ball_vel_history, dt=dt,
            )

            # ---- Dynamic Tensor Intercept (LOS-based anisotropic gain) ----
            if self.use_pn_intercept:
                ball_pos = current[-1:, 22:23, 0:2]
                ball_vel = current[-1:, 22:23, 2:4]
                for p in range(B * P):
                    ppos = next_state[p:p+1, :22, 0:2]
                    pvel = next_state[p:p+1, :22, 2:4]
                    r_rel = ball_pos - ppos
                    v_rel = ball_vel - pvel
                    los_dist = torch.norm(r_rel, dim=-1, keepdim=True) + 1e-6
                    los_angle = torch.atan2(r_rel[:,:,1], r_rel[:,:,0])
                    los_unit = r_rel / los_dist
                    cross2d = r_rel[:,:,0:1] * v_rel[:,:,1:2] - r_rel[:,:,1:2] * v_rel[:,:,0:1]
                    los_rate = cross2d / (los_dist ** 2 + 1e-6)
                    abs_los_rate = torch.clamp(torch.abs(los_rate), max=5.0)
                    Vc = torch.clamp(-(r_rel * v_rel).sum(dim=-1, keepdim=True) / los_dist, min=0.01)
                    t_go = torch.clamp(los_dist / Vc, max=self.intercept_horizon * dt)
                    intercept = ball_pos + ball_vel * t_go
                    to_int = intercept - ppos
                    to_int_n = to_int / (torch.norm(to_int, dim=-1, keepdim=True) + 1e-6)
                    speed = torch.norm(next_state[p:p+1, :22, 2:4], dim=-1, keepdim=True)
                    # Build dynamic tensor: radial = 1.0, lateral = 1 + k_lateral * |los_rate|
                    k_r = 1.0
                    k_l = 1.0 + self.pn_k_lateral * abs_los_rate
                    c = torch.cos(los_angle).unsqueeze(-1)  # (1, 22, 1)
                    s = torch.sin(los_angle).unsqueeze(-1)
                    A_xx = k_r * c**2 + k_l * s**2
                    A_xy = (k_r - k_l) * c * s
                    A_yy = k_r * s**2 + k_l * c**2
                    v_dir = pvel / (torch.norm(pvel, dim=-1, keepdim=True) + 1e-6)
                    v_new_x = A_xx * v_dir[:,:,0:1] + A_xy * v_dir[:,:,1:2]
                    v_new_y = A_xy * v_dir[:,:,0:1] + A_yy * v_dir[:,:,1:2]
                    v_new = torch.cat([v_new_x, v_new_y], dim=-1)
                    v_new = v_new / (torch.norm(v_new, dim=-1, keepdim=True) + 1e-6) * speed
                    # Blend with original network velocity
                    next_state[p, :22, 2:4] = (self.pn_beta * next_state[p, :22, 2:4]
                                               + (1 - self.pn_beta) * v_new[0])
            # ---- Simple intercept-aware velocity correction (fallback) ----
            elif self.use_intercept:
                ball_pos = current[-1:, 22:23, 0:2]
                ball_vel = current[-1:, 22:23, 2:4]
                ball_future = ball_pos + ball_vel * self.intercept_horizon * dt
                for p in range(B * P):
                    player_pos = next_state[p:p+1, :22, 0:2]
                    to_ball = ball_future - player_pos
                    to_ball_norm = torch.norm(to_ball, dim=-1, keepdim=True)
                    speed = torch.norm(next_state[p:p+1, :22, 2:4], dim=-1, keepdim=True)
                    v_intercept = to_ball / (to_ball_norm + 1e-6) * speed
                    next_state[p, :22, 2:4] = (self.intercept_beta * next_state[p, :22, 2:4]
                                               + (1 - self.intercept_beta) * v_intercept[0])

            # ---- Fluid ball: Langevin particle noise per PF particle ----
            if self.use_fluid_ball and P > 1:
                f_pred = step - burn_in + 2
                if self.use_ocsvm_ball and self.ocsvm_model is not None:
                    gamma_eff = self._compute_ocsvm_gamma(next_state, current, dt)
                else:
                    gamma_eff = (self.fluid_ball_gamma_target + (self.fluid_ball_gamma - self.fluid_ball_gamma_target)
                                 * math.exp(-f_pred / self.fluid_ball_gamma_tau))
                ball_drag = (1 - gamma_eff * dt)
                ball_noise = (torch.randn(B * P, 1, 2, device=device)
                              * self.fluid_ball_sigma * math.sqrt(dt))
                ball_lorentz = next_state[:, 22:23, 2:4] * ball_drag + ball_noise

                if self.use_dynamic_fallback:
                    ball_baseline = next_state[:, 22:23, 2:4].clone()
                    ball_pos = next_state[:, 22:23, 0:2]
                    dist_norm = (torch.norm(ball_pos, dim=-1) / self.field_half_diag).clamp(0.0, 1.0)
                    speed_norm = torch.sigmoid((torch.norm(ball_baseline, dim=-1) - 10.0) / 3.0)
                    f_pred = step - burn_in + 2
                    F_pred = horizon - burn_in + 1
                    step_frac = (f_pred - 1) / max(F_pred - 1, 1)
                    prev_ball_vel = current[0:1, 22:23, 2:4]
                    ball_accel = torch.norm(ball_baseline[0:1] - prev_ball_vel, dim=-1) / max(dt, 1e-8)
                    accel_norm = (ball_accel / self.fallback_accel_max).clamp(0.0, 1.0)
                    alpha = (self.fallback_w_dist * dist_norm
                             + self.fallback_w_speed * speed_norm
                             + self.fallback_w_horizon * step_frac
                             + self.fallback_w_accel * accel_norm).clamp(0.0, 1.0)
                    alpha = alpha.view(-1, 1).unsqueeze(-1)  # (B*P, 1, 1)
                    # Ball always gets full Lorentz correction; players use dynamic blend
                    ball_final = ball_lorentz
                    next_state[:, 22:23, 2:4] = ball_final
                    self.alpha_trace.append({
                        'scene': B, 'frame': f_pred,
                        'alpha': alpha[0, 0, 0].item(),
                        'alpha_ball': 1.0,
                        'dist_norm': dist_norm[0, 0].item(),
                        'speed_norm': speed_norm[0, 0].item(),
                        'accel_norm': accel_norm[0, 0].item(),
                    })
                else:
                    next_state[:, 22:23, 2:4] = ball_lorentz

                # Integrator for ball position
                if self.integrator == 'legacy':
                    next_state[:, 22:23, 0:2] = next_state[:, 22:23, 0:2] + ball_noise * dt * 0.5
                else:
                    ball_vel_t = current[0:1, 22:23, 2:4]  # v_t
                    ball_pos_t = current[0:1, 22:23, 0:2]  # x_t
                    n = ball_lorentz.shape[0]  # B*P
                    v_tp1 = ball_lorentz  # damped velocity at t+1
                    v_t_exp = ball_vel_t.expand(n, 1, 2).clone()

                    if self.integrator == 'euler':
                        integrated_vel = v_tp1
                    elif self.integrator == 'heun':
                        integrated_vel = (v_t_exp + v_tp1) / 2.0
                    elif self.integrator == 'simpson':
                        self._ball_vel_buffer.append(ball_vel_t.detach().clone())
                        if len(self._ball_vel_buffer) >= 3:
                            v_prev2 = self._ball_vel_buffer[-3].expand(n, 1, 2)
                            v_prev1 = self._ball_vel_buffer[-2].expand(n, 1, 2)
                            integrated_vel = (v_prev2 + 4.0 * v_prev1 + v_tp1) / 6.0
                        else:
                            integrated_vel = v_tp1
                    elif self.integrator == 'ab2':
                        self._ball_vel_buffer.append(ball_vel_t.detach().clone())
                        if len(self._ball_vel_buffer) >= 2:
                            v_prev = self._ball_vel_buffer[-2].expand(n, 1, 2)
                            integrated_vel = (3.0 * v_t_exp - v_prev) / 2.0
                        else:
                            integrated_vel = v_tp1
                    else:
                        integrated_vel = v_tp1

                    ball_pos_t_exp = ball_pos_t.expand(n, 1, 2)
                    next_state[:, 22:23, 0:2] = ball_pos_t_exp + integrated_vel * dt

            # ---- Hybrid ball: regime-aware damping ----
            if self.use_hybrid_ball and P > 1:
                ball_vel = next_state[:, 22:23, 2:4].squeeze(1)
                ball_pos = next_state[:, 22:23, 0:2].squeeze(1)
                prev_ball_vel = current[0:1, 22:23, 2:4].squeeze(1).expand_as(ball_vel)

                speed = torch.norm(ball_vel, dim=-1, keepdim=True)
                acc_norm = torch.norm(ball_vel - prev_ball_vel, dim=-1, keepdim=True) / max(dt, 1e-8)

                # LOW speed: keep moving (γ=0.1, almost no damping)
                gamma_low = 0.1
                # NORMAL: standard damping (γ=0.6)
                gamma_norm = self.hybrid_gamma
                # HIGH accel: trust network more (γ=0.3, less damping)
                gamma_high = 0.3

                # Soft transition between regimes
                w_low = torch.sigmoid((self.hybrid_linear_speed - speed) * 20.0)
                w_high = torch.sigmoid((acc_norm - self.hybrid_fluid_accel) * 10.0)
                w_norm = (1.0 - w_low) * (1.0 - w_high)

                gamma_eff = w_low * gamma_low + w_norm * gamma_norm + w_high * gamma_high
                ball_drag = 1.0 - gamma_eff * dt
                ball_noise = (torch.randn(B * P, 2, device=device) * 0.02 * math.sqrt(dt))
                new_vel = ball_vel * ball_drag + ball_noise

                next_state[:, 22:23, 2:4] = new_vel.unsqueeze(1)
                next_state[:, 22:23, 0:2] = ball_pos.unsqueeze(1) + new_vel.unsqueeze(1) * dt

            # ---- Per-step weights (velocity deviation + intercept alignment) ----
            p_vel = next_state[:, :22, 2:4].reshape(B, P, 22, 2)    # (B, P, 22, 2)
            mean_vel = p_vel.mean(dim=1, keepdim=True)               # (B, 1, 22, 2)
            vel_dev = torch.norm(p_vel - mean_vel, dim=-1).mean(dim=-1)  # (B, P)

            if self.use_pn_intercept or self.use_intercept:
                ball_pos = current[-1:, 22:23, 0:2]
                ball_vel = current[-1:, 22:23, 2:4]
                if self.use_pn_intercept:
                    p_pos = next_state[:, :22, 0:2].reshape(B, P, 22, 2)
                    r_rel = ball_pos - p_pos
                    los_d = torch.norm(r_rel, dim=-1, keepdim=True) + 1e-6
                    Vc = torch.clamp(-(r_rel * (ball_vel - p_vel)).sum(dim=-1, keepdim=True) / los_d, min=0.01)
                    t_go = torch.clamp(los_d / Vc, max=self.intercept_horizon * dt)
                    to_intercept = ball_pos + ball_vel * t_go - p_pos
                else:
                    ball_future = ball_pos + ball_vel * self.intercept_horizon * dt
                    to_intercept = ball_future - next_state[:, :22, 0:2].reshape(B, P, 22, 2)
                to_int_unit = to_intercept / (torch.norm(to_intercept, dim=-1, keepdim=True) + 1e-6)
                v_unit = p_vel / (torch.norm(p_vel, dim=-1, keepdim=True) + 1e-6)
                align = (v_unit * to_int_unit).sum(dim=-1).mean(dim=-1)
                vel_dev = vel_dev - self.intercept_weight * align

            w_step = torch.exp(-vel_dev)
            w_step = w_step / (w_step.sum(dim=-1, keepdim=True) + 1e-8)  # (B, P)

            # ---- Weighted consensus (raw) ----
            for b in range(B):
                lo, hi = b * P, (b + 1) * P
                w_b = w_step[b:b+1, :, None, None]  # (1, P, 1, 1)
                weighted = (next_state[lo:hi] * w_b).sum(dim=1, keepdim=True)  # (1, 23, 4)
                raw_traj[b, step + 1] = weighted[0]

                # Exponentially smoothed trajectory
                if step - burn_in + 1 < 2:
                    smooth_traj[b, step + 1] = raw_traj[b, step + 1]
                else:
                    smooth_traj[b, step + 1] = (c3 * raw_traj[b, step + 1]
                                                + c2 * smooth_traj[b, step]
                                                + c1 * smooth_traj[b, step - 1])

            # ---- Use smoothed state as input for next step ----
            for b in range(B):
                lo, hi = b * P, (b + 1) * P
                smooth_state = smooth_traj[b, step + 1].unsqueeze(0)  # (1, 23, 4)
                # Correct all particles toward the smoothed state
                for p in range(lo, hi):
                    next_state[p] = (self.recursive_alpha * next_state[p]
                                     + (1 - self.recursive_alpha) * smooth_state)

            # ---- Advance ----
            vel_history.append(current[:, :22, 2:4].detach())
            ball_vel_history.append(current[0:1, 22:23, 2:4].detach())
            latent_history.append(latent.detach())
            current = next_state

        # ---- Format output (smoothed trajectory) ----
        out = smooth_traj.permute(1, 2, 0, 3).reshape(T, 1, B, 92)
        return out

    # ------------------------------------------------------------------
    #  Main simulation
    # ------------------------------------------------------------------
    def simulate_trajectories(self, initial_state, burn_in, horizon, fs=1.0):
        # Dispatch to stepwise simulation when recursive memory is active
        if self.use_recursive_memory and self.use_perturbation and self.P > 1:
            return self._stepwise_simulate(initial_state, burn_in, horizon, fs)

        device = initial_state.device
        batch_size = initial_state.size(2)
        P = self.num_particles
        dt = 1.0 / fs if fs > 0 else 1.0
        T = horizon
        n_agents = 23
        mem_steps = getattr(self.model, 'memory_steps', 3)

        # ---------- allocate ----------
        traj = torch.zeros(T, batch_size * P, n_agents, 4, device=device)

        for t in range(burn_in):
            traj[t] = initial_state[t, 0].repeat_interleave(P, dim=0).view(-1, n_agents, 4)

        # Current state at burn_in point
        current = traj[burn_in - 1].clone()            # (B*P, 23, 4)

        # Save the deterministic (unperturbed) base state for weight computation
        base_state = initial_state[burn_in - 1, 0].view(-1, 23, 4)     # (B, 23, 4)

        # ----- Monte Carlo: perturb initial velocity (ONCE) -----
        if self.use_perturbation and P > 1:
            players_mask = torch.zeros(1, n_agents, 1, device=device)
            players_mask[:, :22] = 1.0

            # Base random noise
            event = (torch.rand(batch_size * P, n_agents, 1, device=device) < self.p_event).float()
            noise = torch.randn(batch_size * P, n_agents, 2, device=device) * self.noise_scale

            # ---- Volterra-guided bias ----
            if self.use_volterra_mc:
                # For each batch, get the mean direction from memory
                for b in range(batch_size):
                    lo, hi = b * P, (b + 1) * P
                    curr_pos = current[lo, :22, 0:2]          # (22, 2) — first particle as reference
                    direction = self.memory.get_mean_direction(curr_pos)
                    if direction is not None:
                        # Bias particles in this batch toward memory direction
                        bias = direction[:22, :] * self.noise_scale * 0.5
                        noise[lo:hi, :22, :] = noise[lo:hi, :22, :] + bias.unsqueeze(0)

            current[:, :, 2:4] = current[:, :, 2:4] + noise * event * players_mask

        # ----- Pre-fill latent history and velocity history -----
        latent_history: list[torch.Tensor] = []
        vel_history: list[torch.Tensor] = []
        if hasattr(self.model, '_encode_latent'):
            with torch.no_grad():
                for t in range(burn_in - 1):
                    latent = self.model._encode_latent(traj[t])
                    latent_history.append(latent.detach())
                    vel_history.append(traj[t, :, :22, 2:4].detach())
                latent = self.model._encode_latent(current)
                latent_history.append(latent.detach())

        # ----- Propagate all particles -----
        for step in range(burn_in - 1, horizon - 1):
            if len(latent_history) >= 2:
                recent = latent_history[-(mem_steps - 1):]
                hist = recent
            else:
                hist = None

            with torch.no_grad():
                next_state, latent, _ = self.model.predict_state(
                    current, latent_history=hist, vel_history=vel_history, dt=dt,
                )

            # ---- Fluid ball with Lorentz-inspired drag ----
            if self.use_fluid_ball and P > 1:
                # Lorentz factor: grows with prediction horizon
                f_pred = step - burn_in + 2  # 1-indexed prediction frame
                F_pred = horizon - burn_in + 1
                if self.use_ocsvm_ball and self.ocsvm_model is not None:
                    gamma_eff = self._compute_ocsvm_gamma(next_state, current, dt)
                else:
                    gamma_eff = (self.fluid_ball_gamma_target + (self.fluid_ball_gamma - self.fluid_ball_gamma_target)
                                 * math.exp(-f_pred / self.fluid_ball_gamma_tau))
                if f_pred >= F_pred:
                    ball_drag = 0.0
                else:
                    gamma_lorentz = 1.0 / math.sqrt(max(1e-8, 1.0 - (f_pred / F_pred)**2))
                    ball_drag = gamma_eff / gamma_lorentz
                ball_noise = (torch.randn(batch_size * P, 1, 2, device=device)
                              * self.fluid_ball_sigma * math.sqrt(dt))
                ball_lorentz = next_state[:, 22:23, 2:4] * ball_drag + ball_noise

                if self.use_dynamic_fallback:
                    ball_baseline = next_state[:, 22:23, 2:4].clone()
                    ball_pos = next_state[:, 22:23, 0:2]
                    dist_norm = (torch.norm(ball_pos, dim=-1) / self.field_half_diag).clamp(0.0, 1.0)
                    speed_norm = torch.sigmoid((torch.norm(ball_baseline, dim=-1) - 10.0) / 3.0)
                    step_frac = (f_pred - 1) / max(F_pred - 1, 1)
                    # Acceleration: sudden direction change → more correction
                    prev_ball_vel = current[0:1, 22:23, 2:4]
                    ball_accel = torch.norm(ball_baseline[0:1] - prev_ball_vel, dim=-1) / max(dt, 1e-8)
                    accel_norm = (ball_accel / self.fallback_accel_max).clamp(0.0, 1.0)
                    # Weighted combination with configurable coefficients
                    alpha = (self.fallback_w_dist * dist_norm
                             + self.fallback_w_speed * speed_norm
                             + self.fallback_w_horizon * step_frac
                             + self.fallback_w_accel * accel_norm).clamp(0.0, 1.0)
                    alpha = alpha.view(-1, 1).unsqueeze(-1)  # (B*P, 1, 1)
                    # Ball always gets full Lorentz correction; players use dynamic blend
                    ball_final = ball_lorentz
                    next_state[:, 22:23, 2:4] = ball_final
                    # Trace alpha (log first batch element, first particle)
                    self.alpha_trace.append({
                        'scene': batch_size, 'frame': f_pred,
                        'alpha': alpha[0, 0, 0].item(),
                        'alpha_ball': 1.0,
                        'dist_norm': dist_norm[0, 0].item(),
                        'speed_norm': speed_norm[0, 0].item(),
                        'accel_norm': accel_norm[0, 0].item(),
                    })
                else:
                    next_state[:, 22:23, 2:4] = ball_lorentz
                next_state[:, 22:23, 0:2] = next_state[:, 22:23, 0:2] + ball_noise * dt * 0.5

            # ---- Hybrid ball (second path) ----
            if self.use_hybrid_ball and P > 1:
                ball_vel = next_state[:, 22:23, 2:4].squeeze(1)
                ball_pos = next_state[:, 22:23, 0:2].squeeze(1)
                prev_ball_vel = current[0:1, 22:23, 2:4].squeeze(1).expand_as(ball_vel)

                speed = torch.norm(ball_vel, dim=-1, keepdim=True)
                acc_norm = torch.norm(ball_vel - prev_ball_vel, dim=-1, keepdim=True) / max(dt, 1e-8)

                gamma_low = 0.1
                gamma_norm = self.hybrid_gamma
                gamma_high = 0.3

                w_low = torch.sigmoid((self.hybrid_linear_speed - speed) * 20.0)
                w_high = torch.sigmoid((acc_norm - self.hybrid_fluid_accel) * 10.0)
                w_norm = (1.0 - w_low) * (1.0 - w_high)

                gamma_eff = w_low * gamma_low + w_norm * gamma_norm + w_high * gamma_high
                ball_drag = 1.0 - gamma_eff * dt
                ball_noise = (torch.randn(batch_size * P, 2, device=device) * 0.02 * math.sqrt(dt))
                new_vel = ball_vel * ball_drag + ball_noise

                next_state[:, 22:23, 2:4] = new_vel.unsqueeze(1)
                next_state[:, 22:23, 0:2] = ball_pos.unsqueeze(1) + new_vel.unsqueeze(1) * dt

            traj[step + 1] = next_state
            vel_history.append(current[:, :22, 2:4].detach())
            latent_history.append(latent.detach())
            current = next_state

        # ----- Compute weights -----
        if self.use_perturbation and P > 1:
            weights = self._compute_weights(traj, base_state, batch_size, P)
        else:
            weights = None

        # ----- Weighted average output -----
        out = torch.zeros(T, 1, batch_size, 92, device=device)
        for b in range(batch_size):
            lo, hi = b * P, (b + 1) * P
            if weights is not None:
                w = weights[lo:hi]
                w = w / (w.sum() + 1e-8)
                w_4d = w.view(1, P, 1, 1)
                weighted = (traj[:, lo:hi] * w_4d).sum(dim=1)
            else:
                weighted = traj[:, lo]
            out[:, 0, b] = weighted.reshape(T, 92)

        # ----- EnKF correction -----
        if self.use_enkf and self.use_perturbation and P > 1:
            out = self._enkf_correct(out, traj, batch_size, P, burn_in)
            # ----- Predictor–Corrector (velocity consistency) -----
            if self.use_predictor_corrector:
                out = self._predictor_corrector(out)

        # ----- Update Volterra memory -----
        if self.use_perturbation and P > 1 and weights is not None:
            self._update_memory(traj, weights, batch_size, P, out)

        return out
