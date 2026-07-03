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
                 fluid_ball_sigma=0.5):
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
        self.fluid_ball_gamma = fluid_ball_gamma  # drag coefficient
        self.fluid_ball_sigma = fluid_ball_sigma  # noise amplitude

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
                ball_drag = (1 - self.fluid_ball_gamma * dt)
                ball_noise = (torch.randn(B * P, 1, 2, device=device)
                              * self.fluid_ball_sigma * math.sqrt(dt))
                next_state[:, 22:23, 2:4] = (next_state[:, 22:23, 2:4] * ball_drag + ball_noise)
                next_state[:, 22:23, 0:2] = next_state[:, 22:23, 0:2] + ball_noise * dt * 0.5

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

            # ---- Fluid ball perturbation (non-stepwise path) ----
            if self.use_fluid_ball and P > 1:
                ball_drag = (1 - self.fluid_ball_gamma * dt)
                ball_noise = (torch.randn(batch_size * P, 1, 2, device=device)
                              * self.fluid_ball_sigma * math.sqrt(dt))
                next_state[:, 22:23, 2:4] = (next_state[:, 22:23, 2:4] * ball_drag + ball_noise)
                next_state[:, 22:23, 0:2] = next_state[:, 22:23, 0:2] + ball_noise * dt * 0.5

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
