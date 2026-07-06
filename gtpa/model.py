import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import gc
import pywt
from gtpa.network import NeuralPropensityNet
from gtpa.clifford_net import CliffordEncoder
from gtpa.particle_filter import ParticleFilter
from gtpa.integrators import rk2_step
from gtpa.transformer_memory import CausalTransformer
from gtpa.geometry import compute_knn_adjacency
from gtpa.event_head import (
    EventClassifier,
    EventConditionedBallDynamics,
    compute_event_loss,
    NUM_EVENTS,
)


class GTPAModel(nn.Module):
    def __init__(self, params, parser=None):
        super(GTPAModel, self).__init__()
        self.params = params

        self.hidden_dim = params.get('h_dim', 64)
        self.latent_dim = params.get('latent_dim', self.hidden_dim)

        self.use_rk2 = params.get('USE_RK2', True)
        self.use_ab3 = params.get('USE_AB3', True)
        self.predict_acceleration = params.get('PREDICT_ACCELERATION', True)

        self.rollout_loss_weight = params.get('ROLLOUT_LOSS_WEIGHT', 0.3)
        self.rollout_steps = params.get('ROLLOUT_STEPS', 5)
        self.velocity_loss_weight = params.get('VELOCITY_LOSS_WEIGHT', 0.5)
        self.acceleration_loss_weight = params.get('ACCELERATION_LOSS_WEIGHT', 0.1)
        self.accel_clamp = params.get('ACCEL_CLAMP', 0.0)

        self.use_event_head = params.get('USE_EVENT_HEAD', False)
        self.event_loss_weight = params.get('EVENT_LOSS_WEIGHT', 0.2)

        self.scheduled_sampling_start = params.get('SCHEDULED_SAMPLING_START', 0.0)
        self.scheduled_sampling_max = params.get('SCHEDULED_SAMPLING_MAX', 0.9)
        self.scheduled_sampling_anneal_epochs = params.get('SCHEDULED_SAMPLING_ANNEAL_EPOCHS', 15)

        # Graph encoder: Clifford or GAT
        self.use_wavelet = params.get('USE_WAVELET', False)
        self.wavelet_level = params.get('WAVELET_LEVEL', 1)
        self.wavelet_family = params.get('WAVELET_FAMILY', 'db4')

        self.use_clifford = params.get('USE_CLIFFORD', False)
        if self.use_clifford:
            self.network = CliffordEncoder(
                node_in_features=11,
                hidden_dim=self.hidden_dim,
                dropout=0.2,
                n_layers=params.get('CLIFFORD_LAYERS', 2),
            )
        else:
            self.network = NeuralPropensityNet(
                node_in_features=11,
                hidden_dim=self.hidden_dim,
                num_actions=10,
                dropout=0.2
            )

        # Temporal Transformer (substitui Volterra)
        self.memory_steps = params.get('MEMORY_STEPS', 8)
        self.transformer = CausalTransformer(
            d_model=self.hidden_dim,
            nhead=params.get('TRANSFORMER_HEADS', 4),
            num_layers=params.get('TRANSFORMER_LAYERS', 2),
            dim_feedforward=params.get('TRANSFORMER_FF_DIM', self.hidden_dim * 4),
            dropout=0.1,
            max_len=64
        )

        # Dynamics head: latent -> acceleration (residual)
        self.acceleration_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 2),
        )
        for m in self.acceleration_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Multi-task event head (NMSTPP-inspired)
        if self.use_event_head:
            self.event_module = EventClassifier(
                hidden_dim=self.hidden_dim,
                num_events=NUM_EVENTS,
                dropout=0.1,
            )

        # Particle filter (inferência apenas)
        num_particles = params.get('num_particles', 10)
        self.particle_filter = ParticleFilter(
            self,
            num_particles=num_particles,
            num_actions=10,
            noise_scale=params.get('NOISE_SCALE', 0.02),
            p_event=params.get('P_EVENT', 0.3),
            use_perturbation=params.get('USE_PERTURBATION', False),
            alpha=params.get('PF_ALPHA', 1.0),
            beta=params.get('PF_BETA', 0.5),
            gamma=params.get('PF_GAMMA', 2.0),
            memory_capacity=params.get('MEMORY_CAPACITY', 10),
            use_enkf=params.get('USE_ENKF', False),
            enkf_r=params.get('ENKF_R', 1.0),
            enkf_adaptive=params.get('ENKF_ADAPTIVE', False),
            use_volterra_mc=params.get('USE_VOLTERRA_MC', False),
            use_predictor_corrector=params.get('USE_PC', False),
            field_scale=params.get('FIELD_SCALE', 105.0),
            use_recursive_memory=params.get('USE_RECURSIVE_MEMORY', False),
            recursive_alpha=params.get('RECURSIVE_ALPHA', 0.7),
            use_interception=params.get('USE_INTERCEPTION', False),
            intercept_lambda=params.get('INTERCEPT_LAMBDA', 0.2),
            use_intercept=params.get('USE_INTERCEPT', False),
            intercept_beta=params.get('INTERCEPT_BETA', 0.7),
            intercept_horizon=params.get('INTERCEPT_HORIZON', 5),
            intercept_weight=params.get('INTERCEPT_WEIGHT', 0.5),
            use_pn_intercept=params.get('USE_PN_INTERCEPT', False),
            pn_beta=params.get('PN_BETA', 0.7),
            pn_N=params.get('PN_N', 4.0),
            pn_k_lateral=params.get('PN_K_LATERAL', 2.0),
            use_fluid_ball=params.get('USE_FLUID_BALL', False),
            fluid_ball_gamma=params.get('FLUID_BALL_GAMMA', 0.5),
            fluid_ball_sigma=params.get('FLUID_BALL_SIGMA', 0.5),
            fluid_ball_gamma_target=params.get('FLUID_BALL_GAMMA_TARGET', None),
            fluid_ball_gamma_tau=params.get('FLUID_BALL_GAMMA_TAU', 3.0),
            use_hybrid_ball=params.get('USE_HYBRID_BALL', False),
            hybrid_gamma=params.get('HYBRID_GAMMA', 0.6),
            hybrid_linear_speed=params.get('HYBRID_LINEAR_SPEED', 0.3),
            hybrid_fluid_accel=params.get('HYBRID_FLUID_ACCEL', 0.5),
            use_ocsvm_ball=params.get('USE_OCSVM_BALL', False),
            ocsvm_model_path=params.get('OCSVM_MODEL_PATH', 'weights/ocsvm_ball.pkl'),
            integrator=params.get('INTEGRATOR', 'legacy'),
            use_dynamic_fallback=params.get('USE_DYNAMIC_FALLBACK', False),
            fallback_w_dist=params.get('FALLBACK_W_DIST', 0.5),
            fallback_w_speed=params.get('FALLBACK_W_SPEED', 0.3),
            fallback_w_horizon=params.get('FALLBACK_W_HORIZON', 0.2),
            fallback_w_accel=params.get('FALLBACK_W_ACCEL', 0.0),
            fallback_accel_max=params.get('FALLBACK_ACCEL_MAX', 30.0),
            use_wavelet=params.get('USE_WAVELET', False),
            wavelet_level=params.get('WAVELET_LEVEL', 1),
            wavelet_family=params.get('WAVELET_FAMILY', 'db4'),
        )

    def _build_state_components(self, state_curr):
        batch_size = state_curr.size(0)
        device = state_curr.device

        curr_pos = state_curr[:, :22, 0:2]
        curr_vel = state_curr[:, :22, 2:4]
        stamina = torch.ones(batch_size, 22, device=device) * 8000.0
        ball_pos = state_curr[:, 22, 0:2]
        goal_pos = torch.tensor([[52.5, 0.0]], device=device).repeat(batch_size, 1)

        adj_batch = compute_knn_adjacency(curr_pos, k=6)

        node_features = torch.zeros(batch_size, 22, 11, device=device)
        node_features[:, :, 0:2] = curr_pos
        node_features[:, :, 2:4] = curr_vel
        node_features[:, :, 4] = stamina / 8000.0

        b_diff = ball_pos.unsqueeze(1) - curr_pos
        b_dist = torch.norm(b_diff, dim=-1, keepdim=True)
        node_features[:, :, 5:6] = b_dist
        node_features[:, :, 6:8] = b_diff / (b_dist + 1e-5)

        g_diff = goal_pos.unsqueeze(1) - curr_pos
        g_dist = torch.norm(g_diff, dim=-1, keepdim=True)
        node_features[:, :, 8:9] = g_dist
        node_features[:, :, 9:11] = g_diff / (g_dist + 1e-5)

        return node_features, adj_batch, curr_pos, curr_vel, ball_pos

    def _encode_latent(self, state_curr):
        node_features, adj_batch, _, _, _ = self._build_state_components(state_curr)
        return self.network.encode(node_features, adj_batch)

    def _predict_dynamics(self, state_curr, latent_history=None):
        """
        Graph encoder + Temporal Transformer -> acceleration prediction (residual).
        latent_history: list of [B, 22, H] tensors from previous steps.
        """
        node_features, adj_batch, curr_pos, curr_vel, ball_pos = self._build_state_components(state_curr)
        latent = self.network.encode(node_features, adj_batch)

        # Temporal Transformer over history
        if latent_history is not None and len(latent_history) >= 2:
            history = latent_history[-(self.memory_steps - 1):]
            history_tensor = torch.stack(history + [latent], dim=1)
            latent = self.transformer(history_tensor)

        # Residual acceleration: model learns dv/dt = f(state)
        accel = self.acceleration_head(latent)
        return accel, latent

    def _assemble_state(self, template_state, players_pos, players_vel, ball_pos, ball_vel):
        new_state = template_state.clone()
        new_state[:, :22, 0:2] = players_pos
        new_state[:, :22, 2:4] = players_vel
        new_state[:, 22, 0:2] = ball_pos
        new_state[:, 22, 2:4] = ball_vel
        return new_state

    def _advance_ball(self, ball_pos, ball_vel, dt, ball_accel=None):
        next_ball_pos = ball_pos + ball_vel * dt + 0.5 * (ball_accel if ball_accel is not None else 0) * dt**2
        next_ball_vel = ball_vel * 0.995 + (ball_accel if ball_accel is not None else 0) * dt
        return next_ball_pos, next_ball_vel

    def predict_state(self, state_curr, latent_history=None, vel_history=None, ball_vel_history=None, dt=None):
        if dt is None:
            dt = self.params.get('fs', 1.0)

        curr_pos = state_curr[:, :22, 0:2]
        curr_vel = state_curr[:, :22, 2:4]
        ball_pos = state_curr[:, 22, 0:2]
        ball_vel = state_curr[:, 22, 2:4]

        accel, latent = self._predict_dynamics(state_curr, latent_history=latent_history)

        # Clamp acceleration magnitude to prevent unstable initial predictions
        if self.accel_clamp > 0:
            accel_norm = torch.norm(accel, dim=-1, keepdim=True)
            scale = torch.clamp(self.accel_clamp / (accel_norm + 1e-8), max=1.0)
            accel = accel * scale

        if self.use_rk2:
            def accel_fn(position, velocity):
                trial_state = state_curr.clone()
                trial_state[:, :22, 0:2] = position
                trial_state[:, :22, 2:4] = velocity
                trial_state[:, 22, 0:2] = ball_pos + 0.5 * dt * ball_vel
                trial_state[:, 22, 2:4] = ball_vel
                trial_accel, _ = self._predict_dynamics(trial_state, latent_history=latent_history)
                return trial_accel

            next_pos, next_vel = rk2_step(curr_pos, curr_vel, accel_fn, dt)
        else:
            next_vel = curr_vel + dt * accel
            if self.use_ab3 and vel_history is not None and len(vel_history) >= 3:
                v_t = curr_vel
                v_tm1, v_tm2 = vel_history[-1], vel_history[-2]
                next_pos = curr_pos + (dt / 12.0) * (23 * v_t - 16 * v_tm1 + 5 * v_tm2)
            else:
                next_pos = curr_pos + dt * next_vel

        # Estimate ball acceleration from recent ball velocity history
        ball_accel = None
        if ball_vel_history is not None and len(ball_vel_history) >= 2:
            ball_accel = (ball_vel - ball_vel_history[-1]) / dt

        # Event-conditioned ball dynamics
        if self.use_event_head and hasattr(self, 'event_module') and latent_history is not None and len(latent_history) >= 1:
            with torch.no_grad():
                event_logits, _, _ = self.event_module(latent)
                event_logits_mean = event_logits.mean(dim=1).detach()
            next_ball_vel = EventConditionedBallDynamics.apply(ball_vel, event_logits_mean, dt=dt)
            next_ball_pos = ball_pos + next_ball_vel * dt
        else:
            next_ball_pos, next_ball_vel = self._advance_ball(ball_pos, ball_vel, dt, ball_accel=ball_accel)

        next_state = self._assemble_state(state_curr, next_pos, next_vel, next_ball_pos, next_ball_vel)
        return next_state, latent, accel

    def forward(self, states, rollout, train, hp=None):
        device = states.device
        len_time = states.size(0)
        batch_size = states.size(2)
        dt = self.params.get('fs', 1.0)

        out = {'L_rec': torch.zeros(1, device=device)}
        out2 = {'e_pos': torch.zeros(1, device=device), 'e_vel': torch.zeros(1, device=device)}

        extra_losses = {
            'L_acc': torch.zeros(1, device=device),
            'L_rollout': torch.zeros(1, device=device),
            'L_vel': torch.zeros(1, device=device),
        }

        hp = hp or {}
        rollout_steps = int(hp.get('rollout_steps', self.rollout_steps))
        rollout_weight = float(hp.get('rollout_loss_weight', self.rollout_loss_weight))
        scheduled_sampling_prob = float(hp.get('scheduled_sampling_prob', 0.0))
        velocity_weight = float(hp.get('velocity_loss_weight', self.velocity_loss_weight))
        acceleration_weight = float(hp.get('acceleration_loss_weight', self.acceleration_loss_weight))

        latent_history = []
        vel_history = []
        ball_vel_history = []
        prev_pred_state = None

        for t in range(len_time - 1):
            state_curr = states[t, 0].view(batch_size, 23, 4)

            if train and scheduled_sampling_prob > 0 and prev_pred_state is not None:
                teacher_mask = torch.rand(batch_size, 1, 1, device=device) > scheduled_sampling_prob
                teacher_mask = teacher_mask.expand(-1, 23, 4)
                state_curr = torch.where(teacher_mask, state_curr, prev_pred_state)

            state_next = states[t + 1, 0].view(batch_size, 23, 4)

            pred_state, latent, accel_pred = self.predict_state(
                state_curr, latent_history=latent_history, vel_history=vel_history,
                ball_vel_history=ball_vel_history, dt=dt
            )

            if train:
                prev_pred_state = pred_state.detach()

            latent_history.append(latent.detach())
            vel_history.append(state_curr[:, :22, 2:4].detach())
            ball_vel_history.append(state_curr[:, 22, 2:4].detach())

            target_pos = state_next[:, :22, 0:2]
            target_vel = state_next[:, :22, 2:4]
            target_acc = (target_vel - state_curr[:, :22, 2:4]) / dt

            pos_loss = F.mse_loss(pred_state[:, :22, 0:2], target_pos, reduction='mean')
            vel_loss = F.mse_loss(pred_state[:, :22, 2:4], target_vel, reduction='mean')

            step_loss = pos_loss

            if velocity_weight > 0.0:
                step_loss = step_loss + velocity_weight * vel_loss
                extra_losses['L_vel'] += vel_loss.detach()

            if acceleration_weight > 0.0:
                acc_loss = F.mse_loss(accel_pred, target_acc, reduction='mean')
                step_loss = step_loss + acceleration_weight * acc_loss
                extra_losses['L_acc'] += acc_loss.detach()

            if rollout_weight > 0.0 and t + rollout_steps < len_time:
                free_state = state_curr.clone()
                free_history = [h.clone() for h in latent_history]
                free_vel_history = [v.clone() for v in vel_history]
                rollout_loss = torch.zeros(1, device=device)

                for r in range(rollout_steps):
                    pred_free, free_latent, _ = self.predict_state(
                        free_state, latent_history=free_history, vel_history=free_vel_history, dt=dt
                    )
                    free_state = pred_free
                    free_history.append(free_latent.detach())
                    free_vel_history.append(free_state[:, :22, 2:4].detach())

                    target_free = states[t + r + 1, 0].view(batch_size, 23, 4)
                    rollout_loss = rollout_loss + F.mse_loss(
                        pred_free[:, :22, 0:2], target_free[:, :22, 0:2], reduction='mean'
                    )

                step_loss = step_loss + rollout_weight * (rollout_loss / rollout_steps)
                extra_losses['L_rollout'] += rollout_loss.detach()

            out['L_rec'] += step_loss

            with torch.no_grad():
                out2['e_vel'] += torch.mean(torch.norm(pred_state[:, :22, 2:4] - target_vel, dim=-1))
                out2['e_pos'] += torch.mean(torch.norm(pred_state[:, :22, 0:2] - target_pos, dim=-1))

        num_steps = max(1, len_time - 1)

        out['L_rec'] /= num_steps
        out2['e_vel'] /= num_steps
        out2['e_pos'] /= num_steps

        for key, value in extra_losses.items():
            if value.numel() > 0:
                out[key] = value / num_steps

        # ---- Event loss (multi-task auxiliary, added after normalization) ----
        if self.use_event_head and hasattr(self, 'event_module'):
            collected_latents = []
            for t in range(len_time - 1):
                state_t = states[t, 0].view(batch_size, 23, 4)
                node_feat, adj, _, _, _ = self._build_state_components(state_t)
                latent_t = self.network.encode(node_feat, adj)
                collected_latents.append(latent_t.unsqueeze(0))
            latent_seq = torch.cat(collected_latents, dim=0)
            ce_loss, fric_loss, spd_loss, _, _ = compute_event_loss(
                states.view(len_time, batch_size, 23, 4),
                latent_seq,
                self.event_module,
                dt=dt,
            )
            out['L_event_ce'] = ce_loss.detach()
            event_total = ce_loss + 0.5 * fric_loss + 0.3 * spd_loss
            out['L_event'] = (self.event_loss_weight * event_total).detach()
            out['L_rec'] = out['L_rec'] + self.event_loss_weight * event_total

        return out, out2

    def sample(self, states, rollout, burn_in=0, n_sample=1, TEST=False, Challenge=False):
        device = states.device
        real_horizon = states.size(0)
        fs = self.params.get('fs', 1.0)

        with torch.inference_mode():
            selected = self.particle_filter.simulate_trajectories(
                states, burn_in=burn_in, horizon=real_horizon, fs=fs
            )

        out = {'L_rec': torch.zeros(1, device=device)}
        out2 = {
            'e_pos': torch.zeros(1, device=device),
            'e_vel': torch.zeros(1, device=device),
            'e_e_p': torch.zeros(1, device=device),
            'e_e_v': torch.zeros(1, device=device)
        }

        if not Challenge:
            actual_batch = selected.size(2)
            total_elem = selected.numel()
            expected = real_horizon * actual_batch * 23 * 4

            if total_elem == expected:
                sel_state = selected.squeeze(1).contiguous().view(real_horizon, actual_batch, 23, 4)
                gt_state = states.squeeze(1).contiguous().view(real_horizon, actual_batch, 23, 4)
            else:
                flat_sel = selected.squeeze(1).reshape(-1, 92)
                flat_gt = states.squeeze(1).reshape(-1, 92)
                num_timesteps = flat_sel.shape[0] // actual_batch
                total_needed = num_timesteps * actual_batch
                sel_state = flat_sel[:total_needed].reshape(num_timesteps, actual_batch, 23, 4)
                gt_state = flat_gt[:total_needed].reshape(num_timesteps, actual_batch, 23, 4)
                real_horizon = num_timesteps

            num_timesteps = real_horizon - burn_in
            if num_timesteps > 0:
                pos_err = torch.norm(sel_state[burn_in:, :, :, 0:2] - gt_state[burn_in:, :, :, 0:2], dim=-1)
                vel_err = torch.norm(sel_state[burn_in:, :, :, 2:4] - gt_state[burn_in:, :, :, 2:4], dim=-1)

                out2['e_pos'][0] = torch.mean(pos_err)
                out2['e_vel'][0] = torch.mean(vel_err)
                out2['e_e_p'][0] = torch.mean(pos_err[-1])
                out2['e_e_v'][0] = torch.mean(vel_err[-1])
                out['L_rec'][0] = out2['e_e_p'][0]

                del pos_err, vel_err

            del sel_state, gt_state

        # ---- Wavelet denoising (post-processing) ----
        if self.use_wavelet:
            selected = self._apply_wavelet_denoise(selected)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return selected, out, out2

    @torch.no_grad()
    def _apply_wavelet_denoise(self, traj: torch.Tensor) -> torch.Tensor:
        """Wavelet denoising of position trajectories (x, y per agent).

        DWT -> zero out highest-frequency coefficients -> IDWT.
        Works along the time dimension for each agent's x and y separately.
        """
        T = traj.size(0)
        if T < 4:
            return traj

        traj = traj.clone()
        # Reshape: (T, 1, B, 92) -> (T, B, 23, 4)
        B = traj.size(2)
        traj_4d = traj.squeeze(1).reshape(T, B, 23, 4)

        level = min(self.wavelet_level, int(math.log2(T)) - 1)
        if level < 1:
            return traj

        for b in range(B):
            for a in range(23):
                for coord in range(2):  # x, y positions
                    signal = traj_4d[:, b, a, coord].cpu().numpy()
                    if np.std(signal) < 1e-6:
                        continue
                    coeffs = pywt.wavedec(signal, self.wavelet_family, level=level)
                    # Soft-threshold detail coefficients
                    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
                    for i in range(-1, -level - 1, -1):
                        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
                    reconstructed = pywt.waverec(coeffs, self.wavelet_family)
                    # waverec may be longer; trim
                    traj_4d[:, b, a, coord] = torch.from_numpy(reconstructed[:T]).to(traj.device)

        return traj_4d.reshape(T, 1, B, 92)
