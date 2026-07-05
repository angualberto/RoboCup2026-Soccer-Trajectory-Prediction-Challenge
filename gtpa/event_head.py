import torch
import torch.nn as nn
import torch.nn.functional as F
import math


EVENT_CLASSES = {
    0: "idle",
    1: "run",
    2: "pass",
    3: "dribble",
    4: "shot",
    5: "tackle",
}
NUM_EVENTS = len(EVENT_CLASSES)


def _compute_ball_labels(traj):
    """Heuristic event labels from trajectory data.

    For each time step, classify ball/player state into one of NUM_EVENTS
    based on ball speed, acceleration, and proximity to players.

    Args:
        traj: (T, B, 23, 4) — positions + velocities for 22 players + ball.

    Returns:
        labels: (T-1, B) integer event labels.
    """
    T, B = traj.size(0), traj.size(1)
    device = traj.device

    ball_pos = traj[:, :, 22, 0:2]   # (T, B, 2)
    ball_vel = traj[:, :, 22, 2:4]
    ball_speed = torch.norm(ball_vel, dim=-1)  # (T, B)

    labels = torch.full((T - 1, B), 0, device=device, dtype=torch.long)

    for t in range(1, T):
        for b in range(B):
            speed = ball_speed[t, b].item()
            prev_speed = ball_speed[t - 1, b].item() if t > 0 else 0.0

            if speed > 2.5:
                labels[t - 1, b] = 4  # shot
            elif speed > 1.2:
                labels[t - 1, b] = 2  # pass
            elif speed > 0.3:
                labels[t - 1, b] = 3  # dribble
            else:
                labels[t - 1, b] = 0  # idle

    return labels


class EventClassifier(nn.Module):
    """Multi-head event predictor (NMSTPP-inspired).

    Encodes latent state and predicts:
      1. Event type distribution (classification head)
      2. Ball friction modifier (regression head)
      3. Next ball speed residual (regression head)
    """

    def __init__(self, hidden_dim, num_events=NUM_EVENTS, dropout=0.1):
        super().__init__()
        self.num_events = num_events

        self.event_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.event_head = nn.Linear(hidden_dim // 2, num_events)

        self.friction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.speed_residual_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, latent):
        """latent: (B*P, 22, H) or (B, 22, H) — player-wise latents."""
        batch_shape = latent.shape[:-1]
        H = latent.shape[-1]

        x = latent.reshape(-1, H)
        x = self.event_encoder(x)
        logits = self.event_head(x)
        friction_mod = self.friction_head(x)
        speed_residual = self.speed_residual_head(x)

        logits = logits.reshape(*batch_shape, self.num_events)
        friction_mod = friction_mod.reshape(*batch_shape)
        speed_residual = speed_residual.reshape(*batch_shape)

        return logits, friction_mod, speed_residual


class EventConditionedBallDynamics:
    """Ball dynamics modulated by predicted event type.

    Instead of fixed decay (0.995), uses event-dependent decay:
      - shot:   low decay (ball keeps speed)
      - pass:   medium decay
      - dribble: higher decay (ball under close control)
      - idle:   high decay (ball stopping)
    """

    BASE_DECAY = 0.995
    EVENT_DECAY = {
        0: 0.980,  # idle — ball slows quickly
        1: 0.990,  # run
        2: 0.992,  # pass — moderate decay
        3: 0.985,  # dribble — ball under close control
        4: 0.970,  # shot — lower decay, ball keeps speed
        5: 0.988,  # tackle
    }

    @staticmethod
    def apply(ball_vel, event_logits, dt=1.0):
        """Apply event-conditioned decay to ball velocity.

        Args:
            ball_vel: (..., 2) ball velocity
            event_logits: (..., C) event class logits
            dt: time step

        Returns:
            new_ball_vel: (..., 2) decayed velocity
        """
        probs = F.softmax(event_logits, dim=-1)
        weights = probs.clone()
        for cls_id, decay in EventConditionedBallDynamics.EVENT_DECAY.items():
            weights[..., cls_id] = probs[..., cls_id] * decay
        effective_decay = weights.sum(dim=-1).unsqueeze(-1)
        decay_factor = effective_decay ** dt
        return ball_vel * decay_factor


def compute_event_loss(traj, latent, event_module, dt=1.0):
    """Compute multi-task event prediction loss.

    Loss = CE(event_type) + MSE(friction) + MSE(speed_residual)

    Args:
        traj: (T, B, 23, 4) ground truth trajectory
        latent: (T, B, 22, H) latent representations
        event_module: EventClassifier instance

    Returns:
        loss_ce: cross-entropy loss
        loss_friction: friction prediction loss
        loss_speed: speed residual loss
        total_loss: combined loss
        labels: (T-1, B) ground truth labels
    """
    T = traj.size(0)
    labels = _compute_ball_labels(traj)

    logits_list = []
    friction_list = []
    speed_list = []

    for t in range(T - 1):
        logits_t, friction_t, speed_t = event_module(latent[t])
        logits_list.append(logits_t.unsqueeze(0))
        friction_list.append(friction_t.unsqueeze(0))
        speed_list.append(speed_t.unsqueeze(0))

    logits_all = torch.cat(logits_list, dim=0)          # (T-1, B, 22, C)
    logits_scene = logits_all.mean(dim=2)                # (T-1, B, C)
    friction_all = torch.cat(friction_list, dim=0)       # (T-1, B, 22)
    friction_scene = friction_all.mean(dim=2)            # (T-1, B)
    speed_all = torch.cat(speed_list, dim=0)             # (T-1, B, 22)
    speed_scene = speed_all.mean(dim=2)                  # (T-1, B)

    ball_pos = traj[:, :, 22, 0:2]
    ball_vel = traj[:, :, 22, 2:4]
    ball_speed = torch.norm(ball_vel, dim=-1)

    target_speed_next = ball_speed[1:]
    target_speed_curr = ball_speed[:-1]

    # CE loss for event classification (scene-level: aggregate over players)
    loss_ce = F.cross_entropy(
        logits_scene.view(-1, logits_scene.size(-1)),
        labels.view(-1),
        reduction="mean",
    )

    # Friction target: actual decay between steps
    decay_actual = (ball_vel[1:] / (ball_vel[:-1] + 1e-8)).norm(dim=-1)
    decay_target = torch.clamp(decay_actual, 0.9, 1.0)
    loss_friction = F.mse_loss(friction_scene, decay_target)

    # Speed residual
    speed_change = (target_speed_next - target_speed_curr) / (dt + 1e-8)
    speed_change = torch.clamp(speed_change, -5.0, 5.0)
    loss_speed = F.mse_loss(speed_scene, speed_change)

    total_loss = loss_ce + 0.5 * loss_friction + 0.3 * loss_speed

    return loss_ce, loss_friction, loss_speed, total_loss, labels
