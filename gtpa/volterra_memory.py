from __future__ import annotations

from typing import Optional

import torch


class VolterraMemory:
    """Ring buffer of high-weight trajectories for particle re-weighting.

    Stores (trajectory, weight, quality) triples.

    `weight` = particle weight from the weight formula (higher = better).
    `quality` = additional score (e.g. negative validation error) for
    selecting trajectories that historically minimised the prediction error.

    When `query()` is called, returns the minimum L2 distance (final positions)
    to any stored trajectory, weighted by quality so that high-quality
    trajectories contribute more.

    Attributes
    ----------
    capacity : int
        Maximum number of stored trajectories.
    trajectories : list[(Tensor, float, float)]
        Each entry is (trajectory, weight, quality) where trajectory has shape
        (horizon, 23, 4).
    """

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.trajectories: list[tuple[torch.Tensor, float, float]] = []

    @torch.no_grad()
    def add(self, trajectory: torch.Tensor, weight: float, quality: float = 0.0) -> None:
        """Add a trajectory with its weight and optional quality score."""
        if len(self.trajectories) < self.capacity:
            self.trajectories.append((trajectory.detach().cpu(), weight, quality))
        else:
            # Replace the worst (highest weight = worst)
            worst_idx = max(range(len(self.trajectories)),
                            key=lambda i: self.trajectories[i][1])
            self.trajectories[worst_idx] = (trajectory.detach().cpu(), weight, quality)

    @torch.no_grad()
    def query(self, trajectory_final_pos: torch.Tensor) -> torch.Tensor:
        """Minimum L2 distance, weighted by quality.

        Distance is discounted by quality: d_eff = dist / (1 + quality).
        High-quality stored trajectories thus have a larger effective
        influence (smaller effective distance), while low-quality ones
        are ignored.

        Args:
            trajectory_final_pos: (23, 2)  — final positions of 22 players + ball.

        Returns:
            Scalar tensor: the minimum quality-weighted Euclidean distance.
        """
        if not self.trajectories:
            return torch.zeros(1, device=trajectory_final_pos.device).squeeze()

        best_score = torch.tensor(float('inf'), device=trajectory_final_pos.device)
        for stored_traj, _, quality in self.trajectories:
            stored_final = stored_traj[-1, :, :2].to(trajectory_final_pos.device)
            dist = torch.norm(stored_final - trajectory_final_pos, dim=-1).mean()
            # Effective distance discounted by quality
            d_eff = dist / (1.0 + quality)
            if d_eff < best_score:
                best_score = d_eff
        return best_score

    @torch.no_grad()
    def query_best(self, trajectory_final_pos: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the highest-quality stored trajectory's final positions.

        Args:
            trajectory_final_pos: (23, 2)

        Returns:
            (23, 2) tensor of the best stored trajectory's final positions,
            or None if memory is empty.
        """
        if not self.trajectories:
            return None

        best_score = float('inf')
        best_final = None
        for stored_traj, _, quality in self.trajectories:
            stored_final = stored_traj[-1, :, :2]
            dist = torch.norm(stored_final - trajectory_final_pos, dim=-1).mean().item()
            d_eff = dist / (1.0 + quality)
            if d_eff < best_score:
                best_score = d_eff
                best_final = stored_final.clone()
        return best_final

    @torch.no_grad()
    def get_mean_direction(self, current_pos: torch.Tensor) -> Optional[torch.Tensor]:
        """Average direction from current positions to stored final positions.

        `current_pos` should contain only the 22 player positions (not the ball).
        Returns a (22, 2) tensor of mean displacement vectors, or None if
        memory is empty.  Used to bias the Monte Carlo perturbation toward
        regions that historically produced good trajectories.
        """
        if not self.trajectories:
            return None

        n_players = current_pos.size(0)
        displacements = []
        total_w = 0.0
        for stored_traj, weight, quality in self.trajectories:
            stored_final = stored_traj[-1, :n_players, :2].to(current_pos.device)
            disp = stored_final - current_pos
            w = weight * (1.0 + quality)
            displacements.append(disp * w)
            total_w += w

        if total_w > 0:
            return sum(displacements) / total_w
        return None

    # ------------------------------------------------------------------
    #  Recursive prefix matching
    # ------------------------------------------------------------------
    @torch.no_grad()
    def query_next_step(self, prefix: torch.Tensor, t: int) -> Optional[torch.Tensor]:
        """Given a partial trajectory prefix up to step t, return the
        next step (t+1) of the closest matching stored trajectory.

        Args:
            prefix: (T_current, 23, 4) — the trajectory so far
            t: current step index (prefix has length t+1)

        Returns:
            (23, 4) tensor of the next step from the best matching stored
            trajectory, or None if memory is empty.
        """
        if not self.trajectories:
            return None

        prefix = prefix.detach().cpu()
        best_score = float('inf')
        best_next = None

        for stored_traj, weight, quality in self.trajectories:
            stored_traj = stored_traj.detach().cpu()
            # Compare first t+1 steps (or fewer if stored is shorter)
            len_compare = min(prefix.size(0), stored_traj.size(0), t + 1)
            stored_prefix = stored_traj[:len_compare]
            query_prefix = prefix[:len_compare]

            # L2 distance over all agents x features
            dist = torch.norm(stored_prefix - query_prefix).item()
            d_eff = dist / (1.0 + quality)

            # Only consider if stored has a next step beyond what we're comparing
            if d_eff < best_score and stored_traj.size(0) > len_compare:
                best_score = d_eff
                best_next = stored_traj[len_compare].clone()

        return best_next
