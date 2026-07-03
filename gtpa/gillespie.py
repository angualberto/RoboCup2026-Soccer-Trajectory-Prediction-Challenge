import torch


class GillespiePerturbation:
    """
    Generates stochastic perturbations for particle diversity.

    Each agent independently experiences a random velocity perturbation
    at each step with probability p_event.  The perturbation is drawn from
    a centred Gaussian with scale = noise_scale * speed.

    This does NOT replace the network prediction — it only adds controlled
    noise so that particles explore different plausible futures.
    """

    def __init__(self, noise_scale: float = 0.03, p_event: float = 0.4) -> None:
        self.noise_scale = noise_scale
        self.p_event = p_event

    def perturb(self, velocity: torch.Tensor, players_mask: torch.Tensor) -> torch.Tensor:
        """
        velocity:  (..., 2)
        players_mask:  (..., 1)  — 1 for players, 0 for ball (ball is not perturbed)
        Returns perturbed velocity of the same shape.
        """
        event = (torch.rand_like(velocity[:, :, 0:1]) < self.p_event).float()
        speed = velocity.norm(dim=-1, keepdim=True).clamp(min=0.3)
        noise = torch.randn_like(velocity) * speed * self.noise_scale
        return velocity + noise * event * players_mask
