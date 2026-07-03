from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor


AccelerationFn = Callable[[Tensor, Tensor], Tensor]


def rk2_step(
    position: Tensor,
    velocity: Tensor,
    acceleration_fn: AccelerationFn,
    dt: float,
) -> Tuple[Tensor, Tensor]:
    """Midpoint RK2 step for second-order motion.

    Args:
        position: Tensor [..., 2]
        velocity: Tensor [..., 2]
        acceleration_fn: Function that returns acceleration given (position, velocity).
        dt: Time step.

    Returns:
        new_position, new_velocity
    """
    a1 = acceleration_fn(position, velocity)
    mid_velocity = velocity + 0.5 * dt * a1
    mid_position = position + 0.5 * dt * velocity
    a2 = acceleration_fn(mid_position, mid_velocity)

    new_velocity = velocity + dt * a2
    new_position = position + dt * mid_velocity
    return new_position, new_velocity


def rk4_step(
    position: Tensor,
    velocity: Tensor,
    acceleration_fn: AccelerationFn,
    dt: float,
) -> Tuple[Tensor, Tensor]:
    """Classical RK4 step for second-order motion.

    Provided as a utility for experiments; the current codebase uses RK2 by
    default for efficiency unless configured otherwise.
    """
    a1 = acceleration_fn(position, velocity)
    k1_v = a1
    k1_x = velocity

    a2 = acceleration_fn(position + 0.5 * dt * k1_x, velocity + 0.5 * dt * k1_v)
    k2_v = a2
    k2_x = velocity + 0.5 * dt * k1_v

    a3 = acceleration_fn(position + 0.5 * dt * k2_x, velocity + 0.5 * dt * k2_v)
    k3_v = a3
    k3_x = velocity + 0.5 * dt * k2_v

    a4 = acceleration_fn(position + dt * k3_x, velocity + dt * k3_v)
    k4_v = a4
    k4_x = velocity + dt * k3_v

    new_position = position + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    new_velocity = velocity + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return new_position, new_velocity
