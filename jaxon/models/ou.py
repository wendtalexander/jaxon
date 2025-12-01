"""Model for simulating a Ornstein-Uhlenbeck process."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from jaxon.utils.output import SIMOutput


@jax.tree_util.register_dataclass
@dataclass
class OUParams:
    """LIF parameter class.

    Attributes
    ----------
    fs : int
        Sampling intervall
    gamma:
        stokes friction coefficient
    noise_strength : float
        Noise
    """

    fs: int
    gamma: float
    noise_strength: float


def simulate(key: ArrayLike, duration: float, params: OUParams) -> jnp.ndarray:
    """Simulate a Ornstein-Uhlenbeck process.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for noise generation.
    params: OUParams
        Ornstein-Uhlenbeck parameter class

    Returns
    -------
    voltage: jnp.ndarray
        Simulated voltage trace

    Notes
    -----
    Ornstein Uhlenbeck process follows a linear differential equation.

    See the tutorial [Ornstein-Uhlenbeck](/ou.qmd)

    """
    n = int(duration * params.fs)
    deltat = 1 / params.fs
    # Prepare noise
    noise_key, init_key = jax.random.split(key, 2)
    v_mem_init = jax.random.uniform(init_key)
    noise = jax.random.normal(noise_key, (n,))

    def step(carry: ArrayLike, input: ArrayLike) -> jnp.ndarray:
        """Solve the current step with the Euler method.

        Parameters
        ----------
        carry : ArrayLike
            Membrane voltage

        input : ArrayLike
            Gaussian white noise

        Returns
        -------
        tuple[jax.Array] voltage
            voltage

        """
        v_mem_new = (
            carry * (1 - params.gamma * deltat)
            + jnp.sqrt(2 * params.noise_strength * deltat) * input
        )
        return v_mem_new, v_mem_new

    _, v_mem = jax.lax.scan(
        step,
        1,
        noise,
    )
    return v_mem
