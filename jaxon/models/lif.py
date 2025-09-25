from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.tree_util.register_dataclass
@dataclass
class LIFParams:
    """LIF parameter class.

    Attributes
    ----------
    fs : int
        Sampling intervall
    threshold : flaot
        Threshold for spike
    v_offset : float
        Volatge offset
    tau_mem : float
        Membrane time constant
    v_base : float
        Volatge base
    noise_strength : float
        Noise
    """

    fs: int
    threshold: float
    v_offset: float
    tau_mem: float
    v_base: float
    noise_strength: float


def simulate(key: ArrayLike, stimulus: ArrayLike, params: LIFParams) -> jax.Array:
    """Simulate a LIF Model in JAX.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for noise generation.
    stimulus : 1-D jnp.ndarray
        Input stimulus.
    params: LIFParams
        LIF parameter class

    Returns
    -------
    spike_times : 1-D jnp.ndarray
        Indices where spikes occurred.
    """
    n = stimulus.shape[0]
    deltat = 1 / params.fs
    # Prepare noise
    noise_key, init_key = jax.random.split(key, 2)
    v_mem_init = jax.random.uniform(init_key)
    noise = jax.random.normal(noise_key, (n,)) * (params.noise_strength / jnp.sqrt(deltat))

    def step(carry, inputs):
        v_mem = carry
        stim, n = inputs
        dv = (-v_mem + params.v_offset + stim + n) / params.tau_mem * deltat
        v_mem_new = v_mem + dv
        spike = v_mem_new > params.threshold
        v_mem_new = jnp.where(spike, params.v_base, v_mem_new)
        return v_mem_new, spike

    _, spikes = jax.lax.scan(
        step,
        v_mem_init,
        (stimulus, noise),
    )
    # Get spike indices
    return spikes.astype(jnp.int32)
