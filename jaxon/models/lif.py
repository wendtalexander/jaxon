"""Model for simulating a leaky-Integrated-and-Fire Model."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from jaxon.utils.output import SIMOutput


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


def simulate(key: ArrayLike, stimulus: ArrayLike, params: LIFParams) -> SIMOutput:
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
    SIMOutput: class
        Simulation output class which holds both binary spikes and membrane voltage

    Notes
    -----
    Leaky-integrated-and-fire Model follows linear differential equation.

    See the tutorial [LIF](/lif.qmd)

    """
    n = stimulus.shape[0]
    deltat = 1 / params.fs
    # Prepare noise
    noise_key, init_key = jax.random.split(key, 2)
    v_mem_init = jax.random.uniform(init_key)
    noise = jax.random.normal(noise_key, (n,)) * (params.noise_strength / jnp.sqrt(deltat))

    def step(carry: ArrayLike, inputs: tuple[ArrayLike]) -> tuple[jax.Array]:
        """Solve the current LIF step with the Euler method.

        Parameters
        ----------
        carry : ArrayLike
            Membrane voltage

        inputs : tuple[ArrayLike] (stimulus, noise)
            Stimulus and Noise

        Returns
        -------
        tuple[jax.Array] (spikes, membrane voltage)
            Boolen spike times and membrane voltage

        """
        v_mem = carry
        stim, n = inputs
        dv = (-v_mem + params.v_offset + stim + n) / params.tau_mem * deltat
        v_mem_new = v_mem + dv
        spike = v_mem_new > params.threshold
        v_mem_new = jnp.where(spike, params.v_base, v_mem_new)
        return v_mem_new, (spike, v_mem_new)

    _, (spikes, v_mem) = jax.lax.scan(
        step,
        v_mem_init,
        (stimulus, noise),
    )
    return SIMOutput(spikes.astype(int), v_mem)


def simulate_spikes(key: ArrayLike, stimulus: ArrayLike, params: LIFParams) -> jax.Array:
    """Simulate binary spike train.

    Helper function that only returns the binary spike train.
    Helpful for vmapping over trials.

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
    jax.Array
        Binary spike train

    """
    return simulate(key, stimulus, params).spikes


def simulate_mem_v(key: ArrayLike, stimulus: ArrayLike, params: LIFParams) -> jax.Array:
    """Simulate membrane voltage.

    Helper function that only returns the membrane voltage.
    Helpful for vmapping over trials.

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
    jax.Array
        Membrane Voltage

    """
    return simulate(key, stimulus, params).v_mem
