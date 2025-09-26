"""Model for simulating a leaky-Integrated-and-Fire Model with adaptation and refractory period."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from jaxon.utils.output import SIMOutput


@jax.tree_util.register_dataclass
@dataclass
class ALIFRefParams:
    """LIF with adaptation and refractory period parameter class.

    Attributes
    ----------
    fs : int
        Sampling interval
    threshold : float
        Threshold for spike
    v_offset : float
        Voltage offset
    tau_mem : float
        Membrane time constant
    tau_adapt : float
        Adaptation time constant
    a_zero: float
        Initial adaptation current
    deltat_adapt: float
        Time scale for adaptation
    ref_period: float
        Refectory period where no spikes are generated
    v_base : float
        Voltage base
    noise_strength : float
        Noise
    """

    fs: int
    threshold: float
    v_offset: float
    tau_mem: float
    tau_adapt: float
    a_zero: float
    deltat_adapt: float
    ref_period: float
    v_base: float
    noise_strength: float


def simulate(key: ArrayLike, stimulus: ArrayLike, params: ALIFRefParams) -> SIMOutput:
    """Simulate a leaky-Integrated-and-fire- Model with adaptation and refractory period.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for noise generation.
    stimulus : 1-D jnp.ndarray
        Input stimulus.
    params : ALIFRefParams
        Parameter class

    Returns
    -------
    SIMOutput: class
        Simulation output class which holds both binary spikes and membrane voltage
    """
    n = stimulus.shape[0]
    deltat = 1.0 / params.fs
    # Prepare noise
    noise_key, init_key = jax.random.split(key)
    v_mem_init = jax.random.uniform(init_key)
    noise = jax.random.normal(noise_key, (n,)) * (params.noise_strength / jnp.sqrt(deltat))

    def step(carry, inputs):
        v_mem, adapt, last_spike_time, time_index = carry
        stim, n = inputs
        current_time = time_index * deltat

        dv = (-v_mem + params.v_offset + stim - adapt + n) / params.tau_mem * deltat
        v_mem_new = v_mem + dv

        adapt_new = adapt - (adapt / params.tau_adapt) * deltat

        # Check for refractory period and reset membrane potential if needed
        is_refractory = (last_spike_time >= 0) & (
            current_time - last_spike_time < params.ref_period + deltat / 2
        )
        v_mem_ref = jnp.where(is_refractory, params.v_base, v_mem_new)

        spike_fired = v_mem_ref > params.threshold

        final_v_mem = jnp.where(spike_fired, params.v_base, v_mem_ref)
        final_adapt = jnp.where(
            spike_fired, adapt_new + (params.deltat_adapt / params.tau_adapt), adapt_new
        )
        final_last_spike_time = jnp.where(spike_fired, current_time, last_spike_time)

        new_carry = (
            final_v_mem,
            final_adapt,
            final_last_spike_time,
            time_index + 1,
        )

        return new_carry, (spike_fired, final_v_mem)

    # Run scan
    _, (spikes, vmem) = jax.lax.scan(
        step,
        (v_mem_init, params.a_zero, -1, 0),
        (stimulus, noise),
    )
    return SIMOutput(spikes.astype(jnp.int32), vmem)


def simulate_spikes(key: ArrayLike, stimulus: ArrayLike, params: ALIFRefParams) -> jax.Array:
    """Simulate binary spike train.

    Helper function that only returns the binary spike train.
    Helpful for vmapping over trials.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for noise generation.
    stimulus : 1-D jnp.ndarray
        Input stimulus.
    params: ALIFRefParams
        LIF parameter class

    Returns
    -------
    jax.Array
        Binary spike train

    """
    return simulate(key, stimulus, params).spikes


def simulate_mem_v(key: ArrayLike, stimulus: ArrayLike, params: ALIFRefParams) -> jax.Array:
    """Simulate membrane voltage.

    Helper function that only returns the membrane voltage.
    Helpful for vmapping over trials.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for noise generation.
    stimulus : 1-D jnp.ndarray
        Input stimulus.
    params: ALIFRefParams
        LIF parameter class

    Returns
    -------
    jax.Array
        Membrane Voltage

    """
    return simulate(key, stimulus, params).v_mem
