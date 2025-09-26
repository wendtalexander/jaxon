"""Simulate an Adaptive Exponential Integrate-and-Fire (AdEx) model."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from jaxon.utils.output import SIMOutput


@jax.tree_util.register_dataclass
@dataclass
class AdExParams:
    """Parameters for the Adaptive Exponential Integrate-and-Fire (AdEx) model.

    Attributes
    ----------
    fs : int
        Sampling frequency.
    tau_mem : float
        Membrane time constant.
    v_base : float
        Reset potential after a spike.
    threshold : float
        Spike detection threshold.

    exp_threshold : float
        Exponential spike-initiation voltage.
    exp_deltat : float
        Sharpness of the exponential spike initiation.

    tau_adapt : float
        Adaptation time constant.
    deltat_adapt : float
        Adaptation integration timestep.
    a : float
        Subthreshold adaptation coupling .
    b : float
        Spike-triggered adaptation increment.

    ref_period : float
        Absolute refractory period (ms).
    noise_strength : float
        Noise std dev added to the membrane input.
    """

    # Simulation
    fs: int

    # Membrane
    tau_mem: float
    v_base: float
    threshold: float

    # Exponential spike-initiation
    exp_threshold: float
    exp_deltat: float

    # Adaptation
    tau_adapt: float
    deltat_adapt: float
    a: float
    b: float

    # Other
    ref_period: float
    noise_strength: float


def simulate(key: ArrayLike, stimulus: ArrayLike, params: AdExParams) -> SIMOutput:
    """Simulate an Adaptive Exponential Integrate-and-Fire (AdEx) model.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for noise generation.
    stimulus : 1-D jnp.ndarray
        Input current
    params : AdExParams class
        parameter

    Returns
    -------
    SIMOutput: class
        Simulation output class which holds both binary spikes and membrane voltage
    """
    n_steps = stimulus.shape[0]
    dt = 1.0 / params.fs

    # Prepare noise as a white current (nA) with std ~ noise_strength/sqrt(dt)
    noise_key, init_key_v0, init_key_w0 = jax.random.split(key, 3)
    noise = jax.random.normal(noise_key, (n_steps,)) * (params.noise_strength / jnp.sqrt(dt))

    # Initial conditions
    v0 = jax.random.uniform(init_key_v0)
    w0 = 0.0

    def step(carry, inputs):
        v, adapt, last_spike_time, time_index = carry
        stim, noise = inputs
        t = time_index * dt

        exp_arg = (v - params.exp_threshold) / params.exp_deltat
        exp_arg = jnp.clip(exp_arg, a_min=-500.0, a_max=500.0)
        exp_term = params.exp_deltat * jnp.exp(exp_arg)

        dv = (-v + exp_term - adapt + stim + noise) / params.tau_mem * dt
        v_new = v + dv

        adapt_new = adapt + (dt / params.tau_adapt) * (params.a * (v - params.v_base) - adapt)

        is_refr = (last_spike_time >= 0.0) & (t - last_spike_time < params.ref_period + 0.5 * dt)
        v_eff = jnp.where(is_refr, params.v_base, v_new)

        spike = jnp.float32(v_eff) > params.threshold

        # Reset on spike
        v_next = jnp.where(spike, params.v_base, jnp.float32(v_eff))
        w_next = jnp.where(spike, adapt_new + params.b, adapt_new)
        last_spike_next = jnp.where(spike, t, last_spike_time)

        carry_next = (v_next, w_next, last_spike_next, time_index + 1)
        return carry_next, (spike, v_next)

    carry, (spikes, vmem) = jax.lax.scan(
        step,
        (v0, w0, -1.0, 0),
        (stimulus, noise),
    )
    return SIMOutput(spikes.astype(jnp.int32), vmem)


def simulate_spikes(key: ArrayLike, stimulus: ArrayLike, params: AdExParams) -> jax.Array:
    """Simulate binary spike train.

    Helper function that only returns the binary spike train.
    Helpful for vmapping over trials.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for noise generation.
    stimulus : 1-D jnp.ndarray
        Input stimulus.
    params: AdExParams
        parameter class

    Returns
    -------
    jax.Array
        Binary spike train

    """
    return simulate(key, stimulus, params).spikes


def simulate_mem_v(key: ArrayLike, stimulus: ArrayLike, params: AdExParams) -> jax.Array:
    """Simulate membrane voltage.

    Helper function that only returns the membrane voltage.
    Helpful for vmapping over trials.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for noise generation.
    stimulus : 1-D jnp.ndarray
        Input stimulus.
    params: AdExParams
        LIF parameter class

    Returns
    -------
    jax.Array
        Membrane Voltage

    """
    return simulate(key, stimulus, params).v_mem
