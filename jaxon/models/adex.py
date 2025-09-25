from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass
class AdExParams:
    """AdEx paramater class.

    Attributes
    ----------
    fs : int
        sampling period ex 30_000 Hz
    ref_period : float
        refactory period
    noise_strength : float
        noise
    tau_m : float
        membran time constant
    exp_threshold : float
        exponetial threshold
    exp_deltat : float
        exponetial sampling period
    v_base : float
        voltage reset
    threshold : float
        threshold for spike firing
    tau_adapt : flaot
        Adapation time constant
    deltat_adapt : flaot
        Adaptation sampling period
    a : float
        parameter
    b : float
        parameter
    """

    fs: int
    ref_period: float
    noise_strength: float
    tau_m: float
    exp_threshold: float
    exp_deltat: float
    v_base: float
    threshold: float
    tau_adapt: float
    deltat_adapt: float
    a: float
    b: float


def simulate(key: jax.Array, stimulus: jax.Array, params: AdExParams) -> jax.Array:
    """Simulate an AdEx neuron in JAX.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for noise generation.
    stimulus : 1-D jnp.ndarray
        Input current
    params : AdExParams parameter

    Returns
    -------
    spikes : 1-D jnp.ndarray
        0/1 array with spikes at each time step.
    """
    n_steps = stimulus.shape[0]
    dt = 1.0 / params.fs

    # Prepare noise as a white current (nA) with std ~ noise_strength/sqrt(dt)
    noise_key, init_key_v0, init_key_w0 = jax.random.split(key, 3)
    noise = jax.random.normal(noise_key, (n_steps,)) * (
        params.noise_strength / jnp.sqrt(dt)
    )

    # Initial conditions
    v0 = jax.random.uniform(init_key_v0)
    w0 = jax.random.uniform(init_key_w0)

    def step(carry, inputs):
        v, adapt, last_spike_time, time_index = carry
        stim, noise = inputs
        t = time_index * dt

        exp_arg = (v - params.exp_threshold) / params.exp_deltat
        exp_arg = jnp.clip(exp_arg, a_min=-50.0, a_max=50.0)
        exp_term = params.exp_deltat * jnp.exp(exp_arg)

        dv = (-v + exp_term - adapt + stim + noise) / params.tau_m * dt
        v_new = v + dv

        adapt_new = adapt - (params.a * (adapt / params.tau_adapt * dt))

        is_refr = (last_spike_time >= 0.0) & (
            t - last_spike_time < params.ref_period + 0.5 * dt
        )
        v_eff = jnp.where(is_refr, params.v_base, v_new)

        spike = jnp.float32(v_eff) > params.threshold

        # Reset on spike
        v_next = jnp.where(spike, params.v_base, jnp.float32(v_eff))
        w_next = jnp.where(spike, adapt_new + params.b, adapt_new)
        last_spike_next = jnp.where(spike, t, last_spike_time)

        carry_next = (v_next, w_next, last_spike_next, time_index + 1)
        return carry_next, (spike, v_next)

    carry, (spikes, v) = jax.lax.scan(
        step,
        (v0, w0, -1.0, 0),
        (stimulus, noise),
    )
    return spikes.astype(jnp.int32)
