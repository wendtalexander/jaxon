from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.tree_util.register_dataclass
@dataclass
class PUnitParams:
    """Parameter class for p-unit model

    Attributes
    ----------
    a_zero : float
        Adaptation resting value
    delta_a : float
        Sampling interval of adaptation current
    deltat : float
        Sampling interval
    dend_tau : float
        Time constant of dendritic
    input_scaling : flaot
        Scalar multiplier for external currents
    mem_tau : float
        Time constant for membrane
    noise_strength : float
        Noise added to the system
    ref_period : float
        Refactory period
    tau_a : float
        Time constant for adaptation
    threshold : float
        Threshold for spiking
    v_base : float
        The reset potential after a spike
    v_offset : float
        Voltage offset
    v_zero : float
        Resting potential
    """

    a_zero: float
    delta_a: float
    deltat: float
    dend_tau: float
    input_scaling: float
    mem_tau: float
    noise_strength: float
    ref_period: float
    tau_a: float
    threshold: float
    v_base: float
    v_offset: float
    v_zero: float


def simulate(key: ArrayLike, stimulus: ArrayLike, params: PUnitParams) -> jax.Array:
    """Simulate a P-unit using JAX, returning a binary spike train."""
    noise = jax.random.normal(key, shape=stimulus.shape)
    noise *= params.noise_strength / jnp.sqrt(params.deltat)

    stimulus = jnp.maximum(stimulus, 0.0)

    def _scan_body(carry, x):
        v_mem, v_dend, adapt, last_spike_time, time_index = carry
        stim_i, noise_i = x
        current_time = time_index * params.deltat

        # Update dendritic voltage
        v_dend_new = v_dend + (-v_dend + stim_i) / params.dend_tau * params.deltat

        dv_mem = (
            (
                params.v_base
                - v_mem
                + params.v_offset
                + (v_dend_new * params.input_scaling)
                - adapt
                + noise_i
            )
            / params.mem_tau
            * params.deltat
        )

        # Update membrane potential
        v_mem_new = v_mem + dv_mem

        adapt_new = adapt - (adapt / params.tau_a) * params.deltat

        # Check for refractory period and reset membrane potential if needed
        is_refractory = (last_spike_time >= 0) & (
            current_time - last_spike_time < params.ref_period + params.deltat / 2
        )
        v_mem_ref = jnp.where(is_refractory, params.v_base, v_mem_new)

        # Check for threshold crossing
        spike_fired = v_mem_ref > params.threshold

        # Apply spike effects conditionally using jnp.where
        final_v_mem = jnp.where(spike_fired, params.v_base, v_mem_ref)
        final_adapt = jnp.where(
            spike_fired, adapt_new + params.delta_a / params.tau_a, adapt_new
        )
        final_last_spike_time = jnp.where(spike_fired, current_time, last_spike_time)

        new_carry = (
            final_v_mem,
            v_dend_new,
            final_adapt,
            final_last_spike_time,
            time_index + 1,
        )
        return new_carry, spike_fired

    # Initial conditions for the scan
    initial_carry = (params.v_zero, stimulus[0], params.a_zero, -1.0, 0)

    # Run the simulation using jax.lax.scan
    _, spike_fired_series = jax.lax.scan(_scan_body, initial_carry, (stimulus, noise))

    return spike_fired_series.astype(jnp.int32)
