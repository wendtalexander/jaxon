from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


class SIMOutput(NamedTuple):
    """Simulation output class.

    Attributes
    ----------
    spikes : jax.Array
        Binary spike train
    v_mem : jax.Array
        Membrane voltage
    """

    spikes: jax.Array
    v_mem: jax.Array


def to_spikes_times(bin_spikes: ArrayLike, fs: int) -> jnp.ndarray:
    """Simple function to convert  a binaray spike train to spike times."""
    time = jnp.arange(bin_spikes.shape[0]) / fs
    spike_index = jnp.nonzero(bin_spikes)
    return time[spike_index]
