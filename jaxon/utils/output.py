from typing import NamedTuple

import jax


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
