"""Functions for spectral analysis."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax.typing import ArrayLike


def spectra(spikes:ArrayLike, stimulus:ArrayLike, fs:int):

