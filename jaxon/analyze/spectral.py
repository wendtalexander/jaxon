"""Functions for spectral analysis."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax.typing import ArrayLike


def spectra(spikes: ArrayLike, stimulus: ArrayLike, fs: int, nfft: int):
    f, pyy = jsp.signal.welch(stimulus, fs=fs, nperseg=nfft, noverlap=nfft // 2)
    _, pxx = jsp.signal.welch(spikes - jnp.mean(spikes), fs=fs, nperseg=nfft, noverlap=nfft // 2)
    _, pxy = jsp.signal.csd(
        spikes - jnp.mean(spikes), stimulus, fs=fs, nperseg=nfft, noverlap=nfft // 2
    )
    return f, pyy, pxx, pxy
