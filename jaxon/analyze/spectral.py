"""Functions for spectral analysis."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax.typing import ArrayLike


def spectra(
    spikes: ArrayLike, stimulus: ArrayLike, fs: int, nfft: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate spectral properties of the given spikes and stimulus.

    Uses the jax scipy welch method

    Parameters
    ----------
    spikes : ArrayLike
        Binaray spike train

    stimulus : ArrayLike
        Stimulus array

    fs : int
        Sampling frequency in Hertz like 30_000 Hz

    nfft : int
        Window size for the fast Fourier transfomation

    Returns
    -------
    f, pyy, pxx, pxy: jnp.ndarray
        frequency, stimulus power spectra, spike power spectra and cross spectra

    """
    f, pyy = jsp.signal.welch(stimulus, fs=fs, nperseg=nfft, noverlap=nfft // 2)
    _, pxx = jsp.signal.welch(spikes - jnp.mean(spikes), fs=fs, nperseg=nfft, noverlap=nfft // 2)
    _, pxy = jsp.signal.csd(
        spikes - jnp.mean(spikes), stimulus, fs=fs, nperseg=nfft, noverlap=nfft // 2
    )
    return f, pyy, pxx, pxy
