"""Functions for analyzing spike trains."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax.typing import ArrayLike


def cyclic_rate(
    spikes: ArrayLike, cycles: float | ArrayLike, sigma: float = 0.05
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Kernel density estimate of spike times relative to a periodic signal.

    Computes for each spike time its phase $\varphi_i$ within the
    period of `cycles`. Kernel density estimate is then from these
    phases.

    Parameters
    ----------
    spikes: nparray of floats
        Spike times.
    cycles: float or nparray of floats
        Times of full periods. A single number indicates the period
        of the periodic signal.
    sigma: float
        Standard deviation of Gaussian kernel used for the kde
        of interspike intervals.
        Between 0 and 1, 1 corresponds to a full period.

    Returns
    -------
    phases: ndarray of floats
        Phases at which the kde is computed.
        Step size is set to a tenth of `sigma`.
    kde: ndarray of floats
        The kernel density estimate of the spike times within periods of `cycles`.
    """
    if jnp.isscalar(cycles):
        cycles = jnp.arange(0, spikes[-1] + 10 * cycles, cycles)
    phases = jnp.arange(0, 1.005, 0.1 * sigma) * 2 * np.pi

    rate = jnp.zeros(len(phases))
    n = 0
    for i, spike in enumerate(spikes):
        k = cycles.searchsorted(spike) - 1
        if k + 1 >= len(cycles):
            break
        cycle = cycles[k]
        period = cycles[k + 1] - cycles[k]
        phase = 2 * np.pi * (spike - cycle) / period
        cycle_spikes = np.array([phase - 2 * np.pi, phase, phase + 2 * np.pi])
        kernel = jsp.stats.gaussian_kde(
            cycle_spikes, 2 * jnp.pi * sigma / jnp.std(cycle_spikes, ddof=1)
        )
        cycle_rate = kernel(phases)
        rate += cycle_rate
        n += 1
    return phases, rate / n


def interval_statistics(
    spikes: ArrayLike, sigma: float = 1e-4, maxisi: float = 0.1
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Statistics and kde of interspike intervals.

    Parameters
    ----------
    spikes : ArrayLike
        Binary spike train.

    fs : int
        Sampling frequency

    sigma : float
        Standard deviation of Gaussian kernel used for for kde
        of interspike intervals. Same unit as `spikes`.

    maxisi : float
        Maximum interspike interval for kde. If None or 0, use maximum interval.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]

    isis: ndarray of floats
        Interspike intervals for kde.
    kde: ndarray of floats
        Kernel density estimate of interspike intervals for each `isis`.
        Plot it like this:
        ```
        ax.fill_between(isis, kde)
        ```
    rate: float
        Mean baseline firing rate as inverse mean interspike interval.
    cv: float
        Coefficient of variation (std divided by mean) of the interspike intervals.
    """
    intervals = jnp.diff(spikes)
    if not maxisi:
        maxisi = jnp.max(intervals) + 2 * sigma
    isis = jnp.arange(0.0, maxisi, 0.1 * sigma)
    kernel = jsp.stats.gaussian_kde(intervals, sigma / jnp.std(intervals, ddof=1))
    kde = kernel(isis)
    mean_isi = jnp.mean(intervals)
    std_isi = jnp.std(intervals)
    rate = 1 / mean_isi
    cv = std_isi / mean_isi

    return isis, kde, rate, cv


def burst_fraction(
    spikes: ArrayLike, eodf: float, lthresh: float = 0.1, rthresh: float = -0.05
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Burst fraction based on ISI distribution."""
    if not jnp.isfinite(eodf) or eodf <= 0:
        return jnp.nan, jnp.nan, jnp.nan
    intervals = jnp.diff(spikes)
    maxisi = jnp.max(intervals)
    maxisi = max(maxisi, 10 / eodf)

    bins = jnp.arange(0.5 / eodf, maxisi, 1 / eodf)
    bins.at[0].set(0.0)
    counts, _ = jnp.histogram(intervals, bins)
    bf = counts[0] / len(intervals)
    diff = jnp.diff(counts) / (counts[:-1] + 1)
    mask = (diff[:-1] < -lthresh) & (diff[1:] > rthresh)
    if jnp.sum(mask) > 0:
        idx = jnp.argmax(mask)
        thresh = ((idx + 1) + 0.5) / eodf
        bft = jnp.sum(counts[: idx + 1]) / len(intervals)
        if bft > 0.95 or bft < 0.01 or bf < 0.05:
            thresh = 0
            bft = 0
    else:
        thresh = 0
        bft = 0
    return bf, bft, thresh


def vector_strength(spikes: ArrayLike, cycles: float | ArrayLike) -> jnp.ndarray:
    r"""Vector strength of spike times relative to a periodic signal.

    Computes for each spike time its phase $\varphi_i$
    within the period of `cycles`.
    Vector strength is then
    $$vs = \left|\frac[1}{n} \sum_{i=1}^n e^{i\varphi_i} \right|$$

    Parameters
    ----------
    spikes: nparray of floats
        Spike times.
    cycles: float or nparray of floats
        Times of full periods. A single number indicates the period
        of the periodic signal.

    Returns
    -------
    vs: float
        Computed vector strength.
    """
    if np.isscalar(cycles):
        if np.isfinite(cycles) and len(spikes) > 0:
            cycles = np.arange(0, spikes[-1] + 10 * cycles, cycles)
        else:
            return np.nan
    vectors = np.zeros(len(spikes), dtype=complex)
    for i, spike in enumerate(spikes):
        k = cycles.searchsorted(spike) - 1
        if k + 1 >= len(cycles):
            vectors = vectors[:i]
            break
        cycle = cycles[k]
        period = cycles[k + 1] - cycles[k]
        phase = 2 * np.pi * (spike - cycle) / period
        vectors[i] = np.exp(1j * phase)
    return np.abs(np.mean(vectors))


if __name__ == "__main__":
    import pandas as pd
    from IPython import embed

    from jaxon.models import punit
    from jaxon.utils.output import to_spikes_times

    duration = 15
    fs = 30_000
    time = jnp.arange(0, duration, 1 / fs)
    stimulus = jnp.ones(duration * fs)
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 100)
    df = pd.read_csv("jaxon/models/parameters_punits.csv")
    punit_spikes = jax.vmap(jax.jit(punit.simulate_spikes), in_axes=[0, None, None])
    for m in df.iterrows():
        model = m[1].to_dict()
        cell = model.pop("cell")
        eodf = model.pop("EODf")
        model["deltat"] = 1 / 30_000
        baseline = jnp.sin(2 * jnp.pi * time * eodf)
        params = punit.PUnitParams(**model)
        spikes = punit_spikes(keys, baseline, params)
        isis = []
        rates = np.zeros(spikes.shape[0])
        eod_period = 1 / eodf
        for i, trial in enumerate(spikes):
            spike_times = to_spikes_times(trial, fs)
            isi, kde, rate, cv = interval_statistics(
                spike_times, sigma=0.05 * eod_period, maxisi=15.5 * eod_period
            )
            rates[i] = rate
            res = burst_fraction(spike_times, eodf)
            vs = vector_strength(spike_times, eod_period)
            cphase, crate = cyclic_rate(spike_times, eod_period, sigma=0.02)
