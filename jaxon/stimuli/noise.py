import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def whitenoise(
    key: ArrayLike,
    cflow: float,
    cfup: float,
    fs: int,
    duration: float,
    scaling: float = 1.0,
) -> jax.Array:
    """
    JAX-native, JIT-compatible band-limited white noise generator.

    Generates white noise with a flat power spectrum between `cflow` and
    `cfup` Hertz.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The random key for generation.
    cflow: float
        Lower cutoff frequency in Hertz.
    cfup: float
        Upper cutoff frequency in Hertz.
    fs: int
        Sampling rate in Hz.
    duration: float
        Total duration of the resulting array in seconds.
    scaling: float
        Final scaling factor for the noise.

    Returns
    -------
    noise: 1-D JAX array
        Band-limited white noise.
    """
    n = int(duration * fs)
    nn = int(2 ** jnp.ceil(jnp.log2(n)))
    inx0 = int(round(cflow * nn / fs))
    inx1 = int(round(cfup * nn / fs))

    inx0 = max(0, inx0)
    inx1 = min(nn // 2, inx1)

    whitef = jnp.zeros(nn // 2 + 1, dtype=jnp.complex64)
    if inx0 == 0:
        whitef = whitef.at[0].set(0)
        inx0 = 1
    if inx1 >= nn // 2:
        whitef = whitef.at[nn // 2].set(1)
        inx1 = nn // 2 - 1

    num_phases = inx1 - inx0 + 1
    phases = jax.random.uniform(key, shape=(num_phases,)) * 2 * jnp.pi
    complex_vals = jnp.exp(1j * phases)
    whitef = whitef.at[inx0 : inx1 + 1].set(complex_vals)

    noise = jnp.fft.irfft(whitef)

    norm_factor = nn / jnp.sqrt(2.0 * float(inx1 - inx0))

    return noise[:n] * norm_factor * scaling
