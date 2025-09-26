import jax.scipy as jsp


def spike_rate(binary_spike_train, kernel):
    return jsp.signal.fftconvolve(binary_spike_train, kernel, "same")
