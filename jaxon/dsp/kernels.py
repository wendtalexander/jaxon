import jax.numpy as jnp


def gauss_kernel(sigma, dt, k_time):
    x = jnp.arange(-k_time * sigma, k_time * sigma, dt)
    return jnp.exp(-0.5 * (x / sigma) ** 2) / jnp.sqrt(2.0 * jnp.pi) / sigma
