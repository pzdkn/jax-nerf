import typing as t
import jax.numpy as jnp

def exc_cumprod(factors: jnp.ndarray) -> jnp.ndarray:
    cumprod = jnp.cumprod(factors, axis=-1)
    cumprod = jnp.roll(cumprod, shift=1, axis=-1)
    cumprod = cumprod.at[0].set(1.0)
    return cumprod

def render_volume_density(radience_field: jnp.ndarray,
                          depth_values: jnp.ndarray) -> t.Tuple[jnp.ndarray]:
    """ Render rgb & depth from radiance field """
    sigma = radience_field[..., 3]
    rgb = radience_field[..., :3]

    one_e_10 = jnp.array([1e10])
    dist = jnp.concatenate((depth_values[..., 1:] - depth_values[..., :-1], 
                            jnp.broadcast_to(one_e_10, depth_values.shape[:2] + (1,))), axis=-1)
    alpha = 1. - jnp.exp(- sigma * dist)
    weights = alpha  * exc_cumprod(1. - alpha + 1e-10)

    color = jnp.sum(weights[..., jnp.newaxis] * rgb, axis=-2)
    depth = jnp.sum(weights * depth_values, axis=-1)
    return color, depth