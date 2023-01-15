import pytest
import jax.numpy as jnp

@pytest.fixture()
def camera_intrinsics() -> jnp.array:
    """Camera transformation: rotation of 60° around x and translation of 1 in y

    Returns:
        jnp.array: _description_
    """
    theta = jnp.pi / 3
    rot = jnp.array([[1, 0, 0], 
                    [0 , jnp.cos(theta), -jnp.sin(theta)],
                     [0, jnp.sin(theta), jnp.cos(theta)]])
    translation = jnp.expand_dims(jnp.array([0, 1, 0]), axis=-1)
    T = jnp.block([[rot, translation], [0, 0, 0, 1]])
    return T