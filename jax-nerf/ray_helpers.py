import typing as t 
import jax.numpy as jnp

def get_ray_bundle(height: int, width: int, focal_length: float, camera_intrisics: jnp.array) -> t.Tuple[jnp.array, jnp.array]:
    """Generate ray bundle

    Args:
        height (int): _description_
        width (int): _description_
        focal_length (float): _description_
        camera_intrisics (jnp.array): _description_

    Returns:
        t.Tuple[jnp.array, jnp.array]: _description_
    """
    xx, yy = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="xy")
    directions_c = jnp.stack((xx - .5 * width) / focal_length, 
                            (yy - .5 * height) / focal_length,
                             -jnp.ones(height, width), axis=-1) # HxWx3
    directions_w =  jnp.dot(directions_c, camera_intrisics[:3, :3])  #HxWx3
    origins_w =  jnp.broadcast_to(camera_intrisics[:3, -1], directions_c.shape) # HxWx3
    return origins_w, directions_w
