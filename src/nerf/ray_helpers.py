import typing as t 
import jax.numpy as jnp
import jax.random as rand

def get_ray_bundle(height: int, width: int, focal_length: float, cam2world: jnp.array) -> t.Tuple[jnp.array, jnp.array]:
    """Generate ray bundle expressed in world coordinates

    Args:
        height (int): _description_
        width (int): _description_
        focal_length (float): _description_
        cam2world (jnp.array): _description_

    Returns:
        t.Tuple[jnp.array, jnp.array]: _description_
    """
    xx, yy = jnp.meshgrid(jnp.arange(width), jnp.arange(height), indexing="xy")
    directions_c = jnp.stack([(xx - .5 * width) / focal_length, 
                              (yy - .5 * height) / focal_length,
                              -jnp.ones_like(xx)], axis=-1) # HxWx3
    directions_w =  jnp.dot(directions_c, cam2world[:3, :3])  #HxWx3
    origins_w =  jnp.broadcast_to(cam2world[:3, -1], directions_c.shape) # HxWx3
    return origins_w, directions_w

def sample_query_points(origins: jnp.array,
                        directions: jnp.array,
                        near_thres: float,
                        far_thres: float, 
                        num_samples: int,
                        rand_key: t.Optional[rand.KeyArray] = None) -> t.Tuple[jnp.array, jnp.array]:
    """_summary_

    Args:
        origins (jnp.array): _description_
        directions (jnp.array): _description_
        near_thres (float): _description_
        far_thres (float): _description_
        num_samples (int): _description_
        rand_key (t.Optional[rand.KeyArray], optional): _description_. Defaults to None.

    Returns:
        t.Tuple[jnp.array, jnp.array]: _description_
    """
    depth_values = jnp.linspace(near_thres, far_thres, num_samples)  # (num_samples, )
    if rand_key is not None:
        # (H, W, num_samples) = (1, 1, num_samples) + (H, W, num_samples)
        depth_values = depth_values[jnp.newaxis, jnp.newaxis, :] + rand.uniform(rand_key, 
                                                                    shape=(*origins.shape[:-1], num_samples)) / num_samples
    else:
        depth_values = jnp.expand_dims(depth_values, axis=(0, 1))
    # (H, W, num_samples, 3) = (H, W, 1, 3) + (H, W, 1, 3) * (1, 1, num_samples, 1) or (H, W, num_samples, 1)
    query_points = origins[..., jnp.newaxis, :] + directions[..., jnp.newaxis, :] * depth_values[..., jnp.newaxis]
    return query_points, depth_values

def positional_encoding(query_points: jnp.array, num_encodings: int) -> jnp.array:
    """ Create sinusioidal positional encodings from query_points

    Args:
        query_points (jnp.array): query_points flattened => (H*W*num_samples, 3)
        num_encodings (int): number of encodings/frequency bands

    Returns:
        jnp.array: encoded feature vec (H*W*num_samples, 3*num_encodings)
    """
    encodings = []
    freq_bands = 2**jnp.linspace(0, num_encodings, num_encodings)
    for freq in freq_bands:
        for fn in [jnp.sin, jnp.cos]:
            encodings.append(fn(freq * query_points))
    encodings = jnp.concatenate(encodings, axis=-1)
    return encodings
