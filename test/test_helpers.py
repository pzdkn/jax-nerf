import sys

import jax.numpy as jnp
import jax.random as rnd
from nerf import get_ray_bundle, sample_query_points, positional_encoding

def test_ray_bundle(camera_intrinsics: jnp.array) -> None:
    height, width, focal_length = 100, 100, 2.
    origins, directions = get_ray_bundle(height, width, focal_length, camera_intrinsics)
    assert origins.shape == (height, width, 3)
    assert directions.shape == (height, width, 3)

def test_query_points(camera_intrinsics: jnp.array) -> None:
    height, width, focal_length = 100, 100, 2.
    origins, directions = get_ray_bundle(height, width, focal_length, camera_intrinsics)
    near_thres, far_thres, num_samples  = 0.1, 1.0, 100
    rng_key = rnd.PRNGKey(42)
    query_points, depth_values = sample_query_points(origins, 
                                                    directions,
                                                    near_thres, 
                                                    far_thres, 
                                                    num_samples,
                                                    rng_key)
    assert query_points.shape == (height, width, 3, num_samples)
    assert depth_values.shape == (height, width, 1, num_samples) 

def test_encoding_fn(camera_intrinsics: jnp.array) -> None:
    height, width, focal_length = 100, 100, 2.
    origins, directions = get_ray_bundle(height, width, focal_length, camera_intrinsics)
    near_thres, far_thres, num_samples  = 0.1, 1.0, 100
    rng_key = rnd.PRNGKey(42)
    query_points, depth_values = sample_query_points(origins, 
                                                    directions,
                                                    near_thres, 
                                                    far_thres, 
                                                    num_samples,
                                                    rng_key)
    query_points = jnp.reshape(query_points, (-1, 3))

    num_encodings = 4
    encodings = positional_encoding(query_points, num_encodings=num_encodings)
    assert encodings.shape == (height * width * num_samples, 3, num_encodings * 2)

