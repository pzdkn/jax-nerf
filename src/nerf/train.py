import typing as t
from pathlib import Path

import jax.random as rd
import jax.numpy as jnp
import numpy as np
import flax 
import jax
import optax
import matplotlib.pyplot as plt
from flax.training import train_state

from nerf.ray_helpers import get_ray_bundle, sample_query_points, positional_encoding
from nerf.model import MLP
from nerf.rendering import render_volume_density
from nerf.dataloading import load_image_data

class TrainState(train_state.TrainState):
    rngs: rd.KeyArray

def create_train_state(module: flax.linen.Module,
                       lr: float, 
                       pseudo_input: t.Optional[jnp.ndarray],
                       rng: rd.KeyArray) -> TrainState:
    rngs = rd.split(rng, num=3)
    params = module.init(rngs[0], pseudo_input)["params"]
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)
    return TrainState(apply_fn=nerf_predict,
                      params=params,
                      tx=tx,
                      rngs=rngs[1:],
                      opt_state=opt_state,
                      step=0
                      )

def nerf_predict(params: jnp.ndarray,
                 image: jnp.ndarray, 
                 cam2world: jnp.array,
                 focal_length: float,
                 model: flax.linen.Module,
                 batch_size:int,
                 rng: rd.KeyArray) -> jnp.ndarray:
    """Run one iteration of nerf"""
    new_key, _ = rd.split(key=rng)
    height, width, _ = image.shape
    ray_origins, ray_dirs = get_ray_bundle(height=height, 
                                            width=width, 
                                            focal_length=focal_length, 
                                            cam2world=cam2world)
                            
    query_points, depth_values = sample_query_points(origins=ray_origins, 
                                       directions=ray_dirs,
                                       near_thres=0.1, 
                                       far_thres=6., 
                                       num_samples=2, 
                                       rand_key=new_key)
    enc_points = positional_encoding(query_points.reshape(-1, 3), num_encodings=2)
    
    predictions = []
    for i in range(0, enc_points.shape[0], batch_size):
        batch = enc_points[i:i+batch_size, :]
        pred = model.apply({"params": params}, batch)
        predictions.append(pred)
    predictions = jnp.concatenate(predictions, axis=0)
    predictions = predictions.reshape(query_points.shape[:-1] + (4,))

    rgb_predicted, _ = render_volume_density(radience_field=predictions,
                                          depth_values=depth_values)

    return rgb_predicted



def train_step(state: TrainState,
               image: jnp.ndarray, 
               cam2world: jnp.array,
               focal_length,
               model: flax.linen.Module,
               batch_size: int):
    "Run one training step"
    def nerf_loss(params) -> float:
        predicted_img = state.apply_fn(params, 
                                       image, 
                                       cam2world,
                                       focal_length,
                                       model, 
                                       batch_size, 
                                       state.rngs[0])

        return jnp.sum((image - predicted_img) ** 2) / 2
    grad_fn = jax.grad(nerf_loss)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state.replace(rngs=rd.split(state.rngs[1]))
    return state


def main():
    # DATA
    DATA_PATH = Path("./data/tiny_nerf_data.npz")
    SAVE_PATH = Path("./results/figs")
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    # PARAMETERS 
    NEAR, FAR = 1., 9.
    ENCODINS_FNS = 2
    DEPTH_SAMPLES_PER_RAY = 8
    BATCH_SIZE = 128
    LR = 1e-3
    NUM_ITERS = 3000
    SEED = 42
    # Random
    rng_key = rd.PRNGKey(SEED)
    # Load
    images, cam2world, focal = load_image_data(DATA_PATH)
    image = images[0]
    plt.imshow(image)
    plt.show()
    # Model
    model = MLP([128, 128], 4)
    h, w = image.shape[:2]
    pseudo_input = jnp.ones([h, w, 3*2*ENCODINS_FNS])
    # Optimizer State
    rng_keys = rd.split(rng_key, num=3)
    state = create_train_state(module=model,
                              lr=LR, 
                              pseudo_input=pseudo_input,
                              rng=rng_keys[0])

    for i in range(NUM_ITERS):
        target_img_idx = rd.randint(rng_keys[1], (1,) , 0, images.shape[0]).item()
        target_img = images[target_img_idx]
        cam2world = cam2world[target_img_idx]

        state = train_step(state, 
                           target_img,
                           cam2world,
                           focal,
                           model,
                           batch_size=BATCH_SIZE
                          )
        if i % 100 == 0:
            predicted_img = nerf_predict(state.params,
                                         image,
                                         cam2world,
                                         focal,
                                         model,
                                         batch_size=BATCH_SIZE,
                                         rng=rng_keys[2])
            plt.imshow(predicted_img)
            plt.savefig(SAVE_PATH / f"{str(i).zfill(10)}.png")
    

if __name__ == "__main__":
    main()