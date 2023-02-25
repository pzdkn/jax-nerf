import typing as t
import argparse
from pathlib import Path
from functools import partial

import jax.random as rd
import jax.numpy as jnp
import numpy as np
import flax 
import jax
import optax
import matplotlib.pyplot as plt
from flax.training import train_state
from jax.tree_util import Partial
from tqdm import tqdm

from nerf.ray_helpers import get_ray_bundle, sample_query_points, positional_encoding
from nerf.model import MLP
from nerf.rendering import render_volume_density
from nerf.dataloading import load_image_data

class TrainState(train_state.TrainState):
    rngs: rd.KeyArray = None

def create_train_state(module: flax.linen.Module,
                       lr: float, 
                       image_shape: t.Tuple[int, int],
                       num_encodings: int,
                       rng: rd.KeyArray) -> TrainState:
    rngs = rd.split(rng, num=3)
    h, w = image_shape
    params = module.init(rngs[0], jnp.ones([h, w, 3*2*num_encodings  + 3]))["params"]
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)
    return TrainState(apply_fn=module.apply,
                      params=params,
                      tx=tx,
                      rngs=rngs[1:],
                      opt_state=opt_state,
                      step=0
                      )

def nerf_predict(apply_fn: t.Callable,
                 params: jnp.ndarray,
                 image_shape: jnp.ndarray, 
                 cam2world: jnp.ndarray,
                 focal_length: float,
                 near_thres: float,
                 far_thres: float,
                 num_samples: int,
                 num_encodings: int,
                 batch_size:int,
                 rng: rd.KeyArray) -> jnp.ndarray:
    """Run one iteration of nerf"""
    new_key, _ = rd.split(key=rng)
    height, width, = image_shape
    ray_origins, ray_dirs = get_ray_bundle(height=height, 
                                           width=width, 
                                           focal_length=focal_length, 
                                           cam2world=cam2world)
                            
    query_points, depth_values = sample_query_points(origins=ray_origins, 
                                                    directions=ray_dirs,
                                                    near_thres=near_thres, 
                                                    far_thres=far_thres, 
                                                    num_samples=num_samples, 
                                                    rand_key=new_key)
    enc_points = positional_encoding(query_points.reshape(-1, 3), num_encodings=num_encodings)
    
    predictions = []
    for i in tqdm(range(0, enc_points.shape[0], batch_size), desc="RENDERING"):
        batch = enc_points[i:i+batch_size, :]
        pred = apply_fn({"params": params}, batch)
        predictions.append(pred)
    predictions = jnp.concatenate(predictions, axis=0)
    predictions = predictions.reshape(query_points.shape[:-1] + (4,))

    rgb_predicted, _ = render_volume_density(radience_field=predictions,
                                            depth_values=depth_values)

    return rgb_predicted

@partial(jax.jit, 
        static_argnames=["image_shape", "near_thres",
                         "far_thres", "num_samples", 
                         "num_encodings", "batch_size"])
def train_step(state: TrainState,
               image: jnp.ndarray, 
               cam2world: jnp.ndarray,
               focal_length,
               near_thres: float,
               far_thres: float,
               num_samples: int,
               num_encodings: int,
               batch_size: int):
    "Run one training step"
    def nerf_loss(params) -> float:
        predicted_img = nerf_predict(apply_fn=state.apply_fn,
                                     params=params,
                                     image_shape=image.shape[:2],
                                     cam2world=cam2world,
                                     focal_length=focal_length,
                                     near_thres=near_thres,
                                     far_thres=far_thres,
                                     num_samples=num_samples,
                                     num_encodings=num_encodings,
                                     batch_size=batch_size,
                                     rng=state.rngs[0]      
                                     )
        return jnp.sum((image - predicted_img) ** 2) / 2
    grad_fn = jax.grad(nerf_loss)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state.replace(rngs=rd.split(state.rngs[1]))
    return state


def main():
    parser = argparse.ArgumentParser()
    #FOLDERS
    data_group = parser.add_argument_group("data")
    data_group.add_argument("--data_path", type=str, default="./data/tiny_nerf_data.npz")
    data_group.add_argument("--res_dir", type=str, default="./results")
    data_group.add_argument("--fig_dir", type=str, default="./results/figs")

    # NERF
    nerf_group = parser.add_argument_group("nerf")
    nerf_group.add_argument("--near_far_bounds", nargs=2, type=float, default=[2., 6.])
    nerf_group.add_argument("--num_encodings", default=6, type=int)
    nerf_group.add_argument("--samples_per_ray", type=int, default=32)
    
    # TRAIN
    train_group = parser.add_argument_group("train")
    train_group.add_argument("--batch_size", type=int, default=2)
    train_group.add_argument("--lr", type=float, default=1e-3)
    train_group.add_argument("--n_iters", type=int, default=3000)

    # OTHER 
    other_group = parser.add_argument_group("other")
    other_group.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # FOLDER OPS
    data_path = Path(args.data_path)
    assert data_path.exists(), f"No data found under {data_path.resolve()}"

    res_dir = Path(args.res_dir)
    fig_dir = Path(args.fig_dir)
    to_create = [res_dir, fig_dir]
    for folder in to_create:
        folder.mkdir(exist_ok=True, parents=True)
    
    # SET SEED
    rng_key = rd.PRNGKey(args.seed)
    
    # LOAD IMAGES
    images, cam2worlds, focal = load_image_data(data_path)
    image = images[0]
    
    # INIT MODEL
    model = MLP([128, 128], 4)
    # Optimizer State and Model State
    rng_keys = rd.split(rng_key, num=3)
    state = create_train_state(module=model,
                              lr=args.lr, 
                              image_shape=image.shape[:2],
                              num_encodings=args.num_encodings,
                              rng=rng_keys[0])
    jit_nerf_predict = jax.jit(nerf_predict, 
                              static_argnames=["apply_fn","image_shape", "near_thres",
                                                "far_thres", "num_samples",    
                                                "num_encodings", "batch_size"])
    for i in tqdm(range(args.n_iters), desc="TRAINING"):
        target_img_idx = rd.randint(rng_keys[1], (1,) , 0, images.shape[0]).item()
        target_img = images[target_img_idx]
        cam2world = cam2worlds[target_img_idx]

        state = train_step(state, 
                           target_img,
                           cam2world=cam2world,
                           focal_length=focal,
                           near_thres=args.near_far_bounds[0],
                           far_thres=args.near_far_bounds[1],
                           num_samples=args.samples_per_ray,
                           num_encodings=args.num_encodings,
                           batch_size=args.batch_size
                          )
        if i % 100 == 0 or i == args.n_iters - 1:
            predicted_img = jit_nerf_predict(apply_fn=state.apply_fn,
                                            params=state.params,
                                            image_shape=image.shape[:2],
                                            cam2world=cam2world,
                                            focal_length=focal,
                                            near_thres=args.near_far_bounds[0],
                                            far_thres=args.near_far_bounds[1],
                                            num_samples=args.samples_per_ray,
                                            num_encodings=args.num_encodings,
                                            batch_size=args.batch_size,    
                                            rng=rng_keys[2])
            plt.imshow(predicted_img)
            plt.savefig(fig_dir/ f"{str(i).zfill(10)}.png")
    

if __name__ == "__main__":
    main()