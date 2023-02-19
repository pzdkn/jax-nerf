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
from tqdm import tqdm

from nerf.ray_helpers import get_ray_bundle, sample_query_points, positional_encoding
from nerf.model import MLP
from nerf.rendering import render_volume_density
from nerf.dataloading import load_image_data

class TrainState(train_state.TrainState):
    rngs: rd.KeyArray = None

def create_train_state(module: flax.linen.Module,
                       lr: float, 
                       pseudo_input: t.Optional[jnp.ndarray],
                       near_thres: float,
                       far_thres: float,
                       num_samples: int,
                       num_encodings: int,
                       rng: rd.KeyArray) -> TrainState:
    rngs = rd.split(rng, num=3)
    params = module.init(rngs[0], pseudo_input)["params"]
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)
    return TrainState(apply_fn=partial(nerf_predict,
                                       near_thres=near_thres, 
                                       far_thres=far_thres, 
                                       num_samples=num_samples,
                                       num_encodings=num_encodings),
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
                 near_thres: float,
                 far_thres: float,
                 num_samples: int,
                 num_encodings: int,
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
                                       near_thres=near_thres, 
                                       far_thres=far_thres, 
                                       num_samples=num_samples, 
                                       rand_key=new_key)
    enc_points = positional_encoding(query_points.reshape(-1, 3), num_encodings=num_encodings)
    
    predictions = []
    for i in tqdm(range(0, enc_points.shape[0], batch_size), desc="RENDERING"):
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
        predicted_img = state.apply_fn(params=params, 
                                       image=image, 
                                       cam2world=cam2world,
                                       focal_length=focal_length,
                                       model=model, 
                                       batch_size=batch_size, 
                                       rng=state.rngs[0])

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
    nerf_group.add_argument("--near_far_bounds", nargs=2, type=float, default=[1., 9.])
    nerf_group.add_argument("--num_encodings", default=2)
    nerf_group.add_argument("--samples_per_ray", type=int, default=2)
    
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
    images, cam2world, focal = load_image_data(data_path)
    image = images[0]
    
    # INIT MODEL
    model = MLP([128, 128], 4)
    h, w = image.shape[:2]
    pseudo_input = jnp.ones([h, w, 3*2*args.num_encodings])

    # Optimizer State and Model State
    rng_keys = rd.split(rng_key, num=3)
    state = create_train_state(module=model,
                              lr=args.lr, 
                              pseudo_input=pseudo_input,
                              rng=rng_keys[0],
                              near_thres=args.near_far_bounds[0],
                              far_thres=args.near_far_bounds[1],
                              num_samples=args.samples_per_ray,
                              num_encodings=args.num_encodings)

    for i in tqdm(range(args.n_iters), desc="TRAINING"):
        target_img_idx = rd.randint(rng_keys[1], (1,) , 0, images.shape[0]).item()
        target_img = images[target_img_idx]
        cam2world = cam2world[target_img_idx]

        state = train_step(state, 
                           target_img,
                           cam2world,
                           focal,
                           model,
                           batch_size=args.batch_size
                          )
        if i % 100 == 0 or i == args.n_iters - 1:
            predicted_img = nerf_predict(state.params,
                                         image,
                                         cam2world,
                                         focal,
                                         model,
                                         batch_size=args.batch_size,
                                         near_thres=args.near_far_bounds[0],
                                         far_thres=args.near_far_bounds[1],
                                         num_samples=args.samples_per_ray,
                                         num_encodings=args.num_encodings,     
                                         rng=rng_keys[2])
            plt.imshow(predicted_img)
            plt.savefig(fig_dir/ f"{str(i).zfill(10)}.png")
    

if __name__ == "__main__":
    main()