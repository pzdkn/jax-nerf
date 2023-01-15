from pathlib import Path
import typing as t
import numpy as np

def load_image_data(image_data_path: Path) -> t.Dict[str, np.ndarray]:
    data = np.load(image_data_path)
    images = data["images"]
    cam2world = data["poses"]
    focal_length = data["focal"]
    return images, cam2world,  focal_length