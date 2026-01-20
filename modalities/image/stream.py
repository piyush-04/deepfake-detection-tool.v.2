from datasets import load_dataset
from PIL import Image
import numpy as np

def stream_real_images():
    ds = load_dataset(
        "imagenet-1k",
        split="train",
        streaming=True
    )

    for row in ds:
        img = row["image"]
        if isinstance(img, Image.Image):
            yield img



