import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from modalities.image.model import ImageForensicsModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Resolve model path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "image_model.pt")

model = ImageForensicsModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])


@torch.no_grad()
def analyze_image(input_data):
    """
    input_data can be:
    - file path (str)
    - numpy.ndarray (H, W, C)
    Returns: fake probability in [0,1]
    """

    if isinstance(input_data, str):
        img = Image.open(input_data)

    elif isinstance(input_data, np.ndarray):
        img = Image.fromarray(input_data)

    else:
        raise TypeError("analyze_image expects file path or numpy array")

    x = transform(img).unsqueeze(0).to(DEVICE)
    logit = model(x)
    return torch.sigmoid(logit).item()



