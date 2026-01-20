import cv2
import numpy as np
from PIL import Image

def generate_fake_image(img: Image.Image):
    img = np.array(img)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img = cv2.resize(img, None, fx=2.0, fy=2.0)
    return Image.fromarray(img)



