import cv2
import numpy as np

def generate_fake_video_frame(frame):
    """
    Simple but real manipulation:
    - motion blur
    - frame resampling
    """
    h, w = frame.shape[:2]

    # motion blur kernel
    kernel = np.zeros((15, 15))
    kernel[7, :] = np.ones(15)
    kernel /= 15

    blurred = cv2.filter2D(frame, -1, kernel)

    # downscale + upscale (compression artifact)
    small = cv2.resize(blurred, (w // 2, h // 2))
    fake = cv2.resize(small, (w, h))

    return fake



