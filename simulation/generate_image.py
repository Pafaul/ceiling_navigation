import numpy as np


def generate_canvas(height, width):
    return np.ones((height, width, 3), np.uint8)
