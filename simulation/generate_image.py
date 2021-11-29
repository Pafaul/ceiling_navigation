import numpy as np


def generate_canvas(height, width, coefficients):
    return np.ones(
        (
            int(height * coefficients[0]),
            int(width * coefficients[1]),
            3
        ),
        np.uint8
    )
