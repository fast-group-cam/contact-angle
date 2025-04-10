import numpy as np
from .droplet.center_coordinates import center_coordinates

#==================================================================================================
# Utility: trigonometric functions in degrees

def sin_deg(x):
    return np.sin(x * np.pi / 180)

def cos_deg(x):
    return np.cos(x * np.pi / 180)
