#==================================================================================================
# This submodule exists only to ensure a consistent definition of 2D grid points throughout the
# `graphene` module.
#==================================================================================================

import numpy as np

def castable_to_int(x):
    try:
        int(x)
        return True
    except:
        return False

def cast_to_gridsize(N):
    return ((N, N) if castable_to_int(N) else N[0:2])

def generate_grid(
        N: tuple[int, int] | int,
        cell_xy: np.ndarray | tuple[float, float]
    ) -> np.ndarray:
    """This function generates a 2D grid of points, and returns their x and y coordinates in a
np.NDArray of shape (N_x, N_y, 2). The points span the first periodic unit cell, centred on the
origin; hence, the x-coordinates run in ascending order within the interval (-cell_x/2, cell_x/2),
and the y-coordinates run in ascending order within the interval (-cell_y/2, cell_y/2). The
coordinates are symmetric, i.e. the represent the centres of rectangles of width cell_x/N_x and
height cell_y/N_y.

The size N may either be a tuple (N_x, N_y), or a single integer N_x = N_y = N.

Note that this ordering of axes contradicts MatPlotLib's imshow axes, so np.swapaxes(..., 0, 1)
should be used when generating plots via imshow etc.."""

    N_x, N_y = cast_to_gridsize(N)
    x_coords = np.linspace(-0.5 * cell_xy[0], 0.5 * cell_xy[0], N_x) + (0.5 * cell_xy[0]/N_x)
    y_coords = np.linspace(-0.5 * cell_xy[1], 0.5 * cell_xy[1], N_y) + (0.5 * cell_xy[1]/N_y)
    return np.stack(np.meshgrid(x_coords, y_coords, indexing='ij'), axis=-1)
