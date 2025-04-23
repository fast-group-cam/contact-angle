#==================================================================================================
# This submodule exists only to ensure a consistent definition of 2D grid points throughout the
# `graphene` module.
#==================================================================================================

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator

#==================================================================================================

def castable_to_int(x):
    try:
        int(x)
        return True
    except:
        return False

#==================================================================================================

def cast_to_gridsize(N):
    return ((N, N) if castable_to_int(N) else N[0:2])

#==================================================================================================

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

#==================================================================================================

def smooth_sheet(
        carbons: np.ndarray,
        cell_xy: np.ndarray | tuple[float, float],
        N: tuple[int, int] | int = 80
        ) -> tuple[np.ndarray, np.ndarray]:
    """This function takes in the coordinates of the carbon atoms of a graphene sheet, and
calculates a 'smooth' sheet using the Clough-Tocher interpolator on the standard grid given by
`generate_grid`. The inputs are:

    - `carbons`: The Cartesian coordinates of the carbon atoms at a single given instant, with
    shape (N_carbon, 3).
    - `cell_xy`: The cell parameters along the x- and y-axes, expressed as [cell_x, cell_y].
    - `N`: The size of the grid, either specified as a tuple of integers (N_x, N_y), or given as a
    single integer N_x = N_y.

The output is a np.NDArray of shape (N_x, N_y), representing the smooth surface z(x, y) evaluated
at the gridpoints given by `generate_grid`."""
    
    cell_xy = np.array(cell_xy[0:2], dtype=float)
    carbons[:,0:2] -= cell_xy * np.round(carbons[:,0:2] / cell_xy)
    carbons[:,2] -= np.mean(carbons[:,2])

    real_grid = generate_grid(N, cell_xy)
    interp = CloughTocher2DInterpolator(carbons[:,0:2], carbons[:,2])
    interp_z = interp(real_grid[:,:,0], real_grid[:,:,1])
    return np.nan_to_num(interp_z)
