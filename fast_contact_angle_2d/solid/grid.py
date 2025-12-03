#==================================================================================================
# This submodule exists only to ensure a consistent definition of 2D grid points throughout the
# `solid` module.
#==================================================================================================

import numpy as np

#==================================================================================================

def _castable_to_int(x):
    """Checks if the input can be safely casted to an integer.
    """
    try:
        int(x)
        return True
    except:
        return False

#==================================================================================================

def cast_to_gridsize(N):
    """Interprets a 2D size specification (N_x, N_y) either as a tuple of integers, or a single
    integer N_x = N_y.
    """
    return ((N, N) if _castable_to_int(N) else N[0:2])

#==================================================================================================

def generate_grid(
        N: tuple[int, int] | int,
        cell_xy: np.ndarray | tuple[float, float]
    ) -> np.ndarray:
    """Generates a 2D grid of points.
    
    Parameters
    ----------
    N : tuple[int, int] or int
        The number of grid points, either specified as a tuple of integers (N_x, N_y), or given as
        a single integer N_x = N_y.
    cell_xy : array_like
        The cell parameters along the x- and y-axes, expressed as [cell_x, cell_y].
    
    Returns
    -------
    grid : ndarray
        The x and y coordinates of the grid point, with shape (N_x, N_y, 2). The points span the
        first periodic unit cell, centred on the origin; hence the x-coordinates run within the
        interval -cell_x/2 to cell_x/2, and the y-coordinates run within the interval -cell_y/2 to
        cell_y/2. The grid points are symmetric about the origin, i.e. they represents the centers
        (not the corners) of the rectangles of width cell_x/N_x and height cell_y/N_y tiling the
        plane.
    
    Notes
    -----
    The ordering of axes (N_x, N_y) contradicts the order of MatPlotLib's `imshow` axes, so
    `np.swapaxes(..., 0, 1)` should be used when generating plots etc..
    """

    N_x, N_y = cast_to_gridsize(N)
    d_x = cell_xy[0] / N_x
    d_y = cell_xy[1] / N_y
    x_coords = np.linspace((d_x - cell_xy[0]) / 2.0, (cell_xy[0] - d_x) / 2.0, N_x)
    y_coords = np.linspace((d_y - cell_xy[1]) / 2.0, (cell_xy[1] - d_y) / 2.0, N_y)
    return np.stack(np.meshgrid(x_coords, y_coords, indexing='ij'), axis=-1)

