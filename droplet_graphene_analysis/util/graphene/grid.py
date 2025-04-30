#==================================================================================================
# This submodule exists only to ensure a consistent definition of 2D grid points throughout the
# `graphene` module.
#==================================================================================================

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from .sheet import C_C_DISTANCE

#==================================================================================================

def castable_to_int(x):
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
    return ((N, N) if castable_to_int(N) else N[0:2])

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
    x_coords = np.linspace(-0.5 * cell_xy[0], 0.5 * cell_xy[0], N_x) + (0.5 * cell_xy[0]/N_x)
    y_coords = np.linspace(-0.5 * cell_xy[1], 0.5 * cell_xy[1], N_y) + (0.5 * cell_xy[1]/N_y)
    return np.stack(np.meshgrid(x_coords, y_coords, indexing='ij'), axis=-1)

#==================================================================================================

def smooth_sheet(
        carbons: np.ndarray,
        cell_xy: np.ndarray | tuple[float, float],
        N: tuple[int, int] | int = 80, *,
        margin: float = (3 * C_C_DISTANCE)
        ) -> np.ndarray:
    """Calculates a 'smooth' sheet from the coordinates of the carbon atoms of a graphene sheet,
    using the Clough-Tocher interpolator on the standard grid given by `generate_grid`.

    Parameters
    ----------
    carbons : ndarray
        The Cartesian coordinates of the carbon atoms at a single given instant, with shape
        (N_carbon, 3).
    cell_xy : array_like
        The cell parameters along the x- and y-axes, expressed as [cell_x, cell_y].
    N : tuple[int, int] or int, optional
        The number of grid points, either specified as a tuple of integers (N_x, N_y), or given as
        a single integer N_x = N_y. Defaults to (80, 80).
    margin : float, optional
        To preserve the periodic boundary conditions, a copy of the carbon atoms within `margin`
        angstroms of each boundary is tiled across the opposite boundary, before performing the
        Clough-Tocher interpolation. Thus, `margin` should be large enough to capture at least one
        graphene lattice unit per boundary, ensuring that the convex hull of the tiled carbon atoms
        fully encapsulates all grid points; however increasingly `margin` excessively leads to
        performance costs from the Clough-Tocher interpolator. Defaults to 4.278.

    Returns
    -------
    z : ndarray
        The smooth surface z(x, y) evaluated at the grid points given by `generate_grid`, with
        shape (N_x, N_y).
    """
    
    cell_xy = np.array(cell_xy[0:2], dtype=float)
    cell_params = np.array((cell_xy[0], cell_xy[1], 0), dtype=float)
    carbons[:,0:2] -= cell_xy * np.round(carbons[:,0:2] / cell_xy)
    carbons[:,2] -= np.mean(carbons[:,2])

    shifts = np.array([[i, j, 0] for i in [-1, 0, 1] for j in [-1, 0, 1]]) * cell_params
    tiled_carbons = (carbons[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    tiled_carbons = tiled_carbons[np.abs(tiled_carbons[:, 0]) < (0.5 * cell_xy[0]) + margin]
    tiled_carbons = tiled_carbons[np.abs(tiled_carbons[:, 1]) < (0.5 * cell_xy[1]) + margin]

    real_grid = generate_grid(N, cell_xy)
    interp = CloughTocher2DInterpolator(tiled_carbons[:,0:2], tiled_carbons[:,2])
    interp_z = interp(real_grid[:,:,0], real_grid[:,:,1])
    return np.nan_to_num(interp_z)
