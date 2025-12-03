import numpy as np
from scipy.special import j1
from scipy.interpolate import CloughTocher2DInterpolator
from .grid import cast_to_gridsize, generate_grid
from ..interpolate import PeriodicGridInterpolator

#==================================================================================================
# Default parameters

DISK_RADIUS = 4.5       # Radius of the disk function for heightmap regularization (in angstroms)
DEFAULT_MARGIN = 5.0    # Safety margin for tiling solid particles (in angstroms)

#==================================================================================================

def _z_grid(
        cell_xy: np.ndarray, 
        sol: np.ndarray,
        res: tuple[int, int],
        margin: float
        ) -> np.ndarray:
    """Internal function which converts an instantaneous frame of solid particle coordinates to a
    grid of interpolated z-values (using the Clough-Tocher interpolator), as raw input to a
    PeriodicGridInterpolator."""

    sol_copy = np.array(sol, copy=True)
    sol_copy[:,0:2] -= cell_xy * np.round(sol_copy[:,0:2] / cell_xy)
    sol_copy[:,2] -= np.mean(sol[:,2])

    cell_p3d = np.array((cell_xy[0], cell_xy[1], 0), dtype=float)
    shifts = np.array([[i, j, 0] for i in [-1, 0, 1] for j in [-1, 0, 1]]) * cell_p3d
    tiled_sol = (sol_copy[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    tiled_sol = tiled_sol[np.abs(tiled_sol[:, 0]) < (0.5 * cell_xy[0]) + margin]
    tiled_sol = tiled_sol[np.abs(tiled_sol[:, 1]) < (0.5 * cell_xy[1]) + margin]

    real_grid = generate_grid(res, cell_xy)
    interp = CloughTocher2DInterpolator(tiled_sol[:,0:2], tiled_sol[:,2])
    interp_z = interp(real_grid[:,:,0], real_grid[:,:,1])
    return np.nan_to_num(interp_z)

#==================================================================================================

def _z_grid_regularized(
        cell_xy: np.ndarray, 
        sol: np.ndarray,
        res: tuple[int, int],
        disk_radius: float,
        margin: float
        ) -> np.ndarray:
    """Internal function which does the same thing as _z_grid, and then regularizes it by
    convolving with the disk function of radius `cutoff_radius`."""

    z_values = _z_grid(cell_xy, sol, res, margin)

    d_x = cell_xy[0] / res[0]
    d_y = cell_xy[1] / res[1]
    k_x = np.fft.fftfreq(res[0], d_x)
    k_y = np.fft.rfftfreq(res[1], d_y)
    k_x, k_y = np.meshgrid(k_x, k_y, indexing='ij', sparse=True)
    k_mag_scaled = np.sqrt((k_x**2) + (k_y**2)) * disk_radius
    fourier = np.fft.rfft2(z_values)
    filter = np.divide(2 * j1(k_mag_scaled), k_mag_scaled, out=np.ones_like(k_mag_scaled),
                       where=(k_mag_scaled > 0.0))
    return np.fft.irfft2(fourier * filter, z_values.shape)

#==================================================================================================

def instantaneous_heightmap(
        cell_params: np.ndarray, 
        sol: np.ndarray,
        N: tuple[int, int] | int = 100, *,
        margin: float = DEFAULT_MARGIN
        ) -> PeriodicGridInterpolator:
    """Calculates the instantaneous heightmap of the solid surface (nominally aligned to the xy
    plane) from the coordinates of the solid particles, using the Clough-Tocher interpolator on the
    standard grid given by `generate_grid`.

    Parameters
    ----------
    cell_params : array_like
        The cell parameters, expressed as [cell_x, cell_y, cell_z].
    sol : ndarray
        The Cartesian coordinates of the solid particles at a single instantaneous frame, with shape
        (N_sol, 3).
    N : tuple[int, int] or int, optional
        The resolution of the output heightmap, either specified as a tuple of integers (N_x, N_y),
        or given as a single integer N_x = N_y. Increasing this parameter increases the accuracy of
        the output PeriodicGridInterpolator, but also increases computational costs for both
        constructing and later inferring from the PeriodicGridInterpolator. Defaults to (100, 100).
    margin : float, optional
        To preserve the periodic boundary conditions, a copy of the solid particles within `margin`
        angstroms of each boundary is tiled across the opposite boundary, before performing the
        Clough-Tocher interpolation. Thus, `margin` should be large enough to capture at least one
        additional layer of particles per boundary, ensuring that the convex hull of the tiled
        solid particles fully encapsulates all grid points; however increasing `margin` excessively
        leads to performance costs from the Clough-Tocher interpolator. Defaults to 5, which is an
        entirely arbitrary choice (and may not be safe if the interparticle distance is large!).

    Returns
    -------
    h : PeriodicGridInterpolator
        The instantaneous heightmap h(x, y; t), with resolution (N_x, N_y) and periodic unit cell
        of lengths [cell_x, cell_y].
    """
    
    if ((len(sol.shape) != 2) or (sol.shape[-1] != 3)):
        raise RuntimeError(f'Wrong shape {sol.shape} for "sol", expected input for a single frame'
                           ' should be of shape (N_sol, 3) only.')

    cell_xy = np.asarray(cell_params[0:2], dtype=float)
    res = cast_to_gridsize(N)
    z_values = _z_grid(cell_xy, sol, res, margin)
    return PeriodicGridInterpolator(cell_xy, z_values)

#==================================================================================================

def regularized_instantaneous_heightmap(
        cell_params: np.ndarray, 
        sol: np.ndarray,
        N: tuple[int, int] | int = 100,
        disk_radius = DISK_RADIUS, *,
        margin: float = DEFAULT_MARGIN
        ) -> PeriodicGridInterpolator:
    """Performs the same action as `instantaneous_heightmap`, except that the heightmap is
    additionally regularized by convolving it with a disk function.

    Parameters
    ----------
    cell_params : array_like
        The cell parameters, expressed as [cell_x, cell_y, cell_z].
    sol : ndarray
        The Cartesian coordinates of the solid particles at a single instantaneous frame, with shape
        (N_sol, 3).
    N : tuple[int, int] or int, optional
        The resolution of the output heightmap, either specified as a tuple of integers (N_x, N_y),
        or given as a single integer N_x = N_y. Increasing this parameter increases the accuracy of
        the output PeriodicGridInterpolator, but also increases computational costs for both
        constructing and later inferring from the PeriodicGridInterpolator. Defaults to (100, 100).
    disk_radius : float, optional
        The radius of the disk function, in angstroms, to convolve the heightmap with in order to
        regularize it. Defaults to 4.5.
    margin : float, optional
        To preserve the periodic boundary conditions, a copy of the solid particles within `margin`
        angstroms of each boundary is tiled across the opposite boundary, before performing the
        Clough-Tocher interpolation. Thus, `margin` should be large enough to capture at least one
        additional layer of particles per boundary, ensuring that the convex hull of the tiled
        solid particles fully encapsulates all grid points; however increasing `margin` excessively
        leads to performance costs from the Clough-Tocher interpolator. Defaults to 5, which is an
        entirely arbitrary choice (and may not be safe if the interparticle distance is large!).

    Returns
    -------
    h : PeriodicGridInterpolator
        The regularized instantaneous heightmap h(x, y; t), with resolution (N_x, N_y) and periodic
        unit cell of lengths [cell_x, cell_y].
    """
    
    if ((len(sol.shape) != 2) or (sol.shape[-1] != 3)):
        raise RuntimeError(f'Wrong shape {sol.shape} for "sol", expected input for a single frame'
                           ' should be of shape (N_sol, 3) only.')

    cell_xy = np.asarray(cell_params[0:2], dtype=float)
    res = cast_to_gridsize(N)
    z_values = _z_grid_regularized(cell_xy, sol, res, disk_radius, margin)
    return PeriodicGridInterpolator(cell_xy, z_values)

#==================================================================================================

def time_averaged_heightmap(
        cell_params: np.ndarray, 
        sol: np.ndarray,
        N: tuple[int, int] | int = 100, *,
        margin: float = DEFAULT_MARGIN
        ) -> PeriodicGridInterpolator:
    """Calculates the time-averaged heightmap of the solid surface (nominally aligned to the xy
    plane) from the coordinates of the solid particles, using the Clough-Tocher interpolator on the
    standard grid given by `generate_grid`.

    Parameters
    ----------
    cell_params : array_like
        The cell parameters, expressed as [cell_x, cell_y, cell_z].
    sol : ndarray
        The Cartesian coordinates of the solid particles over a trajectory, with shape (N_frames,
        N_sol, 3).
    N : tuple[int, int] or int, optional
        The resolution of the output heightmap, either specified as a tuple of integers (N_x, N_y),
        or given as a single integer N_x = N_y. Increasing this parameter increases the accuracy of
        the output PeriodicGridInterpolator, but also increases computational costs for both
        constructing and later inferring from the PeriodicGridInterpolator. Defaults to (100, 100).
    margin : float, optional
        To preserve the periodic boundary conditions, a copy of the solid particles within `margin`
        angstroms of each boundary is tiled across the opposite boundary, before performing the
        Clough-Tocher interpolation. Thus, `margin` should be large enough to capture at least one
        additional layer of particles per boundary, ensuring that the convex hull of the tiled
        solid particles fully encapsulates all grid points; however increasing `margin` excessively
        leads to performance costs from the Clough-Tocher interpolator. Defaults to 5, which is an
        entirely arbitrary choice (and may not be safe if the interparticle distance is large!).

    Returns
    -------
    h : PeriodicGridInterpolator
        The time-averaged heightmap <h(x, y)>, with resolution (N_x, N_y) and periodic unit cell of
        lengths [cell_x, cell_y].
    """
    
    if ((len(sol.shape) != 3) or (sol.shape[-1] != 3)):
        raise RuntimeError(f'Wrong shape {sol.shape} for "sol", expected input for a trajectory'
                           ' should be of shape (N_frames, N_sol, 3) only.')

    cell_xy = np.asarray(cell_params[0:2], dtype=float)
    res = cast_to_gridsize(N)
    N_frames = sol.shape[0]
    z_values = np.empty((N_frames, res[0], res[1]), dtype=float)
    for i in range(N_frames):
        z_values[i] = _z_grid(cell_xy, sol[i], res, margin)
    return PeriodicGridInterpolator(cell_xy, np.mean(z_values, axis=0))
    
#==================================================================================================

def regularized_time_averaged_heightmap(
        cell_params: np.ndarray, 
        sol: np.ndarray,
        N: tuple[int, int] | int = 100,
        disk_radius = DISK_RADIUS, *,
        margin: float = DEFAULT_MARGIN
        ) -> PeriodicGridInterpolator:
    """Performs the same action as `time_averaged_heightmap`, except that the heightmap is
    additionally regularized by convolving it with a disk function.

    Parameters
    ----------
    cell_params : array_like
        The cell parameters, expressed as [cell_x, cell_y, cell_z].
    sol : ndarray
        The Cartesian coordinates of the solid particles over a trajectory, with shape (N_frames,
        N_sol, 3).
    N : tuple[int, int] or int, optional
        The resolution of the output heightmap, either specified as a tuple of integers (N_x, N_y),
        or given as a single integer N_x = N_y. Increasing this parameter increases the accuracy of
        the output PeriodicGridInterpolator, but also increases computational costs for both
        constructing and later inferring from the PeriodicGridInterpolator. Defaults to (100, 100).
    disk_radius : float, optional
        The radius of the disk function, in angstroms, to convolve the heightmap with in order to
        regularize it. Defaults to 4.5.
    margin : float, optional
        To preserve the periodic boundary conditions, a copy of the solid particles within `margin`
        angstroms of each boundary is tiled across the opposite boundary, before performing the
        Clough-Tocher interpolation. Thus, `margin` should be large enough to capture at least one
        additional layer of particles per boundary, ensuring that the convex hull of the tiled
        solid particles fully encapsulates all grid points; however increasing `margin` excessively
        leads to performance costs from the Clough-Tocher interpolator. Defaults to 5, which is an
        entirely arbitrary choice (and may not be safe if the interparticle distance is large!).

    Returns
    -------
    h : PeriodicGridInterpolator
        The regularized time-averaged heightmap <h(x, y)>, with resolution (N_x, N_y) and periodic
        unit cell of lengths [cell_x, cell_y].
    """
    
    if ((len(sol.shape) != 3) or (sol.shape[-1] != 3)):
        raise RuntimeError(f'Wrong shape {sol.shape} for "sol", expected input for a trajectory'
                           ' should be of shape (N_frames, N_sol, 3) only.')

    cell_xy = np.asarray(cell_params[0:2], dtype=float)
    res = cast_to_gridsize(N)
    N_frames = sol.shape[0]
    z_values = np.empty((N_frames, res[0], res[1]), dtype=float)
    for i in range(N_frames):
        z_values[i] = _z_grid_regularized(cell_xy, sol[i], res, disk_radius, margin)
    return PeriodicGridInterpolator(cell_xy, np.mean(z_values, axis=0))

