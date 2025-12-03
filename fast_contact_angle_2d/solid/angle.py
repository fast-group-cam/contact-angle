import numpy as np
from .grid import cast_to_gridsize
from .sheet import DISK_RADIUS, DEFAULT_MARGIN, _z_grid_regularized
from ..interpolate import PeriodicGridInterpolator

#==================================================================================================

def instantaneous_inclination_angles(
        cell_params: np.ndarray,
        sol: np.ndarray,
        N: tuple[int, int] | int = 100, *,
        disk_radius: float = DISK_RADIUS,
        margin: float = DEFAULT_MARGIN
        ) -> PeriodicGridInterpolator:
    """Calculates the instantaneous local inclination angles of the solid surface (nominally
    aligned to the xy plane) from the coordinates of the solid particles, on the standard grid
    given by `generate_grid`. The local inclination angle is defined as the angle, in degrees, that
    the tangent of the local neighbourhood of the solid surface makes with the z-axis.

    Parameters
    ----------
    cell_params : array_like
        The cell parameters, expressed as [cell_x, cell_y, cell_z].
    sol : ndarray
        The Cartesian coordinates of the solid particles at a single instantaneous frame, with shape
        (N_sol, 3).
    N : tuple[int, int] or int, optional
        The resolution of the output map of local inclination angles, either specified as a tuple
        of integers (N_x, N_y), or given as a single integer N_x = N_y. Increasing this parameter
        increases the accuracy of the output PeriodicGridInterpolator, but also increases
        computational costs for both constructing and later inferring from the
        PeriodicGridInterpolator. Defaults to (100, 100).

    Returns
    -------
    theta : PeriodicGridInterpolator
        The instantaneous function θ(x, y; t) for the local inclination angles, with resolution
        (N_x, N_y) and periodic unit cell of lengths [cell_x, cell_y].

    Other parameters
    ----------------
    disk_radius : float, optional
        The radius of the disk function, in angstroms, to convolve the heightmap with in order to
        regularize it; this therefore represents the "local neighbourhood" defining the inclination
        of each test point. Defaults to 4.5.
    margin : float, optional
        See documentation for `instantaneous_heightmap`.
    """

    if ((len(sol.shape) != 2) or (sol.shape[-1] != 3)):
        raise RuntimeError(f'Wrong shape {sol.shape} for "sol", expected input for a single frame'
                           ' should be of shape (N_sol, 3) only.')

    cell_xy = np.asarray(cell_params[0:2], dtype=float)
    res = cast_to_gridsize(N)
    z_values = _z_grid_regularized(cell_xy, sol, res, disk_radius, margin)
    return PeriodicGridInterpolator(cell_xy, _calc_inclination_angles_from_z_grid(cell_xy, z_values))

#==================================================================================================

def _calc_inclination_angles_from_z_grid(
        cell_xy: np.ndarray,
        z_values: np.ndarray
        ) -> np.ndarray:
    """Internal function which converts a grid of interpolated z-values into local angles."""
    
    N_x, N_y = z_values.shape
    d_x = cell_xy[0] / N_x
    d_y = cell_xy[1] / N_y
    dz_dx = (np.roll(z_values, -1, axis=0) - np.roll(z_values, 1, axis=0)) / (2 * d_x)
    dz_dy = (np.roll(z_values, -1, axis=1) - np.roll(z_values, 1, axis=1)) / (2 * d_y)
    cosines = np.power(1.0 + (dz_dx**2) + (dz_dy**2), -0.5)
    return (np.arccos(cosines) * 180 / np.pi)


#==================================================================================================

def time_averaged_inclination_angles(
        cell_params: np.ndarray,
        sol: np.ndarray,
        N: tuple[int, int] | int = 100, *,
        disk_radius: float = DISK_RADIUS,
        margin: float = DEFAULT_MARGIN
        ) -> PeriodicGridInterpolator:
    """Calculates the time-averaged local inclination angles of the solid surface (nominally
    aligned to the xy plane) from the coordinates of the solid particles, on the standard grid
    given by `generate_grid`. The local inclination angle is defined as the angle, in degrees, that
    the tangent of the local neighbourhood of the solid surface makes with the z-axis. Note that
    this function calculates the time-average of the instantaneous local inclinations of the
    surface, rather than the local inclinations of the time-average of the surface; use
    `inclination_angles_from_surface` in the latter case.

    Parameters
    ----------
    cell_params : array_like
        The cell parameters, expressed as [cell_x, cell_y, cell_z].
    sol : ndarray
        The Cartesian coordinates of the solid particles over a trajectory, with shape (N_frames,
        N_sol, 3).
    N : tuple[int, int] or int, optional
        The resolution of the output map of local inclination angles, either specified as a tuple
        of integers (N_x, N_y), or given as a single integer N_x = N_y. Increasing this parameter
        increases the accuracy of the output PeriodicGridInterpolator, but also increases
        computational costs for both constructing and later inferring from the
        PeriodicGridInterpolator. Defaults to (100, 100).

    Returns
    -------
    theta : PeriodicGridInterpolator
        The time-averaged function <θ(x, y)> for the local inclination angles, with resolution
        (N_x, N_y) and periodic unit cell of lengths [cell_x, cell_y].

    Other parameters
    ----------------
    disk_radius : float, optional
        The radius of the disk function, in angstroms, to convolve the heightmap with in order to
        regularize it; this therefore represents the "local neighbourhood" defining the inclination
        of each test point. Defaults to 4.5.
    margin : float, optional
        See documentation for `instantaneous_heightmap`.
    """

    if ((len(sol.shape) != 3) or (sol.shape[-1] != 3)):
        raise RuntimeError(f'Wrong shape {sol.shape} for "sol", expected input for a trajectory'
                           ' should be of shape (N_frames, N_sol, 3) only.')
    
    cell_xy = np.asarray(cell_params[0:2], dtype=float)
    res = cast_to_gridsize(N)
    N_frames = sol.shape[0]
    angles = np.empty((N_frames, res[0], res[1]), dtype=float)
    for i in range(N_frames):
        z_values = _z_grid_regularized(cell_xy, sol[i], res, disk_radius, margin)
        angles[i] = _calc_inclination_angles_from_z_grid(cell_xy, z_values)
    return PeriodicGridInterpolator(cell_xy, np.mean(angles, axis=0))

#==================================================================================================

def inclination_angles_from_surface(
        heightmap: PeriodicGridInterpolator
        ) -> PeriodicGridInterpolator:
    """Calculates the local inclination angles of the solid surface (nominally aligned to the xy
    plane) from a continuous heightmap. The local inclination angle is defined as the angle, in
    degrees, that the tangent of the local neighbourhood of the solid surface makes with the
    z-axis.

    Parameters
    ----------
    heightmap : PeriodicGridInterpolator
        The heightmap h(x, y) to calculate the inclination angles from (which should ideally be
        spatially regularized to prevent sharp gradients), whether instantaneous h(x, y; t) or
        time-averaged <h(x, y)>.

    Returns
    -------
    theta : PeriodicGridInterpolator
        The function θ(x, y) for the local inclination angles, with the same resolution and
        periodic unit cell as the supplied heightmap.
    """

    cell_xy = heightmap.cell_params
    if cell_xy.shape[0] != 2:
        raise RuntimeError(f'Wrong shape {heightmap.cell_params.shape} for "heightmap" unit cell.')
    z_values = heightmap.interp.values[1:-1, 1:-1]
    return PeriodicGridInterpolator(cell_xy, _calc_inclination_angles_from_z_grid(cell_xy, z_values))

