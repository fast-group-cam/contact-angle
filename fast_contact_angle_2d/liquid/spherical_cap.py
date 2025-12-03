import numpy as np
import warnings
from ..interpolate import PeriodicGridInterpolator
from .coarse_grain import find_interface, DEFAULT_TOLERANCE

#==================================================================================================
# Default parameters for spherical cap-finding algorithm

N_SPHERE_PTS = 150    # Number of points to use to find best-fit spherical top

#==================================================================================================

def _find_sphere_pts(
        liq: np.ndarray,
        cell_params: np.ndarray,
        sol_heightmap: PeriodicGridInterpolator, *,
        N_pts: int = N_SPHERE_PTS
    ) -> np.ndarray:
    """Internal function for finding points on the spherical cap."""

    # Calculate droplet CoM
    CoM = np.mean(liq, axis=(0, 1)) if len(liq.shape) == 3 else np.mean(liq, axis=0)

    # Calculate largest distance that mean_heightmap is defined on
    max_r = min(cell_params[0] / 2, cell_params[1] / 2)

    # Generate search directions
    phi = np.linspace(0, 2 * np.pi, N_pts, endpoint=False)
    azi = np.c_[np.cos(phi), np.sin(phi)]
    polarc = np.random.random(N_pts)
    polars = np.sqrt(1.0 - (polarc**2))
    sphere_pts = list()
    warning_occurred = 0

    # Find points on the spherical cap, making sure that search directions do not intersect the
    # heightmap
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        for i in range(N_pts):
            try:
                test_r = np.linspace(max_r / N_pts, max_r, N_pts, endpoint=False)
                test_z = sol_heightmap(test_r[:,None] * azi[None,i,:])
                safe_gradient = np.max((test_z - CoM[2]) / test_r)
                if safe_gradient >= 0.0 and polarc[i] < polars[i] * safe_gradient:
                    polars[i] = 1.0 / np.sqrt(1.0 + (safe_gradient**2))
                    polarc[i] = np.sqrt(1.0 - (polars[i]**2))
                search_dir = np.array((azi[i,0] * polars[i], azi[i,1] * polars[i], polarc[i]))
                sphere_pts.append(find_interface(liq, CoM, search_dir))
            except RuntimeWarning:
                warning_occurred += 1
    
    if warning_occurred > 0:
        warnings.warn(f'find_spherical_cap: RuntimeWarning raised {warning_occurred} times(s) ' +
                      'by find_interface', RuntimeWarning)
    return np.array(sphere_pts)
    
#==================================================================================================

def find_spherical_cap(
        liq: np.ndarray,
        cell_params: np.ndarray,
        sol_heightmap: PeriodicGridInterpolator, *,
        N_pts: int = N_SPHERE_PTS
    ) -> dict[str, np.ndarray]:
    """Finds the spherical cap, defined as the best-fit sphere for a series of points on the far
    time-averaged liquid interface. This version assumes rotational symmetry.

    Parameters
    ----------
    liq : ndarray
        The Cartesian coordinates of the liquid particles, with shape (N_frames, N_liq, 3) for a
        collection of frames, representing the liquid droplet in centred coordinates.
    cell_params : array_like
        The cell parameters, expressed as [cell_x, cell_y, cell_z].
    sol_heightmap : PeriodicGridInterpolator
        The time-averaged heightmap of the solid surface, which should be supplied as a
        PeriodicGridInterpolator with domain spanning -cell_x/2 to cell_x/2 along the x-coordinate
        and -cell_y/2 to cell_y/2 along the y-coordinate.
    N_pts : int, optional
        The number of sampling points to fit the sphere onto; also used for averaging the heightmap
        azimuthally to enforce rotational symmetry. Defaults to 150.

    Returns
    -------
    sphere_fit : dict
        A dictionary indicating the properties of the best-fit sphere, with the following fields:
            - 'r': float, the radius of the sphere
            - 'z': float, the z-coordinate of the center of the sphere
            - 'a': float, the radius of the contact line/circle
            - 'angle': float, the contact angle
    """

    # Find points on the spherical cap
    phi = np.linspace(0, 2 * np.pi, N_pts, endpoint=False)
    azi = np.c_[np.cos(phi), np.sin(phi)]
    sphere_pts = _find_sphere_pts(liq, cell_params, sol_heightmap, N_pts=N_pts)

    # Least-squares best-fit spherical cap constrained on z-axis
    A_mat = np.empty((sphere_pts.shape[0], 2), dtype=float)
    A_mat[:,0] = 2 * sphere_pts[:,-1]
    A_mat[:,1] = 1
    f_vec = np.empty((sphere_pts.shape[0], 1), dtype=float)
    f_vec[:,0] = np.sum(sphere_pts**2, axis=-1)
    c_vec, _, _, _ = np.linalg.lstsq(A_mat, f_vec, rcond=None)
    sphere_r = np.sqrt(np.sum(np.square(c_vec[0,0])) + c_vec[1,0])
    sphere_z = c_vec[0,0]

    # Iteratively solve for contact point
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            floor = sol_heightmap(np.zeros((2,), dtype=float))[0]
            for _ in range(10):
                sphere_a = np.sqrt((sphere_r**2) - ((sphere_z - floor)**2))
                floor = np.mean(sol_heightmap(sphere_a * azi))
            sphere_a = np.sqrt((sphere_r**2) - ((sphere_z - floor)**2))
            sphere_angle = 90.0 + (np.arcsin((sphere_z - floor) / sphere_r) * 180.0 / np.pi)
            floor_far = np.mean(sol_heightmap(1.01 * sphere_a * azi))
            floor_near = np.mean(sol_heightmap(0.99 * sphere_a * azi))
            local_grad = (floor_far - floor_near) / (0.02 * sphere_a)
            sphere_angle += np.arctan(local_grad) * 180.0 / np.pi
        except RuntimeWarning:
            sphere_a = 0.0
            sphere_angle = (180.0 if sphere_z > 0.0 else 0.0)

    return {'r': sphere_r, 'z': sphere_z, 'a': sphere_a, 'angle': sphere_angle}

#==================================================================================================

def find_spherical_cap_aniso(
        liq: np.ndarray,
        cell_params: np.ndarray,
        sol_heightmap: PeriodicGridInterpolator, *,
        N_pts: int = N_SPHERE_PTS
    ) -> dict[str, np.ndarray]:
    """Finds the spherical cap, defined as the best-fit sphere for a series of points on the far
    time-averaged liquid interface. This version makes minimal assumptions on rotational symmetry,
    and therefore does not report the intersecting contact line.

    Parameters
    ----------
    liq : ndarray
        The Cartesian coordinates of the liquid particles, with shape (N_frames, N_liq, 3) for a
        collection of frames, representing the liquid droplet in centred coordinates.
    cell_params : array_like
        The cell parameters, expressed as [cell_x, cell_y, cell_z].
    sol_heightmap : PeriodicGridInterpolator
        The time-averaged heightmap of the solid surface, which should be supplied as a
        PeriodicGridInterpolator with domain spanning -cell_x/2 to cell_x/2 along the x-coordinate
        and -cell_y/2 to cell_y/2 along the y-coordinate.
    N_pts : int, optional
        The number of sampling points to fit the sphere onto; also used for averaging the heightmap
        azimuthally to enforce rotational symmetry. Defaults to 150.

    Returns
    -------
    sphere_fit : dict
        A dictionary indicating the properties of the best-fit sphere, with the following fields:
            - 'r': float, the radius of the sphere
            - 'c': ndarray of shape (3,), the coordinates of the center of the sphere
    """

    # Find points on the spherical cap
    sphere_pts = _find_sphere_pts(liq, cell_params, sol_heightmap, N_pts=N_pts)

    # Least-squares best-fit spherical cap with no constraint
    A_mat = np.empty((sphere_pts.shape[0], 4), dtype=float)
    A_mat[:,0:3] = 2 * sphere_pts
    A_mat[:,3] = 1
    f_vec = np.empty((sphere_pts.shape[0], 1), dtype=float)
    f_vec[:,0] = np.sum(sphere_pts**2, axis=-1)
    c_vec, _, _, _ = np.linalg.lstsq(A_mat, f_vec, rcond=None)
    sphere_r = np.sqrt(np.sum(np.square(c_vec[0:3,0])) + c_vec[3,0])
    sphere_c = c_vec[0:3,0]
    
    return {'r': sphere_r, 'c': sphere_c}

#==================================================================================================

def find_sheet_normal(
        liq: np.ndarray,
        sol_heightmap: PeriodicGridInterpolator, *,
        tol: float = DEFAULT_TOLERANCE
    ) -> np.ndarray:
    """Finds nominal direction for the mean normal of the solid heightmap, defined as the direction
    of closest approach between the droplet centre-of-mass (in centred coordinates) and the time-
    averaged solid heightmap surface.

    Parameters
    ----------
    liq : ndarray
        The Cartesian coordinates of the liquid particles, with shape (N_frames, N_liq, 3) for a
        collection of frames, representing the liquid droplet in centred coordinates.
    sol_heightmap : PeriodicGridInterpolator
        The time-averaged heightmap of the solid surface, which should be supplied as a
        PeriodicGridInterpolator.
    tol : float, optional
        The precision tolerance for the position on the solid surface closest to the droplet CoM,
        in angstroms. Defaults to 0.01.
    
    Returns
    -------
    sheet_normal : ndarray
        The mean normal of the graphene sheet, with shape (3,).
    """

    CoM_z = np.mean(liq[...,2])
    grad = sol_heightmap.nabla()

    # Gradient descent (with reduction of descent rate) to minimize distance to CoM
    ratio = 1.0
    point = np.zeros((2,), dtype=float)
    dist_sq = (CoM_z - sol_heightmap(point)[0])**2
    dist_moved = np.inf
    while dist_moved > tol:
        delta_z = CoM_z - sol_heightmap(point)[0]
        gradient = (delta_z * grad(point)[0]) - point
        new_point = point + (ratio * gradient)
        new_dist_sq = np.sum((new_point)**2) + (CoM_z - sol_heightmap(new_point)[0])**2
        dist_moved = ratio * np.linalg.norm(gradient)
        if new_dist_sq >= dist_sq:
            ratio /= 2
        else:
            point = new_point
            dist_sq = new_dist_sq

    sheet_normal = np.array((-point[0], -point[1], CoM_z - sol_heightmap(point)[0]))
    return (sheet_normal / np.linalg.norm(sheet_normal))
