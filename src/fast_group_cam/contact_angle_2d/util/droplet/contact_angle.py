import numpy as np
import warnings
from typing import Any, Callable
from ..interpolate import PeriodicGridInterpolator
from .coarse_grain import find_interface, DEFAULT_TOLERANCE

#==================================================================================================
# Default parameters for droplet foot-finding and spherical cap-finding algorithms

Z_FOOT = 5.0          # Height of the droplet foot (in angstroms)
STEP_BACK = 10.0      # Step back per iteration of foot-finding algorithm (in angstroms)
MAX_ITER = 10         # Maximum number of iteration for foot-finding algorithm
N_SPHERE_PTS = 150    # Number of points to use to find best-fit spherical top

#==================================================================================================

def find_sheet_normal(
        waters: np.ndarray,
        heightmaps: np.ndarray,
        cell_xy: np.ndarray | tuple[float, float], *,
        tol: float = DEFAULT_TOLERANCE
    ) -> np.ndarray:
    """Finds nominal direction for the mean normal of the graphene sheet.

    Parameters
    ----------
    waters : ndarray
        The Cartesian coordinates of the water molecules, with shape (N_frames, N_water, 3) for a
        collection of frames, representing the water droplet.
    heightmaps : ndarray
        The regularized instantaneous heightmaps of the graphene sheet h(x, y; t), as per returned
        by the `util.graphene.sheet` module, with shape (N_frames, N_x, N_y).
    cell_xy : array_like
        The cell parameters along the x- and y-axes, expressed as [cell_x, cell_y].
    tol : float, optional
        The precision tolerance for the position on the graphene sheet closest to the droplet CoM,
        in angstroms. Defaults to 0.01.
    
    Returns
    -------
    sheet_normal : ndarray
        The mean normal of the graphene sheet, with shape (3,).
    """

    # Process input shapes and calculate CoM
    if len(waters.shape) == 3:
        if len(heightmaps.shape) != 3:
            raise ValueError(f'Unrecognized shape of heightmaps: {heightmaps.shape}')
        if heightmaps.shape[0] != waters.shape[0]:
            raise ValueError(f'Shape of heightmaps {heightmaps.shape} does not match waters ' +
                             f'{waters.shape}!')
        CoM = np.mean(waters, axis=(0, 1))
    elif len(waters.shape) == 2:
        heightmaps = np.atleast_3d(heightmaps)
        if heightmaps.shape[0] != 1:
            raise ValueError(f'Shape of heightmaps {heightmaps.shape} does not match waters ' +
                             f'{waters.shape}!')
        CoM = np.mean(waters, axis=0)
    else:
        raise ValueError(f'Unrecognized input shape: waters {waters.shape}')

    # Prepare interpolation functions for mean heightmap
    _, N_x, N_y = heightmaps.shape
    d_x = cell_xy[0] / N_x
    d_y = cell_xy[1] / N_y
    z = np.mean(heightmaps, axis=0)
    dz_dx = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) / (2 * d_x)
    dz_dy = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2 * d_y)
    z = PeriodicGridInterpolator(cell_xy[0:2], z)
    dz_dx = PeriodicGridInterpolator(cell_xy[0:2], dz_dx)
    dz_dy = PeriodicGridInterpolator(cell_xy[0:2], dz_dy)

    # Gradient descent (with smart reduction of descent rate) to minimize distance to CoM
    ratio = 1.0
    point = CoM[0:2]
    dist_sq = (CoM[2] - z(point)[0])**2
    dist_moved = np.inf
    while dist_moved > tol:
        gradient_x = (CoM[0] - point[0]) + ((CoM[2] - z(point)[0]) * dz_dx(point)[0])
        gradient_y = (CoM[1] - point[1]) + ((CoM[2] - z(point)[0]) * dz_dy(point)[0])
        new_point = point + (ratio * np.array((gradient_x, gradient_y)))
        new_dist_sq = np.sum((CoM[0:2] - new_point)**2) + (CoM[2] - z(new_point)[0])**2
        dist_moved = ratio * np.sqrt((gradient_x**2) + (gradient_y**2))
        if new_dist_sq >= dist_sq:
            ratio /= 2
        else:
            point = new_point
            dist_sq = new_dist_sq

    sheet_normal = CoM - np.array((point[0], point[1], z(point)[0]))
    return (sheet_normal / np.linalg.norm(sheet_normal))

#==================================================================================================

def find_droplet_foot(
        waters: np.ndarray,
        mean_heightmap: PeriodicGridInterpolator,
        search_directions: np.ndarray, *,
        tol: float = DEFAULT_TOLERANCE,
        z_foot: float = Z_FOOT,
        step_back: float = STEP_BACK,
        max_iter: int = MAX_ITER
    ) -> tuple[np.ndarray, np.ndarray]:
    """Finds the droplet foot(s), defined as the point on the Willard-Chandler interface with
    z-coordinate equal to the nominal interfacial separation plus z_foot, along a list of given
    search directions.

    Parameters
    ----------
    waters : ndarray
        The Cartesian coordinates of the water molecules, with shape (N_frames, N_water, 3) for a
        collection of frames, representing the water droplet.
    mean_heightmap : PeriodicGridInterpolator
        The time-averaged regularized heightmap of the graphene sheet <h(x, y)>_t, which should
        be supplied as a PeriodicGridInterpolator with domain spanning -cell_x/2 to cell_x/2 along
        the x-coordinate and -cell_y/2 to cell_y/2 along the y-coordinate.
    search_directions : ndarray
        The directions to search for the droplet foot(s), with shape (N_dir, 3). Each search
        direction is expected to be in the form of [cos(phi), sin(phi), 0.0].
    tol : float, optional
        The precision tolerance for the positions of the droplet foot(s), in angstroms. Defaults to
        0.01.
    z_foot : float, optional
        The defining parameter for the positions of the droplet foot(s), in the sense of distance
        to the graphene sheet (plus the nominal interfacial separation), in angstroms. Defaults to
        5.0.
    step_back: float, optional
        The maximal distance which the foot-finding algorithm is allowed to step backwards during
        a search iteration, in angstroms. Defaults to 10.0.
    max_iter: int, optional
        The maximum number of iterations which the foot-finding algorithm is allowed to repeat,
        before halting (even if not yet converged). Defaults to 10.
    
    Returns
    -------
    interfaces : ndarray
        The positions of the droplet foot(s) along the given search directions, with shape
        (N_dir, 3).
    normals: ndarray
        The surface normals of the droplet foot(s) at the corresponding positions, with shape
        (N_dir, 3).
        
    Warns
    -----
    RuntimeWarning
        If `find_interface` raises a RuntimeWarning; in which case a new warning is raised, and the
        corresponding output is padded with NaN.

    Notes
    -----
    The droplet foot is found iteratively, by first scanning for the Willard-Chandler interface in
    the given search direction starting from the CoM; then projecting that point onto the graphene
    sheet, and finding the z-coordinate to search; and then repeating the interface scan at that
    new z-coordinate to find an improved estimate of the droplet foot position. So long as neither
    the graphene sheet nor the interface are rapidly-varying in the neighbourhood of the droplet
    foot, this algorithm will converge stepwise towards the true value.
    """

    # Calculate droplet CoM and floor
    CoM = np.mean(waters, axis=(0, 1)) if len(waters.shape) == 3 else np.mean(waters, axis=0)
    droplet_floor = find_interface(waters, CoM, (0, 0, -1), tol=tol)[2]
    combined_z = z_foot + droplet_floor - mean_heightmap(CoM[0:2])[0]
    
    # Iterate through search directions
    interfaces = list()
    normals = list()
    warning_occurred = 0
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        for search_dir in np.atleast_2d(search_directions):
            try:

                # First guess of interface
                inter = find_interface(waters, CoM, search_dir)
                iter_count = 0
                dist_moved = np.inf
                prev_inter = inter

                # Repeat guesses until converged
                while ((dist_moved > tol) and (iter_count < max_iter)):
                    local_floor = mean_heightmap(inter[0:2])[0]
                    start = np.array((inter[0], inter[1], local_floor + combined_z))
                    start -= min(np.dot(inter, search_dir), step_back) * search_dir
                    inter = find_interface(waters, start, search_dir)
                    iter_count += 1
                    dist_moved = np.linalg.norm(inter - prev_inter)
                    prev_inter = inter

                # Final guess and append result
                local_floor = mean_heightmap(inter[0:2])[0]
                start = np.array((inter[0], inter[1], local_floor + combined_z))
                start -= min(np.dot(inter, search_dir), step_back) * search_dir
                inter, norm = find_interface(waters, start, search_dir, calc_normal=True)
                interfaces.append(inter)
                normals.append(norm)

            except RuntimeWarning:
                warning_occurred += 1
                interfaces.append(np.full((3,), np.nan))
                normals.append(np.full((3,), np.nan))

    if warning_occurred > 0:
        warnings.warn(f'find_droplet_foot: RuntimeWarning raised {warning_occurred} times(s) by ' +
                      'find_interface, output padded with NaN', RuntimeWarning)
    return (np.array(interfaces), np.array(normals))

#==================================================================================================

def _find_sphere_pts(
        waters: np.ndarray,
        cell_params: np.ndarray,
        mean_heightmap: PeriodicGridInterpolator, *,
        N_pts: int = N_SPHERE_PTS
    ) -> np.ndarray:
    """Internal function for finding points on the spherical cap."""

    # Calculate droplet CoM
    CoM = np.mean(waters, axis=(0, 1)) if len(waters.shape) == 3 else np.mean(waters, axis=0)

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
                test_z = mean_heightmap(test_r[:,None] * azi[None,i,:])
                safe_gradient = np.max((test_z - CoM[2]) / test_r)
                if safe_gradient >= 0.0 and polarc[i] < polars[i] * safe_gradient:
                    polars[i] = 1.0 / np.sqrt(1.0 + (safe_gradient**2))
                    polarc[i] = np.sqrt(1.0 - (polars[i]**2))
                search_dir = np.array((azi[i,0] * polars[i], azi[i,1] * polars[i], polarc[i]))
                sphere_pts.append(find_interface(waters, CoM, search_dir))
            except RuntimeWarning:
                warning_occurred += 1
    
    if warning_occurred > 0:
        warnings.warn(f'find_spherical_cap: RuntimeWarning raised {warning_occurred} times(s) ' +
                      'by find_interface', RuntimeWarning)
    return np.array(sphere_pts)
    
#==================================================================================================

def find_spherical_cap(
        waters: np.ndarray,
        cell_params: np.ndarray,
        mean_heightmap: PeriodicGridInterpolator, *,
        N_pts: int = N_SPHERE_PTS
    ) -> tuple[np.ndarray, np.ndarray]:
    """Finds the spherical cap, defined as the best-fit sphere for a series of points on the far
    time-averaged Willard-Chandler interface. This version assumes rotational symmetry.

    Parameters
    ----------
    waters : ndarray
        The Cartesian coordinates of the water molecules, with shape (N_frames, N_water, 3) for a
        collection of frames, representing the water droplet.
    cell_params : array_like
        The cell parameters, expressed as [cell_x, cell_y, cell_z].
    mean_heightmap : PeriodicGridInterpolator
        The time-averaged regularized heightmap of the graphene sheet <h(x, y)>_t, which should
        be supplied as a PeriodicGridInterpolator with domain spanning -cell_x/2 to cell_x/2 along
        the x-coordinate and -cell_y/2 to cell_y/2 along the y-coordinate.
    N_pts : int, optional
        The number of sampling points to fit the sphere onto; also used for averaging the heightmap
        azimuthally to enforce rotational symmetry. Defaults to 100.

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
    CoM = np.mean(waters, axis=(0, 1)) if len(waters.shape) == 3 else np.mean(waters, axis=0)
    phi = np.linspace(0, 2 * np.pi, N_pts, endpoint=False)
    azi = np.c_[np.cos(phi), np.sin(phi)]
    sphere_pts = _find_sphere_pts(waters, cell_params, mean_heightmap, N_pts=N_pts)

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
            floor = mean_heightmap(CoM[0:2])[0]
            for _ in range(10):
                sphere_a = np.sqrt((sphere_r**2) - ((sphere_z - floor)**2))
                floor = np.mean(mean_heightmap(sphere_a * azi))
            sphere_a = np.sqrt((sphere_r**2) - ((sphere_z - floor)**2))
            sphere_angle = 90.0 + (np.arcsin((sphere_z - floor) / sphere_r) * 180.0 / np.pi)
            floor_far = np.mean(mean_heightmap(1.01 * sphere_a * azi))
            floor_near = np.mean(mean_heightmap(0.99 * sphere_a * azi))
            local_grad = (floor_far - floor_near) / (0.02 * sphere_a)
            sphere_angle += np.arctan(local_grad) * 180.0 / np.pi
        except RuntimeWarning:
            sphere_a = 0.0
            sphere_angle = (180.0 if sphere_z > 0.0 else 0.0)

    return {'r': sphere_r, 'z': sphere_z, 'a': sphere_a, 'angle': sphere_angle}

#==================================================================================================

def find_spherical_cap_aniso(
        waters: np.ndarray,
        cell_params: np.ndarray,
        mean_heightmap: Callable[..., Any], *,
        N_pts: int = N_SPHERE_PTS
    ) -> tuple[np.ndarray, np.ndarray]:
    """Finds the spherical cap, defined as the best-fit sphere for a series of points on the far
    time-averaged Willard-Chandler interface. This version makes minimal assumptions on rotational
    symmetry, and therefore does not report the intersecting contact line.

    Parameters
    ----------
    waters : ndarray
        The Cartesian coordinates of the water molecules, with shape (N_frames, N_water, 3) for a
        collection of frames, representing the water droplet.
    cell_params : array_like
        The cell parameters, expressed as [cell_x, cell_y, cell_z].
    mean_heightmap : function (..., 2) -> (...)
        The time-averaged regularized heightmap of the graphene sheet <h(x, y)>_t, which should
        be supplied as a ufunc capable of taking (x, y) coordinates in the form of a ndarray of
        shape (..., 2) and returning the appropriate scalars, with domain spanning -cell_x/2 to
        cell_x/2 along the x-coordinate and -cell_y/2 to cell_y/2 along the y-coordinate.
    N_pts : int, optional
        The number of sampling points to fit the sphere onto; also used for averaging the heightmap
        azimuthally to enforce rotational symmetry. Defaults to 100.
    get_intersection : bool, optional
        If true, returns the radius and contact angle at the contact line/circle, else said fields
        will be missing from the output dictionary. Defaults to True.

    Returns
    -------
    sphere_fit : dict
        A dictionary indicating the properties of the best-fit sphere, with the following fields:
            - 'r': float, the radius of the sphere
            - 'c': ndarray of shape (3,), the coordinates of the center of the sphere
    """

    # Find points on the spherical cap
    sphere_pts = _find_sphere_pts(waters, cell_params, mean_heightmap, N_pts=N_pts)

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
