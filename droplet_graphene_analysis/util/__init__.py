import time
import numpy as np
from .droplet.coordinates import center_coordinates, read_droplet_trajectory

#==================================================================================================

def elapsed_time(time_start: float) -> str:
    """Given a start time measured by ``time.time()``, this function measures the time elapsed
    since and formats it nicely into a string.
    """
    time_taken = time.time() - time_start
    if time_taken > 3600:
        hours = int(time_taken / 3600)
        time_taken -= hours * 3600
        mins = int(time_taken / 60)
        time_taken -= mins * 60
        return f'(hh:mm:ss) {hours:02}:{mins:02}:{round(time_taken):02}'
    if time_taken > 60:
        mins = int(time_taken / 60)
        time_taken -= mins * 60
        return f'{mins}mins{round(time_taken):02}s'
    return f'{time_taken:.3f}s'

#==================================================================================================

def best_fit_sphere(points: np.ndarray, d: int = None) -> tuple[float, np.ndarray]:
    """Calculates the best-fit (d-1)-dimensional sphere, i.e. the surface defined by
    :math:`|r - c| = a`, for a set of d-dimensional points. The sphere is fitted by minimizing the
    squares of the points' squared distances from the centre minus the squared radius, i.e. the sum
    of :math:`(|r_i - c|^2 - a^2)^2`.

    Parameters
    ----------
    points : ndarray
        Cartesian coordinates of the points to fit to, with shape (N_points, d).
    d : None or int
        The number of dimensions; will be deduced from the last axis of `points` if not specified.

    Returns
    -------
    a : float
        The radius of the best-fit sphere; guaranteed positive.
    c : ndarray
        The Cartesian coordinates of the center of the best-fit sphere, with shape (d,).
    """
    dims = (points.shape[-1] if d is None else int(d))
    A_mat = np.empty((points.shape[0], dims+1), dtype=float)
    A_mat[:,0:dims] = 2 * points[:,0:dims]
    A_mat[:,dims] = 1
    f_vec = np.empty((points.shape[0], 1), dtype=float)
    f_vec[:,0] = np.sum((points[:,0:dims])**2, axis=-1)
    c_vec, _, _, _ = np.linalg.lstsq(A_mat, f_vec, rcond=None)
    a = np.sqrt(np.sum(np.square(c_vec[0:dims,0])) + c_vec[dims,0])
    return (a, c_vec[0:dims,0])

#==================================================================================================

def best_fit_axial_sphere(points: np.ndarray) -> tuple[float, float]:
    """Calculates the best-fit (d-1)-dimensional sphere constrained along the last axis, i.e. the
    surface defined by :math:`|r - c| = a` for centre :math:`c = (0, 0, ..., c_z)`, for a set of
    d-dimensional points. The sphere is fitted by minimizing the squares of the points' squared
    distances from the centre minus the squared radius, i.e. the sum of :math:`(|r_i - c|^2 - a^2)^2`.

    Parameters
    ----------
    points : ndarray
        Cartesian coordinates of the points to fit to, with shape (N_points, d).

    Returns
    -------
    a : float
        The radius of the best-fit sphere; guaranteed positive.
    c_z : float
        The last Cartesian coordinate of the center of the best-fit sphere.

    Notes
    -----
    Unlike `best_fit_sphere`, the dimension d cannot be specified for this function, and will
    always use the full shape of `points`.
    """
    A_mat = np.empty((points.shape[0], 2), dtype=float)
    A_mat[:,0] = 2 * points[:,-1]
    A_mat[:,1] = 1
    f_vec = np.empty((points.shape[0], 1), dtype=float)
    f_vec[:,0] = np.sum(points**2, axis=-1)
    c_vec, _, _, _ = np.linalg.lstsq(A_mat, f_vec, rcond=None)
    a = np.sqrt(np.sum(np.square(c_vec[0,0])) + c_vec[1,0])
    return (a, c_vec[0,0])