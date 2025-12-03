import numpy as np
    
#==================================================================================================

def center_coordinates(
        cell_params: np.ndarray,
        liq_raw: np.ndarray = None,
        sol_raw: np.ndarray = None
        ) -> tuple[np.ndarray, ...]:
    """Takes in a single frame of a liquid droplet on a solid surface, and shifts the coordinates
    such that the droplet centre-of-mass is on the z-axis (at x = y = 0), whereas the mean
    z-coordinate of the solid particles is zero.

    Parameters
    ----------
    cell_params : array_like
        The cell parameters, expressed as [cell_x, cell_y, cell_z].
    liq_raw : ndarray, optional
        The raw 3D Cartesian coordinates of the liquid particles at a single instantantaneous
        frame, which is of shape (N_liq, 3).
    sol_raw : ndarray, optional
        The raw 3D Cartesian coordinates of the solid particles at a single instantantaneous
        frame, which is of shape (N_sol, 3).

    Returns
    -------
    liq_centered : ndarray
        The centered Cartesian coordinates of the liquid particles, such that the droplet
        centre-of-mass is at x = y = 0. If `sol_raw` was provided, then the mean solid z-coordinate
        is zero, otherwise the droplet centre-of-mass is at z = 0.
    sol_centered : ndarray
        The centered Cartesian coordinates of the solid particles.
    shift : ndarray
        The vector of shape (3,) representing the active shift of the coordinate system's origin
        with respect to the original raw coordinates. Note that the motion of the droplet
        centre-of-mass with respect to the original raw coordinates is therefore the negative of
        this `shift` vector.
    """

    if sol_raw is None and liq_raw is None:
        raise RuntimeError('Cannot have both "liq_raw" and "sol_raw" missing!')
        
    cell_p = np.asarray(cell_params[0:3])
    sol = np.zeros((0, 3), dtype=float) if sol_raw is None else np.atleast_2d(sol_raw)
    liq = np.zeros((0, 3), dtype=float) if liq_raw is None else np.atleast_2d(liq_raw)
    N_liq = liq.shape[0]
    N_sol = sol.shape[0]
    shift = np.zeros(3, dtype=float)
    
    # If there are no solid particles, return just the liquid particles (centered on unit cell)
    if N_sol == 0:
        if N_liq == 0:
            raise RuntimeError('Both "liq" and "sol" have zero particles!')
        
        # Find CoM using circular average
        angles = 2.0 * np.pi * liq / cell_p
        angles = np.arctan2(np.mean(np.sin(angles), axis=0), np.mean(np.cos(angles), axis=0))
        CoM = 0.5 * cell_p * angles / np.pi
        liq -= CoM
        shift -= CoM
        liq -= cell_p * np.round(liq / cell_p)

        # Find CoM again using traditional average
        CoM = np.mean(liq, axis=0)
        liq -= CoM
        shift -= CoM
        liq -= cell_p * np.round(liq / cell_p)
        return liq, sol, shift
    
    # Otherwise, set middle of solid surface to z = 0
    cell_z = cell_p[2]
    for _ in range(3):
        sol[:,2] -= cell_z * np.round(sol[:,2] / cell_z)
        mean_z_coord = np.mean(sol[:,2])
        sol[:,2] -= mean_z_coord
        liq[:,2] -= mean_z_coord
        shift[2] -= mean_z_coord
    
    # If there are no liquid particles, return just the solid particles (centered on unit cell)
    cell_xy = cell_p[0:2]
    if N_liq == 0:
        sol[:,0:2] -= cell_xy * np.round(sol[:,0:2] / cell_xy)
        return liq, sol, shift

    # Set all liquid particles to the +z side of the solid surface
    liq[:,2] = np.remainder(liq[:,2], cell_z)

    # Find CoM using circular average
    angles = 2.0 * np.pi * liq / cell_p
    angles = np.arctan2(np.mean(np.sin(angles), axis=0), np.mean(np.cos(angles), axis=0))
    CoM = 0.5 * cell_p * angles / np.pi
    CoM[2] = 0.0
    liq -= CoM
    sol -= CoM
    shift -= CoM
    liq -= cell_p * np.round(liq / cell_p)

    # Find CoM again using traditional average
    CoM = np.mean(liq, axis=0)
    CoM[2] = 0.0
    liq -= CoM
    sol -= CoM
    shift -= CoM
    liq[:,0:2] -= cell_xy * np.round(liq[:,0:2] / cell_xy)
    sol[:,0:2] -= cell_xy * np.round(sol[:,0:2] / cell_xy)

    # Fixing a weird bug where the droplet is sometimes set to the wrong side of the unit cell
    CoM_z = np.mean(liq[:,2])
    liq[:,2] -= cell_z * np.round((liq[:,2] - CoM_z) / cell_z)

    return liq, sol, shift

