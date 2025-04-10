import warnings
import numpy as np

#==================================================================================================
# Default parameters for Willard-Chandler coarse-graining function

COARSE_GRAIN_LENGTH = 2.4    # Coarse-graining lengthscale (in angstroms)
CUTOFF_DENSITY = 0.016       # Cutoff density for interface (in angstroms**(-3))
SLICING_CUTOFF = 2           # Cutoff distance for slicing (in multiples of COARSE_GRAIN_LENGTH)
DEFAULT_TOLERANCE = 0.01     # Accuracy tolerance for interface search (in angstroms)

#==================================================================================================

def coarse_grained_density(pos: np.ndarray, waters: np.ndarray, *,
                           coarse_grain_length: float = COARSE_GRAIN_LENGTH) -> float | np.ndarray:
    """This function calculates the coarse-grained density distribution, given the coordinates of
the water molecules, at either one test point or a list of test points. The inputs are:

    pos:                 Either the test point to calculate the density function at, with shape
                           (3,); or the sequence of test points to calculate the density function
                           at, with shape (N_pos, 3).
    waters:              The Cartesian coordinates of the water molecules, with shape (N_water, 3).
    coarse_grain_length: The coarse-graining lengthscale in angstroms.

If pos has shape (3,), then the output is a float representing the density at pos; and if it has
shape (N_pos, 3), then the output is a np.NDArray of shape (N_pos,) representing the densities at
each position."""

    prefactor = np.power(2 * np.pi, -1.5) * np.power(coarse_grain_length, -3)
    scaling = -0.5 / (coarse_grain_length**2)

    if pos.shape == (3,):
        dist = waters - pos
        return np.sum(prefactor * np.exp(scaling * np.sum(np.square(dist), axis=-1)))
    elif len(pos.shape) == 2 and pos.shape[-1] == 3:
        dist = waters[np.newaxis,:,:] - pos[:,np.newaxis,:]
        return np.sum(prefactor * np.exp(scaling * np.sum(np.square(dist), axis=-1)), axis=-1)
    else:
        raise RuntimeError(f'Unrecognized input shape: {pos.shape}')
    
#==================================================================================================

def find_interface(waters: np.ndarray, search_start: np.ndarray = None, axis: np.ndarray = None,
                   tol: float = DEFAULT_TOLERANCE, max_dist: float = None, *,
                   coarse_grain_length: float = COARSE_GRAIN_LENGTH,
                   cutoff_density: float = CUTOFF_DENSITY,
                   slicing_cutoff: float = SLICING_CUTOFF, reverse_search = False) -> np.ndarray:
    """This function takes in the coordinates of water molecules, and finds the Willard-Chandler
instantaneous interface along a given search axis. The search axis should be a ray starting from
within the bulk liquid region, pointing 'outwards' towards the exterior region in some direction.
The inputs are:

    waters:              The Cartesian coordinates of the water molecules, with shape (N_water, 3).
    search_start:        The coordinates to begin searching from, with shape (3,); the search axis
                           is a ray extending from this point. Ideally this point should be within
                           the bulk liquid region, unless reverse_search is turned on.
    axis:                The direction of the search axis to scan along.
    tol:                 The distance tolerance for the position of the intersection along the
                           search axis.
    max_dist:            The maximum distance along the search axis to scan; set to None to search
                           unlimited distances. Do not set to np.inf under any circumstance!
    coarse_grain_length: The coarse-graining lengthscale in angstroms.
    cutoff_density:      The cutoff number density parameter for defining the interface; usually
                           should be set to half of the bulk number density.
    slicing_cutoff:      The cutoff distance for slicing the subset of water molecules relevant to
                           the search axis (to reduce computational costs of calculating the
                           density function), as expressed in multiples of coarse_grain_length;
                           making this parameter smaller speeds up calculation but may introduce
                           inaccuracy. Set to None or np.inf to prevent this slicing.
    reverse_search:      Without reverse_search, it is assumed that the search axis starts from
                           within the bulk liquid region, hence the interface is the first point
                           where the density falls to the value of cutoff_density; turning on
                           reverse_search reverses this behaviour, so that the interface is the
                           first point where the density rises to the value of cutoff_density.

The output is a np.NDArray of shape (3,), representing the position of the intersection between the
Willard-Chandler interface and the search axis. A RuntimeWarning will be issued if the intersection
cannot be found, in which case the output will default to search_start."""

    # Establish default parameters
    slicing_width = (np.inf if slicing_cutoff is None else (slicing_cutoff * coarse_grain_length))
    search_start = np.array(((0, 0, 0) if search_start is None else search_start), dtype=float)
    axis = np.array(((0, 0, 1) if axis is None else axis), dtype=float)
    axis /= np.linalg.norm(axis)

    # Shift positions ("sliced") to set search_start as the origin
    sliced = waters - search_start

    # Rotate positions so that search axis is the z-axis
    tmp = (axis[0]**2) + (axis[1]**2)
    if tmp > 0.01:
        xx = ((axis[2] * (axis[0]**2)) + (axis[1]**2)) / tmp
        yy = ((axis[0]**2) + (axis[2] * (axis[1]**2))) / tmp
        xy = (axis[0] * axis[1] * (axis[2] - 1)) / tmp
        rot_matrix = np.array(((xx,      xy,      -axis[0]),
                               (xy,      yy,      -axis[1]),
                               (axis[0], axis[1], axis[2])), dtype=float)
        sliced = np.einsum('jk,ik->ij', rot_matrix, sliced)
    elif axis[2] < 0:
        sliced[:,(1,2)] *= -1

    # Perform slicing to reduce number of molecules to compute
    sliced = waters[waters[:,0] < slicing_width]
    sliced = sliced[sliced[:,0] > -slicing_width]
    sliced = sliced[sliced[:,1] < slicing_width]
    sliced = sliced[sliced[:,1] > -slicing_width]

    # Maximum z-coordinate to search
    z_ceil = (np.max(sliced[:,2]) + slicing_width if max_dist is None else max_dist)

    # Initial coarse search to determine maximum and minimum densities
    testpoints = np.linspace(0, z_ceil, 30)
    testpoints = np.column_stack(np.broadcast_arrays(0.0, 0.0, testpoints))
    densities = coarse_grained_density(testpoints, sliced)

    if np.max(densities) < cutoff_density:
        warnings.warn('System density lower than cutoff density everywhere, could not identify ' +
                      'any liquid region', RuntimeWarning)
        return search_start
    if np.min(densities) > cutoff_density:
        warnings.warn('System density higher than cutoff density everywhere, could not identify ' +
                      'any exterior region', RuntimeWarning)
        return search_start

    # Perform binary search to find the distance of the interface from the origin
    if reverse_search:
        lower = testpoints[np.argmin(densities), 2]
        upper = z_ceil
        result = (lower + upper) / 2.0
        while (upper - lower) > tol:
            density = coarse_grained_density(np.array((0.0, 0.0, result)), sliced)
            if density > cutoff_density:
                upper = result
            else:
                lower = result
            result = (lower + upper) / 2.0
    else:
        lower = testpoints[np.argmax(densities), 2]
        upper = z_ceil
        result = (lower + upper) / 2.0
        while (upper - lower) > tol:
            density = coarse_grained_density(np.array((0.0, 0.0, result)), sliced)
            if density > cutoff_density:
                lower = result
            else:
                upper = result
            result = (lower + upper) / 2.0

    return search_start + (result * axis)


