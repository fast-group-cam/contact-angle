import warnings
import numpy as np

#==================================================================================================
# Default parameters for Willard-Chandler coarse-graining function

COARSE_GRAIN_LENGTH = 2.4    # Coarse-graining lengthscale (in angstroms)
CUTOFF_DENSITY = 0.016       # Cutoff density for interface (in angstroms**(-3))
BULK_DENSITY = 0.033368      # Bulk density for liquid (in angstroms**(-3))
SLICING_CUTOFF = 3.5         # Cutoff distance for slicing (in multiples of COARSE_GRAIN_LENGTH)
DEFAULT_TOLERANCE = 0.01     # Accuracy tolerance for interface search (in angstroms)

MAX_SIZE = 1e7               # Maximum size of numpy broadcasting operation (in size(float)s) for
                             # memory protection

#==================================================================================================

def coarse_grained_density(
        pos: np.ndarray,
        waters: np.ndarray, *,
        coarse_grain_length: float = COARSE_GRAIN_LENGTH,
        max_size: int = MAX_SIZE
        ) -> float | np.ndarray:
    """This function calculates the coarse-grained density distribution, given the coordinates of
the water molecules, at either one test point or a list of test points. The inputs are:

    - `pos`: Either the test point to calculate the density function at, with shape (3,); or the
    sequence of test points to calculate the density function at, with shape (N_pos, 3).
    - `waters`: The Cartesian coordinates of the water molecules, either with shape (N_water, 3)
    for a single instantaneous frame; or shape (N_frames, N_water, 3) for a collection of frames,
    in which case the density distribution is averaged over the frames.
    - `coarse_grain_length`: The coarse-graining lengthscale in angstroms.

If pos has shape (3,), then the output is a float representing the density at pos; and if it has
shape (N_pos, 3), then the output is a np.NDArray of shape (N_pos,) representing the densities at
each position."""

    if len(waters.shape) == 2 and waters.shape[-1] == 3:
        N_frames = 1
        waters_unrolled = waters
    elif len(waters.shape) == 3 and waters.shape[-1] == 3:
        N_frames = waters.shape[0]
        waters_unrolled = waters.reshape(-1, 3)
    else:
        raise RuntimeError(f'Unregonized input shape: waters {waters.shape}')

    prefactor = np.power(2 * np.pi, -1.5) * np.power(coarse_grain_length, -3) / N_frames
    scaling = -0.5 / (coarse_grain_length**2)

    if pos.shape == (3,):

        dist = waters_unrolled - pos
        return np.sum(prefactor * np.exp(scaling * np.sum(np.square(dist), axis=-1)))
    
    elif len(pos.shape) == 2 and pos.shape[-1] == 3:

        if (pos.shape[0] * waters_unrolled.shape[0] * 3) < max_size:
            dist = waters_unrolled[None,:,:] - pos[:,None,:]
            return np.sum(prefactor * np.exp(scaling * np.sum(np.square(dist), axis=-1)), axis=-1)
        else:
            results = np.empty((pos.shape[0],), dtype=float)
            for i, p in enumerate(pos):
                dist = waters_unrolled - p
                results[i] = np.sum(prefactor * np.exp(scaling * np.sum(np.square(dist), axis=-1)))
            return results

    elif len(pos.shape) == 3 and pos.shape[-1] == 3:

        if (pos.shape[0] * pos.shape[1] * waters_unrolled.shape[0] * 3) < max_size:
            dist = waters_unrolled[None,None,:,:] - pos[:,:,None,:]
            return np.sum(prefactor * np.exp(scaling * np.sum(np.square(dist), axis=-1)), axis=-1)
        elif (pos.shape[1] * waters_unrolled.shape[0] * 3) < max_size:
            results = np.empty((pos.shape[0], pos.shape[1]), dtype=float)
            for i, p in enumerate(pos):
                dist = waters_unrolled[None,:,:] - p[:,None,:]
                results[i] = np.sum(prefactor * np.exp(scaling * np.sum(np.square(dist), axis=-1)), axis=-1)
            return results
        else:
            results = np.empty((pos.shape[0], pos.shape[1]), dtype=float)
            for i, row in enumerate(pos):
                for j, p in enumerate(row):
                    dist = waters_unrolled - p
                    results[i,j] = np.sum(prefactor * np.exp(scaling * np.sum(np.square(dist), axis=-1)))
            return results
        
    else:
        raise RuntimeError(f'Unrecognized input shape: pos {pos.shape}')
    
#==================================================================================================

def coarse_grained_density_grad(
        pos: np.ndarray,
        waters: np.ndarray, *,
        coarse_grain_length: float = COARSE_GRAIN_LENGTH,
        max_size: int = MAX_SIZE
        ) -> float | np.ndarray:
    """This function calculates the gradient of the coarse-grained density distribution, given the
coordinates of the water molecules, at either one test point or a list of test points. See
`coarse_grained_density` for the list of inputs.

If pos has shape (3,), then the output is a np.NDArray of shape (3,) representing the density
gradient at pos; and if it has shape (N_pos, 3), then the output is a np.NDArray of shape
(N_pos, 3) representing the density gradients at each position."""

    if len(waters.shape) == 2 and waters.shape[-1] == 3:
        N_frames = 1
        waters_unrolled = waters
    elif len(waters.shape) == 3 and waters.shape[-1] == 3:
        N_frames = waters.shape[0]
        waters_unrolled = waters.reshape(-1, 3)
    else:
        raise RuntimeError(f'Unregonized input shape: waters {waters.shape}')
    
    prefactor = np.power(2 * np.pi, -1.5) * np.power(coarse_grain_length, -5) / N_frames
    scaling = -0.5 / (coarse_grain_length**2)

    if pos.shape == (3,):

        dist = waters_unrolled - pos
        return np.sum(prefactor * dist[:,:] *
                      np.exp(scaling * np.sum(np.square(dist), axis=-1))[:,None], axis=0)
    
    elif len(pos.shape) == 2 and pos.shape[-1] == 3:

        if (pos.shape[0] * waters_unrolled.shape[0] * 3) < max_size:
            dist = waters_unrolled[None,:,:] - pos[:,None,:]
            return np.sum(prefactor * dist[:,:,:] *
                          np.exp(scaling * np.sum(np.square(dist), axis=-1))[:,:,None], axis=1)
        else:
            results = np.empty((pos.shape[0], 3), dtype=float)
            for i, p in enumerate(pos):
                dist = waters_unrolled - p
                results[i] = np.sum(prefactor * dist[:,:] *
                                    np.exp(scaling * np.sum(np.square(dist), axis=-1))[:,None], axis=0)
            return results
        
    elif len(pos.shape) == 3 and pos.shape[-1] == 3:

        if (pos.shape[0] * pos.shape[1] * waters_unrolled.shape[0] * 3) < max_size:
            dist = waters_unrolled[None,None,:,:] - pos[:,:,None,:]
            return np.sum(prefactor * dist[:,:,:,:] *
                          np.exp(scaling * np.sum(np.square(dist), axis=-1))[:,:,:,None], axis=2)
        elif (pos.shape[1] * waters_unrolled.shape[0] * 3) < max_size:
            results = np.empty((pos.shape[0], pos.shape[1], 3), dtype=float)
            for i, p in enumerate(pos):
                dist = waters_unrolled[None,:,:] - p[:,None,:]
                results[i] = np.sum(prefactor * dist[:,:,:] *
                                    np.exp(scaling * np.sum(np.square(dist), axis=-1))[:,:,None], axis=1)
            return results
        else:
            results = np.empty((pos.shape[0], pos.shape[1], 3), dtype=float)
            for i, row in enumerate(pos):
                for j, p in enumerate(row):
                    dist = waters_unrolled - p
                    results[i,j] = np.sum(prefactor * dist[:,:] *
                                          np.exp(scaling * np.sum(np.square(dist), axis=-1))[:,None], axis=0)
            return results
        
    else:
        raise RuntimeError(f'Unrecognized input shape: pos {pos.shape}')

#==================================================================================================

def find_interface(
        waters: np.ndarray,
        search_start: np.ndarray = None,
        axis: np.ndarray = None,
        tol: float = DEFAULT_TOLERANCE,
        max_dist: float = None,
        calc_normal: bool = False, *,
        coarse_grain_length: float = COARSE_GRAIN_LENGTH,
        cutoff_density: float = CUTOFF_DENSITY,
        slicing_cutoff: float = SLICING_CUTOFF,
        reverse_search = False
        ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """This function takes in the coordinates of water molecules, and finds the Willard-Chandler
instantaneous interface along a given search axis. The search axis should be a ray starting from
within the bulk liquid region, pointing 'outwards' towards the exterior region in some direction.
The inputs are:

    - `waters`: The Cartesian coordinates of the water molecules, either with shape (N_water, 3)
    for a single instantaneous frame; or shape (N_frames, N_water, 3) for multiple frames, which
    will be collated together.
    - `search_start`: The coordinates to begin searching from, with shape (3,); the search axis is
    a ray extending from this point. Ideally this point should be within the bulk liquid region,
    unless reverse_search is turned on.
    - `axis`: The direction of the search axis to scan along.
    - `tol`: The distance tolerance for the position of the intersection along the search axis.
    - `max_dist`: The maximum distance along the search axis to scan; set to None to search
    unlimited distances. Do not set to np.inf under any circumstance!
    - `calc_normal`: Whether the output should include the interface normal or not.
    - `coarse_grain_length`: The coarse-graining lengthscale in angstroms.
    - `cutoff_density`: The cutoff number density parameter for defining the interface; usually
    should be set to half of the bulk number density.
    - `slicing_cutoff`: The cutoff distance for slicing the subset of water molecules relevant to
    the search axis (to reduce computational costs of calculating the density function), as
    expressed in multiples of coarse_grain_length; making this parameter smaller speeds up the
    calculation but may introduce inaccuracy. Set to None or np.inf to prevent this slicing.
    - `reverse_search`: Without reverse_search, it is assumed that the search axis starts from
    within the bulk liquid region, hence the interface is the first point where the density falls
    to the value of cutoff_density; turning on reverse_search reverses this behaviour, so that the
    interface is instead the first point where the density rises to the value of cutoff_density.

If calc_normal is not enabled (default), the output is a np.NDArray of shape (3,) representing the
position of the intersection between the Willard-Chandler interface and the search axis. A
RuntimeWarning will be issued if the intersection cannot be found, in which case the output will
default to search_start.

If calc_normal is enabled, the output is a tuple of two np.NDArrays both of shape (3,), which
represent the position and normal vector of the interface respectively. As before, a RuntimeWarning
will be issued if the intersection cannot be found, in which case the output will default to
(search_start, axis)."""

    # Process water shape and unroll (collate) into a single array
    if len(waters.shape) == 2 and waters.shape[-1] == 3:
        N_frames = 1
        waters_unrolled = waters
    elif len(waters.shape) == 3 and waters.shape[-1] == 3:
        N_frames = waters.shape[0]
        waters_unrolled = waters.reshape(-1, 3)
    else:
        raise RuntimeError(f'Unregonized input shape: waters {waters.shape}')
    
    # Establish default parameters
    slice_width = (np.inf if slicing_cutoff is None else (slicing_cutoff * coarse_grain_length))
    search_start = np.array(((0, 0, 0) if search_start is None else search_start), dtype=float)
    axis = np.array(((0, 0, 1) if axis is None else axis), dtype=float)
    axis /= np.linalg.norm(axis)

    # Shift positions ("sliced") to set search_start as the origin
    sliced = waters_unrolled - search_start

    # Perform slicing to reduce number of molecules to compute
    parallel_components = np.dot(sliced, axis)
    perpendiculars = sliced - (parallel_components[:, None] * axis[None, :])
    perpendiculars = np.sum(np.square(perpendiculars), axis=-1)
    sliced = sliced[perpendiculars < slice_width**2]
    parallel_components = np.dot(sliced, axis)
    sliced = sliced[parallel_components > -slice_width]
    if sliced.shape[0] == 0:
        warnings.warn('No waters encountered along search axis')
        return ((search_start, axis) if calc_normal else search_start)

    # Maximum z-coordinate to search
    if max_dist is None:
        z_ceil = np.max(parallel_components) + (-slice_width if reverse_search else slice_width)
    else:
        z_ceil = max_dist

    # Initial coarse search to determine maximum and minimum densities
    testpoints = np.linspace(0, z_ceil, 30)
    testpoints = testpoints[:, np.newaxis] * axis[np.newaxis, :]
    densities = coarse_grained_density(testpoints, sliced, coarse_grain_length=coarse_grain_length) / N_frames

    if np.max(densities) < cutoff_density:
        warnings.warn('System density lower than cutoff density everywhere, could not identify ' +
                      'any liquid region', RuntimeWarning)
        return ((search_start, axis) if calc_normal else search_start)
    if np.min(densities) > cutoff_density:
        warnings.warn('System density higher than cutoff density everywhere, could not identify ' +
                      'any exterior region', RuntimeWarning)
        return ((search_start, axis) if calc_normal else search_start)

    # Perform binary search to find the distance of the interface from the origin
    if reverse_search:
        lower = np.dot(testpoints[np.argmin(densities)], axis)
        upper = z_ceil
        result = (lower + upper) / 2.0
        while (upper - lower) > tol:
            density = coarse_grained_density(result * axis, sliced,
                                             coarse_grain_length=coarse_grain_length) / N_frames
            if density > cutoff_density:
                upper = result
            else:
                lower = result
            result = (lower + upper) / 2.0
    else:
        lower = np.dot(testpoints[np.argmax(densities)], axis)
        upper = z_ceil
        result = (lower + upper) / 2.0
        while (upper - lower) > tol:
            density = coarse_grained_density(result * axis, sliced,
                                             coarse_grain_length=coarse_grain_length) / N_frames
            if density > cutoff_density:
                lower = result
            else:
                upper = result
            result = (lower + upper) / 2.0

    if calc_normal:
        density_grad = coarse_grained_density_grad(result * axis, sliced,
                                                   coarse_grain_length=coarse_grain_length)
        return (search_start + (result * axis), -density_grad / np.linalg.norm(density_grad))
    return search_start + (result * axis)


