import warnings
import numpy as np
from numba import jit

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
    """Calculates the coarse-grained density distribution, given the water molecules' coordinates,
    at either one test point or a list of test points.

    Parameters
    ----------
    pos : ndarray
        Either the test point to calculate the density function at, with shape (3,); or the
        sequence of test points to calculate the density function at, with shape (N_pos, 3).
    waters : ndarray
        The Cartesian coordinates of the water molecules, either with shape (N_water, 3) for a
        single instantaneous frame; or shape (N_frames, N_water, 3) for a collection of frames,
        in which case the density distribution is averaged over the frames.
    coarse_grain_length : float, optional
        The coarse-graining lengthscale in angstroms. Defaults to 2.4.
    max_size: int, optional
        The maximum size (i.e. number of floats) that broadcasting operations are allowed to create
        during the calculation, to prevent running out of memory. Defaults to 1e7.

    Returns
    -------
    density : ndarray
        An array of shape (N_pos,), representing the densities at each position. A scalar is
        returned if only one position is supplied.
    """
    
    if len(waters.shape) == 2:
        
        prefactor = np.power(2 * np.pi, -1.5) * np.power(coarse_grain_length, -3)
        scaling = -0.5 / (coarse_grain_length**2)
        chunk_size = max(1, max_size // pos.size)

        if len(pos.shape) == 1:
            return _cg_dens_chunked(np.atleast_2d(pos), waters, prefactor, scaling, chunk_size)[0]
        elif len(pos.shape) == 2:
            return _cg_dens_chunked(pos, waters, prefactor, scaling, chunk_size)
        else:
            return _cg_dens_chunked(pos.reshape(-1, pos.shape[-1]), waters, prefactor, scaling,
                                    chunk_size).reshape(pos.shape[:-1])

    elif len(waters.shape) == 3:

        N_frames = waters.shape[0]
        flat_waters = waters.reshape(-1, 3)
        prefactor = np.power(2 * np.pi, -1.5) * np.power(coarse_grain_length, -3) / N_frames
        scaling = -0.5 / (coarse_grain_length**2)
        chunk_size = max(1, max_size // pos.size)

        if len(pos.shape) == 1:
            return _cg_dens_chunked(np.atleast_2d(pos), flat_waters, prefactor, scaling,
                                    chunk_size)[0]
        elif len(pos.shape) == 2:
            return _cg_dens_chunked(pos, flat_waters, prefactor, scaling, chunk_size)
        else:
            return _cg_dens_chunked(pos.reshape(-1, pos.shape[-1]), flat_waters, prefactor,
                                    scaling, chunk_size).reshape(pos.shape[:-1])
        
    else:
        raise RuntimeError(f'Unregonized input shape for waters {waters.shape}!')
    
#--------------------------------------------------------------------------------------------------

@jit('float64[:](float64[:,:], float64[:,:], float64, float64, int32)', nopython=True)
def _cg_dens_chunked(pos: np.ndarray, waters: np.ndarray, prefactor: float, scaling: float,
                     chunk_size: int = MAX_SIZE):
    result = np.zeros(pos.shape[0])
    N_waters = waters.shape[0]
    for start in range(0, N_waters, chunk_size):
        end = min(start + chunk_size, N_waters)
        waters_chunk = waters[start:end]
        disp = waters_chunk[None,:,:] - pos[:,None,:]
        dist_sq = np.sum(disp**2, axis=-1)
        result += prefactor * np.sum(np.exp(scaling * dist_sq), axis=-1)
    return result

#==================================================================================================

def coarse_grained_density_grad(
        pos: np.ndarray,
        waters: np.ndarray, *,
        coarse_grain_length: float = COARSE_GRAIN_LENGTH,
        max_size: int = MAX_SIZE
        ) -> np.ndarray:
    """Calculates the gradient of the coarse-grained density distribution, given the water
    molecules' coordinates, at either one test point or a list of test points.

    Parameters
    ----------
    pos : ndarray
        Either the test point to calculate the density function at, with shape (3,); or the
        sequence of test points to calculate the density function at, with shape (N_pos, 3).
    waters : ndarray
        The Cartesian coordinates of the water molecules, either with shape (N_water, 3) for a
        single instantaneous frame; or shape (N_frames, N_water, 3) for a collection of frames,
        in which case the density distribution is averaged over the frames.
    coarse_grain_length : float, optional
        The coarse-graining lengthscale in angstroms. Defaults to 2.4.
    max_size: int, optional
        The maximum size (i.e. number of floats) that broadcasting operations are allowed to create
        during the calculation, to prevent running out of memory. Defaults to 1e7.

    Returns
    -------
    density_gradient : ndarray
        An array of shape (N_pos, 3), representing the density gradients at each position. An array
        of shape (3,) is returned if only one position is supplied.
    """

    if len(waters.shape) == 2:
        
        prefactor = np.power(2 * np.pi, -1.5) * np.power(coarse_grain_length, -5)
        scaling = -0.5 / (coarse_grain_length**2)
        chunk_size = max(1, max_size // pos.size)

        if len(pos.shape) == 1:
            return _cg_densgrad_chunked(np.atleast_2d(pos), waters, prefactor, scaling,
                                        chunk_size)[0]
        elif len(pos.shape) == 2:
            return _cg_densgrad_chunked(pos, waters, prefactor, scaling, chunk_size)
        else:
            return _cg_densgrad_chunked(pos.reshape(-1, pos.shape[-1]), waters, prefactor, scaling,
                                        chunk_size).reshape(pos.shape)

    elif len(waters.shape) == 3:

        N_frames = waters.shape[0]
        flat_waters = waters.reshape(-1, 3)
        prefactor = np.power(2 * np.pi, -1.5) * np.power(coarse_grain_length, -5) / N_frames
        scaling = -0.5 / (coarse_grain_length**2)
        chunk_size = max(1, max_size // pos.size)

        if len(pos.shape) == 1:
            return _cg_densgrad_chunked(np.atleast_2d(pos), flat_waters, prefactor, scaling,
                                        chunk_size)[0]
        elif len(pos.shape) == 2:
            return _cg_densgrad_chunked(pos, flat_waters, prefactor, scaling, chunk_size)
        else:
            return _cg_densgrad_chunked(pos.reshape(-1, pos.shape[-1]), flat_waters, prefactor,
                                        scaling, chunk_size).reshape(pos.shape)
        
    else:
        raise RuntimeError(f'Unregonized input shape for waters {waters.shape}!')

#--------------------------------------------------------------------------------------------------

@jit('float64[:,:](float64[:,:], float64[:,:], float64, float64, int32)', nopython=True)
def _cg_densgrad_chunked(pos: np.ndarray, waters: np.ndarray, prefactor: float, scaling: float,
                     chunk_size: int = MAX_SIZE):
    result = np.zeros_like(pos)
    N_waters = waters.shape[0]
    for start in range(0, N_waters, chunk_size):
        end = min(start + chunk_size, N_waters)
        waters_chunk = waters[start:end]
        disp = waters_chunk[None,:,:] - pos[:,None,:]
        dist_sq = np.sum(disp**2, axis=-1)
        result += prefactor * np.sum(disp[:,:,:] * np.exp(scaling * dist_sq)[:,:,None], axis=1)
    return result

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
        reverse_search: bool = False
        ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Finds the Willard-Chandler interface, defined as the isosurface of the coarse-grained
    density distribution at the cutoff density, given the water molecules' coordinates and a
    specified search ray.

    Parameters
    ----------
    waters : ndarray
        The Cartesian coordinates of the water molecules, either with shape (N_water, 3) for a
        single instantaneous frame, in which case the found interface represents an instantaneous
        interface; or shape (N_frames, N_water, 3) for a collection of frames, in which case the
        found interface represents the time-averaged interface.
    search_start: array_like, optional
        The coordinates of the starting point to begin searching from, with shape (3,); the search
        ray extends from this point. This point should be within the bulk liquid region, unless
        `reverse_search` is turned on in which case this point should be outside the bulk liquid
        region. Defaults to the origin.
    axis : array_like, optional
        The direction of the search ray to scan along, with shape (3,). Defaults to the z-axis.
    tol : float, optional
        The precision tolerance for the position of the interface intersection along the search
        ray, in angstroms. Defaults to 0.01.
    max_dist : float, optional
        The maximum distance along the search ray to scan, in angstroms; it is important that this
        maximal endpoint (i.e. `search_start` + `max_dist` * `axis`) is outside the bulk liquid
        region if `reverse_search` is turned off / inside the bulk liquid region if
        `reverse_search` is turned on. Setting `max_dist` to None, which is default, allows
        searching unlimited distances. **Warning**: do not set to infinity!
    calc_normal: bool, optional
        Whether the output should include the interface normal or not; defaults to False.
    
    Returns
    -------
    inter : ndarray
        The location of the intersection between the Willard-Chandler interface and the search ray,
        with shape (3,).
    norm: ndarray, only if `calc_normal` is True
        The surface normal of the Willard-Chandler interface at the found intersection, oriented
        to point *out* of the liquid region, with shape (3,).

    Other Parameters
    ----------------
    coarse_grain_length : float, optional
        The coarse-graining lengthscale in angstroms. Defaults to 2.4.
    cutoff_density : float, optional
        The cutoff density defining the interface, i.e. the isosurface where the coarse-grained
        density distribution is equal to this cutoff density, in angstrom^-3. Defaults to 0.016.
    slicing_cutoff : float, optional
        The cutoff distance for slicing the subset of water molecules relevant to the search ray,
        in multiples of `coarse_grain_length`; see notes below. Set to None or np.inf to disable
        this slicing. Defaults to 3.5.
    reverse_search : bool, optional
        The default behavior is that the search ray starts from within the bulk liquid region and
        exits it somewhere, hence the interface is found by searching for the point where the
        density *falls* to the cutoff density; turning on `reverse_search` reverses this behavior,
        so that the search ray starts from outside the bulk liquid region and enters it somewhere,
        and the interface is found by searching for the point where the density *rises* to the
        cutoff density instead.
        
    Warns
    -----
    RuntimeWarning
        If there are no water molecules within the sliced subset; in which case the fallback return
        value of `inter` is `search_start`, and `norm` is `axis`.

    Notes
    -----
    The interface is found by performing a binary search along the search axis to find the point at
    which the coarse-grained density function is equal to the cutoff density; it is therefore
    important that `search_start` and `max_dist` are chosen so that the search ray begins within
    the bulk liquid region, and ends outside the bulk liquid region (leaving `max_dist` as None
    guarantees this), or vice versa in the case of `reverse_search`.

    The behavior of this binary search is ill-defined, and may be sensitive to small changes in
    `search_start` or `max_dist`, if the search ray crosses multiple interfaces (e.g. in the case
    of a void within the liquid).

    To reduce the computational costs of this interface-finding procedure, the coarse-grained
    density function is calculated using only a subset of the water molecules ('slice') within a
    a cutoff distance of the search ray; this cutoff distance is set as `slicing_cutoff` *
    `coarse_grain_length`. Setting the `slicing_cutoff` parameter smaller thus speeds up the
    calculation but may introduce inaccuracies. In typical samples of water, the default value of
    3.5 allows the interface to be calculated to an accuracy of within 0.01 A, and a value of 4 is
    sufficient to achieve an accuracy of within 0.001 A.
    """

    # Process water shape and unroll (collate) into a single array
    if len(waters.shape) == 2 and waters.shape[-1] == 3:
        N_frames = 1
        waters_unrolled = waters
    elif len(waters.shape) == 3 and waters.shape[-1] == 3:
        N_frames = waters.shape[0]
        waters_unrolled = waters.reshape(-1, 3)
    else:
        raise RuntimeError(f'Unrecognized input shape: waters {waters.shape}')
    
    # Establish default parameters
    slice_width = (np.inf if slicing_cutoff is None else (slicing_cutoff * coarse_grain_length))
    search_start = np.asarray(((0, 0, 0) if search_start is None else search_start), dtype=float)
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
        warnings.warn('No waters encountered along search axis', RuntimeWarning)
        return ((search_start, axis) if calc_normal else search_start)

    # Maximum z-coordinate to search
    if max_dist is None:
        z_ceil = np.max(parallel_components) + (-slice_width if reverse_search else slice_width)
    else:
        z_ceil = max_dist

    # Perform binary search to find the distance of the interface from the origin
    if reverse_search:
        lower = 0.0
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
        lower = 0.0
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


