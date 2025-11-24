import numpy as np
import ase
from typing import Literal
from scipy.special import j1
from scipy.interpolate import CloughTocher2DInterpolator
from .grid import cast_to_gridsize, generate_grid

#==================================================================================================
# Default parameters

C_C_DISTANCE = 1.426    # Interatomic C-C distance in graphene (in angstroms)
CUTOFF_RADIUS = 4.5     # Cutoff radius for sheet smoothing (in angstroms)

#==================================================================================================

def raw_heightmap(
        carbons: np.ndarray,
        cell_xy: np.ndarray | tuple[float, float],
        N: tuple[int, int] | int = 80, *,
        margin: float = (3 * C_C_DISTANCE)
        ) -> np.ndarray:
    """Calculates the raw instantaneous heightmap of a graphene sheet (nominally aligned to the xy
    plane) from the coordinates of the carbon atoms, using the Clough-Tocher interpolator on the
    standard grid given by `generate_grid`.

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
        fully encapsulates all grid points; however increasing `margin` excessively leads to
        performance costs from the Clough-Tocher interpolator. Defaults to 4.278, which is thrice
        the C-C interatomic distance in graphene.

    Returns
    -------
    z : ndarray
        The raw heightmap z(x, y) evaluated at the grid points given by `generate_grid`, with
        shape (N_x, N_y).

    Notes
    -----
    The ordering of axes (N_x, N_y) contradicts the order of MatPlotLib's `imshow` axes, so
    `np.swapaxes(..., 0, 1)` should be used when generating plots etc..
    """
    
    cell_xy = np.asarray(cell_xy[0:2], dtype=float)
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

#==================================================================================================

def regularized_heightmap(
        carbons: np.ndarray,
        cell_xy: np.ndarray | tuple[float, float],
        N: tuple[int, int] | int = 80,
        cutoff_radius = CUTOFF_RADIUS, *,
        margin: float = (3 * C_C_DISTANCE)
        ) -> np.ndarray:
    """Calculates the regularized instantaneous heightmap of a graphene sheet (nominally aligned to
    the xy plane), which is simply a coarse-graining of the raw heightmap from `raw_heightmap`.

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
        performance costs from the Clough-Tocher interpolator. Defaults to 4.278, which is thrice
        the C-C interatomic distance in graphene.

    Returns
    -------
    z : ndarray
        The smooth regularized heightmap z(x, y) evaluated at the grid points given by
        `generate_grid`, with shape (N_x, N_y).

    Notes
    -----
    The ordering of axes (N_x, N_y) contradicts the order of MatPlotLib's `imshow` axes, so
    `np.swapaxes(..., 0, 1)` should be used when generating plots etc..
    """

    N_x, N_y = cast_to_gridsize(N)
    d_x = cell_xy[0] / N_x
    d_y = cell_xy[1] / N_y
    k_x = np.fft.fftfreq(N_x, d_x)
    k_y = np.fft.rfftfreq(N_y, d_y)
    k_x, k_y = np.meshgrid(k_x, k_y, indexing='ij', sparse=True)
    k_mag_scaled = np.sqrt((k_x**2) + (k_y**2)) * cutoff_radius

    z = raw_heightmap(carbons, cell_xy, N, margin=margin)
    fourier = np.fft.rfft2(z)
    filter = np.divide(2 * j1(k_mag_scaled), k_mag_scaled, out=np.ones_like(k_mag_scaled),
                       where=(k_mag_scaled > 0.0))
    return np.fft.irfft2(fourier * filter, z.shape)
    

#==================================================================================================

def generate_sheet(
        max_cell_x: float,
        max_cell_y: float = None,
        interatomic_dist: float = C_C_DISTANCE,
        atomic_symbol: str = 'C',
        origin: Literal['corner', 'center'] = 'center'
        ) -> ase.Atoms:
    """Generates a graphene sheet.

    Parameters
    ----------
    max_cell_x : float
        The maximum dimension of cell_x in angstroms.
    max_cell_y : float, optional
        The maximum dimension of cell_y in angstroms, if specified; otherwise taken to be equal to
        `max_cell_x`.
    interatomic_dist : float, optional
        The interatomic distance to generate the sheet with, in angstroms. Defaults to 1.426.
    atomic_symbol : str, optional
        The symbol of the atom used to generate the sheet. Defaults to 'C'.
    origin : {'corner', 'center'}
        If set to 'corner', the sheet will be generated within the x-coordinate range of 0 to
        cell_x, and y-coordinate range of 0 to cell_y; if set to 'center', the sheet will be
        generated within the x-coordinate range of -0.5 * cell_x to 0.5 * cell_x, and y-coordinate
        range of -0.5 * cell_y to 0.5 * cell_y. In either setting, the sheet will be aligned such
        that one atom is placed at (0, 0, 0). Defaults to 'center'.

    Returns
    -------
    atoms : ase.Atoms
        The graphene sheet, initialized with only information about chemical species and position.
        The cell parameters [cell_x, cell_y, cell_z] are set such that cell_x and cell_y are the
        largest possible integer multiples of the hexagonal grid (so that the sheet is correctly
        periodic) that are lesser than or equal to `max_cell_x` and `max_cell_y` respectively,
        while cell_z is set to the geometric mean of cell_x and cell_y.
    """
    
    max_cell_x = float(max_cell_x)
    if max_cell_y is None:
        max_cell_y = max_cell_x

    if origin.casefold() == 'center':
        sheet_centered = True
    elif origin.casefold() == 'corner':
        sheet_centered = False
    else:
        raise RuntimeError(f'Option undefined: {origin}')

    a_x = 1.5 * interatomic_dist
    a_y = interatomic_dist * np.sin(np.pi / 3)
    offset = np.array([interatomic_dist, 0, 0])
    #vec_a = np.array([a_x, a_y, 0])
    vec_b = np.array([a_x, -a_y, 0])

    tmp = max_cell_x / a_x
    units_x_min = (-int(np.floor(0.25 * tmp)) if sheet_centered else 0)
    units_x_max = (int(np.floor(0.25 * tmp)) if sheet_centered else int(np.floor(0.5 * tmp)))
    tmp = max_cell_y / a_y
    units_y_min = (-int(np.floor(0.25 * tmp)) if sheet_centered else 0)
    units_y_max = (int(np.floor(0.25 * tmp)) if sheet_centered else int(np.floor(0.5 * tmp)))

    positions = list()
    for i in range(units_x_min, units_x_max):
        for j in range(units_y_min, units_y_max):
            tmp = np.array([2 * i * a_x, 2 * j * a_y, 0])
            positions.append(tmp)
            positions.append(tmp + offset)
            positions.append(tmp + vec_b)
            positions.append(tmp + vec_b + offset)
    positions = np.array(positions)

    cell_x = (units_x_max - units_x_min) * 2 * a_x
    cell_y = (units_y_max - units_y_min) * 2 * a_y
    cell_z = np.sqrt(cell_x * cell_y)

    return ase.Atoms([atomic_symbol,] * positions.shape[0], positions=positions,
                     cell=[cell_x, cell_y, cell_z], pbc=True)
