import numpy as np
from .grid import cast_to_gridsize, smooth_sheet

#==================================================================================================

def calc_fourier_coefficients(
        carbons: np.ndarray,
        cell_xy: np.ndarray | tuple[float, float],
        N_points: tuple[int, int] | int = 80
        ) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the Fourier coefficients for a smoothened graphene sheet.

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
    
    Returns
    -------
    k_points : ndarray
        The two-dimensional reciprocal space points which the Fourier coefficients were calculated
        at, with shape (N_x, N_y, 2).
    coeffs : ndarray
        The complex Fourier coefficients, with shape (N_x, N_y).
    """

    N_x, N_y = cast_to_gridsize(N_points)
    sheet = smooth_sheet(carbons, cell_xy, N_points)

    coeffs = np.fft.fft2(sheet, s=(N_x, N_y))

    k_x = np.fft.fftfreq(N_x, cell_xy[0]/N_x)
    k_y = np.fft.fftfreq(N_y, cell_xy[1]/N_y)
    k_points = np.stack(np.meshgrid(k_x, k_y, indexing='ij'), axis=-1)

    return (k_points, coeffs)

#==================================================================================================

def calc_power_spectrum(
        carbons: np.ndarray,
        cell_xy: np.ndarray | tuple[float, float],
        N_points: tuple[int, int] | int = 80
        ) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the power spectrum P(k) in terms of the scalar magnitude of k for a smoothened
    graphene sheet.

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
    
    Returns
    -------
    k : ndarray
        The scalar magnitudes of the reciprocal space k-vectors which the spectrum was calculated
        at, with shape (N_x * N_y,).
    power : ndarray
        The real power spectrum, with shape (N_x * N_y,).
    """
    
    k_points, coeffs = calc_fourier_coefficients(carbons, cell_xy, N_points)
    power = (np.conj(coeffs) * coeffs).real.astype(float).flatten()
    k_points = np.sqrt(np.sum(np.square(k_points), axis=-1)).flatten()

    idxs = np.argsort(k_points)

    return (k_points[idxs], power[idxs])

