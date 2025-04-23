import numpy as np
from .grid import cast_to_gridsize, smooth_sheet

#==================================================================================================

def calc_fourier_coefficients(
        carbons: np.ndarray,
        cell_xy: np.ndarray | tuple[float, float],
        N_points: tuple[int, int] | int = 80
        ) -> tuple[np.ndarray, np.ndarray]:
    """This function takes in the coordinates of the carbon atoms of a graphene sheet, and
calculates the Fourier coefficients. The inputs are:

    - `carbons`: The Cartesian coordinates of the carbon atoms at a single given instant, with
    shape (N_carbon, 3).
    - `cell_xy`: The cell parameters along the x- and y-axes, expressed as [cell_x, cell_y].
    - `N_points`: The number of coefficients to calculate in k-space, either specified as a tuple
    of integers (N_x, N_y), or given as a single integer N_x = N_y.

The output is a tuple of two np.NDArrays: the first of shape (N_x, N_y, 2), representing the two-
dimensional reciprocal space points which the coefficients were calculated at, and the second of
shape (N_x, N_y) containing the complex Fourier coefficients."""

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
    """This function takes in the coordinates of the carbon atoms of a graphene sheet, and
calculates the power spectrum P(k) in terms of the scalar magnitude of k. The inputs are:

    - `carbons`: The Cartesian coordinates of the carbon atoms at a single given instant, with
    shape (N_carbon, 3).
    - `cell_xy`: The cell parameters along the x- and y-axes, expressed as [cell_x, cell_y].
    - `N_points`: The number of coefficients to calculate in k-space, either specified as a tuple
    of integers (N_x, N_y), or given as a single integer N_x = N_y.

The output is a tuple of two np.NDArrays: the first of shape (N_x * N_y,), representing the scalar
magnitude k, and the second of shape (N_x * N_y,) containing the real power spectrum."""
    
    k_points, coeffs = calc_fourier_coefficients(carbons, cell_xy, N_points)
    power = (np.conj(coeffs) * coeffs).real.astype(float).flatten()
    k_points = np.sqrt(np.sum(np.square(k_points), axis=-1)).flatten()

    idxs = np.argsort(k_points)

    return (k_points[idxs], power[idxs])

