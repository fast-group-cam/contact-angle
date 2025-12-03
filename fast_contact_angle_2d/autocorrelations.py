import warnings
import numpy as np
from typing import Iterable
from scipy.optimize import curve_fit
from .interpolate import PeriodicGridInterpolator

#==================================================================================================

def autocorrelation(
        functions: Iterable[PeriodicGridInterpolator],
        tau: int = 0
        ) -> np.ndarray:
    """Calculates the temporal autocorrelation function of a list of PeriodicGridInterpolators,
    representing the time evolution f(x, y, ..., t) of an N-dimensional function, at a time
    interval τ.

    Parameters
    ----------
    functions : list or iterable of PeriodicGridInterpolators
        A list of PeriodicGridInterpolators [f1(x, y, ...), f2(x, y, ...), ...] representing the
        time evolution of an N-dimensional function at timesteps t = 1, 2, ...; it is assumed that
        these timesteps are evenly distributed. It is important that the PeriodicGridInterpolators
        all have the same resolution and periodic unit cell.
    tau : int
        The time interval to evaluate the autocorrelation function at, in number of timesteps.

    Returns
    -------
    autocorrelation : PeriodicGridInterpolator
        The autocorrelation function C(x, y, ...) = <f(x, y, ..., t) f(x, y, ..., t + τ)> averaged
        over t, in the form of a PeriodicGridInterpolator with the same resolution and periodic
        unit cell as the supplied PeriodicGridInterpolators.

    Raises
    ------
    RuntimeError
        If the supplied PeriodicGridInterpolators have inconsistent resolution or unit cell, or if
        the specified τ is larger than the length of the supplied PeriodicGridInterpolators.
    """

    N_frames = len(functions)
    cell_params = functions[0].cell_params
    res = functions[0].res
    N_dims = cell_params.shape[0]
    for f in functions:
        if not np.array_equal(f.cell_params, cell_params):
            raise RuntimeError(f'Inconsistent cell parameters ({f.cell_params} vs {cell_params}) detected!')
        if f.res != res:
            raise RuntimeError(f'Inconsistent resolution ({f.res} vs {res}) detected!')
    if tau >= N_frames:
        raise RuntimeError(f'The specified tau ({tau}) is larger than the length of the supplied input ({N_frames})!')
    
    autocorr = np.zeros_like(functions[0].interp.values)
    if tau == 0:
        for f in functions:
            autocorr += np.square(f.interp.values)
        autocorr /= N_frames
    else:
        N_parts = N_frames - tau
        for i in range(N_parts):
            autocorr += functions[i].interp.values * functions[i + tau].interp.values
        autocorr /= N_parts
    
    slices = ((slice(1, -1),) * N_dims) + ((slice(None),) * (len(autocorr.shape) - N_dims))
    autocorr = autocorr[slices]
    return PeriodicGridInterpolator(cell_params, autocorr)
    
#==================================================================================================

def norm_inf_autocorrelation(
        functions: Iterable[PeriodicGridInterpolator],
        max_tau: int = None
        ) -> np.ndarray:
    """Calculates the normalized infinite-time temporal autocorrelation function of a list of
    PeriodicGridInterpolators, representing the time evolution f(x, y, ..., t) of an N-dimensional
    function.

    Parameters
    ----------
    functions : list or iterable of PeriodicGridInterpolators
        A list of PeriodicGridInterpolators [f1(x, y, ...), f2(x, y, ...), ...] representing the
        time evolution of an N-dimensional function at timesteps t = 1, 2, ...; it is assumed that
        these timesteps are evenly distributed. It is important that the PeriodicGridInterpolators
        all have the same resolution and periodic unit cell.
    max_tau : int
        The maximum time interval to evaluate autocorrelation functions at, in number of timesteps.
        Leaving this unspecified means that the longest possible autocorrelation function is
        computed, which may be inefficient as the normalized infinite-time autocorrelation is
        typically dominated by the decay of the autocorrelation function at short times.

    Returns
    -------
    norm_inf_autocorrelation : PeriodicGridInterpolator
        The normalized infinite-time autocorrelation function C = lim(τ→∞) <f(t)f(t + τ)>/<f²>, in
        the form of a PeriodicGridInterpolator with the same resolution and periodic unit cell as
        the supplied PeriodicGridInterpolators.

    Raises
    ------
    RuntimeError
        If the supplied PeriodicGridInterpolators have inconsistent resolution or unit cell.
    """

    N_frames = len(functions)
    cell_params = functions[0].cell_params
    res = functions[0].res
    N_dims = cell_params.shape[0]
    for f in functions:
        if not np.array_equal(f.cell_params, cell_params):
            raise RuntimeError(f'Inconsistent cell parameters ({f.cell_params} vs {cell_params}) detected!')
        if f.res != res:
            raise RuntimeError(f'Inconsistent resolution ({f.res} vs {res}) detected!')
    
    largest_tau = N_frames
    if max_tau is not None:
        largest_tau = min(largest_tau, max_tau)
    if largest_tau < 3:
        raise RuntimeError(f'Not enough frames ({largest_tau}) to extrapolate infinite-time autocorrelation!')

    autocorrs = np.zeros((largest_tau,) + functions[0].interp.values.shape)
    sigma = np.zeros((largest_tau,), dtype=float)
    for f in functions:
        autocorrs[0] += np.square(f.interp.values)
    autocorrs[0] /= N_frames
    sigma[0] = 1 / N_frames
    for tau in range(1, largest_tau):
        N_parts = N_frames - tau
        for i in range(N_parts):
            autocorrs[tau] += functions[i].interp.values * functions[i + tau].interp.values
        autocorrs[tau] /= N_parts * autocorrs[0]
        sigma[tau] = 1 / N_parts
    autocorrs[0] = np.ones_like(autocorrs[0])
    
    def exp_curve(x, c, k):
        return (1.0 - c) * np.exp(-k * x) + c
    
    tau = np.array(range(largest_tau))
    norm_inf_autocorr = np.zeros_like(functions[0].interp.values)
    autocorrs = autocorrs.reshape((largest_tau, -1))
    norm_inf_autocorr = norm_inf_autocorr.reshape(-1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        for i in range(norm_inf_autocorr.shape[0]):
            c_guess = min(max(autocorrs[-1, i], 0.0), 1.0)
            k_guess = min(max((1.0 - c_guess) / np.sum(autocorrs[:,i] - c_guess), 1.0 / largest_tau), 1.0)
            try:
                popt, _ = curve_fit(exp_curve, tau, autocorrs[:,i], p0=(c_guess, k_guess), sigma=sigma)
                norm_inf_autocorr[i] = min(max(popt[0], 0.0), 1.0)
            except RuntimeError:
                norm_inf_autocorr[i] = 0.0

    norm_inf_autocorr = norm_inf_autocorr.reshape(functions[0].interp.values.shape)
    slices = ((slice(1, -1),) * N_dims) + ((slice(None),) * (len(norm_inf_autocorr.shape) - N_dims))
    norm_inf_autocorr = norm_inf_autocorr[slices]
    return PeriodicGridInterpolator(cell_params, norm_inf_autocorr)

