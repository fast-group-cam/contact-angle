import numpy as np
from typing import Self
from pathlib import PurePath
from typing import IO
from scipy.interpolate import RegularGridInterpolator

#==================================================================================================

class PeriodicGridInterpolator:
    """A wrapper of scipy's RegularGridInterpolator, with periodic boundary conditions enforced
    inherently. The domain is assumed to be a N-dimensional orthorhombic unit cell with cell
    lengths [a1, a2, ..., aN], with coordinates ranging from -ai/2 to +ai/2 for the ith axis.
    
    Parameters
    ----------
    cell_params : array_like, shape (N,)
        The cell axis parameters [a1, a2, ..., aN] which define the domain.
    values : array_like, shape (m1, m2, ..., mN, p1, ...)
        The data on the regular grid in N dimensions. The points defining the regular grid are
        constructed based on the shape (m1, ..., mN) and the corresponding cell_params, e.g. the
        ith coordinates of the grid is given by np.linspace(-(ai - di)/2, (ai - di)/2, mi) where
        di = ai/mi is the uniform grid spacing.
    **kwargs
        Extra arguments are passed directly to `RegularGridInterpolator`.
        
    Methods
    -------
    __call__
    min
    max
    derivative
    nabla
    read
    write
    
    Attributes
    ----------
    interp : RegularGridInterpolator
        The underlying RegularGridInterpolator, whose attributes are accessible as usual.
    cell_params : ndarray
        The cell axis parameters [a1, a2, ..., aN] which define the domain.
    res : tuple of ints
        The resolution of the underlying grid (m1, m2, ..., mN).
    shape : tuple of ints
        The shape of the function output (p1, ...).
    """

    def __init__(self, cell_params, values, **kwargs):

        self.cell_params = np.array(cell_params, dtype=float, copy=True)
        if len(self.cell_params.shape) > 1:
            raise ValueError(f'Bad cell_params of shape {self.cell_params.shape}, should only be'
                             ' of the form (N,)!')
        N_dims = self.cell_params.shape[0]

        cast_values = np.asarray(values)
        if len(cast_values.shape) < N_dims:
            raise ValueError(f'Values provided have shape {cast_values.shape}, which does not'
                             f' match number of dimensions {N_dims}!')
        self.res = cast_values.shape[0:N_dims]
        self.shape = (1,) if len(cast_values.shape) == N_dims else cast_values.shape[N_dims:]

        delta = tuple(self.cell_params[i] / self.res[i] for i in range(N_dims))
        points = tuple(np.linspace(-(self.cell_params[i] + delta[i]) / 2.0,
                                   (self.cell_params[i] + delta[i]) / 2.0,
                                   self.res[i] + 2) for i in range(N_dims))
        pad_width = 1 if len(cast_values.shape) == N_dims else ((((1,1),) * N_dims) + (((0,0),) * len(self.shape)))
        self.interp = RegularGridInterpolator(points, np.pad(cast_values, pad_width, 'wrap'), **kwargs)

    #----------------------------------------------------------------------------------------------

    def __call__(self, xi, **kwargs):
        """Interpolation at coordinates.
        
        Parameters
        ----------
        xi : ndarray of shape (..., N)
            The coordinates to evaluate the interpolator at; if out of bounds, will be wrapped
            into the first unit cell.
        **kwargs
            Extra arguments are passed directly to `RegularGridInterpolator.__call__()`.
            
        Returns
        -------
        values_x : ndarray, shape xi.shape[:-1] + (p1, ...)
            Interpolated values at `xi`, with wrapping for periodic boundary conditions. Behaviour
            is otherwise identical to `RegularGridInterpolator.__call__()`.
        """

        centred = xi - (self.cell_params * np.round(xi / self.cell_params))
        return self.interp(centred, **kwargs)
    
    #----------------------------------------------------------------------------------------------

    def min(self) -> float:
        """Returns the global minimum of this function."""
        return np.min(self.interp.values)
    
    #----------------------------------------------------------------------------------------------

    def max(self) -> float:
        """Returns the global maximum of this function."""
        return np.max(self.interp.values)

    #----------------------------------------------------------------------------------------------

    def derivative(self, axis: int) -> Self:
        """Generates a new PeriodicGridInterpolator, representing the derivative of this function
        f(x) with respect to the ith coordinate df(x)/dxi.
        
        Parameters
        ----------
        axis : int, between 0 to N-1
            The axis to differentiate this function along.
            
        Returns
        -------
        derivative : PeriodicGridInterpolator
            The derivative of this function f(x) with respect to the ith coordinate df(x)/dxi,
            expressed as a PeriodicGridInterpolator of the same cell parameters, resolution, and
            shape as the original interpolator.
        """

        N_dims = self.cell_params.shape[0]
        N_output_dims = len(self.interp.values.shape) - N_dims
        if axis >= N_dims:
            raise ValueError(f'Cannot differentiate axis {axis}, only {N_dims} dimensions!')
        slices = ((slice(1, -1),) * N_dims) + ((slice(None),) * N_output_dims)
        f = self.interp.values[slices]
        dx = self.cell_params[axis] / self.res[axis]
        df = (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * dx)
        return PeriodicGridInterpolator(self.cell_params, df)

    #----------------------------------------------------------------------------------------------

    def nabla(self) -> Self:
        """Generates a new PeriodicGridInterpolator, representing the gradient of this function
        f(x) with respect to all coordinates ∇f(x).
            
        Returns
        -------
        gradient : PeriodicGridInterpolator
            The gradient ∇f(x) of this function, expressed as a PeriodicGridInterpolator of the
            same cell parameters and resolution as the original interpolator, and shape
            (p1, ..., N).
        """

        N_dims = self.cell_params.shape[0]
        N_output_dims = len(self.interp.values.shape) - N_dims
        slices = ((slice(1, -1),) * N_dims) + ((slice(None),) * N_output_dims)
        f = self.interp.values[slices]
        df = np.empty(f.shape + (N_dims,), dtype=f.dtype)
        for i in range(N_dims):
            dx = self.cell_params[i] / self.res[i]
            df[...,i] = (np.roll(f, -1, axis=i) - np.roll(f, 1, axis=i)) / (2 * dx)
        return PeriodicGridInterpolator(self.cell_params, df)
    
    #----------------------------------------------------------------------------------------------

    @staticmethod
    def read(file: str | PurePath | IO) -> Self:
        """Generates a PeriodicGridInterpolator from a .npz file."""
        loaded = np.load(file)
        return PeriodicGridInterpolator(loaded['cell_params'], loaded['values'])
    
    #----------------------------------------------------------------------------------------------
    
    def write(self, file: str | PurePath | IO) -> None:
        """Writes a PeriodicGridInterpolator to a .npz file."""
        N_dims = self.cell_params.shape[0]
        N_output_dims = len(self.interp.values.shape) - N_dims
        slices = ((slice(1, -1),) * N_dims) + ((slice(None),) * N_output_dims)
        f = self.interp.values[slices]
        np.savez_compressed(file, cell_params=self.cell_params, values=f)
        return None

