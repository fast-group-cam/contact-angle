import numpy as np
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

