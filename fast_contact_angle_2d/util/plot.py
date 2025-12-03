import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from ..liquid.coarse_grain import (find_interface, coarse_grained_density, COARSE_GRAIN_LENGTH,
                                   CUTOFF_DENSITY, BULK_DENSITY, SLICING_CUTOFF)
from ..solid.grid import cast_to_gridsize
from ..interpolate import PeriodicGridInterpolator

#==================================================================================================

def plot_density_xz_slice(
        liq: np.ndarray,
        sol_heightmap: PeriodicGridInterpolator,
        axis: Axes,
        azi: float = 0.0,
        n_plot_bins: int = 80,
        show_interface: bool = False, *,
        coarse_grain_length: float = COARSE_GRAIN_LENGTH,
        cutoff_density: float = CUTOFF_DENSITY,
        bulk_density: float = BULK_DENSITY,
        slicing_cutoff: float = SLICING_CUTOFF,
        color_sol: tuple[float, ...] | str = (0.6, 0.6, 0.6),
        color_inter: tuple[float, ...] | str = (0.9, 0.45, 0.0)
        ) -> None:
    """Plots the coarse-grained density distribution, taken along the xz plane (assuming the
    droplet has been centered), or some other plane parallel to the z-axis.

    Parameters
    ----------
    liq : ndarray
        The Cartesian coordinates of the liquid particles, either with shape (N_liq, 3) for a
        single instantaneous frame, or with shape (N_frames, N_liq, 3) for a collection of frames,
        representing the liquid droplet in centred coordinates.
    sol_heightmap : PeriodicGridInterpolator
        The time-averaged heightmap of the solid surface, which should be supplied as a
        PeriodicGridInterpolator.
    axis : Axes
        The MatPlotLib Axes object to plot onto.
    azi : float, between 0 to 360, optional
        The azimuthal angle, in degrees, to slice along; setting azi to 0 is equivalent to plotting
        the density along the xz plane, while setting azi to 90 is equivalent to plotting along the
        yz plane, for example. Defaults to 0.
    n_plot_bins : int, optional
        The number of cells to divide the image into for calculating the density distribution; a
        larger number yields higher resolution. Defaults to 80.
    show_interface : bool, optional
        Whether to display the liquid interface. Defaults to False.

    Other Parameters
    ----------------
    coarse_grain_length : float, optional
        This parameter is passed to `coarse_grained_density`.
    cutoff_density : float, optional
        This parameter is passed to `coarse_grained_density`.
    bulk_density: float, optional
        The expected value of the coarse-grained density distribution for bulk liquid. This
        determines the color scale of the plot; specifically, the color map used maps densities
        below `bulk_density` to a blue color #0000FF with alpha (opacity) equal to the relative
        ratio, and densities between one to two times of `bulk_density` to a linear interpolation
        between red #FF0000 and blue #0000FF with alpha 1.
    slicing_cutoff : float, optional
        This parameter is passed on `find_interface`. Furthermore, only water molecules and carbon
        atoms within `slicing_cutoff` * `coarse_grain_length` of the y = 0 plane are considered to
        be part of the xz 'slice', and contribute to the plot.
    color_sol : color, optional
        The color to plot the solid surface with.
    color_inter : color, optional
        The color to plot the liquid interface with, if `show_interface` is turned on.
    """

    # Check input shapes
    if (len(liq.shape) == 2 and liq.shape[-1] == 3):
        pass
    elif (len(liq.shape) == 3 and liq.shape[-1] == 3):
        N_frames = liq.shape[0]
        return plot_density_xz_slice(liq.reshape(-1, 3), sol_heightmap, axis=axis, azi=azi,
                                     n_plot_bins=n_plot_bins, show_interface=show_interface,
                                     coarse_grain_length=coarse_grain_length,
                                     cutoff_density=(N_frames * cutoff_density),
                                     bulk_density=(N_frames * bulk_density),
                                     slicing_cutoff=slicing_cutoff,
                                     color_sol=color_sol, color_inter=color_inter)
    else:
        raise RuntimeError(f'Unregonized input: liq {liq.shape} shaped wrongly!')

    # Pre-slice sampling space
    slice_width = (np.inf if slicing_cutoff is None else (slicing_cutoff * coarse_grain_length))
    rot_matrix = np.array(((np.cos(azi * np.pi / 180.0),  np.sin(azi * np.pi / 180.0), 0.0),
                           (-np.sin(azi * np.pi / 180.0), np.cos(azi * np.pi / 180.0), 0.0),
                           (0.0,                          0.0,                         1.0)))
    rotated = np.einsum('jk,ik->ij', rot_matrix, liq)
    sliced = rotated[rotated[:,1] < slice_width]
    sliced = sliced[sliced[:,1] > -slice_width]

    # Find droplet height and CoM
    droplet_CoM = np.mean(sliced, axis=0)
    droplet_roof = find_interface(sliced, droplet_CoM, (0, 0, 1),
                                  coarse_grain_length=coarse_grain_length,
                                  cutoff_density=cutoff_density, slicing_cutoff=slicing_cutoff)[2]
    droplet_floor = find_interface(sliced, droplet_CoM, (0, 0, -1),
                                   coarse_grain_length=coarse_grain_length,
                                   cutoff_density=cutoff_density, slicing_cutoff=slicing_cutoff)[2]
    droplet_height = droplet_roof - droplet_floor

    # Find interface points
    inter_pts = list()
    search_angles = np.linspace(0, 2 * np.pi, (360 if show_interface else 2), endpoint=show_interface)
    for angle in search_angles:
        inter_pts.append(find_interface(sliced, droplet_CoM, (np.cos(angle), 0, np.sin(angle)),
                                        coarse_grain_length=coarse_grain_length,
                                        cutoff_density=cutoff_density,
                                        slicing_cutoff=slicing_cutoff))
    inter_pts = np.array(inter_pts)

    # Calculate plot bounds
    x_min = 1.5 * np.min(inter_pts[:,0])
    x_max = 1.5 * np.max(inter_pts[:,0])
    z_min = droplet_floor - (0.5 * droplet_height)
    z_max = droplet_roof + (0.5 * droplet_height)

    # Plot density function
    x_space = np.linspace(x_min, x_max, n_plot_bins)
    x_pad = (x_space[1] - x_space[0]) / 2.0
    z_space = np.linspace(z_min, z_max, n_plot_bins)
    z_pad = (z_space[1] - z_space[0]) / 2.0
    xx, zz = np.meshgrid(x_space, z_space)
    testpoints = np.column_stack((xx.ravel(), np.zeros(n_plot_bins**2), zz.ravel()))
    densities = coarse_grained_density(testpoints, sliced, coarse_grain_length=coarse_grain_length)
    densities = np.reshape(densities, (n_plot_bins, n_plot_bins))
    colors = np.zeros((n_plot_bins, n_plot_bins, 4), dtype=float)
    colors[:,:,0] = np.clip((densities / bulk_density) - 1.0, a_min=0.0, a_max=1.0)
    colors[:,:,2] = np.clip(2.0 - (densities / bulk_density), a_min=0.0, a_max=1.0)
    colors[:,:,3] = np.clip((densities / bulk_density), a_min=0.0, a_max=1.0)

    extent = (x_min - x_pad, x_max + x_pad, z_min - z_pad, z_max + z_pad)
    axis.imshow(colors, origin='lower', extent=extent)

    # Plot solid surface
    N_samples = max(100, n_plot_bins)
    x_samples = np.linspace(x_min, x_max, N_samples)
    testpoints = np.c_[x_samples * np.cos(azi * np.pi / 180.0), x_samples * np.sin(azi * np.pi / 180.0)]
    axis.plot(x_samples, sol_heightmap(testpoints), '-', color=color_sol)
    
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(z_min, z_max)
    axis.set_aspect('equal')

    # Plot interface
    if show_interface:
        axis.plot(inter_pts[:,0], inter_pts[:,2], '-', color=color_inter)
    return None

#==================================================================================================

def plot_density_radially_symmetric(
        liq: np.ndarray,
        sol_heightmap: PeriodicGridInterpolator,
        axis: Axes,
        n_azi: int = 150,
        n_plot_bins: int = 80, *,
        coarse_grain_length: float = COARSE_GRAIN_LENGTH,
        cutoff_density: float = CUTOFF_DENSITY,
        bulk_density: float = BULK_DENSITY,
        slicing_cutoff: float = SLICING_CUTOFF,
        color_sol: tuple[float, ...] | str = (0.6, 0.6, 0.6)
        ) -> None:
    """Plots the coarse-grained density distribution, averaged azimuthally.

    Parameters
    ----------
    liq : ndarray
        The Cartesian coordinates of the liquid particles, either with shape (N_liq, 3) for a
        single instantaneous frame, or with shape (N_frames, N_liq, 3) for a collection of frames,
        representing the liquid droplet in centred coordinates.
    sol_heightmap : PeriodicGridInterpolator
        The time-averaged heightmap of the solid surface, which should be supplied as a
        PeriodicGridInterpolator.
    axis : Axes
        The MatPlotLib Axes object to plot onto.
    n_azi : int, optional
        The number of azimuthal directions to average over; defaults to 150.
    n_plot_bins : int, optional
        The number of cells to divide the image into for calculating the density distribution; a
        larger number yields higher resolution. Defaults to 80.

    Other Parameters
    ----------------
    coarse_grain_length : float, optional
        This parameter is passed to `coarse_grained_density`.
    cutoff_density : float, optional
        This parameter is passed to `coarse_grained_density`.
    bulk_density: float, optional
        The expected value of the coarse-grained density distribution for bulk liquid. This
        determines the color scale of the plot; specifically, the color map used maps densities
        below `bulk_density` to a blue color #0000FF with alpha (opacity) equal to the relative
        ratio, and densities between one to two times of `bulk_density` to a linear interpolation
        between red #FF0000 and blue #0000FF with alpha 1.
    slicing_cutoff : float, optional
        This parameter is passed on `find_interface`. Furthermore, only water molecules and carbon
        atoms within `slicing_cutoff` * `coarse_grain_length` of the y = 0 plane are considered to
        be part of the xz 'slice', and contribute to the plot.
    color_sol : color, optional
        The color to plot the solid surface with.
    """

    # Check input shapes
    if (len(liq.shape) == 2 and liq.shape[-1] == 3):
        pass
    elif (len(liq.shape) == 3 and liq.shape[-1] == 3):
        N_frames = liq.shape[0]
        return plot_density_radially_symmetric(liq.reshape(-1, 3), sol_heightmap, axis=axis,
                                               n_azi=n_azi, n_plot_bins=n_plot_bins,
                                               coarse_grain_length=coarse_grain_length,
                                               cutoff_density=(N_frames * cutoff_density),
                                               bulk_density=(N_frames * bulk_density),
                                               slicing_cutoff=slicing_cutoff,
                                               color_sol=color_sol)
    else:
        raise RuntimeError(f'Unregonized input: liq {liq.shape} shaped wrongly!')

    # Find droplet height and CoM
    CoM = np.mean(liq, axis=0)
    droplet_h = find_interface(liq, CoM, (0, 0, 1), coarse_grain_length=coarse_grain_length,
                               cutoff_density=cutoff_density, slicing_cutoff=slicing_cutoff)[2]
    floor = sol_heightmap(CoM[0:2])[0]
    slice_width = (np.inf if slicing_cutoff is None else (slicing_cutoff * coarse_grain_length))

    # Calculate plot bounds
    r_max = 2.0 * find_interface(liq, CoM, (1, 0, 0), coarse_grain_length=coarse_grain_length,
                                 cutoff_density=cutoff_density, slicing_cutoff=slicing_cutoff)[0]
    z_min = ((2.0 * floor) if floor < 0 else 0.0)
    z_max = 1.5 * droplet_h

    # Plot density function
    r_space = np.linspace(0.0, r_max, n_plot_bins)
    r_pad = (r_space[1] - r_space[0]) / 2.0
    z_space = np.linspace(z_min, z_max, n_plot_bins)
    z_pad = (z_space[1] - z_space[0]) / 2.0
    phi = np.linspace(0.0, 2 * np.pi, n_azi, endpoint=False)
    azi = np.c_[np.cos(phi), np.sin(phi)]
    densities = np.empty((n_plot_bins, n_plot_bins))
    for j, z in enumerate(z_space):
        z_slice = liq[liq[:,2] < (z + slice_width)]
        z_slice = z_slice[z_slice[:,2] >= (z - slice_width)]
        radii = np.sqrt(np.sum(z_slice[:,0:2]**2, axis=-1))
        for i, r in enumerate(r_space):
            testpoints = np.c_[r * azi[:,0], r * azi[:,1], np.full(n_azi, z)]
            sliced = z_slice[np.abs(radii - r) < slice_width]
            densities[j,i] = np.mean(coarse_grained_density(testpoints, sliced,
                                                            coarse_grain_length=coarse_grain_length))
    colors = np.zeros((n_plot_bins, n_plot_bins, 4), dtype=float)
    colors[:,:,0] = np.clip((densities / bulk_density) - 1.0, a_min=0.0, a_max=1.0)
    colors[:,:,2] = np.clip(2.0 - (densities / bulk_density), a_min=0.0, a_max=1.0)
    colors[:,:,3] = np.clip((densities / bulk_density), a_min=0.0, a_max=1.0)

    extent = (-r_pad, r_max + r_pad, z_min - z_pad, z_max + z_pad)
    axis.imshow(colors, origin='lower', extent=extent)

    # Plot solid surface
    z_mean = np.empty((n_plot_bins,), dtype=float)
    for i, r in enumerate(r_space):
        z_mean[i] = np.mean(sol_heightmap(r * azi))
    axis.plot(r_space, z_mean, '-', color=color_sol)
    
    axis.set_xlim(0.0, r_max)
    axis.set_ylim(z_min, z_max)
    axis.set_aspect('equal')
    return None

#==================================================================================================

def plot_2d_function(
        f: PeriodicGridInterpolator,
        axis: Axes,
        n_plot_pts: int | tuple[int, int] = 80, *,
        x_range: tuple[float, float] = None,
        y_range: tuple[float, float] = None,
        **kwargs
        ) -> AxesImage:
    """Plots a 2-dimensional function f(x, y), provided as a PeriodicGridInterpolator, to a
    MatPlotLib axis.

    Parameters
    ----------
    f : PeriodicGridInterpolator
        The function f(x, y) to be plotted. This should be supplied as a PeriodicGridInterpolator,
        which may have unit cell parameters [cell_x, cell_y].
    axis : Axes
        The MatPlotLib Axes object to plot onto.
    n_plot_pts : int or tuple[int, int], optional
        The number of grid points, either specified as a tuple of integers (N_x, N_y), or given as
        a single integer N_x = N_y, to plot the function f(x, y) at.
    x_range : tuple[float, float], optional
        The range of x-values to plot over; if unspecified, this defaults to (-cell_x/2, cell_x/2).
    y_range : tuple[float, float], optional
        The range of y-values to plot over; if unspecified, this defaults to (-cell_y/2, cell_y/2).
    **kwargs
        Extra arguments are passed directly to `matplotlib.pyplot.imshow`.
    """

    cell_xy = f.cell_params
    if cell_xy.shape[0] != 2:
        raise RuntimeError(f'Wrong shape {f.cell_params.shape} for "f" unit cell.')

    x_min = (-cell_xy[0] / 2.0 if x_range is None else x_range[0])
    x_max = (cell_xy[0] / 2.0 if x_range is None else x_range[1])
    y_min = (-cell_xy[1] / 2.0 if y_range is None else y_range[0])
    y_max = (cell_xy[1] / 2.0 if y_range is None else y_range[1])
    if x_max < x_min:
        raise RuntimeError(f'x plot range ({x_min, x_max}) badly defined!')
    if y_max < y_min:
        raise RuntimeError(f'y plot range ({y_min, y_max}) badly defined!')

    N_x, N_y = cast_to_gridsize(n_plot_pts)
    d_x = (x_max - x_min) / N_x
    d_y = (y_max - y_min) / N_y
    x_coords = np.linspace(x_min + (d_x / 2.0), x_max - (d_x / 2.0), N_x)
    y_coords = np.linspace(y_min + (d_y / 2.0), y_max - (d_y / 2.0), N_y)
    testpoints = np.stack(np.meshgrid(x_coords, y_coords, indexing='ij'), axis=-1)
    z_values = f(testpoints)

    return axis.imshow(z_values, extent=(x_min, x_max, y_min, y_max), **kwargs)

