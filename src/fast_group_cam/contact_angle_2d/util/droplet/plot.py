from .coarse_grain import *
import numpy as np
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from ..interpolate import PeriodicGridInterpolator

#==================================================================================================

def plot_density_xz_slice(
        waters: np.ndarray,
        carbons: np.ndarray,
        axis: Axes,
        n_plot_bins: int = 80,
        show_interface: bool = False, *,
        show_carbon_dist: bool = False,
        coarse_grain_length: float = COARSE_GRAIN_LENGTH,
        cutoff_density: float = CUTOFF_DENSITY,
        bulk_density: float = BULK_DENSITY,
        slicing_cutoff: float = SLICING_CUTOFF,
        color_carbon: tuple[float, ...] | str = (0.6, 0.6, 0.6),
        color_inter: tuple[float, ...] | str = (0.9, 0.45, 0.0)
        ) -> tuple[Artist, ...]:
    """Plots the coarse-grained density distribution, taken along the xz plane (assuming the
    droplet has been centered).

    Parameters
    ----------
    waters : ndarray
        The Cartesian coordinates of the water molecules, either with shape (N_water, 3) for a
        single instantaneous frame; or shape (N_frames, N_water, 3) for a collection of frames,
        in which case the density distribution is averaged over the frames.
    carbons : ndarray
        The Cartesian coordinates of the carbon atoms, either with shape (N_carbon, 3) for a
        single instantaneous frame; or shape (N_frames, N_carbon, 3) for a collection of frames.
        Note that the number of dimensions must match with `waters`.
    axis : Axes
        The MatPlotLib Axes object to plot onto.
    n_plot_bins : int, optional
        The number of cells to divide the image into for calculating the density distribution; a
        larger number yields higher resolution. Defaults to 80.
    show_interface : bool, optional
        Whether to display the Willard-Chandler interface. Defaults to False.
    
    Returns
    -------
    artists : tuple[Artist, ...]
        A tuple of Artists involved with the plot, which can be used by `update_density_xz_slice`
        to update the plot.

    Other Parameters
    ----------------
    show_carbon_dist : bool, optional
        If a single instantaneous frame is provided and `show_carbon_dist` is False, the carbon
        atoms in the slice (see `slicing_cutoff` below) will be individually rendered as points;
        otherwise, if `show_carbon_dist` is True, or a collection of frames is provided (regardless
        of whether `show_carbon_dist` is True or not), the carbon atoms in the slice will be
        collated and displayed as a distribution of z-coordinates along the x-axis, binned over
        `n_plot_bins` intervals.
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
    color_carbon : color, optional
        The color to plot the carbon atoms with.
    color_inter : color, optional
        The color to plot the Willard-Chandler interface with, if `show_interface` is turned on.
    """

    # Check shapes of waters and carbons
    if (len(waters.shape) == 2 and len(carbons.shape) == 2 and waters.shape[-1] == 3 and
        carbons.shape[-1] == 3):
        pass
    elif (len(waters.shape) == 3 and len(carbons.shape) == 3 and waters.shape[-1] == 3 and
        carbons.shape[-1] == 3):
        N_frames = waters.shape[0]
        return plot_density_xz_slice(waters.reshape(-1, 3), carbons.reshape(-1, 3), axis=axis,
                                     n_plot_bins=n_plot_bins, show_interface=show_interface,
                                     show_carbon_dist=True,
                                     coarse_grain_length=coarse_grain_length,
                                     cutoff_density=(N_frames * cutoff_density),
                                     bulk_density=(N_frames * bulk_density),
                                     slicing_cutoff=slicing_cutoff,
                                     color_carbon=color_carbon, color_inter=color_inter)
    else:
        raise RuntimeError(f'Unregonized input: either waters {waters.shape} or carbons ' +
                           f'{carbons.shape} shaped wrongly!')

    # Pre-slice sampling space
    slice_width = (np.inf if slicing_cutoff is None else (slicing_cutoff * coarse_grain_length))
    sliced = waters[waters[:,1] < slice_width]
    sliced = sliced[sliced[:,1] > -slice_width]
    carbon_slice = carbons[carbons[:,1] < slice_width]
    carbon_slice = carbon_slice[carbon_slice[:,1] > - slice_width]
    if show_carbon_dist:
        carbon_slice = carbon_slice[np.argsort(carbon_slice[:,0])]

    # Find droplet height and CoM
    droplet_CoM = np.mean(sliced, axis=0)
    droplet_h = find_interface(sliced, droplet_CoM, (0, 0, 1),
                               coarse_grain_length=coarse_grain_length,
                               cutoff_density=cutoff_density, slicing_cutoff=slicing_cutoff)[2]

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
    z_min = np.min(carbon_slice[:,2])
    z_max = 1.5 * droplet_h

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

    artists = list()
    artists.append(axis.imshow(colors, origin='lower', extent=(x_min - x_pad, x_max + x_pad,
                                                               z_min - z_pad, z_max + z_pad)))

    # Plot carbons
    if show_carbon_dist:
        N_samples = max(100, n_plot_bins)
        x_samples = np.linspace(x_min, x_max, N_samples)
        z_mean = np.empty((N_samples,), dtype=float)
        z_stdev = np.empty((N_samples,), dtype=float)
        z_bot = np.empty((N_samples,), dtype=float)
        z_top = np.empty((N_samples,), dtype=float)
        for i in range(N_samples):
            sample = carbon_slice[np.abs(carbon_slice[:,0] - x_samples[i]) < coarse_grain_length, 2]
            z_mean[i] = np.mean(sample)
            z_stdev[i] = np.std(sample)
            z_bot[i] = np.min(sample)
            z_top[i] = np.max(sample)
        artists.append(axis.fill_between(x_samples, z_bot, z_top, color=(0.6, 0.6, 0.6, 0.25)))
        artists.append(axis.fill_between(x_samples, z_mean - z_stdev, z_mean + z_stdev,
                                         color=(0.6, 0.6, 0.6, 0.25)))
        artists.append(axis.plot(x_samples, z_mean, '-', color=color_carbon)[0])
    else:
        artists.append(axis.plot(carbon_slice[:,0], carbon_slice[:,2], '.', color=color_carbon)[0])
    
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(z_min, z_max)
    axis.set_aspect('equal')

    # Plot interface
    if show_interface:
        artists.append(axis.plot(inter_pts[:,0], inter_pts[:,2], '-', color=color_inter)[0])
    return tuple(artists)

#==================================================================================================

def update_density_xz_slice(
        waters: np.ndarray,
        carbons: np.ndarray,
        artists: tuple[Artist, ...],
        n_plot_bins: int = 80, *,
        coarse_grain_length: float = COARSE_GRAIN_LENGTH,
        cutoff_density: float = CUTOFF_DENSITY,
        bulk_density: float = BULK_DENSITY,
        slicing_cutoff: float = SLICING_CUTOFF,
        color_carbon: tuple[float, float, float] | str = (0.6, 0.6, 0.6),
        color_inter: tuple[float, float, float] | str = (0.9, 0.45, 0.0)
        ) -> None:
    """Updates the plot generated by `plot_density_xz_slice` using the new parameters passed in.
    Useful for animations etc.."""

    if len(artists) in (4, 5):
        raise NotImplementedError('WIP: update_density_xz_slice with show_carbon_dist not ' +
                                  'implemented yet!')
    if len(artists) not in (2, 3):
        raise RuntimeError(f'Wrong number of artists ({len(artists)}) supplied!')
    show_interface = (len(artists) == 3)

    # Pre-slice sampling space
    slice_width = (np.inf if slicing_cutoff is None else (slicing_cutoff * coarse_grain_length))
    sliced = waters[waters[:,1] < slice_width]
    sliced = sliced[sliced[:,1] > -slice_width]
    carbon_slice = carbons[carbons[:,1] < slice_width]
    carbon_slice = carbon_slice[carbon_slice[:,1] > - slice_width]

    # Find droplet height and CoM
    droplet_CoM = np.mean(sliced, axis=0)
    droplet_h = find_interface(sliced, droplet_CoM, (0, 0, 1),
                               coarse_grain_length=coarse_grain_length,
                               cutoff_density=cutoff_density, slicing_cutoff=slicing_cutoff)[2]

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
    z_min = np.min(carbon_slice[:,2])
    z_max = 1.5 * droplet_h

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
    artists[0].set(data=colors, extent=(x_min - x_pad, x_max + x_pad, z_min - z_pad, z_max + z_pad))

    # Plot carbons
    artists[1].set_xdata(carbon_slice[:,0])
    artists[1].set_ydata(carbon_slice[:,2])
    artists[1].set_color(color_carbon)

    # Plot interface
    if show_interface:
        artists[2].set_xdata(inter_pts[:,0])
        artists[2].set_ydata(inter_pts[:,2])
        artists[2].set_color(color_inter)

#==================================================================================================

def plot_density_radially_symmetric(
        waters: np.ndarray,
        mean_heightmap: PeriodicGridInterpolator,
        axis: Axes,
        n_plot_bins: int = 80,
        n_azi: int = 100, *,
        coarse_grain_length: float = COARSE_GRAIN_LENGTH,
        cutoff_density: float = CUTOFF_DENSITY,
        bulk_density: float = BULK_DENSITY,
        slicing_cutoff: float = SLICING_CUTOFF,
        color_carbon: tuple[float, ...] | str = (0.6, 0.6, 0.6)
        ) -> tuple[Artist, ...]:
    """Plots the coarse-grained density distribution, averaged azimuthally.

    Parameters
    ----------
    waters : ndarray
        The Cartesian coordinates of the water molecules, with shape (N_frames, N_water, 3) for a
        collection of frames; the density distribution is averaged over the frames.
    mean_heightmap : PeriodicGridInterpolator
        The time-averaged regularized heightmap of the graphene sheet <h(x, y)>_t, which should
        be supplied as a PeriodicGridInterpolator with domain spanning -cell_x/2 to cell_x/2 along
        the x-coordinate and -cell_y/2 to cell_y/2 along the y-coordinate.
    axis : Axes
        The MatPlotLib Axes object to plot onto.
    n_plot_bins : int, optional
        The number of cells to divide the image into for calculating the density distribution; a
        larger number yields higher resolution. Defaults to 80.
    n_azi : int, optional
        The number of azimuthal directions to average over; defaults to 100.
    
    Returns
    -------
    artists : tuple[Artist, ...]
        A tuple of Artists involved with the plot.

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
    color_carbon : color, optional
        The color to plot the mean heightmap with.
    """

    # Check shapes of waters and carbons
    if (len(waters.shape) == 2 and waters.shape[-1] == 3):
        pass
    elif (len(waters.shape) == 3 and waters.shape[-1] == 3):
        N_frames = waters.shape[0]
        return plot_density_radially_symmetric(waters.reshape(-1, 3), mean_heightmap, axis=axis,
                                               n_plot_bins=n_plot_bins, n_azi=n_azi,
                                               coarse_grain_length=coarse_grain_length,
                                               cutoff_density=(N_frames * cutoff_density),
                                               bulk_density=(N_frames * bulk_density),
                                               slicing_cutoff=slicing_cutoff,
                                               color_carbon=color_carbon)
    else:
        raise RuntimeError(f'Unregonized input: waters {waters.shape} shaped wrongly!')

    # Find droplet height and CoM
    CoM = np.mean(waters, axis=0)
    droplet_h = find_interface(waters, CoM, (0, 0, 1), coarse_grain_length=coarse_grain_length,
                               cutoff_density=cutoff_density, slicing_cutoff=slicing_cutoff)[2]
    floor = mean_heightmap(CoM[0:2])[0]
    slice_width = (np.inf if slicing_cutoff is None else (slicing_cutoff * coarse_grain_length))

    # Calculate plot bounds
    r_max = 2.0 * find_interface(waters, CoM, (1, 0, 0), coarse_grain_length=coarse_grain_length,
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
    for i, r in enumerate(r_space):
        for j, z in enumerate(z_space):
            testpoints = np.c_[r * azi[:,0], r * azi[:,1], np.full(n_azi, z)]
            sliced = waters[np.abs(waters[:,2] - z) < slice_width]
            radii = np.sqrt(np.sum(sliced[:,0:2]**2, axis=-1))
            sliced = sliced[np.abs(radii - r) < slice_width]
            densities[j,i] = np.mean(coarse_grained_density(testpoints, sliced,
                                                            coarse_grain_length=coarse_grain_length))
    colors = np.zeros((n_plot_bins, n_plot_bins, 4), dtype=float)
    colors[:,:,0] = np.clip((densities / bulk_density) - 1.0, a_min=0.0, a_max=1.0)
    colors[:,:,2] = np.clip(2.0 - (densities / bulk_density), a_min=0.0, a_max=1.0)
    colors[:,:,3] = np.clip((densities / bulk_density), a_min=0.0, a_max=1.0)

    artists = list()
    artists.append(axis.imshow(colors, origin='lower', extent=(-r_pad, r_max + r_pad,
                                                               z_min - z_pad, z_max + z_pad)))

    # Plot carbons
    z_mean = np.empty((n_plot_bins,), dtype=float)
    for i, r in enumerate(r_space):
        z_mean[i] = np.mean(mean_heightmap(r * azi))
    artists.append(axis.plot(r_space, z_mean, '-', color=color_carbon)[0])
    
    axis.set_xlim(0.0, r_max)
    axis.set_ylim(z_min, z_max)
    axis.set_aspect('equal')
    return tuple(artists)