#! /usr/bin/env python

prog_desc_header = '''
===================================================================================================
 This program measures the contact angle of a water droplet on a graphene sheet from a simulated
 trajectory. The input is a file, which must be compatible with ASE's file i/o formats, describing
 either a time evolution or a single snapshot of a water droplet (rotationally symmetric about the
 z-axis) on a graphene sheet aligned aligned to the xy plane. The contact angle is calculated by
 finding the Willard-Chandler interface for a small number of testpoints at the droplet's foot,
 and calculating the direction of the plane.

 The program takes some number of frames `N_frames` from the input file (which is specified by the
 user). The action of the program depends on the user inputs:

   (1) If `N_frames` is 1 (default), the contact angle is measured as an instantaneous 'snapshot'
       from the last frame of the file. The reported value is averaged over some number of
       azimuthal slices (specified by `N_azimuths`), and the reported uncertainty is the standard
       error of the mean. The program also generates a plot of the fit for each azimuthal slice,
       over the water molecules' number density distribution.

   (2) If `N_frames` is greater than 1, and `block-average` is false (default), the contact angle
       is measured for each and every frame, and the reported value is averaged over both azimuthal
       slices and across frames (with reported uncertainty being the standard error of the mean).
       The program then generates a plot of the contact angles measured for each azimuthal slice
       for each frame, and also a plot of the azimuthal and time autocorrelation functions. Note
       that the frames are sliced from the start of the file, with slicing interval specified by
       `--interval`.

   (3) If `N_frames` is greater than 1, `block-average` is true, and `auto` is true or `b` is
       unspecified, the program will proceed as per mode #2, except that, instead of reporting the
       standard error of the mean, the program performs reverse cumulative averaging over a range
       of block sizes to identify a block size with statistical inefficiency within the upper
       quartile of statistical inefficiencies (refer to DOI:10.1063/1.1638996); the reported
       uncertainty is the standard error of the block means at this block size. The program also
       additionally generates a plot of the statistical inefficiency as a function of block size.

   (4) If `N_frames` is greater than 1, `block-average` is true, `auto` is false (default), and
       `b` is specified, the program will proceed as per mode #3 (performing reverse cumulative
       averaging over a range of blocksizes), but the reported uncertainty will be calculated using
       the user-specified blocksize.
===================================================================================================
'''

import sys
import os
import time
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import ase
import ase.io

from contact_angle.util import center_coordinates
from contact_angle.util.droplet import find_interface
from contact_angle.util.droplet.coarse_grain import COARSE_GRAIN_LENGTH, SLICING_CUTOFF, BULK_DENSITY
from contact_angle.util.droplet.plot import plot_density_xz_slice
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize


#==================================================================================================
# Function declaration: contact_angles

def contact_angles(waters: np.ndarray, carbons: np.ndarray, N_azimuths: int, rot_angle: float,
                   fig: Figure = None, axes: list[Axes] = None) -> tuple[np.ndarray, float]:
    """This high-level function processes the contact angles over a range of azimuthal directions,
given the pre-processed coordinates of the system, for a single frame. It also plots the density
distribution to a PyPlot Figure if provided (as needed for program mode (1)). The inputs are:

    waters:      The Cartesian coordinates of the water molecules.
    carbons:     The Cartesian coordinates of the carbon atoms.
    N_azimuths:  The number of azimuthal slices to scan.
    rot_angle:   The rotational angle to rotate between azimuthal slices, in degrees.
    fig:         If provided, the PyPlot figure to render to.
    axes:        If provided, the list of PyPlot axes to render to.

The output is a tuple, where the first item is a np.NDArray of shape (2 * N_azimuths,) representing
the instantaneous contact angles of this frame over the range of azimuthal directions, and the
second item is a float representing the highest value of the coarse-grained density distribution
encountered in this frame.
    """

    # Calculate droplet height
    droplet_h = find_interface(waters, (0, 0, 0), (0, 0, 1))[2]

    # Pre-calculate rotation matrix for iterating through azimuthal slices
    rot_matrix = np.array(((np.cos(rot_angle * np.pi/180), -np.sin(rot_angle * np.pi/180), 0.0),
                           (np.sin(rot_angle * np.pi/180), np.cos(rot_angle * np.pi/180),  0.0),
                           (0.0,                           0.0,                            1.0)))
    
    # Pre-calculate z-coordinate to search for each azimuthal slice
    SLICING_WIDTH = SLICING_CUTOFF * COARSE_GRAIN_LENGTH
    z_floor = np.min(waters[:,2])
    droplet_thickness = droplet_h - z_floor
    highest_scan_z = z_floor + (0.2 * droplet_thickness)
    cutoff_z = highest_scan_z + SLICING_WIDTH

    # Iterate through azimuthal slices
    angles = np.zeros((2 * N_azimuths,), dtype=float)
    highest_density = 0.0
    for azi in range(N_azimuths):

        # Cut out only most relevant waters
        sliced = waters[waters[:,1] < SLICING_WIDTH]
        sliced = sliced[sliced[:,1] > -SLICING_WIDTH]
        sliced = sliced[sliced[:,2] < cutoff_z]
        l_waters = sliced[sliced[:,0] < SLICING_WIDTH]
        r_waters = sliced[sliced[:,0] > -SLICING_WIDTH]
        carbon_slice = carbons[carbons[:,1] < SLICING_WIDTH]
        carbon_slice = carbon_slice[carbon_slice[:,1] > -SLICING_WIDTH]
        
        # Find left interface
        l_inter, l_inter_norm = find_interface(l_waters, (0, 0, highest_scan_z), (-1, 0, 0),
                                               calc_normal=True)

        # Find closest carbon atoms to the left interface, and calculate their normal vector
        l_inter_foot = np.zeros((3,), dtype=float)
        l_inter_foot[0] = l_inter[0] + (l_inter[2] * l_inter_norm[2] / l_inter_norm[0])
        carbon_dists_sq = np.sum(np.square(carbon_slice - l_inter_foot), axis=-1)
        closest_carbons = carbon_slice[np.argpartition(carbon_dists_sq, 30)[:30]]
        l_carbon_c = np.mean(closest_carbons, axis=0)
        l_carbon_l = l_carbon_c[0] - np.min(closest_carbons[:,0])
        l_carbon_r = np.max(closest_carbons[:,0]) - l_carbon_c[0]
        l_carbon_norm = np.linalg.svd((closest_carbons - l_carbon_c).T)[0][:,-1]
        if l_carbon_norm[2] < 0.0:
            l_carbon_norm = -l_carbon_norm

        # Calculate the left contact angle
        cosine = np.dot(l_inter_norm, l_carbon_norm)
        cosine /= np.linalg.norm(l_inter_norm)
        cosine /= np.linalg.norm(l_carbon_norm)
        angles[azi] = np.arccos(cosine) * 180.0 / np.pi
        
        # Find right interface
        r_inter, r_inter_norm = find_interface(r_waters, (0, 0, highest_scan_z), (1, 0, 0),
                                               calc_normal=True)

        # Find closest carbon atoms to the right interface, and calculate their normal vector
        r_inter_foot = np.zeros((3,), dtype=float)
        r_inter_foot[0] = r_inter[0] + (r_inter[2] * r_inter_norm[2] / r_inter_norm[0])
        carbon_dists_sq = np.sum(np.square(carbon_slice - r_inter_foot), axis=-1)
        closest_carbons = carbon_slice[np.argpartition(carbon_dists_sq, 30)[:30]]
        r_carbon_c = np.mean(closest_carbons, axis=0)
        r_carbon_l = r_carbon_c[0] - np.min(closest_carbons[:,0])
        r_carbon_r = np.max(closest_carbons[:,0]) - r_carbon_c[0]
        r_carbon_norm = np.linalg.svd((closest_carbons - r_carbon_c).T)[0][:,-1]
        if r_carbon_norm[2] < 0.0:
            r_carbon_norm = -r_carbon_norm

        # Calculate the right contact angle
        cosine = np.dot(r_inter_norm, r_carbon_norm)
        cosine /= np.linalg.norm(r_inter_norm)
        cosine /= np.linalg.norm(r_carbon_norm)
        angles[azi + N_azimuths] = np.arccos(cosine) * 180.0 / np.pi

        # If available, plot diagram
        if (fig is not None) and (azi < len(axes)):

            # Plot density function, carbons, and full instantaneous interface
            plot_density_xz_slice(waters, carbon_slice, axes[azi], show_interface=True,
                                  color_inter=(1, 0, 1, 0.5))

            # Plot carbon planes
            point_a_x = l_carbon_c[0] - l_carbon_l
            point_a_z = l_carbon_c[2] + (l_carbon_l * l_carbon_norm[0] / l_carbon_norm[2])
            point_b_x = l_carbon_c[0] + l_carbon_r
            point_b_z = l_carbon_c[2] - (l_carbon_r * l_carbon_norm[0] / l_carbon_norm[2])
            axes[azi].plot((point_a_x, point_b_x), (point_a_z, point_b_z), 'k-')
            point_a_x = r_carbon_c[0] - r_carbon_l
            point_a_z = r_carbon_c[2] + (r_carbon_l * r_carbon_norm[0] / r_carbon_norm[2])
            point_b_x = r_carbon_c[0] + r_carbon_r
            point_b_z = r_carbon_c[2] - (r_carbon_r * r_carbon_norm[0] / r_carbon_norm[2])
            axes[azi].plot((point_a_x, point_b_x), (point_a_z, point_b_z), 'k-')

            # Plot water interfaces
            point_b_z = 2.0 * l_inter[2]
            point_b_x = l_inter[0] - (l_inter[2] * l_inter_norm[2] / l_inter_norm[0])
            axes[azi].plot((l_inter_foot[0], point_b_x), (0.0, point_b_z), '-', color=(1.0, 0.0, 1.0))
            point_b_z = 2.0 * r_inter[2]
            point_b_x = r_inter[0] - (r_inter[2] * r_inter_norm[2] / r_inter_norm[0])
            axes[azi].plot((r_inter_foot[0], point_b_x), (0.0, point_b_z), '-', color=(1.0, 0.0, 1.0))

            # Final formatting
            axes[azi].text(0.01, 0.99, (r'$\theta_{left}\;=\;' + str(round(angles[azi], 1))
                           + r'\degree$' + '\n' + r'$\theta_{right}\;=\;' +
                           str(round(angles[azi + N_azimuths], 1)) + r'\degree$'),
                           horizontalalignment='left', verticalalignment='top',
                           transform = axes[azi].transAxes)
            axes[azi].set_title(r'$\varphi = ' + str(round(azi * rot_angle, 1)) + r'\degree$')
            axes[azi].set_xlabel(r'r ($\AA$)')
            axes[azi].set_ylabel(r'z ($\AA$)')
            axes[azi].set_xlim(left=np.min(carbon_slice[:,0]) - COARSE_GRAIN_LENGTH,
                               right=np.max(carbon_slice[:,0]) + COARSE_GRAIN_LENGTH)
            axes[azi].set_ylim(bottom=np.min(carbon_slice[:,2]) - COARSE_GRAIN_LENGTH,
                               top=1.5 * droplet_h)
            axes[azi].set_aspect(1)

        # Rotate azimuthal slice
        waters = np.einsum('jk,ik->ij', rot_matrix, waters)
        carbons = np.einsum('jk,ik->ij', rot_matrix, carbons)

    return (angles, highest_density)


#==================================================================================================
# Start of program flow

if __name__ == "__main__":

    # Generate program description
    prog_desc = ''
    for line in prog_desc_header.splitlines()[2:-1]:
        prog_desc += (line.lstrip(' ') + ' ') if line != '' else '\n\n'

    # Read script inputs from command line interface
    argparser = argparse.ArgumentParser(prog='contact-angle', description=prog_desc,
                                        usage='%(prog)s filename [options]',
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument('filename', help='input file to read data from')
    argparser.add_argument('-N', '--N_frames', '-n', type=int, default=1, dest='N_frames',
                        help='number of frames to extract from the start of the input file')
    argparser.add_argument('-i', '--interval', type=int, default=1, dest='interval',
                        help='slicing interval for extraction of frames')
    argparser.add_argument('--N_azimuths', type=int, default=1, dest='N_azimuths',
                        help='number of azimuthal angles to analyze per frame')
    argparser.add_argument('--block-average', action='store_true', dest='block_average',
                        help=('perform reverse cumulative averaging for unbiased uncertainty '
                                'estimates over varying block sizes'))
    argparser.add_argument('--auto', action='store_true', dest='opt_auto',
                        help=('if --block-average is turned on, enforces the automatic '
                                'determination of block size (overrides --blocksize)'))
    argparser.add_argument('-b', '--blocksize', type=int, default=None, dest='blocksize',
                        help=('if --block-average is turned on, disables automatic determination '
                                'of block size and enforces manually specified block size (unless '
                                '--auto was turned on)'))
    argparser.add_argument('--units', default='A', dest='units',
                        help='length units of coordinates in the input file (default angstroms)')
    argparser.add_argument('-o', default='output.png', dest='output_filename',
                        help='output file to save graphics to')
    argparser.add_argument('--no-save', action='store_false', dest='opt_save',
                        help='disable saving of graphics to output file')
    argparser.add_argument('--no-display', action='store_false', dest='opt_display',
                        help='disable display of graphics')

    prog_args = argparser.parse_args()

    # Checking filename
    if not os.path.isfile(prog_args.filename):
        raise RuntimeError(f'File "{prog_args.filename}" not found.')

    # Checking N_frames and interval
    if prog_args.N_frames < 1:
        raise RuntimeError(f'Number of frames ({prog_args.N_frames}) must be positive!')
    if prog_args.interval < 1:
        raise RuntimeError(f'Frame slicing interval ({prog_args.interval}) must be positive!')
    if prog_args.N_azimuths < 1:
        raise RuntimeError(f'Number of azimuthal angles ({prog_args.N_azimuths}) must be positive!')

    # Checking units
    length_scaling_factor = 1.0
    prog_args.units = prog_args.units.lower()
    if prog_args.units in {'a', 'aa', 'ang', 'angstrom', 'angstroms'}:
        length_scaling_factor = 1.0
    elif prog_args.units in {'f', 'fm', 'fermi', 'femto', 'femtometer', 'femtometre'}:
        length_scaling_factor = 0.00001
    elif prog_args.units in {'p', 'pm', 'pico', 'picometer', 'picometre'}:
        length_scaling_factor = 0.01
    elif prog_args.units in {'n', 'nm', 'nano', 'nanometer', 'nanometre'}:
        length_scaling_factor = 10.0
    elif prog_args.units in {'u', 'um', 'mu', 'micron', 'micro', 'micrometer', 'micrometre'}:
        length_scaling_factor = 10000.0
    else:
        raise RuntimeError(f'Units "{prog_args.units}" not recognized.')

    # Check block averaging options
    if prog_args.blocksize is None:
        prog_args.opt_auto = True
    elif prog_args.blocksize < 1:
        raise RuntimeError(f'Block size ({prog_args.blocksize}) must be positive.')

    print(f'\nReading "{prog_args.filename}"...', end='')
    time_start = time.time()

    #----------------------------------------------------------------------------------------------
    # Program mode (1):  Contact angle is measured as an instantaneous 'snapshot' from the last
    # frame of the file. The angle is averaged over azimuthal slices. The program also generates a
    # plot of the fit for each azimuthal slice over the water density if N_azimuths is less than 7;
    # otherwise the plot is generated only for the first azimuthal slice.

    if prog_args.N_frames == 1:

        atoms = ase.io.read(prog_args.filename)

        # Check if atomic numbers are assigned correctly in the file and reassign if needed
        element_numbers = np.unique(atoms.numbers)
        if np.array_equal(element_numbers, [1, 2, 3]):
            atoms.numbers[atoms.numbers == 1] = 6
            atoms.numbers[atoms.numbers == 2] = 1
            atoms.numbers[atoms.numbers == 3] = 8
        elif np.array_equal(element_numbers, [1, 6, 8]):
            pass
        else:
            raise RuntimeError('Unidentified atomic numbers in file!')
        
        # Process coordinates
        cell_params = atoms.cell.cellpar()[0:3]
        waters, carbons, _ = center_coordinates(atoms, np.array(cell_params))
        time_end = time.time()
        print(f'done in {(time_end - time_start):.3f} s.')

        # Create matplotlib figure to generate plots
        if prog_args.opt_save or prog_args.opt_display:
            if prog_args.N_azimuths == 2:
                fig, fig_ax = plt.subplots(1, 2, layout='constrained')
                axes = [fig_ax[0], fig_ax[1]]
                fig.set_size_inches(13, 3.12)
            elif prog_args.N_azimuths == 3:
                fig, fig_ax = plt.subplots(2, 2, layout='constrained')
                axes = [fig_ax[0][0], fig_ax[0][1], fig_ax[1][0]]
                fig_ax[1][1].set_axis_off()
                fig.set_size_inches(13, 6.24)
            elif prog_args.N_azimuths == 4:
                fig, fig_ax = plt.subplots(2, 2, layout='constrained')
                axes = [fig_ax[0][0], fig_ax[0][1], fig_ax[1][0], fig_ax[1][1]]
                fig.set_size_inches(13, 6.24)
            elif prog_args.N_azimuths == 5:
                fig, fig_ax = plt.subplots(2, 3, layout='constrained')
                axes = [fig_ax[0][0], fig_ax[0][1], fig_ax[0][2], fig_ax[1][0], fig_ax[1][1]]
                fig_ax[1][2].set_axis_off()
                fig.set_size_inches(13, 4.216)
            elif prog_args.N_azimuths == 6:
                fig, fig_ax = plt.subplots(2, 3, layout='constrained')
                axes = [fig_ax[0][0], fig_ax[0][1], fig_ax[0][2],
                        fig_ax[1][0], fig_ax[1][1], fig_ax[1][2]]
                fig.set_size_inches(13, 4.216)
            else:
                fig, fig_ax = plt.subplots(layout='constrained')
                axes = [fig_ax,]
                fig.set_size_inches(13, 6)
            highest_density = 0.0
        
        # Calculate contact angles
        print('Calculating contact angles...', end='')
        time_start = time.time()
        if prog_args.opt_save or prog_args.opt_display:
            angles, highest_density = contact_angles(waters, carbons, prog_args.N_azimuths,
                                                    180 / prog_args.N_azimuths, fig, axes)
        else:
            angles, highest_density = contact_angles(waters, carbons, prog_args.N_azimuths,
                                                    180 / prog_args.N_azimuths)
        time_end = time.time()
        print(f'done in {(time_end - time_start):.3f} s.')

        # Output result
        angle = np.mean(angles)
        error = np.std(angles) / np.sqrt(angles.shape[0] - 1)
        dp = int(np.ceil(-np.log10(error))) + 1
        print(f'\nContact angle = {round(angle, dp)} \u00b1 {round(error, dp)}\u00b0\n')

        # Final formatting of axes etc.
        if prog_args.opt_save or prog_args.opt_display:
            
            # Align all axes to same range
            leftest_left = np.inf
            rightest_right = -np.inf
            bottomest_bottom = np.inf
            toppest_top = -np.inf
            for axis in axes:
                left, right = axis.get_xlim()
                leftest_left = min(leftest_left, left)
                rightest_right = max(rightest_right, right)
                bottom, top = axis.get_ylim()
                bottomest_bottom = min(bottomest_bottom, bottom)
                toppest_top = max(toppest_top, top)
            for axis in axes:
                axis.set_xlim(left=leftest_left, right=rightest_right)
                axis.set_ylim(bottom=bottomest_bottom, top=toppest_top)

            # Custom colorbar for the density function
            if highest_density < BULK_DENSITY:
                cdict = {'red': [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
                            'green': [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
                            'blue': [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)]}
                fig.colorbar(ScalarMappable(norm=Normalize(vmin=0.0, vmax=BULK_DENSITY),
                            cmap=LinearSegmentedColormap('', segmentdata=cdict, N=256)),
                            ax=fig_ax, orientation='vertical', fraction=0.046, pad=0.04, shrink=0.8,
                            label=r'Number density ($\AA^{-3}$)')
            elif highest_density < 2 * BULK_DENSITY:
                extent = BULK_DENSITY / highest_density
                excess = (highest_density / BULK_DENSITY) - 1.0
                cdict = {'red': [(0.0, 1.0, 1.0), (extent, 0.0, 0.0), (1.0, excess, excess)],
                            'green': [(0.0, 1.0, 1.0), (extent, 0.0, 0.0), (1.0, 0.0, 0.0)],
                            'blue': [(0.0, 1.0, 1.0), (extent, 1.0, 1.0), (1.0, 1.0 - excess, 1.0 - excess)]}
                fig.colorbar(ScalarMappable(norm=Normalize(vmin=0.0, vmax=highest_density),
                            cmap=LinearSegmentedColormap('', segmentdata=cdict, N=256)),
                            ax=fig_ax, orientation='vertical', fraction=0.046, pad=0.04, shrink=0.8,
                            label=r'Number density ($\AA^{-3}$)')
            else:
                extent = BULK_DENSITY / highest_density
                cdict = {'red': [(0.0, 1.0, 1.0), (extent, 0.0, 0.0),
                                    (2 * extent, 1.0, 1.0), (1.0, 1.0, 1.0)],
                            'green': [(0.0, 1.0, 1.0), (extent, 0.0, 0.0), (1.0, 0.0, 0.0)],
                            'blue': [(0.0, 1.0, 1.0), (extent, 1.0, 1.0),
                                    (2 * extent, 0.0, 0.0), (1.0, 0.0, 0.0)]}
                fig.colorbar(ScalarMappable(norm=Normalize(vmin=0.0, vmax=highest_density),
                            cmap=LinearSegmentedColormap('', segmentdata=cdict, N=256)),
                            ax=fig_ax, orientation='vertical', fraction=0.046, pad=0.04, shrink=0.8,
                            label=r'Number density ($\AA^{-3}$)')

        # Save output & display (if relevant), then quit
        if prog_args.opt_save:
            fig.savefig(prog_args.output_filename, dpi=fig.dpi)
        if prog_args.opt_display:
            plt.show()
        sys.exit()

    #--------------------------------------------------------------------------------------------------
    # Program modes (2), (3), (4): Contact angle is measured over multiple frames (with multiple
    # azimuthal slices per frame) to obtain statistics of contact angles over both frame number and
    # azimuthal angle; the final reported quantity is derived from these statistics.

    else:

        traj = ase.io.iread(prog_args.filename, index=slice(0, prog_args.N_frames * prog_args.interval,
                                                            prog_args.interval))
        print()
        angles = np.zeros((prog_args.N_frames, 2 * prog_args.N_azimuths), dtype=float)
        cell_params = None
        need_to_reassign = None
        rot_angle = 180 / prog_args.N_azimuths
        frame_counter = 0
        for atoms in traj:

            print(f'    - Processing frame #{frame_counter}...', end='')
            frame_time_start = time.time()

            # Get cell parameters
            if cell_params is None:
                cell_params = np.array(atoms.cell.cellpar()[0:3])

            # Check if atomic numbers are assigned correctly in the file...
            if need_to_reassign is None:
                element_numbers = np.unique(atoms.numbers)
                if np.array_equal(element_numbers, [1, 2, 3]):
                    need_to_reassign = True
                elif np.array_equal(element_numbers, [1, 6, 8]):
                    need_to_reassign = False
                else:
                    raise RuntimeError('Unidentified atomic numbers in file!')
                
            # ...and reassign if needed
            if need_to_reassign:
                atoms.numbers[atoms.numbers == 1] = 6
                atoms.numbers[atoms.numbers == 2] = 1
                atoms.numbers[atoms.numbers == 3] = 8
            
            # Process this frame
            waters, carbons, _ = center_coordinates(atoms, cell_params)
            angles[frame_counter,:], highest_density = contact_angles(waters, carbons,
                                                                    prog_args.N_azimuths, rot_angle)
            frame_time_end = time.time()
            print(f'done in {(frame_time_end - frame_time_start):.3f} s.')
            frame_counter += 1
            
        time_end = time.time()
        print(f'...done in {(time_end - time_start):.3f} s.')
        if prog_args.block_average and frame_counter < 9:
            warnings.warn('Cannot perform block averaging with less than 9 frames', RuntimeWarning)
            prog_args.block_average = False

        # Calculate result and output
        if prog_args.block_average:

            blocksize_max = int(frame_counter / 3)
            blocksize_step = max(int(blocksize_max / 32), 1)
            blocksizes = list(range(1, blocksize_max + 1, blocksize_step))
            if (not prog_args.opt_auto) and (prog_args.blocksize not in blocksizes):
                blocksizes.append(prog_args.blocksize)
                blocksizes.sort()
            
            block_means = np.zeros((len(blocksizes),), dtype=float)
            block_vars = np.zeros((len(blocksizes),), dtype=float)
            block_counts = np.zeros((len(blocksizes),), dtype=int)
            for blocksize_index in range(len(blocksizes)):
                size = blocksizes[blocksize_index]
                start_indices = list(range(0, frame_counter, size))
                block_counts[blocksize_index] = len(start_indices) - 1
                data = np.zeros((block_counts[blocksize_index],), dtype=float)
                for b in range(block_counts[blocksize_index]):
                    data[b] = np.median(angles[start_indices[b]:start_indices[b+1]])
                block_means[blocksize_index] = np.mean(data)
                block_vars[blocksize_index] = np.var(data)
            blocksizes = np.array(blocksizes)

            stat_inefficiencies = blocksizes * block_vars / block_vars[0]
            if prog_args.opt_auto:
                k = int(blocksizes.shape[0] * 0.75)
                chosen_index = np.argpartition(stat_inefficiencies, k)[k]
            else:
                chosen_index = np.argwhere(blocksizes == prog_args.blocksize)[0,0]

            angle = block_means[chosen_index]
            error = np.sqrt(block_vars[chosen_index] / (block_counts[chosen_index] - 1))
            dp = int(np.ceil(-np.log10(error))) + 1
            print(f'\nIdentified ideal block-median size of {blocksizes[chosen_index]} frames')
            print(f'Mean block-medianed contact angle = {round(angle, dp)} \u00b1 {round(error, dp)}\u00b0')
            print(f'(Overall median value: {round(np.median(angles), dp)}\u00b0)\n')

            if prog_args.opt_display:
                fig, axis = plt.subplots(1, 2)
                fig.set_size_inches(13, 8)
                axis[0].plot(blocksizes, stat_inefficiencies, 'bo')
                axis[0].plot((blocksizes[chosen_index],), (stat_inefficiencies[chosen_index],), 'rs')
                axis[0].set_title('Statistical inefficiency against blocksize')
                axis[0].set_xlabel('Blocksize')
                axis[0].set_ylabel('Statistical inefficiency (arb. units)')
                axis[1].errorbar(blocksizes, block_means, yerr=np.sqrt(block_vars / (block_counts - 1)), fmt='bo')
                axis[1].errorbar((blocksizes[chosen_index],), (angle,), yerr=(error,), fmt='rs')
                axis[1].set_title('Mean of block medians against blocksize')
                axis[1].set_xlabel('Blocksize')
                axis[1].set_ylabel('Mean of block medians (\u00b0)')
                plt.show()

        else:
            angle = np.mean(angles)
            error = np.std(angles) / np.sqrt(angles.shape[0] - 1)
            dp = int(np.ceil(-np.log10(error))) + 1
            print(f'\nContact angle = {round(angle, dp)} \u00b1 {round(error, dp)}\u00b0')
            print(f'(Median value: {round(np.median(angles), dp)}\u00b0)\n')

        # If desired, display results
        if prog_args.opt_save or prog_args.opt_display:
            fig, axis = plt.subplots()
            fig.set_size_inches(12, 9)
            graphic = axis.imshow(angles.T, origin='lower', extent=(-0.5, frame_counter - 0.5,
                                                                    -0.5 * rot_angle, 360 - (0.5 * rot_angle)))
            axis.set_aspect(frame_counter / 360)
            axis.set_xlabel('Frame number')
            axis.set_ylabel(r'$\varphi\;\;(\degree)$')
            if prog_args.N_azimuths < 11:
                axis.set_yticks(np.linspace(0, 360, 2 * prog_args.N_azimuths, endpoint=False))
            if frame_counter < 11:
                axis.set_xticks(np.linspace(0, frame_counter, frame_counter, endpoint=False))
            fig.colorbar(graphic, ax=axis)
            if prog_args.opt_save:
                fig.savefig(prog_args.output_filename, dpi=fig.dpi)
            if prog_args.opt_display:
                plt.show()
        sys.exit()
