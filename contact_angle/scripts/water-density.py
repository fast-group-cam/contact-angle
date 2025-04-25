#! /usr/bin/env python

prog_desc_header = '''
===================================================================================================
 Python script which takes a trajectory of a water droplet on graphene, measures its density (using
 traditional binning & counting methods, rather than the coarse-grained density function), and
 displays it in a custom visualization. Use as:

     python water-density.py <input_file> [-o <output_file>] [--index <index>] [--N_bins <N_bins>]

===================================================================================================
'''

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from contact_angle.util import elapsed_time, read_droplet_trajectory
from contact_angle.util.droplet.coarse_grain import (COARSE_GRAIN_LENGTH, BULK_DENSITY,
                                                     find_interface)
from contact_angle.util.droplet.plot import plot_density_xz_slice_multi

if __name__ == "__main__":

    #----------------------------------------------------------------------------------------------
    # Generate program description and parse input arguments

    prog_desc = ''
    for line in prog_desc_header.splitlines()[2:-1]:
        prog_desc += (line.lstrip(' ') + ' ') if line != '' else '\n\n'
    
    parser = argparse.ArgumentParser(prog='water-density', description=prog_desc,
                                     usage='%(prog)s filename [options]',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file', nargs='+')
    parser.add_argument('-o', '--output', default='density.png')
    parser.add_argument('--index', default=':')
    parser.add_argument('-N', '--N_bins', type=int, default=100)
    parser.add_argument('--no-display', action='store_false', dest='opt_display')
    args = parser.parse_args()

    for file in args.input_file:
        if not os.path.isfile(file):
            raise RuntimeError(f'File "{file}" not found.')
    if os.path.isfile(args.output):
        os.remove(args.output)
    
    #----------------------------------------------------------------------------------------------
    # Read input file and save coordinates

    time_start = time.time()
    cell_params, waters, carbons, _ = read_droplet_trajectory(args.input_file, index=args.index)
    if len(args.input_file) == 1:
        print(f'Read "{args.input_file[0]}" in {elapsed_time(time_start)}.')
    else:
        print(f'Read {len(args.input_file)} files in {elapsed_time(time_start)}.')

    #----------------------------------------------------------------------------------------------
    # Calculate outer limits of the droplet

    N_frames = waters.shape[0]
    time_start = time.time()

    z_floor = np.inf
    z_ceil = -np.inf
    r_max = -np.inf

    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    axes = np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))

    for f in range(N_frames):
        CoM = np.mean(waters[f], axis=0)
        for axis in axes:
            inter = find_interface(waters[f], CoM, axis)
            r_max = max(r_max, np.linalg.norm(inter - CoM))
        z_ceil = max(z_ceil, find_interface(waters[f], CoM, (0, 0, 1))[2])
        z_floor = min(z_floor, find_interface(waters[f], CoM, (0, 0, -1))[2])
    
    print(f'Found droplet maximal radius to be {r_max:.2f}\u212b and height to be ' +
          f'{(z_ceil - z_floor):.2f}\u212b (time taken: {elapsed_time(time_start)}).')

    #----------------------------------------------------------------------------------------------
    # Calculate density bins

    r_max = (1.1 * r_max) + COARSE_GRAIN_LENGTH
    z_floor = z_floor + COARSE_GRAIN_LENGTH
    z_ceil = (1.1 * z_ceil) + COARSE_GRAIN_LENGTH
    time_start = time.time()

    radial_bin_edges = np.linspace(0, r_max, args.N_bins + 1, endpoint=True)
    radial_bin_centers = (radial_bin_edges[:-1] + radial_bin_edges[1:]) / 2
    radial_bin_areas = np.pi * (np.square(radial_bin_edges[1:]) - np.square(radial_bin_edges[:-1]))
    radial_bin_counts = np.zeros((args.N_bins,), dtype=float)

    axial_bin_edges = np.linspace(z_floor, z_ceil, args.N_bins + 1, endpoint=True)
    axial_bin_centers = (axial_bin_edges[:-1] + axial_bin_edges[1:]) / 2
    axial_bin_height = axial_bin_edges[1] - axial_bin_edges[0]
    axial_bin_counts = np.zeros((args.N_bins,), dtype=float)

    for f in range(N_frames):

        CoM = np.mean(waters[f,:,2])
        r_coords = np.sqrt(np.sum(np.square(waters[f,:,0:2]), axis=-1))
        radial_slice = r_coords[np.abs(waters[f,:,2] - CoM) < COARSE_GRAIN_LENGTH]
        axial_slice = waters[f, r_coords < COARSE_GRAIN_LENGTH, 2]

        radial_bin_counts += np.histogram(radial_slice, bins=radial_bin_edges)[0]
        axial_bin_counts += np.histogram(axial_slice, bins=axial_bin_edges)[0]

    radial_bin_counts /= N_frames
    radial_density = radial_bin_counts / (2 * COARSE_GRAIN_LENGTH * radial_bin_areas)
    axial_bin_counts /= N_frames
    axial_density = axial_bin_counts / (np.pi * (COARSE_GRAIN_LENGTH**2) * axial_bin_height)

    print(f'Calculated density distribution in {elapsed_time(time_start)}.')

    #----------------------------------------------------------------------------------------------
    # Plot results

    def test_curve(x, A, x0, delta):
        return 0.5 * A * (np.tanh((x0 - x) / delta) + 1)
    
    popt_r, _ = curve_fit(test_curve, radial_bin_centers, radial_density,
                          p0=(BULK_DENSITY, 0.9 * r_max, 1))
    popt_z, _ = curve_fit(test_curve, axial_bin_centers, axial_density,
                          p0=(BULK_DENSITY, 0.9 * z_ceil, 1))
    CoM = np.mean(waters[:,:,2])
    
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(14, 7)
    time_start = time.time()

    if N_frames < 30:
        plot_density_xz_slice_multi(waters, carbons, ax[0][0])
    else:
        plot_density_xz_slice_multi(waters[::int(N_frames / 30)], carbons, ax[0][0])
    ax[0][0].plot([-r_max, r_max], [CoM + COARSE_GRAIN_LENGTH, CoM + COARSE_GRAIN_LENGTH], '-',
                  color=(1, 0, 1))
    ax[0][0].plot([-r_max, r_max], [CoM - COARSE_GRAIN_LENGTH, CoM - COARSE_GRAIN_LENGTH], '-',
                  color=(1, 0, 1))
    ax[0][0].set_xlabel(r'x [$\AA$]')
    ax[0][0].set_ylabel(r'z [$\AA$]')
    ax[0][0].set_title('CoM disk')

    ax[0][1].plot([0, r_max], [BULK_DENSITY, BULK_DENSITY], 'k--')
    ax[0][1].plot(radial_bin_centers, test_curve(radial_bin_centers, *popt_r), 'b-')
    ax[0][1].plot(radial_bin_centers, radial_density, 'b.')
    ax[0][1].set_xlabel(r'r [$\AA$]')
    ax[0][1].set_ylabel(r'$\rho_{O}(r)$ [$\AA^{-3}$]')
    ax[0][1].set_title(r'Density $\rho_{O}$ within CoM disk')
    ax[0][1].annotate(r'$\rho_{O}^{ref} = ' + f'{BULK_DENSITY:.5f}' + r' \,\AA^{-3}$' + '\n' +
                      r'$\rho_{O}^{fit} = ' + f'{popt_r[0]:.5f}' + r' \,\AA^{-3}$',
                      (0.05 * r_max, 0.08 * BULK_DENSITY), ha='left', va='bottom')

    ax[0][2].plot(radial_bin_centers, np.cumsum(radial_bin_counts), 'b.')
    ax[0][2].set_xlabel(r'r [$\AA$]')
    ax[0][2].set_ylabel(r'$N_{O}(r)$')
    ax[0][2].set_title(r'Cumulative count $N_{O}$ within CoM disk')

    if N_frames < 30:
        plot_density_xz_slice_multi(waters, carbons, ax[1][0])
    else:
        plot_density_xz_slice_multi(waters[::int(N_frames / 30)], carbons, ax[1][0])
    ax[1][0].plot([-COARSE_GRAIN_LENGTH, -COARSE_GRAIN_LENGTH],
                  [z_floor - COARSE_GRAIN_LENGTH, z_ceil], '-', color=(1, 0, 1))
    ax[1][0].plot([COARSE_GRAIN_LENGTH, COARSE_GRAIN_LENGTH],
                  [z_floor - COARSE_GRAIN_LENGTH, z_ceil], '-', color=(1, 0, 1))
    ax[1][0].set_xlabel(r'x [$\AA$]')
    ax[1][0].set_ylabel(r'z [$\AA$]')
    ax[1][0].set_title('Central axis')

    ax[1][1].plot([z_floor, z_ceil], [BULK_DENSITY, BULK_DENSITY], 'k--')
    ax[1][1].plot(axial_bin_centers, test_curve(axial_bin_centers, *popt_z), 'b-')
    ax[1][1].plot(axial_bin_centers, axial_density, 'b.')
    ax[1][1].set_xlabel(r'z [$\AA$]')
    ax[1][1].set_ylabel(r'$\rho_{O}(z)$ [$\AA^{-3}$]')
    ax[1][1].set_title(r'Density $\rho_{O}$ along central axis')
    ax[1][1].annotate(r'$\rho_{O}^{ref} = ' + f'{BULK_DENSITY:.5f}' + r' \,\AA^{-3}$' + '\n' +
                      r'$\rho_{O}^{fit} = ' + f'{popt_z[0]:.5f}' + r' \,\AA^{-3}$',
                      (0.05 * z_ceil, 0.08 * BULK_DENSITY), ha='left', va='bottom')

    ax[1][2].plot(axial_bin_centers, np.cumsum(axial_bin_counts), 'b.')
    ax[1][2].set_xlabel(r'z [$\AA$]')
    ax[1][2].set_ylabel(r'$N_{O}(z)$')
    ax[1][2].set_title(r'Cumulative count $N_{O}$ along central axis')

    print(f'Computed plots in {elapsed_time(time_start)}.')
    fig.tight_layout()
    fig.savefig(args.output, dpi=(3*fig.dpi), bbox_inches='tight', pad_inches=0.05)
    if args.opt_display:
        plt.show()
