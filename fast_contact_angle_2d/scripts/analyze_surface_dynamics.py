#! /usr/bin/env python

prog_desc_header = '''
===================================================================================================
 Python script which takes a trajectory of a solid surface (which may or may not have a liquid
 droplet on it), calculates certain observables with regards to local dynamics, and plots them to
 a file. Use as:

 >    python analyze_surface_dynamics.py <input_file(s)> [-o <output_dir>] [--index <index>]
          [--sol_symbol <sol_symbol>] [--sol_number <sol_number>] [--liq_symbol <liq_symbol>]
          [--liq_number <liq_number>] [--N_x <N_x>] [--N_y <N_y>] [--max_tau <max_tau>]
          [--delta_t <delta_t>] [--time_rescale_factor <time_rescale_factor>]
          [--length_rescale_factor <length_rescale_factor>] [--disk_radius <disk_radius>]
          [--margin <margin>] [--z_range <z_range>] [--z_fluc_range <z_fluc_range>]
          [--theta_range <theta_range>] [--autocorr_range <autocorr_range>] [--no-display]

 The generated plots are calculated over a square grid of (x, y) coordinates, with resolution given
 by N_x and N_y, and displays the following: in the top left, the time-averaged heightmap; in the
 top right, the fluctuation widths (i.e. standard deviation) of the heightmap over time; in the
 bottom left, the time-averaged local inclination angles of the surface; and in the bottom right,
 the normalized infinite-time autocorrelations of the local inclination angles of the sheet.
===================================================================================================
'''

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import fast_contact_angle_2d.util.io as io
from fast_contact_angle_2d import elapsed_time
from fast_contact_angle_2d.interpolate import PeriodicGridInterpolator
from fast_contact_angle_2d.solid.sheet import _z_grid, DEFAULT_MARGIN
from fast_contact_angle_2d.solid.angle import (time_averaged_inclination_angles,
                                               instantaneous_inclination_angles, DISK_RADIUS)
from fast_contact_angle_2d.autocorrelations import norm_inf_autocorrelation
from fast_contact_angle_2d.util.plot import plot_2d_function

def main() -> None:

    #----------------------------------------------------------------------------------------------
    # Generate program description and parse input arguments

    prog_desc = ''
    for line in prog_desc_header.splitlines()[2:-1]:
        prog_desc += (line.lstrip(' ') + ' ') if line != '' else '\n\n'
    
    parser = argparse.ArgumentParser(prog='analyze_surface_dynamics', description=prog_desc,
                                     usage='%(prog)s input_file(s) [options]',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file', nargs='+',
                        help='input file(s) to read data from')
    parser.add_argument('-o', '--output', default='surface-dynamics', dest='output_dir',
                        help='output folder to save results and graphical outputs to')
    parser.add_argument('--index', default=':', dest='index',
                        help='slice of indices to take from each input file')
    parser.add_argument('--sol_symbol', type=str, nargs='*', default=None, dest='sol_symbol',
                        help='atomic symbol(s) to interpret as solid particles')
    parser.add_argument('--sol_number', type=int, nargs='*', default=None, dest='sol_number',
                        help='atomic number(s) to interpret as solid particles')
    parser.add_argument('--liq_symbol', type=str, nargs='*', default=None, dest='liq_symbol',
                        help='atomic symbol(s) to interpret as liquid particles')
    parser.add_argument('--liq_number', type=int, nargs='*', default=None, dest='liq_number',
                        help='atomic number(s) to interpret as liquid particles')
    parser.add_argument('-N', '--N_x', type=int, default=100, dest='N_x',
                        help='resolution of plot in x-direction')
    parser.add_argument('--N_y', type=int, default=None, dest='N_y',
                        help='resolution of plot in y-direction (defaults to be the same as N_x)')
    parser.add_argument('-t', '--max_tau', type=float, default=None, dest='max_tau',
                        help='maximum timescale to calculate autocorrelations (in ps)')
    parser.add_argument('--delta_t', type=float, default=None, dest='delta_t',
                        help='time interval between trajectory frames (in ps)')
    parser.add_argument('--time_rescale_factor', type=float, default=None, dest='time_rescale_factor',
                        help='rescaling factor for automatically detected timesteps (to get to ps)')
    parser.add_argument('--length_rescale_factor', type=float, default=1.0, dest='length_rescale_factor',
                        help='rescaling factor for lengths (to get to angstroms)')
    parser.add_argument('--disk_radius', type=float, default=DISK_RADIUS, dest='disk_radius',
                        help='radius of disk function for regularizing local inclination angle')
    parser.add_argument('--margin', type=float, default=DEFAULT_MARGIN, dest='margin',
                        help='safety margin for padding PeriodicGridInterpolator')
    parser.add_argument('--z_range', type=float, default=None, dest='z_range',
                        help='plotting range for mean z-heights')
    parser.add_argument('--z_fluc_range', type=float, default=None, dest='z_fluc_range',
                        help='plotting range for fluctuations of z-heights')
    parser.add_argument('--theta_range', type=float, default=None, dest='theta_range',
                        help='plotting range for mean local inclination angles')
    parser.add_argument('--autocorr_range', type=float, default=None, dest='autocorr_range',
                        help='plotting range for autocorrelations of local inclination angles')
    parser.add_argument('--no-display', action='store_false', dest='opt_display',
                        help='disable display of graphics')
    args = parser.parse_args()

    for file in args.input_file:
        if not os.path.isfile(file):
            raise RuntimeError(f'File "{file}" not found.')
        
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        
    if (args.sol_symbol is None) and (args.sol_number is None):
        args.sol_symbol = 'C'
    if (args.liq_symbol is None) and (args.liq_number is None):
        args.liq_symbol = 'O'

    if args.N_y is None:
        args.N_y = args.N_x
    if args.z_range is not None:
        args.z_range = 0.5 * abs(args.z_range)
    if args.z_fluc_range is not None:
        args.z_fluc_range = 0.5 * abs(args.z_fluc_range)
    if args.theta_range is not None:
        args.theta_range = 0.5 * abs(args.theta_range)
    if args.autocorr_range is not None:
        args.autocorr_range = min(0.5 * abs(args.autocorr_range), 0.5)

    #----------------------------------------------------------------------------------------------
    # Read input file and save coordinates

    file_msg = (f'"{args.input_file[0]}"' if len(args.input_file) == 1 else
                f'{len(args.input_file)} files')
    
    time_start_0 = time.time()
    trajectory = io.read(args.input_file, index=args.index, liq_symbol=args.liq_symbol,
                         sol_symbol=args.sol_symbol, liq_number=args.liq_number,
                         sol_number=args.sol_number, delta_t=args.delta_t,
                         time_rescale_factor=args.time_rescale_factor,
                         length_rescale_factor=args.length_rescale_factor)
    cell_params = trajectory['cell_params']
    sol = trajectory['sol']
    N_frames = sol.shape[0]
    timestep = trajectory.get('delta_t', 1.0)

    print(f'Read {N_frames} frames from {file_msg} in {elapsed_time(time_start_0)}.')
    
    #----------------------------------------------------------------------------------------------
    # Calculate interpolated z-heights

    time_start_1 = time.time()
    z_grids = np.empty((N_frames, args.N_x, args.N_y), dtype=float)
    cell_xy = cell_params[0:2]
    for f in range(N_frames):
        z_grids[f] = _z_grid(cell_xy, sol[f], (args.N_x, args.N_y), margin=args.margin)
    
    heightmap_average = PeriodicGridInterpolator(cell_xy, np.mean(z_grids, axis=0))
    heightmap_fluctuations = PeriodicGridInterpolator(cell_xy, np.std(z_grids, axis=0))
    print(f'Interpolated z-heights in {elapsed_time(time_start_1)}.')

    #----------------------------------------------------------------------------------------------
    # Calculate local inclination angles

    time_start_1 = time.time()
    angles = list()
    for f in range(N_frames):
        angles.append(instantaneous_inclination_angles(cell_params, sol[f], (args.N_x, args.N_y),
                                                       disk_radius=args.disk_radius, margin=args.margin))
    time_averaged_angles = time_averaged_inclination_angles(cell_params, sol, (args.N_x, args.N_y),
                                                            disk_radius=args.disk_radius, margin=args.margin)
    print(f'Calculated local inclination angles in {elapsed_time(time_start_1)}.')

    #----------------------------------------------------------------------------------------------
    # Calculate normalized infinite-time autocorrelations of local inclination angles

    time_start_1 = time.time()

    if args.max_tau is None:
        max_tau_in_frames = N_frames - 1
    else:
        max_tau_in_frames = min(int(args.max_tau / timestep), N_frames - 1)

    angle_norm_inf_autocorr = norm_inf_autocorrelation(angles, max_tau_in_frames)
    print(f'Calculated autocorrelations in {elapsed_time(time_start_1)}.')

    #----------------------------------------------------------------------------------------------
    # Display plots

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(9, 9)
    resolution = (int(3 * args.N_x), int(3 * args.N_y))

    def plot_to_axis(f, axis, vrange, cap_vmin, cap_vmax, clabel, title):
        if vrange is None:
            im = plot_2d_function(f, axis, resolution)
        else:
            midpoint = 0.5 * (f.min() + f.max())
            vmax = midpoint + vrange
            vmin = midpoint - vrange
            if (cap_vmin and (vmin < 0.0)):
                vmax -= vmin
                vmin = 0.0
            if (cap_vmax and (vmax > 1.0)):
                vmin -= (vmax - 1.0)
                vmax = 1.0
            im = plot_2d_function(f, axis, resolution, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
        cbar.set_label(clabel)
        axis.set_xlabel(r'x ($\AA$)')
        axis.set_ylabel(r'y ($\AA$)')
        axis.set_title(title)
        axis.set_aspect('equal')
        
    plot_to_axis(heightmap_average,       ax[0][0], args.z_range,        False, False, r'$\langle z\rangle_t$',        'Time-averaged z-height')
    plot_to_axis(heightmap_fluctuations,  ax[0][1], args.z_fluc_range,   True,  False, r'$\sigma_z$',                  'Fluctuation width of z-height')
    plot_to_axis(time_averaged_angles,    ax[1][0], args.theta_range,    True,  False, r'$\langle\theta\rangle_t$',    'Time-averaged local inclination angle')
    plot_to_axis(angle_norm_inf_autocorr, ax[1][1], args.autocorr_range, False, True,  r'$C_{\theta}(\tau\to\infty)$', 'Normalized long-time autocorrelation of local inclination angle')

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'surface-dynamics.png'), dpi=(3*fig.dpi), bbox_inches='tight', pad_inches=0.05)
    heightmap_average.write(os.path.join(args.output_dir, 'heightmap_average.npz'))
    heightmap_fluctuations.write(os.path.join(args.output_dir, 'heightmap_fluctuations.npz'))
    time_averaged_angles.write(os.path.join(args.output_dir, 'angles_average.npz'))
    angle_norm_inf_autocorr.write(os.path.join(args.output_dir, 'angles_norm-inf-autocorr.npz'))

    print(f'Program completed in {elapsed_time(time_start_0)}.')
    if args.opt_display:
        plt.show()


#==================================================================================================
# Run from src

if __name__ == "__main__":
    main()

