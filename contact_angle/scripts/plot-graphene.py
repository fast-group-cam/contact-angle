#! /usr/bin/env python

prog_desc_header = '''
===================================================================================================
 Python script which takes a trajectory of a graphene sheet (which may or may not have a water
 droplet on it), calculates its local inclination angle autocorrelation function, and plots it to
 a file. Use as:

     python plot-graphene.py <input_file> [-o <output_file>] [--max_tau <max_tau>] [--N_x <N_x>]
         [--N_y <N_y>] [--max_threads <max_threads>] [--z_range <z_range>]
         [--z_fluc_range <z_fluc_range>] [--theta_range <theta_range>]
         [--autocorr_range <autocorr_range>] [--no-display]

===================================================================================================
'''

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from contact_angle.util import elapsed_time, read_droplet_trajectory
from contact_angle.util.graphene import smooth_sheet, calc_inclination_angles

if __name__ == "__main__":

    #----------------------------------------------------------------------------------------------
    # Generate program description and parse input arguments

    prog_desc = ''
    for line in prog_desc_header.splitlines()[2:-1]:
        prog_desc += (line.lstrip(' ') + ' ') if line != '' else '\n\n'
    
    parser = argparse.ArgumentParser(prog='plot-graphene', description=prog_desc,
                                     usage='%(prog)s filename [options]',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file')
    parser.add_argument('-o', '--output', default='graphene.png')
    parser.add_argument('--index', default=':')
    parser.add_argument('-t', '--max_tau', type=int, default=30)
    parser.add_argument('-N', '--N_x', type=int, default=80)
    parser.add_argument('--N_y', type=int, default=None)
    parser.add_argument('--z_range', type=float, default=None)
    parser.add_argument('--z_fluc_range', type=float, default=None)
    parser.add_argument('--theta_range', type=float, default=None)
    parser.add_argument('--autocorr_range', type=float, default=None)
    parser.add_argument('--no-display', action='store_false', dest='opt_display')
    args = parser.parse_args()

    if os.path.isfile(args.output):
        os.remove(args.output)
    if not os.path.isfile(args.input_file):
        raise RuntimeError(f'File "{args.input_file}" not found.')
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

    time_start = time.time()
    cell_params, _, carbons, _ = read_droplet_trajectory(args.input_file, index=args.index)
    cell_xy = cell_params[0:2]
    print(f'Read "{args.input_file}" in {elapsed_time(time_start)}.')
    
    #----------------------------------------------------------------------------------------------
    # Calculate interpolated z-heights

    time_start = time.time()
    sheet = np.zeros((carbons.shape[0], args.N_x, args.N_y), dtype=float)
    for f, carbon_atoms in enumerate(carbons):
        sheet[f] = smooth_sheet(carbon_atoms, cell_xy, (args.N_x, args.N_y))
    print(f'Calculated interpolated z-heights in {elapsed_time(time_start)}.')

    #----------------------------------------------------------------------------------------------
    # Calculate local inclination angles

    time_start = time.time()
    angles = calc_inclination_angles(carbons, cell_xy, (args.N_x, args.N_y))
    print(f'Calculated local inclination angles in {elapsed_time(time_start)}.')

    #----------------------------------------------------------------------------------------------
    # Calculate normalized infinite-time autocorrelations of local inclination angles

    time_start = time.time()
    autocorr = np.zeros((args.max_tau, args.N_x, args.N_y), dtype=float)
    autocorr[0] = np.mean(np.square(angles), axis=0)
    for tau in range(1, args.max_tau):
        autocorr[tau] = np.mean(angles[:-tau] * angles[tau:], axis=0)
    autocorr[:] /= autocorr[0]

    def exp_curve(x, A, k, c):
        return A * np.exp(-k * x) + c

    tau = np.array(range(args.max_tau))
    inf_autoc = np.zeros((args.N_x, args.N_y), dtype=float)
    for i in range(args.N_x):
        for j in range(args.N_y):
            try:
                popt, _ = curve_fit(exp_curve, tau, autocorr[:,i,j], p0=(0.2, 0.5, 0.8))
                inf_autoc[i,j] = max(popt[-1], 0.0)
            except RuntimeError:
                inf_autoc[i,j] = 0.0
    print(f'Calculated autocorrelations in {elapsed_time(time_start)}.')

    #----------------------------------------------------------------------------------------------
    # Display plots

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(9, 9)
    extent = (-0.5 * cell_xy[0], 0.5 * cell_xy[0], -0.5 * cell_xy[1], 0.5 * cell_xy[1])

    data = np.swapaxes(np.mean(sheet, axis=0), 0, 1)
    if args.z_range is None:
        im = ax[0][0].imshow(data, origin='lower', extent=extent)
    else:
        midpoint = 0.5 * (np.min(data) + np.max(data))
        vmax = midpoint + args.z_range
        vmin = midpoint - args.z_range
        im = ax[0][0].imshow(data, vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    cbar = fig.colorbar(im, ax=ax[0][0], fraction=0.046, pad=0.04)
    cbar.set_label(r'$\langle z\rangle_t$')
    ax[0][0].set_xlabel(r'x ($\AA$)')
    ax[0][0].set_ylabel(r'y ($\AA$)')
    ax[0][0].set_title('Time-averaged z-height')
    ax[0][0].set_aspect('equal')
    
    data = np.swapaxes(np.std(sheet, axis=0), 0, 1)
    if args.z_fluc_range is None:
        im = ax[0][1].imshow(data, origin='lower', extent=extent)
    else:
        midpoint = 0.5 * (np.min(data) + np.max(data))
        vmax = midpoint + args.z_fluc_range
        vmin = midpoint - args.z_fluc_range
        if vmin < 0.0:
            vmax -= vmin
            vmin = 0.0
        im = ax[0][1].imshow(data, vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    cbar = fig.colorbar(im, ax=ax[0][1], fraction=0.046, pad=0.04)
    cbar.set_label(r'$\sigma_z$')
    ax[0][1].set_xlabel(r'x ($\AA$)')
    ax[0][1].set_ylabel(r'y ($\AA$)')
    ax[0][1].set_title('Fluctuation width of z-height')
    ax[0][1].set_aspect('equal')

    data = np.swapaxes(np.mean(angles, axis=0), 0, 1)
    if args.theta_range is None:
        im = ax[1][0].imshow(data, origin='lower', extent=extent)
    else:
        midpoint = 0.5 * (np.min(data) + np.max(data))
        vmax = midpoint + args.theta_range
        vmin = midpoint - args.theta_range
        if vmin < 0.0:
            vmax -= vmin
            vmin = 0.0
        im = ax[1][0].imshow(data, vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    cbar = fig.colorbar(im, ax=ax[1][0], fraction=0.046, pad=0.04)
    cbar.set_label(r'$\langle\theta\rangle_t$')
    ax[1][0].set_xlabel(r'x ($\AA$)')
    ax[1][0].set_ylabel(r'y ($\AA$)')
    ax[1][0].set_title('Time-averaged local inclination angle')
    ax[1][0].set_aspect('equal')

    data = np.swapaxes(inf_autoc, 0, 1)
    if args.autocorr_range is None:
        im = ax[1][1].imshow(data, origin='lower', extent=extent)
    else:
        midpoint = 0.5 * (np.min(data) + np.max(data))
        vmax = midpoint + args.autocorr_range
        vmin = midpoint - args.autocorr_range
        if vmax > 1.0:
            vmin -= (vmax - 1.0)
            vmax = 1.0
        im = ax[1][1].imshow(data, vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    cbar = fig.colorbar(im, ax=ax[1][1], fraction=0.046, pad=0.04)
    cbar.set_label(r'$C_{\theta}(\tau\to\infty)$')
    ax[1][1].set_xlabel(r'x ($\AA$)')
    ax[1][1].set_ylabel(r'y ($\AA$)')
    ax[1][1].set_title('Normalized autocorrelation of local inclination angle')
    ax[1][1].set_aspect('equal')

    fig.tight_layout()
    fig.savefig(args.output, dpi=(3*fig.dpi), bbox_inches='tight', pad_inches=0.05)
    if args.opt_display:
        plt.show()
