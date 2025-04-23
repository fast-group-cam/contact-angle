#! /usr/bin/env python

prog_desc_header = '''
===================================================================================================
 Python script which takes a trajectory of a graphene sheet (which may or may not have a water
 droplet on it), calculates its local inclination angle autocorrelation function, and plots it to
 a file. Use as:

     python plot-graphene.py <input_file> [-o <output_file>] [--max_tau <max_tau>] [--N_x <N_x>]
         [--N_y <N_y>] [--max_threads <max_threads>] [--no-display]

===================================================================================================
'''

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from contact_angle.util import elapsed_time, read_droplet_trajectory
from contact_angle.util.graphene import calc_inclination_angles

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
    parser.add_argument('--no-display', action='store_false', dest='opt_display')
    args = parser.parse_args()

    if os.path.isfile(args.output):
        os.remove(args.output)
    if not os.path.isfile(args.input_file):
        raise RuntimeError(f'File "{args.input_file}" not found.')
    if args.N_y is None:
        args.N_y = args.N_x
    
    #----------------------------------------------------------------------------------------------
    # Read input file and save coordinates

    time_start = time.time()
    cell_params, _, carbons, _ = read_droplet_trajectory(args.input_file, index=args.index)
    cell_xy = cell_params[0:2]
    print(f'Read "{args.input_file}" in {elapsed_time(time_start)}.')
    
    #----------------------------------------------------------------------------------------------
    # Calculate local inclination angles and display

    time_start = time.time()
    angles = calc_inclination_angles(carbons, cell_xy, (args.N_x, args.N_y))
    print(f'Calculated local inclination angles in {elapsed_time(time_start)}.')

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
                inf_autoc[i,j] = popt[-1]
            except RuntimeError:
                inf_autoc[i,j] = 0.8
    print(f'Calculated autocorrelations in {elapsed_time(time_start)}.')

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(14, 8)
    extent =  (-0.5 * cell_xy[0], 0.5 * cell_xy[0], -0.5 * cell_xy[1], 0.5 * cell_xy[1])

    im = ax[0].imshow(np.swapaxes(np.mean(angles, axis=0), 0, 1), origin='lower', extent=extent)
    cbar = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    cbar.set_label(r'$\langle\theta\rangle_t$')
    ax[0].set_xlabel(r'x ($\AA$)')
    ax[0].set_ylabel(r'y ($\AA$)')
    ax[0].set_title('Time-averaged local inclination angle')
    ax[0].set_aspect('equal')

    im = ax[1].imshow(np.swapaxes(inf_autoc, 0, 1), origin='lower', extent=extent)
    cbar = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.set_label(r'$C_{\theta}(\tau\to\infty)$')
    ax[1].set_xlabel(r'x ($\AA$)')
    ax[1].set_ylabel(r'y ($\AA$)')
    ax[1].set_title('Normalized autocorrelation of local inclination angle')
    ax[1].set_aspect('equal')

    fig.tight_layout()
    fig.savefig(args.output, dpi=(3*fig.dpi), bbox_inches='tight', pad_inches=0.05)
    if args.opt_display:
        plt.show()
