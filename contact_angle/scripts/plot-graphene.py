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
import ase.io
import matplotlib.pyplot as plt
from contact_angle.util.graphene import inclination_norm_inf_autocor

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
    parser.add_argument('-t', '--max_tau', type=int, default=30)
    parser.add_argument('-N', '--N_x', type=int, default=80)
    parser.add_argument('--N_y', type=int, default=None)
    parser.add_argument('--max_threads', type=int, default=None)
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
    traj = ase.io.iread(args.input_file, index=':')
    cell_xy = None
    elem_to_find = None
    carbons = list()
    
    for atoms in traj:

        # Get cell parameters
        if cell_xy is None:
            cell_xy = np.array(atoms.cell.cellpar()[0:2])
        
        # Check how atomic numbers are assigned in the file
        if elem_to_find is None:
            element_numbers = np.unique(atoms.numbers)
            if np.array_equal(element_numbers, [1, 2, 3]) or np.array_equal(element_numbers, [1,]):
                elem_to_find = 1
            elif np.array_equal(element_numbers, [1, 6, 8]) or np.array_equal(element_numbers, [6,]):
                elem_to_find = 6
            else:
                raise RuntimeError('Unidentified atomic numbers in file!')
            
        # Read coordinates accordingly
        carbons.append(atoms.positions[atoms.numbers == elem_to_find])
    
    #----------------------------------------------------------------------------------------------
    # Calculate autocorrelations and display

    threads = min(32, (os.cpu_count() or 1))
    if args.max_threads is not None:
        threads = min(args.max_threads, threads)
    print(f'Using {threads} workers')

    carbons = np.array(carbons)
    data = inclination_norm_inf_autocor(carbons, cell_xy, args.max_tau, (args.N_x, args.N_y),
                                        max_threads=threads)
    
    time_taken = time.time() - time_start
    hours = int(time_taken / 3600)
    time_taken -= 3600 * hours
    minutes = int(time_taken / 60)
    time_taken -= 60 * minutes
    print(f'Took (hh:mm:ss) {hours:02}:{minutes:02}:{round(time_taken):02} of wall time.')

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 9)
    artist = ax.imshow(np.swapaxes(data, 0, 1), origin='lower', extent=(-0.5 * cell_xy[0],
                                                                        0.5 * cell_xy[0],
                                                                        -0.5 * cell_xy[1],
                                                                        0.5 * cell_xy[1]))
    cbar = fig.colorbar(artist, ax=ax)
    cbar.set_label(r'$C_{\theta}(\tau\to\infty)$')
    ax.set_xlabel(r'x ($\AA$)')
    ax.set_ylabel(r'y ($\AA$)')
    ax.set_title('Normalized autocorrelation of local inclination angle')
    ax.set_aspect('equal')

    fig.savefig(args.output, dpi=(3*fig.dpi), bbox_inches='tight', pad_inches=0.05)
    if args.opt_display:
        plt.show()
