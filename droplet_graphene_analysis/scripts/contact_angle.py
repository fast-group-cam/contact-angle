#! /usr/bin/env python

prog_desc_header = '''
===================================================================================================
 This program measures the contact angle of a water droplet on a graphene sheet from a simulated
 trajectory. The input is a file, which must be compatible with ASE's file i/o formats, describing
 either a time evolution or a single snapshot of a water droplet (rotationally symmetric about the
 z-axis) on a graphene sheet aligned aligned to the xy plane. The contact angle is calculated by
 finding the time-averaged Willard-Chandler interface for a number of testpoints in randomly-
 selected directions not intersecting the graphene sheet, fitting a spherical profile to it, and
 calculating the intersection of the sphere to the graphene sheet. Use as:

 >    python contact_angle.py <input_file(s)> [--index <index>] [-o <output_dir>] [--no-graphics]

 All plots will be saved to an output directory (specified by the -o option), which defaults to
 "contact-angle" within the parent directory that this program is executed from. The --no-graphics
 option disables the generation of these plots. Numerical results are also saved in a results.ini
 file in the output directory.
===================================================================================================
'''

import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from rich.console import Console
from rich.progress import track
from droplet_graphene_analysis import __version__
from droplet_graphene_analysis.util import elapsed_time, read_droplet_trajectory
from droplet_graphene_analysis.util.interpolate import PeriodicGridInterpolator
from droplet_graphene_analysis.util.droplet import find_interface
from droplet_graphene_analysis.util.droplet.contact_angle import find_spherical_cap
from droplet_graphene_analysis.util.droplet.plot import plot_density_radially_symmetric
from droplet_graphene_analysis.util.graphene import raw_heightmap

def main() -> None:

    #----------------------------------------------------------------------------------------------
    # Script default parameters

    from droplet_graphene_analysis.util.graphene.sheet import CUTOFF_RADIUS as CARBON_RADIUS

    #----------------------------------------------------------------------------------------------
    # Generate program description and parse input arguments

    prog_desc = ''
    for line in prog_desc_header.splitlines()[2:-1]:
        prog_desc += (line.lstrip(' ') + ' ') if line != '' else '\n\n'

    parser = argparse.ArgumentParser(prog='contact_angle', description=prog_desc,
                                     usage='%(prog)s input_file [options]',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file', nargs='+',
                        help='input file(s) to read data from')
    parser.add_argument('--index', default=':', dest='index',
                        help='index or slice of indices to take from each input file')
    parser.add_argument('-o', '--output', default='contact-angle', dest='output_dir',
                        help='output folder to save results and graphical outputs to')
    parser.add_argument('--no-graphics', action='store_true', dest='no_graphics',
                        help='disables rendering of graphics (and speeds up the script)')
    args = parser.parse_args()

    for file in args.input_file:
        if not os.path.isfile(file):
            raise RuntimeError(f'File "{file}" not found.')

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    console = Console(highlight=False)
    
    #----------------------------------------------------------------------------------------------
    # Read input file and save coordinates

    file_msg = (f'"{args.input_file[0]}"' if len(args.input_file) == 1 else
                f'{len(args.input_file)} files')
    
    time_start_0 = time.time()
    with console.status(f'[green]Reading {file_msg}...'):
        cell_params, waters, carbons, _ = read_droplet_trajectory(args.input_file, index=args.index)
        N_frames = waters.shape[0]
        N_water = waters.shape[1]

    console.print(f'Read [magenta]{N_frames} frames[/magenta] from [cyan]{file_msg}[/cyan] in ' +
                  f'[green]{elapsed_time(time_start_0)}[/green].')
    
    #----------------------------------------------------------------------------------------------
    # Calculate smoothened carbon sheets for every frame; also calculate nominal interfacial
    # separation for solid-liquid interface

    time_start_1 = time.time()
    
    sheet_Nx = int(np.ceil(6.0 * cell_params[0] / CARBON_RADIUS))
    sheet_Ny = int(np.ceil(6.0 * cell_params[1] / CARBON_RADIUS))
    sheets = np.empty((N_frames, sheet_Nx, sheet_Ny), dtype=float)
    for f in track(range(N_frames), description='Processing graphene sheet...', console=console,
                   transient=True):
        sheets[f] = raw_heightmap(carbons[f], cell_params[0:2], (sheet_Nx, sheet_Ny))
    central_sheet_height = np.mean(sheets, axis=0)[sheet_Nx // 2, sheet_Ny // 2]

    CoM_z = np.mean(waters[:,:,2])
    droplet_roof = find_interface(waters, (0, 0, CoM_z), (0, 0, 1))[2]
    droplet_floor = find_interface(waters, (0, 0, CoM_z), (0, 0, -1))[2]
    console.print(f'Processed graphene sheet in [green]{elapsed_time(time_start_1)}[/green].')

    #----------------------------------------------------------------------------------------------
    # Create output log file

    results_file = open(os.path.join(args.output_dir, 'results.ini'), 'w', encoding='utf-8')
    results_file.write('[General]\n')
    results_file.write(f'No. of frames = {N_frames}\n')
    results_file.write(f'No. of water molecules = {N_water}\n')
    results_file.write(f'Droplet roof [A] = {droplet_roof}\n')
    results_file.write(f'Droplet floor [A] = {droplet_floor}\n')
    results_file.write(f'Droplet CoM z-coordinate [A] = {CoM_z}\n')
    results_file.write(f'Graphene sheet z at origin [A] = {central_sheet_height}\n')
    results_file.write(f'Droplet height [A] = {droplet_roof - droplet_floor}\n')
    results_file.write(f'Nominal interfacial separation [A] = {droplet_floor - central_sheet_height}\n\n')

    #----------------------------------------------------------------------------------------------
    # Calculate time-averaged interface across all frames

    time_start_1 = time.time()
    with console.status('[green]Computing time-averaged interface...'):
        mean_sheet = np.mean(sheets, axis=0)
        mean_heightmap = PeriodicGridInterpolator(cell_params[0:2], mean_sheet)
        spherical_cap = find_spherical_cap(waters, cell_params, mean_heightmap)
        sphere_r = spherical_cap['r']
        sphere_z = spherical_cap['z']
        sphere_a = spherical_cap['a']
        sphere_angle = spherical_cap['angle']

    console.print(f'Computed time-averaged interface in [green]{elapsed_time(time_start_1)}[/green].')
    
    if not args.no_graphics:
        
        time_start_1 = time.time()
        with console.status('[green]Plotting best-fit sphere...'):

            fig, ax = plt.subplots()
            fig.set_size_inches(6, 6)

            if N_frames > 6:
                plot_density_radially_symmetric(waters[::(N_frames//6)], mean_heightmap, ax)
            else:
                plot_density_radially_symmetric(waters, mean_heightmap, ax)

            max_phi = np.arccos(np.clip(-sphere_z / sphere_r, -1.0, 1.0))
            phi = np.linspace(0, max_phi, 100)
            ax.plot(sphere_r * np.sin(phi), sphere_z + (sphere_r * np.cos(phi)), '-',
                    color=(0.9, 0.45, 0.0))
            ax.text(0.99, 0.99, (r'$\theta\;=\;' + f'{sphere_angle:.1f}' + r'\degree$'),
                    ha='right', va='top', transform=ax.transAxes)
            ax.set_xlabel(r'r ($\AA$)')
            ax.set_ylabel(r'z ($\AA$)')
            ax.set_title('Spherical fit over time-averaged droplet')

            fig.savefig(os.path.join(args.output_dir, 'ave_sphere_fit.png'), dpi=(3*fig.dpi),
                        bbox_inches='tight', pad_inches=0.05)
        
        console.print(f'Plotted best-fit sphere in [green]{elapsed_time(time_start_1)}[/green].')

    results_file.write('[Time-Averaged Interface]\n')
    results_file.write(f'Contact angle [deg] = {sphere_angle}\n')
    results_file.write(f'Three-phase line radius [A] = {sphere_a}\n')
    results_file.write(f'Best-fit sphere radius [A] = {sphere_r}\n')
    results_file.write(f'Best-fit sphere z-height [A] = {sphere_z}\n\n')
        
    #----------------------------------------------------------------------------------------------
    # End of program

    final_elapsed_time = elapsed_time(time_start_0)
    results_file.write('[Misc]\n')
    results_file.write('Program type = contact_angle_isotropic\n')
    results_file.write(f'Program version = {__version__}\n')
    results_file.write(f'Program wall time = {final_elapsed_time}\n')
    results_file.close()

    console.print(f'Program completed in [green]{final_elapsed_time}[/green].')
    sys.exit()

#==================================================================================================
# Run from src

if __name__ == "__main__":
    main()
