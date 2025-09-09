#! /usr/bin/env python

prog_desc_header = '''
===================================================================================================
 A variation of the contact_angle program, which calculates only the time-averaged spherical cap
 angle, but in a generally anisotropic manner. Use as:

 >    python contact_angle_anisotropic.py <input_file(s)> [--index <index>]
          [--N_azimuths <N_azimuths] [-o <output_dir>] [--no-graphics]

 All plots will be saved to an output directory (specified by the -o option), which defaults to
 "contact-angle" within the parent directory that this program is executed from. The --no-graphics
 option disables the generation of these plots. Numerical results are also saved in a log.txt file
 in the output directory.
===================================================================================================
'''

import sys
import os
import time
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt

from rich.console import Console
from rich.progress import track
from scipy.interpolate import RegularGridInterpolator
from droplet_graphene_analysis.util import elapsed_time, read_droplet_trajectory
from droplet_graphene_analysis.util.droplet import find_interface
from droplet_graphene_analysis.util.droplet.contact_angle import find_spherical_cap_aniso
from droplet_graphene_analysis.util.droplet.plot import plot_density_xz_slice
from droplet_graphene_analysis.util.graphene import regularized_heightmap

def main() -> None:

    #----------------------------------------------------------------------------------------------
    # Script default parameters

    N_AZIMUTHS = 360 # Number of azimuthal directions to analyze

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
    parser.add_argument('--N_azimuths', type=int, default=N_AZIMUTHS, dest='N_azimuths',
                        help='number of azimuthal angles to analyze per frame')
    parser.add_argument('-o', '--output', default='contact-angle-aniso', dest='output_dir',
                        help='output folder to save log and graphical outputs to')
    parser.add_argument('--no-graphics', action='store_true', dest='no_graphics',
                        help='disables rendering of graphics (and speeds up the script)')
    args = parser.parse_args()

    for file in args.input_file:
        if not os.path.isfile(file):
            raise RuntimeError(f'File "{file}" not found.')

    if args.N_azimuths < 12:
        raise RuntimeError(f'N_azimuths ({args.N_azimuths}) must be positive and at least 12.')

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    # Number of azimuthal directions must be a multiple of 12
    N_azi = 12 * int(round(args.N_azimuths / 12))

    console = Console(highlight=False)

    #----------------------------------------------------------------------------------------------
    # Helper functions

    def progress_bar_iter(N_iter, desc):
        return track(range(N_iter), description=desc, console=console, transient=True)
    
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
    sheet_dx = cell_params[0] / sheet_Nx
    sheet_dy = cell_params[1] / sheet_Ny
    sheet_gridx = np.linspace((sheet_dx - cell_params[0]) / 2.0, (cell_params[0] - sheet_dx) / 2.0, sheet_Nx)
    sheet_gridy = np.linspace((sheet_dy - cell_params[1]) / 2.0, (cell_params[1] - sheet_dy) / 2.0, sheet_Ny)
    sheets = np.empty((N_frames, sheet_Nx, sheet_Ny), dtype=float)
    for f in progress_bar_iter(N_frames, 'Processing graphene sheet...'):
        sheets[f] = regularized_heightmap(carbons[f], cell_params[0:2], (sheet_Nx, sheet_Ny))
    central_sheet_height = np.mean(sheets, axis=0)[sheet_Nx // 2, sheet_Ny // 2]

    CoM = np.mean(waters, axis=(0,1))
    droplet_roof = find_interface(waters, (0, 0, CoM[2]), (0, 0, 1))[2]
    droplet_floor = find_interface(waters, (0, 0, CoM[2]), (0, 0, -1))[2]
    console.print(f'Processed graphene sheet in [green]{elapsed_time(time_start_1)}[/green].')

    #----------------------------------------------------------------------------------------------
    # Create output log file

    log_file = open(os.path.join(args.output_dir, 'log.txt'), 'w', encoding='utf-8')
    log_file.write('-------------------------\n')
    log_file.write(' General\n')
    log_file.write('-------------------------\n\n')
    log_file.write(f'No. of frames = {N_frames}\n')
    log_file.write(f'No. of water molecules = {N_water}\n\n')
    log_file.write(f'Droplet roof = {droplet_roof} [A]\n')
    log_file.write(f'Droplet floor = {droplet_floor} [A]\n')
    log_file.write(f'Graphene sheet z at origin = {central_sheet_height} [A]\n\n')
    log_file.write(f'Droplet height = {droplet_roof - droplet_floor} [A]\n')
    log_file.write(f'Nominal interfacial separation = {droplet_floor - central_sheet_height} [A]\n\n')

    #----------------------------------------------------------------------------------------------
    # Calculate time-averaged interface across all frames

    time_start_1 = time.time()
    with console.status('[green]Computing time-averaged interface...'):

        mean_sheet = np.mean(sheets, axis=0)
        mean_heightmap = RegularGridInterpolator((sheet_gridx, sheet_gridy), mean_sheet)
        dh_dx = (np.roll(mean_sheet, -1, axis=0) - np.roll(mean_sheet, 1, axis=0)) / (2 * sheet_dx)
        dh_dy = (np.roll(mean_sheet, -1, axis=1) - np.roll(mean_sheet, 1, axis=1)) / (2 * sheet_dy)
        dh_dx = RegularGridInterpolator((sheet_gridx, sheet_gridy), dh_dx)
        dh_dy = RegularGridInterpolator((sheet_gridx, sheet_gridy), dh_dy)

        sphere_results = find_spherical_cap_aniso(waters, cell_params, mean_heightmap)
        sphere_r = sphere_results['r']
        sphere_c = sphere_results['c']

        azi = np.linspace(0, 2 * np.pi, N_azi, endpoint=False)
        search_directions = np.c_[np.cos(azi), np.sin(azi)]
        sphere_intersections = np.empty((N_azi, 2), dtype=float)
        sphere_angles = np.empty(N_azi, dtype=float)
        for i in range(N_azi):
            search_dir = search_directions[i]
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    floor = mean_heightmap(sphere_c[0:2])[0]
                    for _ in range(10):
                        a = np.sqrt((sphere_r**2) - ((sphere_c[2] - floor)**2))
                        test_pt = (a * search_dir) + sphere_c[0:2]
                        floor = mean_heightmap(test_pt)[0]
                    a = np.sqrt((sphere_r**2) - ((sphere_c[2] - floor)**2))
                    test_pt = (a * search_dir) + sphere_c[0:2]
                    angle = 90.0 + (np.arcsin((sphere_c[2] - floor) / sphere_r) * 180.0 / np.pi)
                    local_grad = (dh_dx(test_pt)[0] * search_dir[0]) + (dh_dy(test_pt)[0] * search_dir[1])
                    angle += np.arctan(local_grad) * 180.0 / np.pi
                except RuntimeWarning:
                    a = 0.0
                    angle = (180.0 if sphere_c[2] > 0.0 else 0.0)
            sphere_intersections[i,0] = a
            sphere_intersections[i,1] = floor - sphere_c[2]
            sphere_angles[i] = angle

    console.print(f'Computed time-averaged interface in [green]{elapsed_time(time_start_1)}[/green].')
    
    if not args.no_graphics:

        fig, ax = plt.subplots(2, 3)
        fig.set_size_inches(15, 5)
        interval = max(int(N_frames * N_water / 2e5), 1)

        time_start_1 = time.time()
        for i in progress_bar_iter(6, 'Plotting time-averaged density functions...'):

            idx = i * (N_azi // 12)
            angle = i * np.pi / 6
            rot_matrix = np.array(((np.cos(angle),  np.sin(angle), 0.0),
                                   (-np.sin(angle), np.cos(angle), 0.0),
                                   (0.0,            0.0,           1.0)))
            rot_waters = np.einsum('kl,ijl->ijk', rot_matrix, waters[::interval])
            rot_carbons = np.einsum('kl,ijl->ijk', rot_matrix, carbons[::interval])
            plot_density_xz_slice(rot_waters, rot_carbons, ax[i // 3][i % 3], show_interface=True)

            rot_sphere_c = rot_matrix @ sphere_c
            phi_l = np.arctan2(sphere_intersections[idx + (N_azi // 2), 0], sphere_intersections[idx + (N_azi // 2), 1])
            phi_r = np.arctan2(sphere_intersections[idx, 0], sphere_intersections[idx, 1])
            phi = np.linspace(-phi_l, phi_r, 180, endpoint=True)
            ax[i // 3][i % 3].plot(rot_sphere_c[0] + (sphere_r * np.sin(phi)), sphere_c[2] + (sphere_r * np.cos(phi)),
                                   '--', color=(0.0, 0.8, 0.0))
            ax[i // 3][i % 3].plot((rot_sphere_c[0],), (sphere_c[2],), '.', color=(0.0, 0.8, 0.0))
            ax[i // 3][i % 3].plot((0.0,), (CoM[2],), '.', color=(1.0, 0.0, 1.0))
            ax[i // 3][i % 3].text(0.01, 0.99, (r'$\theta_{left}\;=\;' +
                                                f'{sphere_angles[idx + (N_azi // 2)]:.1f}' +
                                                r'\degree$' + '\n' + r'$\theta_{right}\;=\;' +
                                                f'{sphere_angles[idx]:.1f}' + r'\degree$'),
                                                ha='left', va='top',
                                                transform=ax[i // 3][i % 3].transAxes)

            ax[i // 3][i % 3].set_title(r'$\varphi\;=\;' + f'{(angle * 180 / np.pi):.0f}' + r'\degree$')

        fig.suptitle('Azimuthal cross-sections of time-averaged droplet')
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, 'ave_cross_sections.png'), dpi=(3*fig.dpi),
                    bbox_inches='tight', pad_inches=0.05)
        
        console.print('Plotted time-averaged density functions in ' +
                    f'[green]{elapsed_time(time_start_1)}[/green].')

    log_file.write('-----------------------------------------\n')
    log_file.write(' Time-averaged anisotropic spherical cap\n')
    log_file.write('-----------------------------------------\n\n')
    log_file.write(f'Center-of-mass z-height = {CoM[2]} [A]\n')
    log_file.write(f'Best-fit sphere z-height = {sphere_c[2]} [A]\n')
    log_file.write(f'Best-fit sphere xy-coords = ({sphere_c[0]}, {sphere_c[1]}) [A]\n')
    log_file.write(f'Best-fit sphere radius = {sphere_r} [A]\n')
    log_file.write(f'Best-fit sphere contact angle = {np.mean(sphere_angles)} [deg]\n')
    log_file.write(f'Best-fit sphere sheet-intersecting radius = {np.mean(sphere_intersections[:,0])} [A]\n\n')
    log_file.write(f'Dist. of sphere contact angles, min = {np.min(sphere_angles)} [deg]\n')
    log_file.write(f'Dist. of sphere contact angles, max = {np.max(sphere_angles)} [deg]\n')
    log_file.write(f'Dist. of sphere contact angles, std = {np.std(sphere_angles)} [deg]\n\n')

    np.save(os.path.join(args.output_dir, 'sphere_angles.npy'), sphere_angles)
        
    #----------------------------------------------------------------------------------------------
    # End of program

    final_elapsed_time = elapsed_time(time_start_0)

    log_file.write('-------\n')
    log_file.write(' Misc.\n')
    log_file.write('-------\n\n')
    log_file.write(f'Program wall time = {final_elapsed_time}\n\n')
    log_file.close()

    print(f'Program completed in {final_elapsed_time}.')
    sys.exit()

#==================================================================================================
# Run from src

if __name__ == "__main__":
    main()
