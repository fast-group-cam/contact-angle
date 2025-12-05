#! /usr/bin/env python

prog_desc_header = '''
===================================================================================================
 This program measures the contact angle of a liquid droplet on a solid surface from a simulated
 trajectory. The input is a file, or list of files (which must be compatible with ASE's file i/o
 formats), describing either a time evolution or a single snapshot of the simulation in the NVE or
 NVT ensemble. The solid surface should be nominally aligned to the xy plane. The contact angle is
 calculated by finding the time-averaged interface for a number of testpoints in randomly-selected
 directions not intersecting the solid surface, fitting a spherical profile to it, and calculating
 the intersection of the sphere to the time-averaged solid surface heightmap. Use as:

 >    python contact_angle.py <input_file(s)> [-o <output_dir>] [--index <index>]
          [--sol_symbol <sol_symbol>] [--sol_number <sol_number>] [--liq_symbol <liq_symbol>]
          [--liq_number <liq_number>] [--delta_t <delta_t>]
          [--time_rescale_factor <time_rescale_factor>]
          [--length_rescale_factor <length_rescale_factor>] [--anisotropic]
          [--N_azimuths <N_azimuths>] [--no-graphics]

 All plots will be saved to an output directory (specified by the -o option), which defaults to
 "contact-angle" within the parent directory that this program is executed from. The --no-graphics
 option disables the generation of these plots. Numerical results are also saved in a results.ini
 file in the output directory.

 By default, the program assumed rotational symmetry around the z-axis, which is useful for the
 average contact angles under the long time limit in axisymmetric systems. The option --anisotropic
 disables this assumption, and causes the program to scan for contact angles in <N_azimuths>
 different azimuthal directions from the droplet centre-of-mass.
===================================================================================================
'''

import sys
import os
import time
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt

import fast_contact_angle_2d.util.io as io
from fast_contact_angle_2d import __version__, elapsed_time
from fast_contact_angle_2d.liquid.coarse_grain import find_interface
from fast_contact_angle_2d.liquid.spherical_cap import find_spherical_cap, find_spherical_cap_aniso
from fast_contact_angle_2d.solid.sheet import time_averaged_heightmap
from fast_contact_angle_2d.util.plot import plot_density_xz_slice, plot_density_radially_symmetric

def main() -> None:

    #----------------------------------------------------------------------------------------------
    # Generate program description and parse input arguments

    prog_desc = ''
    for line in prog_desc_header.splitlines()[2:-1]:
        prog_desc += (line.lstrip(' ') + ' ') if line != '' else '\n\n'

    parser = argparse.ArgumentParser(prog='contact_angle', description=prog_desc,
                                     usage='%(prog)s input_file(s) [options]',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file', nargs='+',
                        help='input file(s) to read data from')
    parser.add_argument('--index', default=':', dest='index',
                        help='index or slice of indices to take from each input file')
    parser.add_argument('-o', '--output', default='contact-angle', dest='output_dir',
                        help='output folder to save results and graphical outputs to')
    parser.add_argument('--sol_symbol', type=str, nargs='*', default=None, dest='sol_symbol',
                        help='atomic symbol(s) to interpret as solid particles')
    parser.add_argument('--sol_number', type=int, nargs='*', default=None, dest='sol_number',
                        help='atomic number(s) to interpret as solid particles')
    parser.add_argument('--liq_symbol', type=str, nargs='*', default=None, dest='liq_symbol',
                        help='atomic symbol(s) to interpret as liquid particles')
    parser.add_argument('--liq_number', type=int, nargs='*', default=None, dest='liq_number',
                        help='atomic number(s) to interpret as liquid particles')
    parser.add_argument('--delta_t', type=float, default=None, dest='delta_t',
                        help='time interval between trajectory frames (in ps)')
    parser.add_argument('--time_rescale_factor', type=float, default=None, dest='time_rescale_factor',
                        help='rescaling factor for automatically detected timesteps (to get to ps)')
    parser.add_argument('--length_rescale_factor', type=float, default=1.0, dest='length_rescale_factor',
                        help='rescaling factor for lengths (to get to angstroms)')
    parser.add_argument('--anisotropic', '--aniso', action='store_true', dest='anisotropic',
                        help='set the script to anisotropic mode')
    parser.add_argument('--N_azimuths', type=int, default=360, dest='N_azimuths',
                        help='number of azimuthal directions to scan, if in anisotropic mode')
    parser.add_argument('--no-graphics', action='store_true', dest='no_graphics',
                        help='disables rendering of graphics (and speeds up the script)')
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
    
    if args.anisotropic:
        if args.N_azimuths < 12:
            raise RuntimeError(f'N_azimuths ({args.N_azimuths}) must be positive and at least 12.')
        N_azi = 12 * int(round(args.N_azimuths / 12)) # Number of azimuthal directions must be a multiple of 12

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
    liq = trajectory['liq']
    sol = trajectory['sol']
    N_frames = liq.shape[0]
    N_liq = liq.shape[1]
    N_sol = sol.shape[1]
    timestep = trajectory.get('delta_t', 1.0)

    if N_liq == 0:
        raise RuntimeError('Found no liquid particles! Make sure either "liq_symbol" or "liq_number"'
                           ' are specified correctly.')
    if N_sol == 0:
        raise RuntimeError('Found no solid particles! Make sure either "sol_symbol" or "sol_number"'
                           ' are specified correctly.')

    print(f'Read {N_frames} frames from {file_msg} in {elapsed_time(time_start_0)}.')
    
    #----------------------------------------------------------------------------------------------
    # Calculate time-averaged heightmap for solid surface; also calculate nominal droplet roof &
    # floor z-coordinates

    time_start_1 = time.time()

    sol_interspacing = np.sqrt(cell_params[0] * cell_params[1] / N_sol)
    sheet_Nx = int(np.ceil(6.0 * cell_params[0] / sol_interspacing))
    sheet_Ny = int(np.ceil(6.0 * cell_params[1] / sol_interspacing))

    mean_heightmap = time_averaged_heightmap(cell_params, sol, (sheet_Nx, sheet_Ny),
                                             margin=(3.0 * sol_interspacing))
    central_sheet_height = mean_heightmap(np.zeros(2))[0]
    CoM_z = np.mean(liq[:,:,2])
    droplet_roof = find_interface(liq, (0, 0, CoM_z), (0, 0, 1))[2]
    droplet_floor = find_interface(liq, (0, 0, CoM_z), (0, 0, -1))[2]
    print(f'Processed solid surface in {elapsed_time(time_start_1)}.')

    #----------------------------------------------------------------------------------------------
    # Create output log file

    results_file = open(os.path.join(args.output_dir, 'results.ini'), 'w', encoding='utf-8')
    results_file.write('[General]\n')
    results_file.write(f'No. of frames = {N_frames}\n')
    results_file.write(f'Timestep per frame [ps] = {timestep}\n')
    results_file.write(f'No. of liquid particles = {N_liq}\n')
    results_file.write(f'No. of solid particles = {N_sol}\n')
    results_file.write(f'Droplet roof [A] = {droplet_roof}\n')
    results_file.write(f'Droplet floor [A] = {droplet_floor}\n')
    results_file.write(f'Droplet CoM z-coordinate [A] = {CoM_z}\n')
    results_file.write(f'Solid surface z at origin [A] = {central_sheet_height}\n')
    results_file.write(f'Droplet height [A] = {droplet_roof - droplet_floor}\n')
    results_file.write(f'Nominal interfacial separation [A] = {droplet_floor - central_sheet_height}\n\n')

    #----------------------------------------------------------------------------------------------
    # Calculate contact angle for best-fit sphere of time-averaged liquid interface
    # (Anisotropic case)

    if args.anisotropic:

        time_start_1 = time.time()

        spherical_cap = find_spherical_cap_aniso(liq, cell_params, mean_heightmap)
        sphere_r = spherical_cap['r']
        sphere_c = spherical_cap['c']
        dh_dx = mean_heightmap.derivative(0)
        dh_dy = mean_heightmap.derivative(1)

        azi = np.linspace(0, 2.0 * np.pi, N_azi, endpoint=False)
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

        print(f'Computed time-averaged interface in {elapsed_time(time_start_1)}.')
    
        if not args.no_graphics:

            time_start_1 = time.time()
            fig, ax = plt.subplots(2, 3)
            fig.set_size_inches(15, 5)
            interval = max(int(N_frames * N_liq / 2e5), 1)

            for i in range(6):

                idx = i * (N_azi // 12)
                angle = i * 30.0
                rot_matrix = np.array(((np.cos(angle * np.pi / 180),  np.sin(angle * np.pi / 180), 0.0),
                                       (-np.sin(angle * np.pi / 180), np.cos(angle * np.pi / 180), 0.0),
                                       (0.0,                          0.0,                         1.0)))
                
                plot_density_xz_slice(liq[::interval], mean_heightmap, ax[i // 3][i % 3],
                                      azi=angle, show_interface=True)

                rot_sphere_c = rot_matrix @ sphere_c
                phi_l = np.arctan2(sphere_intersections[idx + (N_azi // 2), 0], sphere_intersections[idx + (N_azi // 2), 1])
                phi_r = np.arctan2(sphere_intersections[idx, 0], sphere_intersections[idx, 1])
                phi = np.linspace(-phi_l, phi_r, 180, endpoint=True)
                ax[i // 3][i % 3].plot(rot_sphere_c[0] + (sphere_r * np.sin(phi)), sphere_c[2] + (sphere_r * np.cos(phi)),
                                    '--', color=(0.0, 0.75, 0.0))
                ax[i // 3][i % 3].plot((rot_sphere_c[0],), (sphere_c[2],), '.', color=(0.0, 0.75, 0.0))
                ax[i // 3][i % 3].plot((0.0,), (CoM_z,), '.', color=(1.0, 0.0, 1.0))
                ax[i // 3][i % 3].text(0.01, 0.99, (r'$\theta_{left}\;=\;' +
                                                    f'{sphere_angles[idx + (N_azi // 2)]:.1f}' +
                                                    r'\degree$' + '\n' + r'$\theta_{right}\;=\;' +
                                                    f'{sphere_angles[idx]:.1f}' + r'\degree$'),
                                                    ha='left', va='top',
                                                    transform=ax[i // 3][i % 3].transAxes)

                ax[i // 3][i % 3].set_title(r'$\varphi\;=\;' + f'{angle:.0f}' + r'\degree$')

            fig.suptitle('Azimuthal cross-sections of time-averaged droplet')
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, 'cross_sections.png'), dpi=(3*fig.dpi),
                        bbox_inches='tight', pad_inches=0.05)
            
            print(f'Plotted cross-sections of time-averaged density functions in {elapsed_time(time_start_1)}.')

        results_file.write('[Time-Averaged Interface]\n')
        results_file.write(f'Contact angle [deg] = {np.mean(sphere_angles)}\n')
        results_file.write(f'Three-phase line radius [A] = {np.mean(sphere_intersections[:,0])}\n')
        results_file.write(f'Best-fit sphere radius [A] = {sphere_r}\n')
        results_file.write(f'Best-fit sphere z-height [A] = {sphere_c[2]}\n')
        results_file.write(f'Best-fit sphere x-coords [A] = {sphere_c[0]}\n')
        results_file.write(f'Best-fit sphere y-coords [A] = {sphere_c[1]}\n')
        results_file.write(f'Dist. of contact angles, min [deg] = {np.min(sphere_angles)}\n')
        results_file.write(f'Dist. of contact angles, max [deg] = {np.max(sphere_angles)}\n')
        results_file.write(f'Dist. of contact angles, std [deg] = {np.std(sphere_angles)}\n\n')

        np.save(os.path.join(args.output_dir, 'sphere_angles.npy'), sphere_angles)

    #----------------------------------------------------------------------------------------------
    # Calculate contact angle for best-fit sphere of time-averaged liquid interface
    # (Isotropic case)

    else:

        time_start_1 = time.time()

        spherical_cap = find_spherical_cap(liq, cell_params, mean_heightmap)
        sphere_r = spherical_cap['r']
        sphere_z = spherical_cap['z']
        sphere_a = spherical_cap['a']
        sphere_angle = spherical_cap['angle']

        print(f'Computed time-averaged interface in {elapsed_time(time_start_1)}.')
        
        if not args.no_graphics:
            
            time_start_1 = time.time()
            fig, ax = plt.subplots()
            fig.set_size_inches(6, 6)

            if N_frames > 6:
                plot_density_radially_symmetric(liq[::(N_frames//6)], mean_heightmap, ax)
            else:
                plot_density_radially_symmetric(liq, mean_heightmap, ax)

            max_phi = np.arccos(np.clip(-sphere_z / sphere_r, -1.0, 1.0))
            phi = np.linspace(0, max_phi, 100)
            ax.plot(sphere_r * np.sin(phi), sphere_z + (sphere_r * np.cos(phi)), '-',
                    color=(0.9, 0.45, 0.0))
            ax.text(0.99, 0.99, (r'$\theta\;=\;' + f'{sphere_angle:.1f}' + r'\degree$'),
                    ha='right', va='top', transform=ax.transAxes)
            ax.set_xlabel(r'r ($\AA$)')
            ax.set_ylabel(r'z ($\AA$)')
            ax.set_title('Spherical fit over time-averaged droplet')

            fig.savefig(os.path.join(args.output_dir, 'radially_symmetrized.png'), dpi=(3*fig.dpi),
                        bbox_inches='tight', pad_inches=0.05)
            
            print(f'Plotted radially-symmetrized best-fit sphere in {elapsed_time(time_start_1)}.')

            time_start_1 = time.time()
            fig, ax = plt.subplots(2, 3)
            fig.set_size_inches(15, 5)
            interval = max(int(N_frames * N_liq / 2e5), 1)
            phi = np.linspace(-max_phi, max_phi, 200)
            for i in range(6):
                plot_density_xz_slice(liq[::interval], mean_heightmap, ax[i // 3][i % 3],
                                      azi=(i * 30.0), show_interface=True)
                ax[i // 3][i % 3].plot(sphere_r * np.sin(phi), sphere_z + (sphere_r * np.cos(phi)),
                                       '--', color=(0.0, 0.75, 0.0))
                ax[i // 3][i % 3].plot((0.0,), (sphere_z,), '.', color=(0.0, 0.75, 0.0))
                ax[i // 3][i % 3].plot((0.0,), (CoM_z,), '.', color=(1.0, 0.0, 1.0))
                ax[i // 3][i % 3].text(0.99, 0.99, (r'$\theta\;=\;' + f'{sphere_angle:.1f}' +
                                                    r'\degree$'), ha='right', va='top',
                                                    transform=ax[i // 3][i % 3].transAxes)
                ax[i // 3][i % 3].set_title(r'$\varphi\;=\;' + f'{(i * 30.0):.0f}' + r'\degree$')

            fig.suptitle('Azimuthal cross-sections of time-averaged droplet')
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, 'cross_sections.png'), dpi=(3*fig.dpi),
                        bbox_inches='tight', pad_inches=0.05)
            
            print(f'Plotted cross-sections of time-averaged density functions in {elapsed_time(time_start_1)}.')

        results_file.write('[Time-Averaged Interface]\n')
        results_file.write(f'Contact angle [deg] = {sphere_angle}\n')
        results_file.write(f'Three-phase line radius [A] = {sphere_a}\n')
        results_file.write(f'Best-fit sphere radius [A] = {sphere_r}\n')
        results_file.write(f'Best-fit sphere z-height [A] = {sphere_z}\n\n')
        
    #----------------------------------------------------------------------------------------------
    # End of program

    final_elapsed_time = elapsed_time(time_start_0)
    results_file.write('[Misc]\n')
    results_file.write('Program type = contact_angle\n')
    results_file.write(f'Program version = {__version__}\n')
    results_file.write(f'Program wall time = {final_elapsed_time}\n')
    results_file.close()

    print(f'Program completed in {final_elapsed_time}.')
    sys.exit()

#==================================================================================================
# Run from src

if __name__ == "__main__":
    main()
