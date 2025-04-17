#! /usr/bin/env python

prog_desc_header = '''
===================================================================================================
 Python script which takes a trajectory of a water droplet on graphene, displays it in a custom
 visualization, and renders the movie to a MP4 file using FFMPEG. Use as:

     python make-droplet-movie.py <input_file> [-o <output_file>] [--index <index>]

 The trajectory is assumed to be in the NVT ensemble with periodic boundary conditions (i.e. the
 simulation box lengths are fixed).
===================================================================================================
'''

import os
import time
import argparse
import numpy as np
import ase.io
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from contact_angle.util.droplet.center_coordinates import center_coordinates
from contact_angle.util.droplet.plot import plot_density_xz_slice, update_density_xz_slice

if __name__ == "__main__":

    #----------------------------------------------------------------------------------------------
    # Script constants

    RADIUS_CARBON = 0.73
    RADIUS_HYDROGEN = 0.31
    RADIUS_OXYGEN = 0.66

    #----------------------------------------------------------------------------------------------
    # Generate program description and parse input arguments

    prog_desc = ''
    for line in prog_desc_header.splitlines()[2:-1]:
        prog_desc += (line.lstrip(' ') + ' ') if line != '' else '\n\n'
    
    parser = argparse.ArgumentParser(prog='make-droplet-movie', description=prog_desc,
                                     usage='%(prog)s filename [options]',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file')
    parser.add_argument('-o', '--output', default='movie.mp4')
    parser.add_argument('--index', default=':')
    args = parser.parse_args()

    if os.path.isfile(args.output):
        os.remove(args.output)

    #----------------------------------------------------------------------------------------------
    # Function to convert timings into a nicely formatted string

    def time_diff(time_start):
        time_taken = time.time() - time_start
        if time_taken > 3600:
            hours = int(time_taken / 3600)
            time_taken -= hours * 3600
            mins = int(time_taken / 60)
            time_taken -= mins * 60
            return f'(hh:mm:ss) {hours:02}:{mins:02}:{round(time_taken):02}'
        if time_taken > 60:
            mins = int(time_taken / 60)
            time_taken -= mins * 60
            return f'{mins}mins{round(time_taken):02}s'
        return f'{time_taken:.3f}s'

    time_start_0 = time.time()

    #----------------------------------------------------------------------------------------------
    # Read input file and iterate through frames

    carbons = list()
    hydrogens = list()
    oxygens = list()
    frame_counter = 0
    cell_params = None
    lammps_elem_order = None

    print(f'Reading "{args.input_file}"...')
    time_start_1 = time.time()

    traj = ase.io.iread(args.input_file, index=args.index)
    for atoms in traj:

        # Timing purposes
        time_start_2 = time.time()

        # Get cell parameters
        if cell_params is None:
            cell_params = np.array(atoms.cell.cellpar()[0:3])
            cell_xy = cell_params[0:2]
            cell_z = cell_params[2]

        # Check if atomic numbers are assigned according to LAMMPS ordering or true atomic numbers
        if lammps_elem_order is None:
            element_numbers = np.unique(atoms.numbers)
            if np.array_equal(element_numbers, [1, 2, 3]):
                lammps_elem_order = True
            elif np.array_equal(element_numbers, [1, 6, 8]):
                lammps_elem_order = False
            else:
                raise RuntimeError(f'Unidentified atomic numbers in file "{args.input_file}"!')

        # Reassign if necessary
        if lammps_elem_order:
            atoms.numbers[atoms.numbers == 1] = 6
            atoms.numbers[atoms.numbers == 2] = 1
            atoms.numbers[atoms.numbers == 3] = 8

        # Save positions of atoms to memory
        coords = center_coordinates(atoms, cell_params)
        oxygens.append(coords[0])
        carbons.append(coords[1])
        hydrogens.append(coords[2])

        # Next iteration
        print(f'   - processed frame {frame_counter} in {time_diff(time_start_2)}.')
        frame_counter += 1

    carbons = np.array(carbons)
    hydrogens = np.array(hydrogens)
    oxygens = np.array(oxygens)
    print(f'read {frame_counter} frames from "{args.input_file}" in {time_diff(time_start_1)}.\n')

    #----------------------------------------------------------------------------------------------
    # Generate movie

    print('\nPrerender processing...', end='')

    N_frames = carbons.shape[0]
    N_carbons = carbons.shape[1]
    N_hydrogens = hydrogens.shape[1]
    N_oxygens = oxygens.shape[1]

    scatterpoints = np.zeros((N_frames, N_carbons + N_hydrogens + N_oxygens, 3), dtype=float)
    scattersizes = np.zeros((N_carbons + N_hydrogens + N_oxygens,), dtype=float)
    scattercolors = np.zeros((N_frames, N_carbons + N_hydrogens + N_oxygens, 3), dtype=float)

    scatterpoints[:, 0:N_carbons] = carbons
    scattersizes[0:N_carbons] = (2 * RADIUS_CARBON)**2
    carbon_width = np.max(carbons[:,:,2]) - np.min(carbons[:,:,2])
    z_devs = 0.6 * carbons[:,:,2] / carbon_width
    scattercolors[:, 0:N_carbons, 0] = 0.5 + z_devs
    scattercolors[:, 0:N_carbons, 1] = 0.5 - (np.square(z_devs) / 0.3)
    scattercolors[:, 0:N_carbons, 2] = 0.5 - z_devs

    scatterpoints[:, N_carbons:(N_carbons + N_hydrogens)] = hydrogens
    scattersizes[N_carbons:(N_carbons + N_hydrogens)] = (2 * RADIUS_HYDROGEN)**2
    scattercolors[:, N_carbons:(N_carbons + N_hydrogens)] = np.array((0.9, 0.9, 0.9))

    scatterpoints[:, (N_carbons + N_hydrogens):(N_carbons + N_hydrogens + N_oxygens)] = oxygens
    scattersizes[(N_carbons + N_hydrogens):(N_carbons + N_hydrogens + N_oxygens)] = (2 * RADIUS_OXYGEN)**2
    scattercolors[:, (N_carbons + N_hydrogens):(N_carbons + N_hydrogens + N_oxygens)] = np.array((1, 0, 0))

    fig = plt.figure(figsize=(14, 7), dpi=300)
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)

    ax3d.set_xlim([np.min(scatterpoints[:,:,0]), np.max(scatterpoints[:,:,0])])
    ax3d.set_ylim([np.min(scatterpoints[:,:,1]), np.max(scatterpoints[:,:,1])])
    ax3d.set_zlim([np.min(scatterpoints[:,:,2]), np.max(scatterpoints[:,:,2])])
    ax3d.view_init(elev=15, azim=45, roll=0)
    ax3d.set_proj_type('ortho')
    ax3d.set_axis_off()
    ax3d.set_aspect('equal')

    from mpl_toolkits.mplot3d import proj3d
    def transform(x, y, z):
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax3d.get_proj())
        return ax3d.transData.transform((x2, y2))
    magic_matrix = np.zeros((3, 3), dtype=float)
    magic_matrix[(0,1),0] = transform(1, 0, 0) - transform(0, 0, 0)
    magic_matrix[(0,1),1] = transform(0, 1, 0) - transform(0, 0, 0)
    magic_matrix[(0,1),2] = transform(0, 0, 1) - transform(0, 0, 0)
    eigv_x, eigv_y, _ = np.linalg.eigvals(magic_matrix)
    lscale = np.sqrt((eigv_x * eigv_y).real)
    #scattersizes *= (lscale**2)
    scattersizes *= lscale

    render3d = ax3d.scatter(scatterpoints[0,:,0], scatterpoints[0,:,1], scatterpoints[0,:,2],
                            s=scattersizes, c=scattercolors[0], depthshade=False, linewidth=0.1)
    render3d.set_edgecolor('black')

    render2d = plot_density_xz_slice(oxygens[0], carbons[0], ax2d, show_interface=True)
    ax2d.set_xlim([np.min(scatterpoints[:,:,0]), np.max(scatterpoints[:,:,0])])
    ax2d.set_ylim([np.min(scatterpoints[:,:,2]), np.max(scatterpoints[:,:,2])])
    ax2d.set_xlabel(r'x [$\AA$]')
    ax2d.set_ylabel(r'z [$\AA$]')
    ax2d.set_title('Density plot along xz slice')

    def update(frame):
        render3d._offsets3d = scatterpoints[frame].T
        render3d.set_facecolor(scattercolors[frame])
        update_density_xz_slice(oxygens[frame], carbons[frame], render2d)
        print(f'    - rendered frame {frame} / {N_frames}...')
        return (render3d, *render2d)

    print('complete, now rendering...')
    render_time_start = time.time()
    animation = anim.FuncAnimation(fig=fig, func=update, frames=N_frames, interval=30, blit=False)
    animation.save(filename=args.output, writer='ffmpeg')
    print(f'...done in {time_diff(render_time_start)}.')
    print(f'\nProgram completed in {time_diff(time_start_0)}.')

