#! /usr/bin/env python

prog_desc_header = '''
===================================================================================================
 Python script which takes a trajectory of a water droplet on graphene, displays it in a custom
 visualization, and renders the movie to a MP4 file using FFMPEG. Use as:

 >    python make-droplet-movie.py <input_file(s)> [-o <output_file>] [--index <index>]
          [--fps <framerate>]

 The trajectory is assumed to be in the NVT ensemble with periodic boundary conditions (i.e. the
 simulation box lengths are fixed).
===================================================================================================
'''

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from droplet_graphene_analysis.util import elapsed_time, read_droplet_trajectory
from droplet_graphene_analysis.util.droplet.plot import (plot_density_xz_slice,
                                                         update_density_xz_slice)

def main() -> None:

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
                                     usage='%(prog)s input_file [options]',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file', nargs='+',
                        help='input file(s) to read data from')
    parser.add_argument('-o', '--output', default='movie.mp4', dest='output',
                        help='output filename to save the movie to')
    parser.add_argument('--index', default=':', dest='index',
                        help='slice of indices to take from each input file')
    parser.add_argument('--fps', type=float, default=30, dest='framerate',
                        help='framerate to render the movie at')
    args = parser.parse_args()

    for file in args.input_file:
        if not os.path.isfile(file):
            raise RuntimeError(f'File "{file}" not found.')
    if os.path.isfile(args.output):
        os.remove(args.output)

    #----------------------------------------------------------------------------------------------
    # Read input file and iterate through frames

    if len(args.input_file) == 1:
        print(f'Reading "{args.input_file[0]}"...', end='')
    else:
        print(f'Reading {len(args.input_file)} files...', end='')
    time_start_0 = time.time()
    time_start_1 = time.time()
    cell_params, oxygens, carbons, hydrogens = read_droplet_trajectory(args.input_file,
                                                                       index=args.index)
    N_frames = carbons.shape[0]
    print(f'read {N_frames} frames in {elapsed_time(time_start_1)}.')

    #----------------------------------------------------------------------------------------------
    # Generate movie

    print('Prerender processing...', end='')

    NC = carbons.shape[1]
    NH = hydrogens.shape[1]
    NO = oxygens.shape[1]

    scatterpoints = np.zeros((N_frames, NC + NH + NO, 3), dtype=float)
    scattersizes = np.zeros((NC + NH + NO,), dtype=float)
    scattercolors = np.zeros((N_frames, NC + NH + NO, 3), dtype=float)

    scatterpoints[:, 0:NC] = carbons
    scattersizes[0:NC] = (2 * RADIUS_CARBON)**2
    carbon_width = np.max(carbons[:,:,2]) - np.min(carbons[:,:,2])
    z_devs = 0.6 * carbons[:,:,2] / carbon_width
    scattercolors[:, 0:NC, 0] = 0.5 + z_devs
    scattercolors[:, 0:NC, 1] = 0.5 - (np.square(z_devs) / 0.3)
    scattercolors[:, 0:NC, 2] = 0.5 - z_devs

    scatterpoints[:, NC:(NC + NH)] = hydrogens
    scattersizes[NC:(NC + NH)] = (2 * RADIUS_HYDROGEN)**2
    scattercolors[:, NC:(NC + NH)] = np.array((0.9, 0.9, 0.9))

    scatterpoints[:, (NC + NH):(NC + NH + NO)] = oxygens
    scattersizes[(NC + NH):(NC + NH + NO)] = (2 * RADIUS_OXYGEN)**2
    scattercolors[:, (NC + NH):(NC + NH + NO)] = np.array((1, 0, 0))

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

    frame_label = ax3d.annotate('Frame #0', (0, 1), xycoords='axes fraction', ha='left', va='top')

    render2d = plot_density_xz_slice(oxygens[0], carbons[0], ax2d, show_interface=True)
    ax2d.set_xlim([np.min(scatterpoints[:,:,0]), np.max(scatterpoints[:,:,0])])
    ax2d.set_ylim([np.min(scatterpoints[:,:,2]), np.max(scatterpoints[:,:,2])])
    ax2d.set_xlabel(r'x [$\AA$]')
    ax2d.set_ylabel(r'z [$\AA$]')
    ax2d.set_title('Density plot along xz slice')

    def update(frame):
        render3d._offsets3d = scatterpoints[frame].T
        render3d.set_facecolor(scattercolors[frame])
        frame_label.set(text=f'Frame #{frame}')
        update_density_xz_slice(oxygens[frame], carbons[frame], render2d)
        print(f'    - rendered frame {frame} / {N_frames}...')
        return (render3d, frame_label, *render2d)

    print('complete, now rendering...')
    time_start_1 = time.time()
    animation = anim.FuncAnimation(fig=fig, func=update, frames=N_frames,
                                   interval=int(np.round(1000 / args.framerate)), blit=False)
    animation.save(filename=args.output, writer='ffmpeg')
    print(f'...done in {elapsed_time(time_start_1)}.')
    print(f'Program completed in {elapsed_time(time_start_0)}.')

#==================================================================================================
# Run from src

if __name__ == "__main__":
    main()
