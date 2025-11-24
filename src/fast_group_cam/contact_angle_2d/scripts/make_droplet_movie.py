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

from rich.console import Console
from rich.progress import Progress
from droplet_graphene_analysis.util import elapsed_time, read_droplet_trajectory
from droplet_graphene_analysis.util.droplet.plot import (plot_density_xz_slice,
                                                         update_density_xz_slice)

def main() -> None:

    #----------------------------------------------------------------------------------------------
    # Script constants

    RADIUS_CARBON = 0.73
    RADIUS_HYDROGEN = 0.31
    RADIUS_OXYGEN = 0.66
    MARGIN = 12.5
    PLOT_SPACING = 5.0

    #----------------------------------------------------------------------------------------------
    # Generate program description and parse input arguments

    prog_desc = ''
    for line in prog_desc_header.splitlines()[2:-1]:
        prog_desc += (line.lstrip(' ') + ' ') if line != '' else '\n\n'
    
    parser = argparse.ArgumentParser(prog='make_droplet_movie', description=prog_desc,
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

    console = Console(highlight=False)

    #----------------------------------------------------------------------------------------------
    # Read input file and save coordinates

    file_msg = (f'"{args.input_file[0]}"' if len(args.input_file) == 1 else
                f'{len(args.input_file)} files')
    
    time_start_0 = time.time()
    with console.status(f'[green]Reading {file_msg}...'):
        cell_params, oxygens, carbons, hydrogens = read_droplet_trajectory(args.input_file, index=args.index)
        cell_boundary_x = cell_params[0] / 2
        cell_boundary_y = cell_params[1] / 2
        N_frames = oxygens.shape[0]

    console.print(f'Read [magenta]{N_frames} frames[/magenta] from [cyan]{file_msg}[/cyan] in ' +
                  f'[green]{elapsed_time(time_start_0)}[/green].')

    #----------------------------------------------------------------------------------------------
    # Generate movie

    time_start_1 = time.time()
    console.print('Prerender processing...', end='')

    NC = carbons.shape[1]
    NM = 2 * NC
    NH = hydrogens.shape[1]
    NO = oxygens.shape[1]

    scatterpoints = np.zeros((N_frames, NC + NM + NH + NO, 3), dtype=float)
    scattersizes = np.zeros((NC + NM + NH + NO,), dtype=float)
    scattercolors = np.zeros((N_frames, NC + NM + NH + NO, 4), dtype=float)

    scatterpoints[:, 0:NC] = carbons
    scattersizes[0:(NC + NM)] = (2 * RADIUS_CARBON)**2
    carbon_width = max(np.max(carbons[:,:,2]) - np.min(carbons[:,:,2]), 1.0)
    z_devs = 0.6 * carbons[:,:,2] / carbon_width
    scattercolors[:, 0:NC, 0] = np.clip(0.5 + z_devs, a_min=0.0, a_max=1.0)
    scattercolors[:, 0:NC, 1] = np.clip(0.5 - (np.square(z_devs) / 0.3), a_min=0.0, a_max=1.0)
    scattercolors[:, 0:NC, 2] = np.clip(0.5 - z_devs, a_min=0.0, a_max=1.0)
    scattercolors[:, 0:NC, 3] = np.ones(NC)

    for t in range(N_frames):

        ghosts = carbons[t, carbons[t,:,0] > (cell_boundary_x - MARGIN)]
        ghosts[:,0] -= cell_params[0]
        if ghosts.shape[0] > NM:
            ghosts = ghosts[np.argpartition(-ghosts[:,0], NM)[0:NM]]
        idx = ghosts.shape[0]
        scatterpoints[t, NC:(NC + idx)] = ghosts
        z_devs = 0.6 * ghosts[:,2] / carbon_width
        scattercolors[t, NC:(NC + idx), 0] = np.clip(0.5 + z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, NC:(NC + idx), 1] = np.clip(0.5 - (np.square(z_devs) / 0.3), a_min=0.0, a_max=1.0)
        scattercolors[t, NC:(NC + idx), 2] = np.clip(0.5 - z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, NC:(NC + idx), 3] = np.clip(1.0 + ((cell_boundary_x + ghosts[:,0]) / MARGIN), a_min=0.0, a_max=1.0)

        ghosts = carbons[t, carbons[t,:,0] < (MARGIN - cell_boundary_x)]
        ghosts[:,0] += cell_params[0]
        if ghosts.shape[0] > (NM - idx):
            ghosts = ghosts[np.argpartition(ghosts[:,0], NM - idx)[0:(NM - idx)]]
        new_idx = idx + ghosts.shape[0]
        scatterpoints[t, (NC + idx):(NC + new_idx)] = ghosts
        z_devs = 0.6 * ghosts[:,2] / carbon_width
        scattercolors[t, (NC + idx):(NC + new_idx), 0] = np.clip(0.5 + z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 1] = np.clip(0.5 - (np.square(z_devs) / 0.3), a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 2] = np.clip(0.5 - z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 3] = np.clip(1.0 + ((cell_boundary_x - ghosts[:,0]) / MARGIN), a_min=0.0, a_max=1.0)

        idx = new_idx
        ghosts = carbons[t, carbons[t,:,1] > (cell_boundary_y - MARGIN)]
        ghosts[:,1] -= cell_params[1]
        if ghosts.shape[0] > (NM - idx):
            ghosts = ghosts[np.argpartition(-ghosts[:,1], NM - idx)[0:(NM - idx)]]
        new_idx = idx + ghosts.shape[0]
        scatterpoints[t, (NC + idx):(NC + new_idx)] = ghosts
        z_devs = 0.6 * ghosts[:,2] / carbon_width
        scattercolors[t, (NC + idx):(NC + new_idx), 0] = np.clip(0.5 + z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 1] = np.clip(0.5 - (np.square(z_devs) / 0.3), a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 2] = np.clip(0.5 - z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 3] = np.clip(1.0 + ((cell_boundary_y + ghosts[:,1]) / MARGIN), a_min=0.0, a_max=1.0)

        idx = new_idx
        ghosts = carbons[t, carbons[t,:,1] < (MARGIN - cell_boundary_y)]
        ghosts[:,1] += cell_params[1]
        if ghosts.shape[0] > (NM - idx):
            ghosts = ghosts[np.argpartition(ghosts[:,1], NM - idx)[0:(NM - idx)]]
        new_idx = idx + ghosts.shape[0]
        scatterpoints[t, (NC + idx):(NC + new_idx)] = ghosts
        z_devs = 0.6 * ghosts[:,2] / carbon_width
        scattercolors[t, (NC + idx):(NC + new_idx), 0] = np.clip(0.5 + z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 1] = np.clip(0.5 - (np.square(z_devs) / 0.3), a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 2] = np.clip(0.5 - z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 3] = np.clip(1.0 + ((cell_boundary_y - ghosts[:,1]) / MARGIN), a_min=0.0, a_max=1.0)

        idx = new_idx
        corner = np.array((-cell_boundary_x, -cell_boundary_y))
        ghosts = carbons[t, np.sum((carbons[t,:,0:2] - corner)**2, axis=-1) < MARGIN**2]
        ghosts = ghosts[ghosts[:,0] > -cell_boundary_x]
        ghosts = ghosts[ghosts[:,1] > -cell_boundary_y]
        ghosts[:,0] += cell_params[0]
        ghosts[:,1] += cell_params[1]
        dists = np.sqrt(np.sum((ghosts[:,0:2] + corner)**2, axis=-1))
        if ghosts.shape[0] > (NM - idx):
            ghosts = ghosts[np.argpartition(dists, NM - idx)[0:(NM - idx)]]
            dists = np.sqrt(np.sum((ghosts[:,0:2] + corner)**2, axis=-1))
        new_idx = idx + ghosts.shape[0]
        scatterpoints[t, (NC + idx):(NC + new_idx)] = ghosts
        z_devs = 0.6 * ghosts[:,2] / carbon_width
        scattercolors[t, (NC + idx):(NC + new_idx), 0] = np.clip(0.5 + z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 1] = np.clip(0.5 - (np.square(z_devs) / 0.3), a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 2] = np.clip(0.5 - z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 3] = np.clip(1.0 - (dists / MARGIN), a_min=0.0, a_max=1.0)

        idx = new_idx
        corner = np.array((cell_boundary_x, -cell_boundary_y))
        ghosts = carbons[t, np.sum((carbons[t,:,0:2] - corner)**2, axis=-1) < MARGIN**2]
        ghosts = ghosts[ghosts[:,0] < cell_boundary_x]
        ghosts = ghosts[ghosts[:,1] > -cell_boundary_y]
        ghosts[:,0] -= cell_params[0]
        ghosts[:,1] += cell_params[1]
        dists = np.sqrt(np.sum((ghosts[:,0:2] + corner)**2, axis=-1))
        if ghosts.shape[0] > (NM - idx):
            ghosts = ghosts[np.argpartition(dists, NM - idx)[0:(NM - idx)]]
            dists = np.sqrt(np.sum((ghosts[:,0:2] + corner)**2, axis=-1))
        new_idx = idx + ghosts.shape[0]
        scatterpoints[t, (NC + idx):(NC + new_idx)] = ghosts
        z_devs = 0.6 * ghosts[:,2] / carbon_width
        scattercolors[t, (NC + idx):(NC + new_idx), 0] = np.clip(0.5 + z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 1] = np.clip(0.5 - (np.square(z_devs) / 0.3), a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 2] = np.clip(0.5 - z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 3] = np.clip(1.0 - (dists / MARGIN), a_min=0.0, a_max=1.0)

        idx = new_idx
        corner = np.array((-cell_boundary_x, cell_boundary_y))
        ghosts = carbons[t, np.sum((carbons[t,:,0:2] - corner)**2, axis=-1) < MARGIN**2]
        ghosts = ghosts[ghosts[:,0] > -cell_boundary_x]
        ghosts = ghosts[ghosts[:,1] < cell_boundary_y]
        ghosts[:,0] += cell_params[0]
        ghosts[:,1] -= cell_params[1]
        dists = np.sqrt(np.sum((ghosts[:,0:2] + corner)**2, axis=-1))
        if ghosts.shape[0] > (NM - idx):
            ghosts = ghosts[np.argpartition(dists, NM - idx)[0:(NM - idx)]]
            dists = np.sqrt(np.sum((ghosts[:,0:2] + corner)**2, axis=-1))
        new_idx = idx + ghosts.shape[0]
        scatterpoints[t, (NC + idx):(NC + new_idx)] = ghosts
        z_devs = 0.6 * ghosts[:,2] / carbon_width
        scattercolors[t, (NC + idx):(NC + new_idx), 0] = np.clip(0.5 + z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 1] = np.clip(0.5 - (np.square(z_devs) / 0.3), a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 2] = np.clip(0.5 - z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 3] = np.clip(1.0 - (dists / MARGIN), a_min=0.0, a_max=1.0)

        idx = new_idx
        corner = np.array((cell_boundary_x, cell_boundary_y))
        ghosts = carbons[t, np.sum((carbons[t,:,0:2] - corner)**2, axis=-1) < MARGIN**2]
        ghosts = ghosts[ghosts[:,0] < cell_boundary_x]
        ghosts = ghosts[ghosts[:,1] < cell_boundary_y]
        ghosts[:,0] -= cell_params[0]
        ghosts[:,1] -= cell_params[1]
        dists = np.sqrt(np.sum((ghosts[:,0:2] + corner)**2, axis=-1))
        if ghosts.shape[0] > (NM - idx):
            ghosts = ghosts[np.argpartition(dists, NM - idx)[0:(NM - idx)]]
            dists = np.sqrt(np.sum((ghosts[:,0:2] + corner)**2, axis=-1))
        new_idx = idx + ghosts.shape[0]
        scatterpoints[t, (NC + idx):(NC + new_idx)] = ghosts
        z_devs = 0.6 * ghosts[:,2] / carbon_width
        scattercolors[t, (NC + idx):(NC + new_idx), 0] = np.clip(0.5 + z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 1] = np.clip(0.5 - (np.square(z_devs) / 0.3), a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 2] = np.clip(0.5 - z_devs, a_min=0.0, a_max=1.0)
        scattercolors[t, (NC + idx):(NC + new_idx), 3] = np.clip(1.0 - (dists / MARGIN), a_min=0.0, a_max=1.0)

        scatterpoints[t, (NC + new_idx):(NC + NM)] = np.array((0.0, 0.0, 0.0))
        scattercolors[t, (NC + new_idx):(NC + NM)] = np.array((0.0, 0.0, 0.0, 0.0))


    scatterpoints[:, (NC + NM):(NC + NM + NH)] = hydrogens
    scattersizes[(NC + NM):(NC + NM + NH)] = (2 * RADIUS_HYDROGEN)**2
    scattercolors[:, (NC + NM):(NC + NM + NH)] = np.array((0.9, 0.9, 0.9, 1.0))

    scatterpoints[:, (NC + NM + NH):(NC + NM + NH + NO)] = oxygens
    scattersizes[(NC + NM + NH):(NC + NM + NH + NO)] = (2 * RADIUS_OXYGEN)**2
    scattercolors[:, (NC + NM + NH):(NC + NM + NH + NO)] = np.array((1.0, 0.0, 0.0, 1.0))

    scatteredges = np.zeros((N_frames, NC + NM + NH + NO, 4), dtype=float)
    scatteredges[:,:,3] = scattercolors[:,:,3]

    fig = plt.figure(figsize=(12, 8), dpi=300, constrained_layout=True)
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    ax2d = ax3d.inset_axes([0.5, 0.65, 0.5, 0.25])

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
                            s=scattersizes, facecolors=scattercolors[0], edgecolors=scatteredges[0],
                            depthshade=False, linewidths=0.1)
    ax3d.set_xlim([np.min(scatterpoints[:,:,0]), np.max(scatterpoints[:,:,0])])
    ax3d.set_ylim([np.min(scatterpoints[:,:,1]), np.max(scatterpoints[:,:,1])])
    ax3d.set_zlim([np.min(scatterpoints[:,:,2]), np.max(scatterpoints[:,:,2])])

    frame_label = ax3d.annotate('Frame #0', (0, 0.8), xycoords='axes fraction', ha='left', va='top')

    render2d = plot_density_xz_slice(oxygens[0], carbons[0], ax2d, show_interface=True)
    ax2d.set_xlim([-cell_boundary_x, cell_boundary_x])
    ax2d.set_ylim([np.min(scatterpoints[:,:,2]) - PLOT_SPACING, np.max(scatterpoints[:,:,2]) + PLOT_SPACING])
    ax2d.set_xlabel(r'x [$\AA$]')
    ax2d.set_ylabel(r'z [$\AA$]')
    ax2d.set_title('Density plot along xz slice')

    fig.set_layout_engine('none')
    ax3d.set_position([-0.08, -0.16, 1.16, 1.16])

    console.print(f'complete in [green]{elapsed_time(time_start_1)}[/green].')
    progress_bar = Progress(console=console, transient=True)
    progress_bar.start()
    progress_bar_task = progress_bar.add_task('Rendering...', total=N_frames)

    def update(frame):
        render3d._offsets3d = scatterpoints[frame].T
        render3d.set_facecolor(scattercolors[frame])
        render3d.set_edgecolor(scatteredges[frame])
        frame_label.set(text=f'Frame #{frame}')
        update_density_xz_slice(oxygens[frame], carbons[frame], render2d)
        progress_bar.update(progress_bar_task, advance=1)
        return (render3d, frame_label, *render2d)

    time_start_1 = time.time()
    animation = anim.FuncAnimation(fig=fig, func=update, frames=N_frames,
                                   interval=int(np.round(1000 / args.framerate)), blit=False)
    animation.save(filename=args.output, writer='ffmpeg')
    progress_bar.stop()
    console.print(f'Rendered in [green]{elapsed_time(time_start_1)}[/green].')
    console.print(f'Program completed in [green]{elapsed_time(time_start_0)}[/green].')

#==================================================================================================
# Run from src

if __name__ == "__main__":
    main()
