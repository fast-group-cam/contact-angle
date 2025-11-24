#! /usr/bin/env python

prog_desc_header = '''
===================================================================================================
 This program measures the diffusive motion of a water droplet on a graphene sheet from a simulated
 trajectory. The input is a file, which must be compatible with ASE's file i/o formats, describing
 either a time evolution or a single snapshot of a water droplet (rotationally symmetric about the
 z-axis) on a graphene sheet aligned aligned to the xy plane. Use as:

 >    python droplet_diffusion.py <input_file(s)> [-o <output_dir>] [--index <index>]
          [--max_tau <max_tau>] [--delta_t <delta_t>] [--no-display]

 The program tracks the motion of the droplet's CoM across the xy plane, and calculates the
 autocorrelation function to obtain the diffusive motion and drift velocity.
===================================================================================================
'''

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from rich.console import Console
from rich.progress import track
from droplet_graphene_analysis import __version__
from droplet_graphene_analysis.util import elapsed_time, read_droplet_trajectory

def main() -> None:

    #----------------------------------------------------------------------------------------------
    # Generate program description and parse input arguments

    prog_desc = ''
    for line in prog_desc_header.splitlines()[2:-1]:
        prog_desc += (line.lstrip(' ') + ' ') if line != '' else '\n\n'
    
    parser = argparse.ArgumentParser(prog='droplet_diffusion', description=prog_desc,
                                     usage='%(prog)s input_file [options]',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file', nargs='+',
                        help='input file(s) to read data from')
    parser.add_argument('-o', '--output', default='droplet-diffusion', dest='output_dir',
                        help='output folder to save results and graphical outputs to')
    parser.add_argument('--index', default=':', dest='index',
                        help='slice of indices to take from each input file')
    parser.add_argument('-t', '--max_tau', type=int, default=30, dest='max_tau',
                        help='maximum timescale to compute autocorrelations (in number of frames)')
    parser.add_argument('-d', '--delta_t', type=float, default=0.1, dest='delta_t',
                        help='time interval between frames (in ps)')
    parser.add_argument('--no-display', action='store_false', dest='opt_display',
                        help='disable display of graphics')
    args = parser.parse_args()

    for file in args.input_file:
        if not os.path.isfile(file):
            raise RuntimeError(f'File "{file}" not found.')
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if args.max_tau < 2:
        raise RuntimeError(f'Max tau ({args.max_tau}) must be at least 2.')
    if args.delta_t <= 0.0:
        raise RuntimeError(f'Delta t ({args.delta_t}) must be positive.')
    
    console = Console(highlight=False)

    #----------------------------------------------------------------------------------------------
    # Read input file and save coordinates

    file_msg = (f'"{args.input_file[0]}"' if len(args.input_file) == 1 else
                f'{len(args.input_file)} files')
    
    time_start_0 = time.time()
    with console.status(f'[green]Reading {file_msg}...'):
        cell_params, _, _, _, trajectory = read_droplet_trajectory(args.input_file,
                                                                   return_shift_trajectory=True,
                                                                   index=args.index)
        cell_xy = cell_params[0:2]
        N_frames = trajectory.shape[0]
        traj = trajectory[:,0:2]
        traj -= cell_xy * np.round(traj / cell_xy)

    console.print(f'Read [magenta]{N_frames} frames[/magenta] from [cyan]{file_msg}[/cyan] in ' +
                  f'[green]{elapsed_time(time_start_0)}[/green].')
    
    #----------------------------------------------------------------------------------------------
    # Calculate autocorrelation functions

    time_start_1 = time.time()

    max_tau = min(N_frames - 1, args.max_tau)
    autocorr = np.empty(max_tau, dtype=float)
    autocorr[0] = 0.0

    for tau in track(range(1, max_tau), description='Calculating autocorrelation functions...',
                     console=console, transient=True):
        diffs = traj[tau:] - traj[:-tau]
        diffs -= cell_xy * np.round(diffs / cell_xy)
        autocorr[tau] = np.mean(np.sum(diffs**2, axis=-1))

    tau_range = np.arange(max_tau) * args.delta_t
    fit_y = autocorr[1:] / tau_range[1:]
    fit_p, fit_cov = np.polyfit(tau_range[1:], fit_y, deg=1, cov=True)
    if fit_p[0] >= 0.0:
        drift_vel = np.sqrt(fit_p[0])
        drift_vel_unc = np.power(fit_cov[0,0], 0.25)
        diffusion = fit_p[1] / 4
        diffusion_unc = np.sqrt(fit_cov[1,1]) / 4
    else:
        drift_vel = 0.0
        drift_vel_unc = 0.0
        diffusion = np.mean(fit_y) / 4
        diffusion_unc = np.std(fit_y) / (4 * np.sqrt(max_tau - 2))

    console.print(f'Calculated autocorrelations in [green]{elapsed_time(time_start_1)}[/green].')

    #----------------------------------------------------------------------------------------------
    # Display plots

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(14, 7)

    traj -= traj[0]
    traj -= cell_xy * np.round(traj / cell_xy)
    extent = max(np.max(traj[:,0]) - np.min(traj[:,0]), np.max(traj[:,1]) - np.min(traj[:,1]))
    center = np.mean(traj, axis=0)
    width = 0.6 * extent

    scatterplot = ax[0].scatter(traj[:,0], traj[:,1], s=np.linspace(0, 2, N_frames),
                                c=(np.arange(N_frames) * args.delta_t), cmap='YlGnBu')
    fig.colorbar(scatterplot, ax=ax[0], label=r'$t\,\,[ps]$')
    ax[0].set_title('Droplet CoM trajectory')
    ax[0].set_xlabel(r'$x\,\,[\AA]$')
    ax[0].set_ylabel(r'$y\,\,[\AA]$')
    ax[0].set_xlim(center[0] - width, center[0] + width)
    ax[0].set_ylim(center[1] - width, center[1] + width)

    def fmt_str(x):
        exponent = (int(np.floor(np.log10(abs(x)))) if x != 0.0 else 0)
        if exponent == 0:
            return str(x)
        mantissa = x / (10**exponent)
        return f'{mantissa:.2f}' + r'\times 10^{' + str(exponent) + r'}'

    ax[1].plot(tau_range, (4 * diffusion * tau_range) + ((drift_vel * tau_range)**2), 'k--')
    ax[1].plot(tau_range, autocorr, 'b-')
    ax[1].set_title('Positional autocorrelation against time')
    ax[1].set_xlabel(r'$\tau\,\,[ps]$')
    ax[1].set_ylabel(r'$\langle \mathbf{r}(0)\cdot\mathbf{r}(\tau)\rangle\,\,[\AA^2]$')
    ax[1].annotate(r'$v_{0}\,=\,' + fmt_str(drift_vel) + r'\,\AA /ps$' + '\n' + r'$D\,=\,' +
                   fmt_str(diffusion) + r'\,\AA^2 /ps$', (0.01, 0.99), xycoords='axes fraction',
                   ha='left', va='top')
    
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'droplet-diffusion.png'), dpi=(3*fig.dpi),
                bbox_inches='tight', pad_inches=0.05)
    if args.opt_display:
        plt.show()

    #----------------------------------------------------------------------------------------------
    # Write results

    final_elapsed_time = elapsed_time(time_start_0)
    results_file = open(os.path.join(args.output_dir, 'results.ini'), 'w', encoding='utf-8')
    results_file.write('[General]\n')
    results_file.write(f'No. of frames = {N_frames}\n\n')
    results_file.write('[Diffusion]\n')
    results_file.write(f'Drift velocity [A/ps] = {drift_vel}\n')
    results_file.write(f'Drift velocity uncertainty [A/ps] = {drift_vel_unc}\n')
    results_file.write(f'Diffusion coefficient [A^2/ps] = {diffusion}\n')
    results_file.write(f'Diffusion coefficient uncertainty [A^2/ps] = {diffusion_unc}\n\n')
    results_file.write('[Misc]\n')
    results_file.write('Program type = droplet_diffusion\n')
    results_file.write(f'Program version = {__version__}\n')
    results_file.write(f'Program wall time = {final_elapsed_time}\n')
    results_file.close()
    
    console.print(f'Program completed in [green]{final_elapsed_time}[/green].')


#==================================================================================================
# Run from src

if __name__ == "__main__":
    main()
