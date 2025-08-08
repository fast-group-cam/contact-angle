#! /usr/bin/env python

prog_desc_header = '''
===================================================================================================
 This program measures the contact angle of a water droplet on a graphene sheet from a simulated
 trajectory. The input is a file, which must be compatible with ASE's file i/o formats, describing
 either a time evolution or a single snapshot of a water droplet (rotationally symmetric about the
 z-axis) on a graphene sheet aligned aligned to the xy plane. The contact angle is calculated by
 finding the Willard-Chandler interface for a small number of testpoints at the droplet's foot,
 and calculating the direction of the plane. Use as:

 >    python contact_angle.py <input_file(s)> [--index <index>] [--N_azimuths <N_azimuths]
          [--local] [--z_foot <z_foot>] [--max_tau <max_tau>] [--block_average]
          [--blocksize <blocksize>] [-o <output_dir>]

 The program performs the following actions in sequence:

 (1) It calculates the instantaneous interface at every frame of the provided trajectory, and plots
 the dynamic evolution of the surface as a function of time and azimuth;

 (2) It calculates the time-averaged interface across the entire trajectory, and plots selected
 cross-sections of the time-averaged density function;

 (3) If --block_average is enabled, it splits the trajectories into continuous blocks and finds the
 time-averaged interface for each block, to obtain unbiased estimates of the uncertainties of the
 full trajectory time-averaged observables.

 All plots will be saved to an output directory (specified by the -o option), which defaults to
 "contact-angle" within the parent directory that this program is executed from.
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
from rich.progress import track, Progress
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from droplet_graphene_analysis.util import elapsed_time, read_droplet_trajectory
from droplet_graphene_analysis.util.droplet import find_interface
from droplet_graphene_analysis.util.droplet.contact_angle import find_droplet_foot, find_spherical_cap
from droplet_graphene_analysis.util.droplet.plot import plot_density_xz_slice, plot_density_radially_symmetric
from droplet_graphene_analysis.util.graphene import regularized_heightmap

def main() -> None:

    #----------------------------------------------------------------------------------------------
    # Script default parameters

    N_AZIMUTHS = 60       # Number of azimuthal directions to analyze per frame
    MAX_TAU = 25          # Maximum timescale to calculate autocorrelations (in number of frames)
    N_BLOCKSIZES = 30     # Number of blocksizes to scan for automatic determination

    from droplet_graphene_analysis.util.droplet.contact_angle import Z_FOOT
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
    parser.add_argument('--z_foot', type=float, default=Z_FOOT, dest='z_foot',
                        help='height (in angstroms) of the droplet foot above the graphene sheet')
    parser.add_argument('--max_tau', type=int, default=MAX_TAU, dest='max_tau',
                        help='maximum timescale (in no. of frames) to calculate autocorrelations')
    parser.add_argument('--block_average', action='store_true', dest='block_average',
                        help=('perform reverse cumulative averaging for unbiased uncertainty ' +
                              'estimates over varying block sizes'))
    parser.add_argument('-b', '--blocksize', type=int, default=None, dest='blocksize',
                        help=('if --block-average is turned on, disables automatic block sizing ' +
                              'and enforces specified block size'))
    parser.add_argument('--N_blocksizes', type=int, default=N_BLOCKSIZES, dest='N_blocksizes',
                        help=('the number of different block sizes to try (for automatic ' +
                              'block sizing) if --block-average is turned on but --blocksize is ' +
                              'not specified'))
    parser.add_argument('-o', '--output', default='contact-angle', dest='output_dir',
                        help='output folder to save log and graphical outputs to')
    parser.add_argument('--no-graphics', action='store_true', dest='no_graphics',
                        help='disables rendering of graphics (and speeds up the script)')
    args = parser.parse_args()

    for file in args.input_file:
        if not os.path.isfile(file):
            raise RuntimeError(f'File "{file}" not found.')

    if args.N_azimuths < 12:
        raise RuntimeError(f'N_azimuths ({args.N_azimuths}) must be positive and at least 12.')
    if args.z_foot < 0.0:
        raise RuntimeError(f'z_foot ({args.z_foot}) must be positive.')
    if args.max_tau < 2:
        raise RuntimeError(f'Max tau ({args.max_tau}) must be at least 2.')
    if args.blocksize is not None and args.blocksize < 1:
        raise RuntimeError(f'Block size ({args.blocksize}) must be positive.')
    if args.N_blocksizes < 2:
        raise RuntimeError(f'Number of block sizes ({args.N_blocksize}) must be at least 2.')

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    # Number of azimuthal directions must be a multiple of 12
    N_azi = 12 * int(round(args.N_azimuths / 12))

    console = Console(highlight=False)

    #----------------------------------------------------------------------------------------------
    # Helper functions
    
    def arccos_deg(x):
        return np.arccos(x) * 180 / np.pi
    
    def cot_deg(x):
        return -np.tan((x + 90.0) * np.pi / 180)

    def has_nan(x: np.ndarray) -> bool:
        return np.isnan(np.sum(x))

    def exp_curve(x, A, k, c):
        return A * np.exp(-k * x) + c

    def exp_fit(data, k0 = 0.5, c0 = 0.5):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                popt, _ = curve_fit(exp_curve, list(range(data.shape[0])), data, p0=(1 - c0, k0, c0))
            return popt
        except RuntimeError:
            mu = np.mean(data)
            return np.array((1 - mu, np.inf, mu))

    def plot_against_time_and_azimuth(fig, ax, data, title, var_label):
        N_x = data.shape[0]
        N_y = data.shape[1]
        im = ax.imshow(data.T, origin='lower', extent=(-0.5, N_x - 0.5, -180/N_y, 360 - (180/N_y)))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(var_label)
        ax.set_aspect(N_x / 360)
        ax.set_xlabel('Frame number')
        ax.set_ylabel(r'$\varphi\;\;(\degree)$')
        ax.set_title(title)
        if N_x < 16:
            ax.set_xticks(list(range(N_x)))
        if N_y < 16:
            ax.set_yticks(np.linspace(0, 360, N_y, endpoint=False))

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
    # Calculate smoothened carbon sheets for every frame (if --local option is turned on); also
    # calculate nominal interfacial separation for solid-liquid interface

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

    CoM_z = np.mean(waters[:,:,2])
    droplet_roof = find_interface(waters, (0, 0, CoM_z), (0, 0, 1))[2]
    droplet_floor = find_interface(waters, (0, 0, CoM_z), (0, 0, -1))[2]
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
    # Helper function for finding observables of interest across a range of frames

    def calculate_observables(start_frame, end_frame, sphere_fit = False):

        results = argparse.Namespace()

        mean_sheet = np.mean(sheets[start_frame:end_frame], axis=0)
        mean_heightmap = RegularGridInterpolator((sheet_gridx, sheet_gridy), mean_sheet)
        dh_dx = (np.roll(mean_sheet, -1, axis=0) - np.roll(mean_sheet, 1, axis=0)) / (2 * sheet_dx)
        dh_dy = (np.roll(mean_sheet, -1, axis=1) - np.roll(mean_sheet, 1, axis=1)) / (2 * sheet_dy)
        dh_dx = RegularGridInterpolator((sheet_gridx, sheet_gridy), dh_dx)
        dh_dy = RegularGridInterpolator((sheet_gridx, sheet_gridy), dh_dy)

        azi = np.linspace(0, 2 * np.pi, N_azi, endpoint=False)
        search_dirs = np.c_[np.cos(azi), np.sin(azi), np.zeros(N_azi)]
        search_perp = np.c_[-np.sin(azi), np.cos(azi)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # hack!
            interfaces, normals = find_droplet_foot(waters[start_frame:end_frame], mean_heightmap,
                                                    search_dirs, z_foot=args.z_foot)
        results.interfaces = interfaces
        results.normals = normals
        
        local_sheet_c = np.empty((N_azi, 3), dtype=float)
        local_sheet_n = np.empty((N_azi, 3), dtype=float)
        contact_angles = np.empty((N_azi,), dtype=float)
        for i, inter in enumerate(interfaces):
            if has_nan(inter):
                local_sheet_c[i] = np.full((3,), np.nan)
                local_sheet_n[i] = np.full((3,), np.nan)
                contact_angles[i] = 90.0
            else:
                inter -= cell_params * np.round(inter / cell_params)
                local_sheet_c[i] = np.array((inter[0], inter[1], mean_heightmap(inter[0:2])[0]))
                normal_vec = np.array((-dh_dx(inter[0:2])[0], -dh_dy(inter[0:2])[0], 1.0))
                local_sheet_n[i] = normal_vec / np.linalg.norm(normal_vec)
                contact_angles[i] = arccos_deg(np.dot(normals[i], local_sheet_n[i]))
        results.local_sheet_c = local_sheet_c
        results.local_sheet_n = local_sheet_n
        results.contact_angles = np.nan_to_num(contact_angles, copy=False, nan=90.0)
        results.foot_r = np.nanmean(np.linalg.norm(interfaces[:,0:2], axis=-1) +
                                    (args.z_foot * cot_deg(contact_angles)))

        proj_normals = np.power(np.sum(normals[:,0:2]**2, axis=-1), -0.5)[:,None] * normals[:,0:2]
        ooplane_angles = np.arcsin(np.sum(proj_normals * search_perp, axis=-1)) * 180 / np.pi
        results.ooplane_angles = np.nan_to_num(ooplane_angles, copy=False)

        if sphere_fit:
            sphere_results = find_spherical_cap(waters[start_frame:end_frame], mean_heightmap)
            results.sphere_r = sphere_results['r']
            results.sphere_z = sphere_results['z']
            results.sphere_a = sphere_results['a']
            results.sphere_angle = sphere_results['angle']
        
        return results

    #----------------------------------------------------------------------------------------------
    # Calculate instantaneous interfaces for every frame and measure autocorrelations etc.

    contact_angles = np.empty((N_frames, N_azi), dtype=float)
    ooplane_angles = np.empty((N_frames, N_azi), dtype=float)

    time_start_1 = time.time()
    for f in progress_bar_iter(N_frames, 'Computing instantaneous interfaces...'):
        results = calculate_observables(f, f+1, sphere_fit=False)
        contact_angles[f] = results.contact_angles
        ooplane_angles[f] = results.ooplane_angles
    
    console.print(f'Computed instantaneous interfaces in [green]{elapsed_time(time_start_1)}[/green].')
    
    time_start_1 = time.time()
    with console.status('[green]Computing correlations...'):

        max_tau = min(N_frames - 1, args.max_tau)
        contact_angle_time_corrs = np.empty((max_tau,), dtype=float)
        contact_angle_time_corrs[0] = np.mean(np.square(contact_angles), axis=(0,1))
        for tau in range(1, max_tau):
            contact_angle_time_corrs[tau] = np.mean(contact_angles[:-tau] * contact_angles[tau:], axis=(0,1))
        contact_angle_time_corrs[:] /= contact_angle_time_corrs[0]
        contact_angle_time_popt = exp_fit(contact_angle_time_corrs, k0=1.0, c0=0.998)

        ooplane_angle_time_corrs = np.empty((max_tau,), dtype=float)
        ooplane_angle_time_corrs[0] = np.mean(np.square(ooplane_angles), axis=(0,1))
        for tau in range(1, max_tau):
            ooplane_angle_time_corrs[tau] = np.mean(ooplane_angles[:-tau] * ooplane_angles[tau:], axis=(0,1))
        ooplane_angle_time_corrs[:] /= ooplane_angle_time_corrs[0]
        ooplane_angle_time_popt = exp_fit(ooplane_angle_time_corrs, k0=0.5, c0=0.1)

        max_a = int(N_azi / 2)
        contact_angle_azi_corrs = np.empty((max_a,), dtype=float)
        contact_angle_azi_corrs[0] = np.mean(np.square(contact_angles), axis=(0,1))
        for a in range(1, max_a):
            contact_angle_azi_corrs[a] = np.mean(contact_angles[:,:-a] * contact_angles[:,a:], axis=(0,1))
        contact_angle_azi_corrs[:] /= contact_angle_azi_corrs[0]
        contact_angle_azi_popt = exp_fit(contact_angle_azi_corrs, k0=1.0, c0=0.978)
        contact_angle_azi_popt[1] *= N_azi / 360.0

        ooplane_angle_azi_corrs = np.empty((max_a,), dtype=float)
        ooplane_angle_azi_corrs[0] = np.mean(np.square(ooplane_angles), axis=(0,1))
        for a in range(1, max_a):
            ooplane_angle_azi_corrs[a] = np.mean(ooplane_angles[:,:-a] * ooplane_angles[:,a:], axis=(0,1))
        ooplane_angle_azi_corrs[:] /= ooplane_angle_azi_corrs[0]
        ooplane_angle_azi_corrs = np.abs(ooplane_angle_azi_corrs)
        ooplane_angle_azi_popt = exp_fit(ooplane_angle_azi_corrs, k0=10.0, c0=0.05)
        ooplane_angle_azi_popt[1] *= N_azi / 360.0

    console.print(f'Computed correlations in [green]{elapsed_time(time_start_1)}[/green].')

    if not args.no_graphics:

        fig = plt.figure(figsize=(14, 7), layout='constrained')
        subfigs = fig.subfigures(1, 2, wspace=0.06, width_ratios=[1.5, 1.0])
        plot_against_time_and_azimuth(subfigs[0], subfigs[0].subplots(), contact_angles,
                                    'Instantaneous contact angles', r'$\theta(t,\varphi)$')
        ax = subfigs[1].subplots(2, 1)
        tau = np.linspace(0, max_tau, 3 * max_tau, endpoint=False)
        ax[0].plot(tau, exp_curve(tau, *contact_angle_time_popt), 'b--')
        ax[0].plot(list(range(max_tau)), contact_angle_time_corrs, 'r.')
        ax[0].set_xlabel(r'$\tau$  (frames)')
        ax[0].set_ylabel(r'$C_{\theta}(\tau)$')
        ax[0].set_title('Normalized autocorrelation of contact angle against time')
        phi = np.linspace(0, 180, 3 * max_a, endpoint=False)
        ax[1].plot(phi, exp_curve(phi, *contact_angle_azi_popt), 'b--')
        ax[1].plot(np.linspace(0, 180, max_a, endpoint=False), contact_angle_azi_corrs, 'r.')
        ax[1].set_xlabel(r'$\varphi\;\;(\degree)$')
        ax[1].set_ylabel(r'$C_{\theta}(\varphi)$')
        ax[1].set_title('Normalized autocorrelation of contact angle against azimuth')
        fig.savefig(os.path.join(args.output_dir, 'inst_contact_angles.png'), dpi=(3*fig.dpi),
                    bbox_inches='tight', pad_inches=0.05)
        
        fig = plt.figure(figsize=(14, 7), layout='constrained')
        subfigs = fig.subfigures(1, 2, wspace=0.06, width_ratios=[1.5, 1.0])
        plot_against_time_and_azimuth(subfigs[0], subfigs[0].subplots(), ooplane_angles,
                                    'Instantaneous interfacial out-of-plane angles', r'$\delta(t,\varphi)$')
        ax = subfigs[1].subplots(2, 1)
        ax[0].plot(tau, exp_curve(tau, *ooplane_angle_time_popt), 'b--')
        ax[0].plot(list(range(max_tau)), ooplane_angle_time_corrs, 'r.')
        ax[0].set_xlabel(r'$\tau$  (frames)')
        ax[0].set_ylabel(r'$C_{\delta}(\tau)$')
        ax[0].set_title('Normalized autocorrelation of out-of-plane angle against time')
        ax[1].plot(phi, exp_curve(phi, *ooplane_angle_azi_popt), 'b--')
        ax[1].plot(np.linspace(0, 180, max_a, endpoint=False), ooplane_angle_azi_corrs, 'r.')
        ax[1].set_xlabel(r'$\varphi\;\;(\degree)$')
        ax[1].set_ylabel(r'$C_{\delta}(\varphi)$')
        ax[1].set_title('Normalized autocorrelation of out-of-plane angle against azimuth')
        fig.savefig(os.path.join(args.output_dir, 'inst_out-of-plane_angles.png'), dpi=(3*fig.dpi),
                    bbox_inches='tight', pad_inches=0.05)

    log_file.write('-------------------------\n')
    log_file.write(' Instantaneous interface\n')
    log_file.write('-------------------------\n\n')
    log_file.write(f'Mean contact angle = {np.mean(contact_angles)} [deg]\n')
    log_file.write(f'Median contact angle = {np.median(contact_angles)} [deg]\n')
    log_file.write(f'Contact angle stdev = {np.std(contact_angles)} [deg]\n\n')
    log_file.write(f'Contact angle correlation time = {1 / contact_angle_time_popt[1]} [frames]\n')
    log_file.write(f'Contact angle azimuthal correlation scale = {1 / contact_angle_azi_popt[1]} [deg]\n\n')
    log_file.write(f'Mean out-of-plane angle = {np.mean(ooplane_angles)} [deg]\n')
    log_file.write(f'Median out-of-plane angle = {np.median(ooplane_angles)} [deg]\n')
    log_file.write(f'Out-of-plane angle stdev = {np.std(ooplane_angles)} [deg]\n\n')
    log_file.write(f'Out-of-plane angle correlation time = {1 / ooplane_angle_time_popt[1]} [frames]\n')
    log_file.write(f'Out-of-plane angle azimuthal correlation scale = {1 / ooplane_angle_azi_popt[1]} [deg]\n\n')

    #contact_angle_inst_var = np.var(np.mean(contact_angles, axis=-1))

    #----------------------------------------------------------------------------------------------
    # Calculate time-averaged interface across all frames

    time_start_1 = time.time()
    with console.status('[green]Computing time-averaged interface...'):
        results = calculate_observables(0, N_frames, sphere_fit=True)

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
                
            for k in (0, int(N_azi / 2)):
                if not (has_nan(results.interfaces[idx + k]) or has_nan(results.normals[idx + k])):
                    rot_carbon_c = rot_matrix @ results.local_sheet_c[idx + k]
                    rot_carbon_n = rot_matrix @ results.local_sheet_n[idx + k]
                    a_x = rot_carbon_c[0] - CARBON_RADIUS
                    a_z = rot_carbon_c[2] + (CARBON_RADIUS * rot_carbon_n[0] / rot_carbon_n[2])
                    b_x = rot_carbon_c[0] + CARBON_RADIUS
                    b_z = rot_carbon_c[2] - (CARBON_RADIUS * rot_carbon_n[0] / rot_carbon_n[2])
                    ax[i // 3][i % 3].plot((a_x, b_x), (a_z, b_z), 'k-')
                    rot_inter = rot_matrix @ results.interfaces[idx + k]
                    rot_norm = rot_matrix @ results.normals[idx + k]
                    a_x = rot_inter[0] + (rot_inter[2] * rot_norm[2] / rot_norm[0])
                    b_x = rot_inter[0] - (2 * rot_inter[2] * rot_norm[2] / rot_norm[0])
                    ax[i // 3][i % 3].plot((a_x, b_x), (0.0, 3 * rot_inter[2]), '-', color=(1.0, 0.0, 1.0))
            ax[i // 3][i % 3].plot((0.0,), (CoM_z,), '.', color=(1.0, 0.0, 1.0))
            ax[i // 3][i % 3].text(0.01, 0.99, (r'$\theta_{left}\;=\;' +
                                                f'{results.contact_angles[idx + (N_azi // 2)]:.1f}' +
                                                r'\degree$' + '\n' + r'$\theta_{right}\;=\;' +
                                                f'{results.contact_angles[idx]:.1f}' + r'\degree$'),
                                                ha='left', va='top',
                                                transform=ax[i // 3][i % 3].transAxes)

            ax[i // 3][i % 3].set_title(r'$\varphi\;=\;' + f'{(angle * 180 / np.pi):.0f}' + r'\degree$')

        fig.suptitle('Azimuthal cross-sections of time-averaged droplet')
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, 'ave_cross_sections.png'), dpi=(3*fig.dpi),
                    bbox_inches='tight', pad_inches=0.05)
        
        console.print('Plotted time-averaged density functions in ' +
                    f'[green]{elapsed_time(time_start_1)}[/green].')
        
        time_start_1 = time.time()
        with console.status('[green]Plotting best-fit sphere...'):

            fig, ax = plt.subplots()
            fig.set_size_inches(6, 6)

            mean_sheet = np.mean(sheets, axis=0)
            mean_heightmap = RegularGridInterpolator((sheet_gridx, sheet_gridy), mean_sheet)
            plot_density_radially_symmetric(waters, mean_heightmap, ax)

            max_phi = np.arccos(np.clip(-results.sphere_z / results.sphere_r, -1.0, 1.0))
            phi = np.linspace(0, max_phi, 100)
            ax.plot(results.sphere_r * np.sin(phi), results.sphere_z + (results.sphere_r * np.cos(phi)),
                    '-', color=(0.9, 0.45, 0.0))
            ax.text(0.99, 0.99, (r'$\theta\;=\;' + f'{results.sphere_angle:.1f}' + r'\degree$'),
                    ha='right', va='top', transform=ax.transAxes)
            ax.set_xlabel(r'r ($\AA$)')
            ax.set_ylabel(r'z ($\AA$)')
            ax.set_title('Spherical fit over time-averaged droplet')

            fig.savefig(os.path.join(args.output_dir, 'ave_sphere_fit.png'), dpi=(3*fig.dpi),
                        bbox_inches='tight', pad_inches=0.05)
        
        console.print(f'Plotted best-fit sphere in [green]{elapsed_time(time_start_1)}[/green].')

    log_file.write('-------------------------\n')
    log_file.write(' Time-averaged interface\n')
    log_file.write('-------------------------\n\n')
    log_file.write(f'Mean contact angle = {np.mean(results.contact_angles)} [deg]\n')
    log_file.write(f'Median contact angle = {np.median(results.contact_angles)} [deg]\n')
    log_file.write(f'Contact angle stdev = {np.std(results.contact_angles)} [deg]\n\n')
    log_file.write(f'Mean out-of-plane angle = {np.mean(results.ooplane_angles)} [deg]\n')
    log_file.write(f'Median out-of-plane angle = {np.median(results.ooplane_angles)} [deg]\n')
    log_file.write(f'Out-of-plane angle stdev = {np.std(results.ooplane_angles)} [deg]\n\n')
    log_file.write(f'Center-of-mass z-height = {CoM_z} [A]\n')
    log_file.write(f'Best-fit interfacial circular radius = {results.foot_r} [A]\n\n')
    log_file.write(f'Best-fit sphere z-height = {results.sphere_z} [A]\n')
    log_file.write(f'Best-fit sphere radius = {results.sphere_r} [A]\n')
    log_file.write(f'Best-fit sphere contact angle = {results.sphere_angle} [deg]\n')
    log_file.write(f'Best-fit sphere sheet-intersecting radius = {results.sphere_a} [A]\n\n')

    #----------------------------------------------------------------------------------------------
    # Calculate block-averages of time-averaged interfaces (for user-specified blocksize)

    if args.block_average and args.blocksize is not None:

        N_blocks = N_frames // args.blocksize
        if N_blocks < 1:
            raise RuntimeError(f'User-specified block size ({args.blocksize}) too large.')
        contact_angle_block_means = np.empty(N_blocks)
        foot_r_block_means = np.empty(N_blocks)
        sphere_angle_block_means = np.empty(N_blocks)
        sphere_a_block_means = np.empty(N_blocks)

        time_start_1 = time.time()
        for i in progress_bar_iter(N_blocks, 'Computing block averages...'):
            results = calculate_observables(i * args.blocksize, (i+1) * args.blocksize, sphere_fit=True)
            contact_angle_block_means[i] = np.mean(results.contact_angles)
            foot_r_block_means[i] = results.foot_r
            sphere_angle_block_means[i] = results.sphere_angle
            sphere_a_block_means[i] = results.sphere_a
        console.print(f'Computed block averages in [green]{elapsed_time(time_start_1)}[/green].')

        log_file.write('---------------------------------------------\n')
        log_file.write(' Block-averaging of time-averaged interfaces\n')
        log_file.write('---------------------------------------------\n\n')
        log_file.write(f'Blocksize (user-specified) = {args.blocksize} [frames]\n')
        log_file.write(f'Number of blocks = {N_blocks}\n\n')
        log_file.write(f'Contact angle, mean of block means = {np.mean(contact_angle_block_means)} [deg]\n')
        log_file.write(f'Contact angle, uncertainty = {np.std(contact_angle_block_means) / np.sqrt(N_blocks - 1)} [deg]\n\n')
        log_file.write(f'Foot edge radius, mean of block means = {np.mean(foot_r_block_means)} [A]\n')
        log_file.write(f'Foot edge radius, uncertainty = {np.std(foot_r_block_means) / np.sqrt(N_blocks - 1)} [A]\n\n')
        log_file.write(f'Best-fit sphere contact angle, mean of block means = {np.mean(sphere_angle_block_means)} [deg]\n')
        log_file.write(f'Best-fit sphere contact angle, uncertainty = {np.std(sphere_angle_block_means) / np.sqrt(N_blocks - 1)} [deg]\n\n')
        log_file.write(f'Best-fit sphere edge radius, mean of block means = {np.mean(sphere_a_block_means)} [A]\n')
        log_file.write(f'Best-fit sphere edge radius, uncertainty = {np.std(sphere_a_block_means) / np.sqrt(N_blocks - 1)} [A]\n\n')

    #----------------------------------------------------------------------------------------------
    # Calculate block-averages of time-averaged interfaces (for automatic blocksizing)

    elif args.block_average and args.blocksize is None:

        if N_frames < (10 * (args.N_blocksizes + 2)):
            raise RuntimeError('Too few frames in the trajectory for automatic blocksizing.')
        N_blocks = np.array([n + 3 for n in range(args.N_blocksizes)], dtype=int)
        blocksizes = np.unique(np.array([N_frames // n for n in N_blocks], dtype=int))
        N_blocks = np.array([N_frames // n for n in blocksizes], dtype=int)
        N_blocksizes = N_blocks.shape[0]
        contact_angle_block_means = np.empty(N_blocksizes, dtype=float)
        contact_angle_block_vars = np.empty(N_blocksizes, dtype=float)
        foot_r_block_means = np.empty(N_blocksizes, dtype=float)
        foot_r_block_vars = np.empty(N_blocksizes, dtype=float)
        sphere_angle_block_means = np.empty(N_blocksizes, dtype=float)
        sphere_angle_block_vars = np.empty(N_blocksizes, dtype=float)
        sphere_a_block_means = np.empty(N_blocksizes, dtype=float)
        sphere_a_block_vars = np.empty(N_blocksizes, dtype=float)

        time_start_1 = time.time()
        progress_bar = Progress(console=console, transient=True)
        progress_bar.start()
        progress_bar_parent_task = progress_bar.add_task('Scanning blocksizes...', total=N_blocksizes)
        for b in range(N_blocksizes):
            block_means = np.empty(N_blocks[b])
            r_block_means = np.empty(N_blocks[b])
            sph_ang_block_means = np.empty(N_blocks[b])
            sph_r_block_means = np.empty(N_blocks[b])
            progress_bar_child_task = progress_bar.add_task('Computing block averages for b = ' +
                                    f'{blocksizes[b]} ({b+1}/{N_blocksizes})...', total=N_blocks[b])
            for i in range(N_blocks[b]):
                results = calculate_observables(i * blocksizes[b], (i+1) * blocksizes[b], sphere_fit=True)
                block_means[i] = np.mean(results.contact_angles)
                r_block_means[i] = results.foot_r
                sph_ang_block_means[i] = results.sphere_angle
                sph_r_block_means[i] = results.sphere_a
                progress_bar.update(progress_bar_child_task, advance=1)
            contact_angle_block_means[b] = np.mean(block_means)
            contact_angle_block_vars[b] = np.var(block_means)
            foot_r_block_means[b] = np.mean(r_block_means)
            foot_r_block_vars[b] = np.var(r_block_means)
            sphere_angle_block_means[b] = np.mean(sph_ang_block_means)
            sphere_angle_block_vars[b] = np.var(sph_ang_block_means)
            sphere_a_block_means[b] = np.mean(sph_r_block_means)
            sphere_a_block_vars[b] = np.var(sph_r_block_means)
            progress_bar.remove_task(progress_bar_child_task)
            progress_bar.update(progress_bar_parent_task, advance=1)
        progress_bar.stop()
        console.print(f'Computed block averages in [green]{elapsed_time(time_start_1)}[/green].')

        #stat_inefficiencies = blocksizes * contact_angle_block_vars / contact_angle_inst_var
        stat_inefficiencies = blocksizes * sphere_angle_block_vars / np.min(sphere_angle_block_vars)
        idx = int(round(0.75 * N_blocksizes))
        idx = np.argpartition(stat_inefficiencies, idx)[idx]

        if not args.no_graphics:

            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(10, 5)
            ax[0].plot(blocksizes, stat_inefficiencies, 'b.')
            ax[0].plot((blocksizes[idx],), (stat_inefficiencies[idx],), 'r.')
            ax[0].set_xlabel('Blocksize')
            ax[0].set_ylabel('Statistical inefficiency')
            ax[0].set_title('Statistical inefficiency against blocksize')
            ax[1].errorbar(blocksizes, contact_angle_block_means,
                        yerr=np.sqrt(contact_angle_block_vars / (N_blocks - 1.0)), fmt='b.')
            ax[1].errorbar((blocksizes[idx],), (contact_angle_block_means[idx],),
                        yerr=(np.sqrt(contact_angle_block_vars[idx] / (N_blocks[idx] - 1.0)),), fmt='r.')
            ax[1].set_xlabel('Blocksize')
            ax[1].set_ylabel(r'$\langle\theta\rangle_{b}\,(\degree)$')
            ax[1].set_title('Block-averaged contact angle against blocksize')

            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, 'blocksizes.png'), dpi=(3*fig.dpi),
                        bbox_inches='tight', pad_inches=0.05)

        log_file.write('---------------------------------------------\n')
        log_file.write(' Block-averaging of time-averaged interfaces\n')
        log_file.write('---------------------------------------------\n\n')
        log_file.write(f'Blocksize = {blocksizes[idx]} [frames]\n')
        log_file.write(f'Number of blocks = {N_blocks[idx]}\n\n')
        log_file.write(f'Contact angle, mean of block means = {contact_angle_block_means[idx]} [deg]\n')
        log_file.write(f'Contact angle, uncertainty = {np.sqrt(contact_angle_block_vars[idx] / (N_blocks[idx] - 1))} [deg]\n\n')
        log_file.write(f'Foot edge radius, mean of block means = {foot_r_block_means[idx]} [A]\n')
        log_file.write(f'Foot edge radius, uncertainty = {np.sqrt(foot_r_block_vars[idx] / (N_blocks[idx] - 1))} [A]\n\n')
        log_file.write(f'Best-fit sphere contact angle, mean of block means = {sphere_angle_block_means[idx]} [deg]\n')
        log_file.write(f'Best-fit sphere contact angle, uncertainty = {np.sqrt(sphere_angle_block_vars[idx] / np.sqrt(N_blocks[idx] - 1))} [deg]\n\n')
        log_file.write(f'Best-fit sphere edge radius, mean of block means = {sphere_a_block_means[idx]} [A]\n')
        log_file.write(f'Best-fit sphere edge radius, uncertainty = {np.sqrt(sphere_a_block_vars[idx] / np.sqrt(N_blocks[idx] - 1))} [A]\n\n')
        
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
