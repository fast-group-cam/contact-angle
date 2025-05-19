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
import matplotlib.colors as mplc

from rich.console import Console
from rich.progress import track, Progress
from scipy.optimize import curve_fit
from matplotlib.cm import ScalarMappable
from droplet_graphene_analysis.util import elapsed_time, read_droplet_trajectory, best_fit_axial_sphere
from droplet_graphene_analysis.util.droplet import find_interface
from droplet_graphene_analysis.util.droplet.plot import plot_density_xz_slice
from droplet_graphene_analysis.util.graphene import generate_grid, smooth_sheet

def main() -> None:

    #----------------------------------------------------------------------------------------------
    # Script default parameters

    N_AZIMUTHS = 60       # Number of azimuthal directions to analyze per frame
    Z_FOOT = 7            # Height of the droplet foot above the graphene sheet (in angstroms)
    STEP_BACK = 10        # Step back per iteration of foot-finding algorithm (in angstroms)
    MAX_TAU = 25          # Maximum timescale to calculate autocorrelations (in number of frames)
    N_SPHERE_PTS = 100    # Number of points to use to find best-fit spherical top
    N_BLOCKSIZES = 30     # Number of blocksizes to scan for automatic determination

    from droplet_graphene_analysis.util.graphene.angle import CUTOFF_RADIUS as CARBON_RADIUS
    CARBON_RADIUS_SQ = CARBON_RADIUS**2

    from droplet_graphene_analysis.util.droplet.coarse_grain import BULK_DENSITY

    #----------------------------------------------------------------------------------------------
    # Generate program description and parse input arguments

    prog_desc = ''
    for line in prog_desc_header.splitlines()[2:-1]:
        prog_desc += (line.lstrip(' ') + ' ') if line != '' else '\n\n'

    parser = argparse.ArgumentParser(prog='contact-angle', description=prog_desc,
                                     usage='%(prog)s input_file [options]',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file', nargs='+',
                        help='input file(s) to read data from')
    parser.add_argument('--index', default=':', dest='index',
                        help='index or slice of indices to take from each input file')
    parser.add_argument('--N_azimuths', type=int, default=N_AZIMUTHS, dest='N_azimuths',
                        help='number of azimuthal angles to analyze per frame')
    parser.add_argument('--local', action='store_true', dest='local',
                        help='use local definition of graphene inclination')
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

    def arcsin(x):
        return np.arcsin(x) * 180 / np.pi
    
    def arccos(x):
        return np.arccos(x) * 180 / np.pi
    
    def arctan(x):
        return np.arctan(x) * 180 / np.pi
    
    def cot(x):
        return -np.tan((x + 90.0) * np.pi / 180)

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
    # Helper function for finding interface at droplet foot

    def droplet_foot_interfaces(waters, carbons, search_directions):

        # Calculate droplet CoM and floor, and flatten carbons array
        CoM = np.mean(waters, axis=(0, 1))
        graphene = carbons.reshape(-1, 3)
        
        # Iterate through search directions
        interfaces = list()
        normals = list()
        for search_dir in search_directions:

            # First guess of interface
            inter = find_interface(waters, (0, 0, CoM[2]), search_dir)

            # Find z-coordinate of graphene underneath first guess
            nearby_Cs = graphene[np.sum((graphene[:,0:2] - inter[0:2])**2, axis=-1) < CARBON_RADIUS_SQ]
            local_floor = np.mean(nearby_Cs[:,2])

            # Second guess of interface (and repeat refinement)
            step_back = min(np.dot(inter, search_dir), STEP_BACK) * search_dir
            inter = find_interface(waters, (inter[0] - step_back[0], inter[1] - step_back[1],
                                            local_floor + args.z_foot), search_dir)
            nearby_Cs = graphene[np.sum((graphene[:,0:2] - inter[0:2])**2, axis=-1) < CARBON_RADIUS_SQ]
            local_floor = np.mean(nearby_Cs[:,2])

            # Third and final guess of interface
            step_back = min(np.dot(inter, search_dir), STEP_BACK) * search_dir
            inter, norm = find_interface(waters, (inter[0] - step_back[0], inter[1] - step_back[1],
                                                  local_floor + args.z_foot), search_dir, calc_normal=True)
            interfaces.append(inter)
            normals.append(norm)

        return (np.array(interfaces), np.array(normals))
    
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
    # Calculate smoothened carbon sheets for every frame (if --local option is turned on)

    if args.local:
        sheet_res_x = int(np.ceil(6.0 * cell_params[0] / CARBON_RADIUS))
        sheet_res_y = int(np.ceil(6.0 * cell_params[1] / CARBON_RADIUS))
        sheet_gridpts = generate_grid((sheet_res_x, sheet_res_y), cell_params[0:2])
        sheets = np.empty((N_frames, sheet_res_x, sheet_res_y), dtype=float)
        time_start_1 = time.time()
        for f in progress_bar_iter(N_frames, 'Processing graphene sheet...'):
            sheets[f] = smooth_sheet(carbons[f], cell_params[0:2], (sheet_res_x, sheet_res_y))
        console.print(f'Processed graphene sheet in [green]{elapsed_time(time_start_1)}[/green].')

    #----------------------------------------------------------------------------------------------
    # Create output log file

    log_file = open(os.path.join(args.output_dir, 'log.txt'), 'w', encoding='utf-8')

    #----------------------------------------------------------------------------------------------
    # Helper function for finding observables of interest across a range of frames

    def calculate_observables(start_frame, end_frame, sphere_fit = False):

        results = argparse.Namespace()

        azi = np.linspace(0, 2 * np.pi, N_azi, endpoint=False)
        search_dirs = np.c_[np.cos(azi), np.sin(azi), np.zeros(N_azi)]
        search_perp = np.c_[-np.sin(azi), np.cos(azi)]
        interfaces, normals = droplet_foot_interfaces(waters[start_frame:end_frame],
                                                      carbons[start_frame:end_frame], search_dirs)
        results.interfaces = interfaces
        results.normals = normals
        results.foot_r = np.mean(np.linalg.norm(interfaces[:,0:2], axis=-1))
        
        if args.local:
            mean_sheet = np.mean(sheets[start_frame:end_frame], axis=0)
            local_sheet_c = np.empty((N_azi, 3), dtype=float)
            local_sheet_n = np.empty((N_azi, 3), dtype=float)
            contact_angles = np.empty((N_azi,), dtype=float)
            for i, inter in enumerate(interfaces):
                nearby_idx = np.sum((sheet_gridpts - inter[0:2])**2, axis=-1) < CARBON_RADIUS_SQ
                nearby = np.concat([sheet_gridpts[nearby_idx], mean_sheet[nearby_idx, None]], axis=-1)
                local_sheet_c[i] = np.mean(nearby, axis=0)
                local_sheet_n[i] = np.linalg.svd(nearby - local_sheet_c[i], full_matrices=False)[2][-1]
                local_sheet_n[i] /= np.linalg.norm(local_sheet_n[i]) * np.sign(local_sheet_n[i,2])
                contact_angles[i] = arccos(np.dot(normals[i], local_sheet_n[i]))
            results.mean_sheet = mean_sheet
            results.local_sheet_c = local_sheet_c
            results.local_sheet_n = local_sheet_n
        else:
            contact_angles = arccos(np.dot(normals, (0, 0, 1)))
        results.contact_angles = contact_angles
        results.foot_r = np.mean(np.linalg.norm(interfaces[:,0:2], axis=-1) + (args.z_foot * cot(contact_angles)))

        proj_normals = np.power(np.sum(normals[:,0:2]**2, axis=-1), -0.5)[:,None] * normals[:,0:2]
        ooplane_angles = np.arcsin(np.sum(proj_normals * search_perp, axis=-1)) * 180 / np.pi
        results.ooplane_angles = ooplane_angles

        if sphere_fit:
            CoM = np.mean(waters[start_frame:end_frame], axis=(0,1))
            phi = 2 * np.pi * np.random.random(N_SPHERE_PTS)
            cosine_theta = np.random.random(N_SPHERE_PTS)
            sine_theta = np.sqrt(1.0 - (cosine_theta**2))
            axes = np.c_[np.cos(phi) * sine_theta, np.sin(phi) * sine_theta, cosine_theta]
            sphere_pts = np.empty((N_SPHERE_PTS, 3))
            for i in range(N_SPHERE_PTS):
                sphere_pts[i,:] = find_interface(waters[start_frame:end_frame], CoM, axes[i])
            sphere_r, sphere_z = best_fit_axial_sphere(sphere_pts)
            if args.local:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        sphere_a = np.sqrt((sphere_r**2) - (sphere_z**2))
                        sheet_radials = np.sqrt(np.sum(sheet_gridpts**2, axis=-1))
                        for _ in range(5):
                            nearby_idx = np.abs(sheet_radials - sphere_a) < CARBON_RADIUS
                            intersection_z = np.mean(mean_sheet[nearby_idx])
                            sphere_a = np.sqrt((sphere_r**2) - ((sphere_z - intersection_z)**2))
                        nearby_idx = np.abs(sheet_radials - sphere_a) < CARBON_RADIUS
                        intersection_z = np.mean(mean_sheet[nearby_idx])
                        sphere_angle = 90.0 + arcsin((sphere_z - intersection_z) / sphere_r)
                        nearby_grad = np.polyfit(sheet_radials[nearby_idx], mean_sheet[nearby_idx], 1)[0]
                        sphere_angle += arctan(nearby_grad) * 180.0 / np.pi
                    except RuntimeWarning:
                        sphere_angle = (180.0 if sphere_z > 0.0 else 0.0)
            else:
                if np.abs(sphere_z) < sphere_r:
                    sphere_a = np.sqrt((sphere_r**2) - (sphere_z**2))
                    sphere_angle = 90.0 + arcsin(sphere_z / sphere_r)
                else:
                    sphere_a = 0.0
                    sphere_angle = (180.0 if sphere_z > 0.0 else 0.0)
            results.sphere_r = sphere_r
            results.sphere_z = sphere_z
            results.sphere_a = sphere_a
            results.sphere_angle = sphere_angle
        
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

    contact_angle_inst_var = np.var(np.mean(contact_angles, axis=-1))

    #----------------------------------------------------------------------------------------------
    # Calculate time-averaged interface across all frames

    time_start_1 = time.time()
    with console.status('[green]Computing time-averaged interface...'):
        CoM_z = np.mean(waters[:,:,2])
        results = calculate_observables(0, N_frames, sphere_fit=True)

    console.print(f'Computed time-averaged interface in [green]{elapsed_time(time_start_1)}[/green].')
    
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
        plot_density_xz_slice(rot_waters, rot_carbons, ax[i // 3][i % 3], show_interface=True,
                              color_inter = (1.0, 0.0, 1.0, 0.4))
            
        for k in (0, int(N_azi / 2)):
            if args.local:
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

        max_r_coord = 1.5 * results.foot_r
        min_z_coord = np.min(carbons[:,:,2])
        max_z_coord = 1.5 * (results.sphere_z + results.sphere_r)
        r_bin_edges = np.linspace(0.0, max_r_coord, 81)
        z_bin_edges = np.linspace(min_z_coord, max_z_coord, 81)
        r_bin_centers = 0.5 * (r_bin_edges[1:] + r_bin_edges[:-1])
        r_bin_pad = 0.5 * (r_bin_edges[1] - r_bin_edges[0])
        z_bin_pad = 0.5 * (z_bin_edges[1] - z_bin_edges[0])
        bin_volumes = 2.0 * np.pi * ((r_bin_edges[1:]**2) - (r_bin_edges[:-1]**2)) * z_bin_pad
        flattened = waters.reshape(-1, 3)
        counts, _, _ = np.histogram2d(np.sqrt(np.sum(flattened[:,0:2]**2, axis=-1)),
                                      flattened[:,2], [r_bin_edges, z_bin_edges])
        counts /= (N_frames * bin_volumes[:,None])
        counts = np.swapaxes(counts, 0, 1)

        colors = np.zeros((80, 80, 4))
        colors[:,:,0] = np.clip((counts / BULK_DENSITY) - 1.0, a_min=0.0, a_max=1.0)
        colors[:,:,2] = np.clip(2.0 - (counts / BULK_DENSITY), a_min=0.0, a_max=1.0)
        colors[:,:,3] = np.clip((counts / BULK_DENSITY), a_min=0.0, a_max=1.0)
        ax.imshow(colors, origin='lower', extent=(-r_bin_pad, max_r_coord + r_bin_pad,
                                                  min_z_coord - z_bin_pad, max_z_coord + z_bin_pad))
        cmap = mplc.LinearSegmentedColormap('water_density', {'red':   [(0.0, 1.0, 1.0),
                                                                        (0.5, 0.0, 0.0),
                                                                        (1.0, 1.0, 1.0)],
                                                              'green': [(0.0, 1.0, 1.0),
                                                                        (0.5, 0.0, 0.0),
                                                                        (1.0, 0.0, 0.0)],
                                                              'blue':  [(0.0, 1.0, 1.0),
                                                                        (0.5, 1.0, 1.0),
                                                                        (1.0, 0.0, 0.0)]})
        norm = mplc.Normalize(vmin=0.0, vmax=(2 * BULK_DENSITY))
        fig.colorbar(ScalarMappable(norm, cmap), ax=ax, label=r'Number density ($\AA^{-3}$)',
                     fraction=0.046, pad=0.04)

        z_mean = np.empty((80,), dtype=float)
        z_stdev = np.empty((80,), dtype=float)
        z_bot = np.empty((80,), dtype=float)
        z_top = np.empty((80,), dtype=float)
        flattened = carbons.reshape(-1, 3)
        radials = np.sqrt(np.sum(flattened[:,0:2]**2, axis=-1))
        for i in range(80):
            sample = flattened[np.abs(radials - r_bin_centers[i]) < r_bin_pad, 2]
            z_mean[i] = np.mean(sample)
            z_stdev[i] = np.std(sample)
            z_bot[i] = np.min(sample)
            z_top[i] = np.max(sample)
        ax.fill_between(r_bin_centers, z_bot, z_top, color=(0.6, 0.6, 0.6, 0.25))
        ax.fill_between(r_bin_centers, z_mean - z_stdev, z_mean + z_stdev, color=(0.6, 0.6, 0.6, 0.25))
        ax.plot(r_bin_centers, z_mean, '-', color=(0.6, 0.6, 0.6))

        max_phi = np.arccos(np.clip(-results.sphere_z / results.sphere_r, -1.0, 1.0))
        phi = np.linspace(0, max_phi, 100)
        ax.plot(results.sphere_r * np.sin(phi), results.sphere_z + (results.sphere_r * np.cos(phi)), 'g-')
        ax.text(0.99, 0.99, (r'$\theta\;=\;' + f'{results.sphere_angle:.1f}' + r'\degree$'),
                ha='right', va='top', transform=ax.transAxes)
        ax.set_xlabel(r'r ($\AA$)')
        ax.set_ylabel(r'z ($\AA$)')
        ax.set_title('Spherical fit over histogram of water density')

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
        if N_blocks < 2:
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

        if N_frames < (5 * (args.N_blocksizes + 4)):
            raise RuntimeError('Too few frames in the trajectory for automatic blocksizing.')
        max_N_blocks = N_frames // 5
        attempted_N_blocks = np.linspace(5, max_N_blocks, args.N_blocksizes)
        N_blocks = np.array([int(round(n)) for n in attempted_N_blocks], dtype=int)
        N_blocks = np.unique(N_blocks)
        blocksizes = np.array([N_frames // n for n in N_blocks])
        blocksizes = np.unique(blocksizes)
        N_blocks = np.array([N_frames // n for n in blocksizes])
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

        stat_inefficiencies = blocksizes * contact_angle_block_vars / contact_angle_inst_var
        idx = int(round(0.75 * N_blocksizes))
        idx = np.argpartition(stat_inefficiencies, idx)[idx]

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
