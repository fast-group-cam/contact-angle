#! /usr/bin/env python

prog_desc_header = '''
===================================================================================================
 This program measures the contact angle of a water droplet on a graphene sheet from a simulated
 trajectory. The input is a file, which must be compatible with ASE's file i/o formats, describing
 either a time evolution or a single snapshot of a water droplet (rotationally symmetric about the
 z-axis) on a graphene sheet aligned aligned to the xy plane. The contact angle is calculated by
 finding the Willard-Chandler interface for a small number of testpoints at the droplet's foot,
 and calculating the direction of the plane.

 The program takes some number of frames `N_frames` from the input file (which is specified by the
 user). The action of the program depends on the user inputs:

   (1) WIP

===================================================================================================
'''

import sys
import os
import time
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from contact_angle.util import elapsed_time, read_droplet_trajectory
from contact_angle.util.droplet import find_interface
from contact_angle.util.droplet.coarse_grain import COARSE_GRAIN_LENGTH
from contact_angle.util.droplet.plot import plot_density_xz_slice

#==================================================================================================
# Helper function

def find_interfaces_and_normals(
        waters: np.ndarray,
        search_directions: np.ndarray,
        z_foot: float
        ) -> tuple[np.ndarray, np.ndarray]:
    """This function takes in a trajectory of water molecules on a graphene sheet, and finds the
time-averaged Willard-Chandler interface and its normals along a list of planar search directions.
The inputs are:

    - `waters`: The coordinates of the water molecules, with shape (N_frames, N_water, 3), which
    will be collated together.
    - `carbons`: The coordinates of the carbon atoms, with shape (N_frames, N_carbon, 3), which
    will be collated together.
    - `search_directions`: The directions to the search along, with shape (N_directions, 3); note
    that they should all be perpendicular to the z-axis!
    - `z_foot`: The defining height of the droplet foot above the droplet floor.
    - `carbon_radius_sq`: The square of the cutoff radius for searching for nearby carbon atoms
    to define the local sheet z-coordinate.

The output is a tuple of two NDArrays, both of shape (N_directions, 3), representing the locations
and normals of the interface points along the search directions."""

    # Sanitizing input shapes
    if len(waters.shape) != 3 or waters.shape[-1] != 3:
        raise RuntimeError(f'Unrecognized input shape: waters {waters.shape}, should be ' +
                           '(N_frames, N_water, 3)')

    # Calculate droplet CoM and floor
    CoM = np.mean(waters, axis=(0, 1))
    droplet_h = find_interface(waters, CoM, (0, 0, 1))[2]
    droplet_floor = find_interface(waters, CoM, (0, 0, -1))[2]
    if droplet_h - droplet_floor < z_foot:
        warnings.warn('Droplet is too short at the CoM, bad behaviour may occur')
    
    # Iterate through search directions
    interfaces = list()
    normals = list()
    for search_dir in search_directions:

        # First guess of interface
        inter = find_interface(waters, (0, 0, droplet_floor + z_foot), search_dir)

        # Find z-coordinate of droplet floor underneath first guess
        local_floor = find_interface(waters, inter, (0, 0, -1))[2]

        # Second guess of interface (and repeat refinement)
        step_back = min(np.dot(inter, search_dir), 2 * COARSE_GRAIN_LENGTH) * search_dir
        inter = find_interface(waters, (inter[0] - step_back[0], inter[1] - step_back[1],
                                        local_floor + z_foot), search_dir)
        local_floor = find_interface(waters, inter, (0, 0, -1))[2]

        # Third and final guess of interface
        step_back = min(np.dot(inter, search_dir), 2 * COARSE_GRAIN_LENGTH) * search_dir
        inter, norm = find_interface(waters, (inter[0] - step_back[0], inter[1] - step_back[1],
                                              local_floor + z_foot), search_dir, calc_normal=True)
        interfaces.append(inter)
        normals.append(norm)

    return (np.array(interfaces), np.array(normals))

#==================================================================================================
# Start of program flow

if __name__ == "__main__":

    #----------------------------------------------------------------------------------------------
    # Script default parameters

    N_AZIMUTHS = 60       # Number of azimuthal directions to analyze per frame
    Z_FOOT = 5.0          # Height of the droplet foot above the graphene sheet (in angstroms)
    MAX_TAU = 25          # Maximum timescale to calculate autocorrelations (in number of frames)
    N_SPHERE_PTS = 100    # Number of points to use to find best-fit spherical top

    from contact_angle.util.graphene.angle import CUTOFF_RADIUS as CARBON_RADIUS
    CARBON_RADIUS_SQ = CARBON_RADIUS**2

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
    parser.add_argument('--block-average', action='store_true', dest='block_average',
                        help=('perform reverse cumulative averaging for unbiased uncertainty ' +
                              'estimates over varying block sizes'))
    parser.add_argument('-b', '--blocksize', type=int, default=None, dest='blocksize',
                        help=('if --block-average is turned on, disables automatic block sizing ' +
                              'and enforces specified block size'))
    parser.add_argument('-o', '--output', default='contact-angle', dest='output',
                        help='output folder to save log and graphical outputs to')
    args = parser.parse_args()

    for file in args.input_file:
        if not os.path.isfile(file):
            raise RuntimeError(f'File "{file}" not found.')

    if args.N_azimuths < 1:
        raise RuntimeError(f'N_azimuths ({args.N_azimuths}) must be positive.')
    if args.z_foot < 0.0:
        raise RuntimeError(f'z_foot ({args.z_foot}) must be positive.')
    if args.max_tau < 2:
        raise RuntimeError(f'Max tau ({args.max_tau}) must be at least 2.')
    if args.blocksize is None:
        args.opt_auto = True
    elif args.blocksize < 1:
        raise RuntimeError(f'Block size ({args.blocksize}) must be positive.')

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    #----------------------------------------------------------------------------------------------
    # Helper functions

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

    def sphere_fit(points):
        A_mat = np.empty((points.shape[0], 4), dtype=float)
        A_mat[:,0:3] = 2 * points
        A_mat[:,3] = 1
        f_vec = np.empty((points.shape[0], 1), dtype=float)
        f_vec[:,0] = np.sum(np.square(points), axis=-1)
        c_vec, _, _, _ = np.linalg.lstsq(A_mat, f_vec)
        radius = np.sqrt(np.sum(np.square(c_vec[0:3,0])) + c_vec[3,0])
        return (radius, c_vec[0:3,0])

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

    #----------------------------------------------------------------------------------------------
    # Read input file and save coordinates

    if len(args.input_file) == 1:
        print(f'Reading "{args.input_file[0]}"...', end='')
    else:
        print(f'Reading {len(args.input_file)} files...', end='')
    time_start_0 = time.time()
    cell_params, waters, carbons, _ = read_droplet_trajectory(args.input_file, index=args.index)
    N_frames = waters.shape[0]
    N_water = waters.shape[1]
    N_carbon = carbons.shape[1]
    print(f'read {N_frames} frames of {N_water} water molecules and {N_carbon} carbon atoms in ' +
          f'{elapsed_time(time_start_0)}.')

    #----------------------------------------------------------------------------------------------
    # Create output log file

    log_file = open(os.path.join(args.output, 'log.txt'), 'w', encoding='utf-8')

    #----------------------------------------------------------------------------------------------
    # Calculate instantaneous interfaces for every frame and measure autocorrelations etc.

    time_start_1 = time.time()
    print('Computing instantaneous interfaces for every frame...')

    azi = np.linspace(0, 2 * np.pi, args.N_azimuths, endpoint=False)
    search_directions = np.c_[np.cos(azi), np.sin(azi), np.zeros_like(azi)]
    search_perp = np.c_[-np.sin(azi), np.cos(azi)]

    contact_angles = np.empty((N_frames, args.N_azimuths), dtype=float)
    ooplane_angles = np.empty((N_frames, args.N_azimuths), dtype=float)

    for f in range(N_frames):

        interfaces, normals = find_interfaces_and_normals(waters[f:f+1], search_directions, args.z_foot)
        
        if args.local:
            for i, inter in enumerate(interfaces):
                nearby = carbons[f, np.sum(np.square(carbons[f,:,0:2] - inter[0:2]), axis=-1) < CARBON_RADIUS_SQ]
                local_norm = np.linalg.svd(nearby - np.mean(nearby, axis=0), full_matrices=False)[2][-1]
                local_norm /= np.linalg.norm(local_norm)
                if local_norm[2] < 0:
                    local_norm *= -1
                contact_angles[f, i] = np.arccos(np.dot(normals[i], local_norm)) * 180 / np.pi
        else:
            contact_angles[f] = np.arccos(np.dot(normals, (0, 0, 1))) * 180 / np.pi

        proj_normals = np.power(np.sum(np.square(normals[:,0:2]), axis=-1), -0.5)[:,None] * normals[:,0:2]
        ooplane_angles[f] = np.arcsin(np.sum(proj_normals * search_perp, axis=-1)) * 180 / np.pi
    
    time_start_2 = time.time()
    print(f'    - computed instantaneous interfaces in {elapsed_time(time_start_1)}...')
    print('    - calculating time correlations...', end='')

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

    print(f'done in {elapsed_time(time_start_2)}.')
    print('    - calculating azimuthal correlations...', end='')
    time_start_2 = time.time()

    max_a = int(args.N_azimuths / 2)
    contact_angle_azi_corrs = np.empty((max_a,), dtype=float)
    contact_angle_azi_corrs[0] = np.mean(np.square(contact_angles), axis=(0,1))
    for a in range(1, max_a):
        contact_angle_azi_corrs[a] = np.mean(contact_angles[:,:-a] * contact_angles[:,a:], axis=(0,1))
    contact_angle_azi_corrs[:] /= contact_angle_azi_corrs[0]
    contact_angle_azi_popt = exp_fit(contact_angle_azi_corrs, k0=1.0, c0=0.978)
    contact_angle_azi_popt[1] *= args.N_azimuths / 360

    ooplane_angle_azi_corrs = np.empty((max_a,), dtype=float)
    ooplane_angle_azi_corrs[0] = np.mean(np.square(ooplane_angles), axis=(0,1))
    for a in range(1, max_a):
        ooplane_angle_azi_corrs[a] = np.mean(ooplane_angles[:,:-a] * ooplane_angles[:,a:], axis=(0,1))
    ooplane_angle_azi_corrs[:] /= ooplane_angle_azi_corrs[0]
    ooplane_angle_azi_corrs = np.abs(ooplane_angle_azi_corrs)
    ooplane_angle_azi_popt = exp_fit(ooplane_angle_azi_corrs, k0=10.0, c0=0.05)
    ooplane_angle_azi_popt[1] *= args.N_azimuths / 360

    print(f'done in {elapsed_time(time_start_2)}.')
    print(f'Done in {elapsed_time(time_start_1)}.')

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
    fig.savefig(os.path.join(args.output, 'inst_contact_angles.png'), dpi=(3*fig.dpi),
                bbox_inches='tight', pad_inches=0.05)
    
    fig = plt.figure(figsize=(14, 7), layout='constrained')
    subfigs = fig.subfigures(1, 2, wspace=0.06, width_ratios=[1.5, 1.0])
    plot_against_time_and_azimuth(subfigs[0], subfigs[0].subplots(), ooplane_angles,
                                  'Instantaneous interfacial out-of-plane angles', r'$\delta$')
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
    fig.savefig(os.path.join(args.output, 'inst_out-of-plane_angles.png'), dpi=(3*fig.dpi),
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

    #----------------------------------------------------------------------------------------------
    # Calculate time-averaged interface across all frames

    time_start_1 = time.time()
    print('Computing time-averaged interface...')

    azi = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    search_directions = np.c_[np.cos(azi), np.sin(azi), np.zeros_like(azi)]
    search_perp = np.c_[-np.sin(azi), np.cos(azi)]
    interfaces, normals = find_interfaces_and_normals(waters, search_directions, args.z_foot)
    if args.local:
        flat_carbons = carbons.reshape(-1, 3)
        local_carbons_c = np.empty((interfaces.shape[0], 3), dtype=float)
        local_carbons_n = np.empty((interfaces.shape[0], 3), dtype=float)
        contact_angles = np.empty((12,), dtype=float)
        for i, inter in enumerate(interfaces):
            down = np.cross(np.array((search_perp[i,0], search_perp[i,1], 0.0)), normals[i])
            foot = inter - ((inter[2] / down[2]) * down)
            nearby = flat_carbons[np.sum(np.square(flat_carbons[:,0:2] - foot[0:2]), axis=-1) < CARBON_RADIUS_SQ]
            local_carbons_c[i] = np.mean(nearby, axis=0)
            local_carbons_n[i] = np.linalg.svd(nearby - local_carbons_c[i], full_matrices=False)[2][-1]
            local_carbons_n[i] /= np.linalg.norm(local_carbons_n[i])
            if local_carbons_n[i,2] < 0:
                local_carbons_n[i] *= -1
            contact_angles[i] = np.arccos(np.dot(normals[i], local_carbons_n[i])) * 180 / np.pi
    else:
        contact_angles = np.arccos(np.dot(normals, (0, 0, 1))) * 180 / np.pi

    proj_normals = np.power(np.sum(np.square(normals[:,0:2]), axis=-1), -0.5)[:,None] * normals[:,0:2]
    ooplane_angles = np.arcsin(np.sum(proj_normals * search_perp, axis=-1)) * 180 / np.pi

    time_start_2 = time.time()
    print(f'    - computed time-averaged interface in {elapsed_time(time_start_1)}.')
    print('    - finding best-fit sphere for upper surface...', end='')

    phi = np.random.random(N_SPHERE_PTS) * 2 * np.pi
    tau = np.random.random(N_SPHERE_PTS)
    search_directions = np.c_[np.sqrt(1 - np.square(tau)) * np.cos(phi), np.sqrt(1 - np.square(tau)) * np.sin(phi), tau]
    sphere_points = list()
    CoM = np.mean(waters, axis=(0,1))
    for i in range(N_SPHERE_PTS):
        sphere_points.append(find_interface(waters, CoM, search_directions[i]))
    sphere_points = np.array(sphere_points)
    sphere_r, sphere_c = sphere_fit(sphere_points)

    print(f'done in {elapsed_time(time_start_2)}.')
    time_start_2 = time.time()
    print('    - plotting time-averaged density functions...', end='')

    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(15, 6)
    interval = max(int(N_frames * N_water / 2e5), 1)
    for i in range(2):
        for j in range(3):

            idx = (3 * i) + j
            angle = azi[idx]
            rot_matrix = np.array(((np.cos(angle),  np.sin(angle), 0.0),
                                   (-np.sin(angle), np.cos(angle), 0.0),
                                   (0.0,            0.0,           1.0)))
            rot_waters = np.einsum('kl,ijl->ijk', rot_matrix, waters[::interval])
            rot_carbons = np.einsum('kl,ijl->ijk', rot_matrix, carbons[::interval])
            plot_density_xz_slice(rot_waters, rot_carbons, ax[i][j], show_interface=True,
                                  color_inter = (1.0, 0.0, 1.0, 0.4))
            
            for k in (0, 6):
                if args.local:
                    rot_carbon_c = rot_matrix @ local_carbons_c[idx + k]
                    rot_carbon_n = rot_matrix @ local_carbons_n[idx + k]
                    a_x = rot_carbon_c[0] - CARBON_RADIUS
                    a_z = rot_carbon_c[2] + (CARBON_RADIUS * rot_carbon_n[0] / rot_carbon_n[2])
                    b_x = rot_carbon_c[0] + CARBON_RADIUS
                    b_z = rot_carbon_c[2] - (CARBON_RADIUS * rot_carbon_n[0] / rot_carbon_n[2])
                    ax[i][j].plot((a_x, b_x), (a_z, b_z), 'k-')
                rot_inter = rot_matrix @ interfaces[idx + k]
                rot_norm = rot_matrix @ normals[idx + k]
                a_x = rot_inter[0] + (rot_inter[2] * rot_norm[2] / rot_norm[0])
                b_x = rot_inter[0] - (2 * rot_inter[2] * rot_norm[2] / rot_norm[0])
                ax[i][j].plot((a_x, b_x), (0.0, 3 * rot_inter[2]), '-', color=(1.0, 0.0, 1.0))
            ax[i][j].plot((0.0,), (CoM[2],), '.', color=(1.0, 0.0, 1.0))
            ax[i][j].text(0.01, 0.99, (r'$\theta_{left}\;=\;' + f'{contact_angles[idx + 6]:.1f}' +
                                       r'\degree$' + '\n' + r'$\theta_{right}\;=\;' +
                                       f'{contact_angles[idx]:.1f}' + r'\degree$'), ha='left',
                                       va='top', transform=ax[i][j].transAxes)

            rot_sphere_c = rot_matrix @ sphere_c
            proj_sphere_r_sq = (sphere_r**2) - (rot_sphere_c[1]**2)
            if proj_sphere_r_sq > 0.0:
                a_x = np.sqrt(proj_sphere_r_sq - (max(CoM[2] - rot_sphere_c[2], 0.0)**2))
                b_x = np.linspace(-a_x, a_x, 100)
                ax[i][j].plot(rot_sphere_c[0] + b_x, rot_sphere_c[2] + np.sqrt(proj_sphere_r_sq - np.square(b_x)),
                              '-.', color=(0.0, 0.5, 0.0, 0.5))
            ax[i][j].plot((rot_sphere_c[0],), (rot_sphere_c[2],), '.', color=(0.0, 0.55, 0.0))

            ax[i][j].set_title(r'$\varphi\;=\;' + f'{(angle * 180 / np.pi):.0f}' + r'\degree$')

    fig.suptitle('Azimuthal cross-sections of time-averaged droplet')
    fig.savefig(os.path.join(args.output, 'ave_cross_sections.png'), dpi=(3*fig.dpi),
                bbox_inches='tight', pad_inches=0.05)
    
    print(f'done in {elapsed_time(time_start_2)}.')
    print(f'Done in {elapsed_time(time_start_1)}.')

    log_file.write('-------------------------\n')
    log_file.write(' Time-averaged interface\n')
    log_file.write('-------------------------\n\n')
    log_file.write(f'Mean contact angle = {np.mean(contact_angles)} [deg]\n')
    log_file.write(f'Median contact angle = {np.median(contact_angles)} [deg]\n')
    log_file.write(f'Contact angle stdev = {np.std(contact_angles)} [deg]\n\n')
    log_file.write(f'Mean out-of-plane angle = {np.mean(ooplane_angles)} [deg]\n')
    log_file.write(f'Median out-of-plane angle = {np.median(ooplane_angles)} [deg]\n')
    log_file.write(f'Out-of-plane angle stdev = {np.std(ooplane_angles)} [deg]\n\n')
    log_file.write(f'Center-of-mass z-height = {CoM[2]} [A]\n')
    log_file.write(f'Best-fit spherical center z-height = {sphere_c[2]} [A]\n')
    log_file.write(f'Best-fit spherical radius = {sphere_r} [A]\n\n')

    #----------------------------------------------------------------------------------------------
    # Calculate time-averaged interface across all frames

    if args.block_average:
        print('Sorry, block averaging not implemented yet -- WIP!')

    #----------------------------------------------------------------------------------------------
    # End of program

    print(f'Program completed in {elapsed_time(time_start_0)}.')
    log_file.close()
    sys.exit()
