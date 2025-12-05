# fast-contact-angle-2d

A Python package for the analysis of liquid droplet contact angles on free-standing 2D materials from atomistic simulation trajectories. This package was written as part of research work under the FAST group, University of Cambridge.

All code is written entirely in Python, and wrapped into an installable Python package. The package contains three scripts, which are ready to run out-of-the-box for "simple" usage; alternatively, the modules can be imported to assist in your own analysis scripts. To use this package and install its dependencies:

1. Create a Python virtual environment and activate it;
2. Install this package by running `pip install .` in the repository directory.

Afterwards, any of the premade scripts can be run as `python /path/to/repo/fast_contact_angle_2d/scripts/<script_name.py> [inputs]`. The scripts can also be launched directly from the command line as just `<script_name> [inputs]` while the virtual environment is active. Consult the help pages (e.g. `<script_name> --help`) for the input arguments of each script.

If you use this package, please cite:

> [awaiting publication]


## `contact_angle`

This script measures the contact angle of a liquid droplet on a solid surface, from the trajectory of an atomistic simulation in either the NVE or NVT ensemble. See `contact_angle --help` for details on the input arguments.

<ins>**Example 1</ins>.** A simulation of water on graphene was performed using ASE, with the trajectory being saved in "trajectory.xyz" every 0.1 ps. To analyze the contact angle in a rotationally-symmetrized manner, the following command can be executed:

```
contact_angle trajectory.xyz --liq_symbol O --sol_symbol C --delta_t 0.1
```

in which case the results will be saved into a folder named "contact-angle".

<ins>**Example 2</ins>.** A simulation of Lennard-Jones fluid on hexagonal boron nitride was performed using LAMMPS, with a timestep of 1 fs, and particle assignments being 1 for the Lennard-Jones fluid, 2 for boron atoms, and 3 for nitrogen atoms. The trajectory was saved once every 0.1 ps, i.e. every 100 simulation steps, and broken up into two files "traj_0.lammpstrj" and "traj_1.lammpstrj". Let's say we want to store the results of the contact angle analysis in a subfolder "analysis/angle"; since the .lammpstrj format does not store information about the atomic identity, and only reports timestamps in multiples of the simulation step, we need to specify the particles based on their internal index, and also the rescaling factor to multiply the timesteps by:

```
contact_angle traj_0.lammpstrj traj_1.lammpstrj -o analysis/angle --liq_number 1 --sol_number 2 3 --time_rescale_factor 0.001
```

Alternatively, specifying `--delta_t=0.1` would also work the same for this specific example. The distiction between the `delta_t` and `time_rescale_factor` arguments is that `delta_t` sets the time interval between frames (in picoseconds) directly, which means that the parameter should be adjusted if the trajectory had been saved using a different dumping interval, whereas using the `time_rescale_factor` causes the script to attempt to detect the timestep information from the source file(s), and then multiply the internal units within the source file(s) to convert to picoseconds, which means that (for LAMMPS dump files at least) the correct value of the `time_rescale_factor` parameter depends on the simulation timestep rather than the dumping interval.

<ins>**Example 3</ins>.** A simulation of water on hexagonal boron nitride using ASE and saved into "trajectory.xyz", but this time the contact angle should be analyzed anisotropically as the droplet is moving with constant velocity across the membrane. To save the results into a folder named "contact-angle-aniso", the following command can be executed:

```
contact_angle trajectory.xyz -o contact-angle-aniso --liq_symbol O --sol_symbol B N --delta_t 0.1 --anisotropic
```


## `analyze_surface_dynamics`

This script calculates certain observables relevant to the spatially-localized dynamics associated to ripples on free-standing 2D materials, from the trajectory of an atomistic simulation in either the NVE or NVT ensemble. The four observables calculated are 1. the time-averaged heightmap (i.e. mean z-coordinate); 2. the standard deviation of instantaneous heightmaps (i.e. fluctuation width of z-coordinates); 3. the time-averaged local inclination angle; and 4. the infinite-time limit of the normalized temporal autocorrelation function of the instantaneous local inclination angles. See `analyze_surface_dynamics --help` for details on the input arguments.

<ins>**Example</ins>.** A simulation of water on graphene was performed using ASE, with the trajectory being saved in "trajectory.xyz" every 0.1 ps. To analyze the rippling dynamics on the sheet in droplet-centred coordinates, the following command can be executed:

```
analyze_surface_dynamics trajectory.xyz --liq_symbol='O' --sol_symbol='C' --delta_t=0.1
```

in which case the results will be saved into a folder named "surface-dynamics".


## `droplet_diffusion`

This program measures the diffusive motion of a liquid droplet on a solid surface, from the trajectory of an atomistic simulation in either the NVE or NVT ensemble. See `droplet_diffusion --help` for details on the input arguments.

<ins>**Example</ins>.** A simulation of water on graphene was performed using ASE, with the trajectory being saved in "trajectory.xyz" every 0.1 ps. To analyze the motion of the droplet on the graphene sheet, the following command can be executed:

```
droplet_diffusion trajectory.xyz --liq_symbol='O' --sol_symbol='C' --delta_t=0.1
```

in which case the results will be saved into a folder named "droplet-diffusion".


## Using the modules in your own code

For direct low-level access to the methods (e.g. usage within your own analysis code), the relevant modules can be imported into Python via `import fast_contact_angle_2d`. The submodules are:

- `fast_contact_angle_2d.liquid` contains methods relevant to the calculation of time-averaged coarse-grained liquid densities, liquid interfaces, and the best-fit sphere of the faraway liquid interface;
- `fast_contact_angle_2d.solid` contains methods relevant to the analysis of free-standing 2D solid materials, in particular the dynamics of the continuous heightmap $h(x,y)$ and the local inclination angle;
- `fast_contact_angle_2d.util.io` contains the necessary file-reading functionality to read and process atomistic simulation trajectory files.

Each method is documented by its docstring (which hopefully is complete enough to figure out the expected syntax etc.). For examples of how to use these methods, it might be a good idea to reverse-engineer the premade "contact_angle" script as a starting point.


## Methods

On the analysis of liquids and liquid interfaces, the starting basis is the time-averaged coarse-grained density function, which is defined as:

```math
\rho (\mathbf{r}) \;=\; \frac{1}{T}\int_{0}^{T} \sum_{i=1}^{N_{\text{liq}}} \left(2\pi\xi^2\right)^{-3/2} \exp\!\left[-\frac{\left|\mathbf{r}-\mathbf{R}_{i}(t)\right|^2}{2\xi^2}\right] \,\mathrm{d}t
```

for coarse-graining parameter $\xi$ and averaging period $T$, where $\mathbf{R}_{i}(t)$ denotes the position of the $i$<sup>th</sup> liquid particle at time $t$. Note that the dual limit $\xi\to 0$ and $T\to\infty$ yields the thermodynamic definition of particle density, whereas the case of a fixed value of $\xi$ and $T$ spanning a single instant recovers the Willard-Chandler instantaneous coarse-grained density (DOI:[10.1021/jp909219k](https://doi.org/10.1021/jp909219k)).

The liquid interface is then defined as the isosurface where $\rho (\mathbf{r})$ reaches half of the mean value of the bulk liquid phase. Again, the dual limit $\xi\to 0$ and $T\to\infty$ yields the thermodynamic definition of the Gibbs equimolar dividing surface, whereas the case of fixed $\xi$ and $T$ spanning a single instant recovers the Willard-Chandler instantaneous interface.

On the analysis of free-standing 2D materials, the starting basis is the instantaneous heightmap function $h(x, y; t)$, which is a $C^2$ smooth function of minimal curvature satisfying $z(t) = h(x(t), y(t); t)$ for all solid particles of time-varying position $(x(t), y(t), z(t))$. The time-averaged heightmap $\langle h \rangle (x, y)$ is thus a useful representation of the structure of the non-flat solid surface in the thermodynamic limit.

From the instantaneous heightmap $h(x, y; t)$, the local inclination angle can be defined as the angle which the normal vector of a local neighbourhood of the surface makes with the z-axis:

```math
\theta (x, y; t) \;=\; \arctan\left( \nabla_{x,y} \tilde{h}(x, y; t)\right)
```

where $\tilde{h} = h \ast \Pi_{\sigma}$ is the instantaneous spatially-regularized heightmap, constructed by convolving the instantaneous heightmap with a uniform disk function $\Pi_{\sigma}$ of radius $\sigma$ in order to smooth out sharp gradients. In particular, the normalized temporal autocorrelation of this local inclination angle is of interest:

```math
\mathcal{C}_{\theta}(x, y; \tau) \;=\; \frac{\left\langle \theta (x, y; t + \tau) \theta (x, y; t) \right\rangle}{\left\langle \left(\theta (x, y) \right)^2 \right\rangle}
```

especially in the long time limit $\tau\to\infty$, as this is a good indicator of phase transitions in vibrational dynamics (see DOI:[10.1073/pnas.2416932122](https://doi.org/10.1073/pnas.2416932122)).

The contact angle between a liquid droplet and a free-standing solid surface can then be defined as the angle of intersection between a best-fit sphere of the faraway portion of the liquid interface, and the time-averaged heightmap of the solid surface.


## Author

**Darren Wayne Lim** (dwl38@cam.ac.uk) as part of the **FAST Group, Theory of Condensed Matter, Cavendish Laboratory, Department of Physics, University of Cambridge** (https://www.fast-group.phy.cam.ac.uk/)
