# fast-contact-angle-2d

A Python package for the analysis of liquid droplet contact angles on 2D materials from atomistic simulation trajectories, written by the FAST group.

TODO: change everything below...

A repository for the analysis of contact angles and other observables, with regards to water droplets on graphene sheets. All code is written entirely in Python. To use this package and install its dependencies:

1. Create a Python virtual environment and activate it;
2. Install this package by running `pip install .` in the repository directory.

Afterwards, any of the analysis scripts can be run as `python /path/to/repo/droplet_graphene_analysis/scripts/<script_name.py> [inputs]`, or directly from the command line as just `<script_name> [inputs]`, while the virtual environment is active. The following scripts are available:

- `contact_angle`: measures the contact angle of a water droplet on a graphene sheet from an NVT trajectory, with both instantaneous dynamics and time-averaged macroscopics.
- `graphene_dynamics`: takes an NVT trajectory of a graphene sheet, and calculates certain observables with regards to local dynamics.
- `make_droplet_movie`: takes an NVT trajectory of a water droplet on graphene, and renders a custom movie to a MP4 file. (N.B. This script requires that FFMPEG is installed and locally available, *separately* from this package's dependencies.)
- `water_density`: measures the density of a water droplet on a graphene sheet from an NVT trajectory.

Consult the help pages (e.g. `<script_name> --help`) for the input arguments of each script.

The module `droplet_graphene_analysis.util` can also be imported directly into Python code, for a direct low-level interface to the methods.


## Methods

If you use this package, please cite:

> [awaiting publication]


## Author

**Darren Wayne Lim** (dwl38@cam.ac.uk) as part of the **FAST Group, Theory of Condensed Matter, Cavendish Laboratory, Department of Physics, University of Cambridge** (https://www.fast-group.phy.cam.ac.uk/)
