"""
fast-group-cam/contact-angle-2d
=====

Package to analyse liquid droplet contact angles on 2D materials from atomistic simulations.

Notes
-----
For simple usage, this package comes with a ready-to-use Python script (`contact_angle.py`), which
can be used out-of-the-box with various options to measure the liquid-solid contact angle from a
simulation trajectory. The script can also be run from the command line as `contact_angle`, if this
package is loaded into the console environment.

For any fancier usage, the modules in this package can be imported, and used in further Python
code. All methods are documented using numpy-style docstrings. Internally, this package relies
almost entirely on handling "raw" numpy ndarrays directly, instead of using a dedicated molecular
dynamics analysis framework (e.g. ASE, MDTraj, MDAnalysis, ...), except for reading and writing
files where the ASE I/O module is used. This has the advantage of being performant and avoiding
weird quirks in how other packages handle data, but admittedly might make it harder to integrate
this package into pre-existing analysis pipelines. If you intend to use this package in your code,
be aware that the interfaces are entirely very "low-level", and deal directly with numpy arrays of
atomic coordinates!

Conventions
-----------
The package always assumes that the simulation was performed in either the NVE or NVT ensemble. The
critical assumption is that the simulation was performed using fixed periodic boundary conditions,
i.e. at constant volume with constant unit cell vectors, and with all boundaries fully periodic,
and a constant number of liquid and solid particles.

The package also assumes that the simulation was set up so that the solid is nominally aligned to
the xy-plane, i.e. z-normal, as the contact angle will be scanned using the z-axis as the axis of
symmetry. Otherwise, the simulation does not need to be re-centred or re-aligned during the time
evolution, and the alignment of the solid surface to the z-normal does not need to be exact nor
instantaneously true for every frame, only approximately true in the long time and space average.

All lengths, coordinates, and positions are measured in angstroms, and all times are measured in
picoseconds.

For a system of N particles, the package expects numpy arrays of shape (N, 3) to describe the
instantaneous positions of the particles in a single frame, and shape (M, N, 3) to describe the
trajectory of the particles across M frames.
"""

from .__version__ import __version__

from .interpolate import PeriodicGridInterpolator
from .autocorrelations import autocorrelation, norm_inf_autocorrelation
from .misc import elapsed_time
from .coordinates import center_coordinates

