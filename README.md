# contact-angle

A repository for the analysis of contact angles and other observables, with regards to water droplets on graphene sheets. All code is written entirely in Python, and only requires `numpy`, `scipy`, `matplotlib`, and `ase` (which are automatically included as package dependencies). To use this package:

1. Create a Python virtual environment and activate it;
2. Install this package by running `pip install .` in the repository directory.

Afterwards, any of the analysis scripts can be run from the command line as `python /path/to/repo/contact_angle/scripts/<script-name.py> [inputs]`. The module `contact_angle.util` can also be imported directly into Python scripts, for a direct low-level interface to the methods.


## scripts/contact-angle.py

A python script, which measures the contact angle of a water droplet on a graphene sheet from a trajectory. This is done by finding the Willard-Chandler interface<sup>[1]</sup> for a small number of testpoints at the droplet's foot, and then calculating the direction of the best-fit plane of the interface. The script can be run by calling the following syntax (with optional arguments):

```
python contact-angle.py <filename> [--N_frames <N_FRAMES>] [--interval <INTERVAL>] [--N_azimuths <N_AZIMUTHS>] [--block-average] [--auto] [--blocksize <BLOCKSIZE>] [--units <UNITS>] [-o <OUTPUT_FILENAME>] [--no-save] [--no-display]
```

The action of the script depends on the parameters:

1. If `N_frames` is 1 (default), the contact angle is measured as an instantaneous 'snapshot' from the **last** frame of the file. The reported value is averaged over some number of azimuthal slices (specified by `N_azimuths`), and the reported uncertainty is the standard error of the mean. The program also generates a plot of the water molecules' number density distribution, and best-fit planes, for each azimuthal slice.

2. If `N_frames` is greater than 1, and `block-average` is false (default), the contact angle is measured for each and every frame, and the reported value is averaged over all azimuthal slices across all frames (with reported uncertainty being the standard error of the mean). The program then generates a plot of the contact angles measured for each azimuthal slice for each frame. Note that the frames are sliced from the **start** of the file, with slicing interval specified by `--interval`.

3. If `N_frames` is greater than 1, `block-average` is true, and `auto` is true or `b` is unspecified, the program will proceed as per mode #2; except that, instead of reporting a na&#239;ve mean across the frames, the program divides the sampled frames into consecutive 'blocks', and reports the mean of the median contact angles within each block. The program decides the optimal blocksize to use by testing a range of possible blocksizes, and selecting a blocksize which yields the 75<sup>th</sup> percentile highest statistical inefficiency<sup>[2]</sup> amongst the possible blocksizes.

4. If `N_frames` is greater than 1, `block-average` is true, `auto` is false (default), and `b` is specified, the program will proceed as per mode #3, but the reported value and uncertainty will be calculated using the user-specified blocksize instead of the automatically-determined blocksize.

The reported value is printed directly to console, and the final graphical output is saved to disk in the working directory.

### Options

- `<filename>` The path to the input file. The script simply calls `ase.io.read(...)` directly on the supplied filename, so the file format **is deduced automatically from the file extension**; as such, the file format and name must be natively compatible with [ASE's I/O formats](https://wiki.fysik.dtu.dk/ase/ase/io/io.html).

- `--N_frames <N_FRAMES>` The number of frames to extract from the start of the input file, if greater than 1. The program will safely truncate sampling if the input file contains fewer frames than specified. If `N_frames` is set to 1 or left unspecified, the program will run in mode #1 and process the end of the input file instead.

- `--interval <INTERVAL>` The slicing interval for the extraction of frames, if `N_frames` is greater than 1. Ignored if `N_frames` is 1.

- `--N_azimuths <N_AZIMUTHS>` The number of azimuthal slices to analyze per frame. Note that, for each azimuthal slice, two azimuthal directions are sampled (in reciprocal pairs), so the total number of samples is twice of `N_azimuths`.

- `--block-average` Enables block averaging, in the form of taking block medians for consecutive blocks of frames, in order to produce an unbiased estimate for a time-correlated series.

- `--auto` Enforces the automatic determination of block size, overriding any user-specified `blocksize`, if `block-average` is turned on.

- `--blocksize <BLOCKSIZE>` Disables the automatic determination of block size and enforces the user-specified block size, if `block-average` is turned on, unless `auto` was turned on.

- `--units <UNITS>` A string specifying the length units of the coordinates in the input file (e.g. "fm"), if ASE cannot automatically recognize the length scale. The script works internally using Angstroms as the default length unit, hence if ASE does not recognize the length scale and `units` is unspecified then the raw data in the input file will be treated as being in Angstroms. **Warning:** this option works by manually rescaling lengths, hence `units` should not be specified if ASE is able to read them automatically, otherwise the conversion factor will be applied twice!

- `-o <OUTPUT_FILENAME>` The output file to save the graphical output to; defaults to "output.png" within the working directory. Ignored if `no-save` is turned on.

- `--no-save` Disables the saving of graphical output to the output file. Note that this is independent from `no-display`, e.g. it is possible to display the graphical output to the screen without saving it to file.

- `--no-display` Disables the display of graphical output to the screen. Note that this is independent from `no-save`, e.g. it is possible to save the graphical output to file without displaying it.


## scripts/make-droplet-movie.py

A Python script which takes a trajectory of a water droplet on graphene, displays it in a custom visualization, and renders the movie to a MP4 file using FFMPEG. Use as:

```
python make-droplet-movie.py <input_file> [-o <output_file>] [--index <index>]
```

The trajectory is assumed to be in the NVT ensemble with periodic boundary conditions (i.e. the simulation box lengths are fixed).


## scripts/plot-graphene.py

A Python script which takes a trajectory of a graphene sheet (which may or may not have a water droplet on it), calculates its local inclination angle autocorrelation function, and plots it to a file. Use as:

```
python plot-graphene.py <input_file> [-o <output_file>] [--max_tau <max_tau>] [--N_x <N_x>] [--N_y <N_y>] [--max_threads <max_threads>] [--no-display]
```


## util/droplet

The module `contact_angle.util.droplet` provides a direct low-level interface to four methods regarding the water droplet and its Willard-Chandler interface<sup>[1]</sup>:

- `center_coordinates`: reads an ASE Atoms object, and recenters the coordinates across periodic boundary conditions so that the graphene sheet is at the z = 0 plane, the droplet's CoM is on the x = y = 0 axis, and all atomic coordinates are within the first periodic image.

- `coarse_grained_density`: calculates the coarse-grained density function defined by Willard and Chandler.

- `coarse_grained_density_grad`: calculates the gradient of the coarse-grained density function.

- `find_interface`: given a "search origin" and "search direction", finds the intersection of the Willard-Chandler interface with a ray extending from the origin in the supplied direction.

Additionally, the `contact_angle.util.droplet.plot` submodule provides functions which plot a coarse-grained density distribution directly to a MatPlotLib axis.


## util/graphene

The module `contact_angle.util.graphene` provides several methods regarding the analysis of a graphene sheet.


## References

<sup>[1]</sup> A.P. Willard, D. Chandler (2010). Instantaneous liquid interfaces. *J. Phys. Chem. B*, 114(5): 1954–1958. [DOI: 10.1021/jp909219k](https://doi.org/10.1021/jp909219k)

<sup>[2]</sup> W. Yang, R. Bitetti-Putzer, M. Karplus (2004). Free energy simulations: Use of reverse cumulative averaging to determine the equilibrated region and the time required for convergence. *J. Chem. Phys.*, 120(6): 2618–2628. [DOI:10.1063/1.1638996](https://doi.org/10.1063/1.1638996)


## Author

Darren Wayne Lim (dwl38@cam.ac.uk)
