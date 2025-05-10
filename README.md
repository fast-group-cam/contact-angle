# contact-angle/droplet-graphene-analysis

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

### contact_angle

Interfacial definitions are based on the work of Willard and Chandler [[1]](#references). In particular, the notion of an instantaneous coarse-grained density is used fairly heavily:

$$\rho(\mathbf{r}, t) = \sum_{i=1}^{N_{oxy}} \frac{1}{\left(2\pi\xi^2 \right)^{3/2}}\exp\left[ -\frac{\left|\mathbf{r}-\mathbf{R}_i(t)\right|^2}{2\xi^2} \right]$$

where $\mathbf{R}_i(t)$ is the position of the $i$<sup>th</sup> oxygen atom at time $t$. This also gives a time-averaged coarse-grained density:

$$\langle\rho(\mathbf{r})\rangle_{t} = \lim_{\tau\to\infty} \frac{1}{\tau}\int_{0}^{\tau}\phi(\mathbf{r}, t') dt'.$$

We expect that $\rho \approx 0.033 &angst;^{-3}$ uniformly and constantly for bulk liquid water. The instantaneous interface, and time-averaged interface, of a collection of water molecules can thus be defined as the isosurfaces of the instantaneous and time-averaged coarse-grained densities respectively:

$$\Phi(t) = \left\\{ \mathbf{r} : \rho(\mathbf{r}, t)=\rho_{c} \right\\} ; \qquad \Phi_{\langle t \rangle} = \left\\{ \mathbf{r} : \langle\rho(\mathbf{r})\rangle_{t} = \rho_{c} \right\\}$$

where the cutoff density $\rho_c$ should be set as half of the bulk liquid value. The contact angle, whether instantaneous or time-averaged, can then be defined as the angle that the normal of the interface at the droplet foot (defined as the point of the droplet $7 &angst;$ above the graphene surface) makes relative to the normal of the graphene surface.

Block averaging, as described by Yang et al. [[2]](#references), is used to calculate unbiased uncertainty estimates.


### graphene_dynamics

The definitions of the local inclination angle, and its normalized infinite-time autocorrelation, were taken from Thiemann et al. [[3]](#references). In particular, for a graphene sheet nominally aligned to the $xy$-plane, the local inclination angle $\theta(x, y)$ is defined as the angle that the best-fit plane to the carbon atoms within $4.5 &angst;$ of $(x, y)$ makes relative to the $z$-axis. Then, the normalized autocorrelation function is defined as:

$$\mathcal{C}_ {\theta}(x, y; \tau) = \frac{\langle \theta(x, y; t + \tau)\theta(x, y; t) \rangle_{t}}{\langle \theta^2(x, y; t) \rangle_{t}}.$$

The infinite-time limit $\mathcal{C}_ {\theta}(\tau\to\infty)$ is of interest for analyzing long-time dynamics, and can be calculated by fitting an exponential decay curve $\mathcal{C}_ {\theta}(\tau)\approx(1-c)\exp(-k\tau)+c$.


## References

[1]: A.P. Willard, D. Chandler (2010). Instantaneous liquid interfaces. *J. Phys. Chem. B*, 114(5): 1954–1958. [DOI: 10.1021/jp909219k](https://doi.org/10.1021/jp909219k)

[2]: W. Yang, R. Bitetti-Putzer, M. Karplus (2004). Free energy simulations: Use of reverse cumulative averaging to determine the equilibrated region and the time required for convergence. *J. Chem. Phys.*, 120(6): 2618–2628. [DOI:10.1063/1.1638996](https://doi.org/10.1063/1.1638996)

[3]: F.L. Thiemann, C. Scalliet, E.A. M&uuml;ller, A. Michaelides (2025). Defects induce phase transition from dynamic to static rippling in graphene. *Proc. Natl. Acad. Sci. U.S.A.*, 122(9): e2416932122. [DOI:10.1073/pnas.2416932122](https://doi.org/10.1073/pnas.2416932122)


## Author

Darren Wayne Lim (dwl38@cam.ac.uk)
