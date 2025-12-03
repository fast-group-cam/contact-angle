import warnings
import numpy as np
import ase
import ase.io
from pathlib import PurePath
from typing import IO, Iterable
from ..coordinates import center_coordinates

#==================================================================================================

def read(
        filename: str | PurePath | IO | Iterable,
        liq_symbol: str = None,
        sol_symbol: str = None,
        liq_number: int = None,
        sol_number: int = None, *,
        delta_t: float = None,
        time_rescale_factor: float = None,
        length_rescale_factor: float = 1.0,
        **kwargs
        ) -> dict[str, np.ndarray]:
    """Reads a file (or list of files) containing a NVT trajectory of a water droplet on graphene,
    and parses it automatically (including centering the coordinates via `center_coordinates`).
    This function uses ASE to read molecular dynamics trajectory files, from any format compatible
    with the `ase.io` module.

    Parameters
    ----------
    filename : file, str, list of files, or of str
        The trajectory file, or list of trajectory files, to read from.
    liq_symbol : str, optional
        The atomic symbol to identify with liquid particles, following ASE conventions. Mutually
        exclusive with `liq_number`.
    liq_number : int, optional
        The atomic number to identify with liquid particles. Mutually exclusive with `liq_symbol`.
    sol_symbol : str, optional
        The atomic symbol to identify with solid particles, following ASE conventions. Mutually
        exclusive with `sol_number`.
    sol_number : int, optional
        The atomic number to identify with solid particles. Mutually exclusive with `sol_symbol`.
    **kwargs
        Extra arguments are passed directly to `ase.io.iread`.

    Returns
    -------
    trajectory : dict
        A dictionary containing all information about the time evolution of the system's trajectory
        in droplet-centered coordinates, with the following fields:
            - 'cell_params': ndarray of shape (3,), the orthorhombic periodic unit cell parameters
                             in angstroms.
            - 'liq':         ndarray of shape (N_frames, N_liq, 3), the trajectory of the liquid
                             particles in droplet-centered coordinates. Note that N_liq may be
                             zero, e.g. if neither `liq_symbol` nor `liq_number` were provided.
            - 'sol':         ndarray of shape (N_frames, N_sol, 3), the trajectory of the solid
                             particles in droplet-centered coordinates. Note that N_sol may be
                             zero, e.g. if neither `sol_symbol` nor `sol_number` were provided.
            - 'shifts':      ndarray of shape (N_frames, 3), the shift of the droplet-centered
                             coordinate system's origin with respect to the original coordinates
                             across the frames.
            - 'delta_t':     float, the timestep between frames in picoseconds. This field will
                             only be present if the source file contains information about
                             timestepping and ASE is able to parse it, or if manually specified
                             via the `delta_t` optional parameter.

    Other Parameters
    ----------------
    delta_t : float, optional
        The timestep between frames in picoseconds, to override any automatic detection from the
        source file. Mutually exclusive with `time_rescale_factor`.
    time_rescale_factor : float, optional
        Rescaling factor for automatically detected timestep from the source file, to convert it
        from whichever original units to picoseconds. Mutually exclusive with `delta_t`.
    length_rescale_factor : float, optional
        Rescaling factor for lengths and positions from the source file, to convert it from
        whichever original units to angstroms. Defaults to 1.

    Raises
    ------
    TypeError
        If both `liq_symbol` and `liq_number` are specified; or if both `sol_symbol` and
        `sol_number` are specified; or if both `delta_t` and `time_rescale_factor` are specified.
        
    Warns
    -----
    RuntimeWarning
        If the automatically detected timestep is inconsistent between frames.

    Notes
    -----
    The file(s) must either have parseable atomic species, such that `ase.io.read` correctly gives
    the atomic species back; or have atomic species specified by an internal index (e.g. in the
    LAMMPS style). The trajectory is then parsed into liquid and solid particles based on the input
    parameters `liq_symbol` (or `liq_number`) and `sol_symbol` (or `sol_number`) respectively.

    Specifying one of `liq_symbol` or `liq_number` while leaving both `sol_symbol` and `sol_number`
    unspecified causes the trajectory to be parsed as containing only liquid, in which case the
    coordinates will be fully centered around the droplet centre-of-mass.

    On the other hand, specifying one of `liq_symbol` or `liq_number` and also one of `sol_symbol`
    or `sol_number` causes the trajectory to be parsed as containing a liquid droplet in contact
    with a solid surface, in which case the coordinates will be centered so that the droplet
    centre-of-mass is at x = y = 0 and the mean solid z-coordinate is zero.

    Specifying one of `sol_symbol` or `sol_number` while leaving both `liq_symbol` and `liq_number`
    unspecified causes the trajectory to be parsed as containing only solid.

    In all cases, it is assumed that the trajectory is fixed-volume with orthorhombic cell
    parameters, and has fully periodic boundary conditions.
    """

    # Check if provided options are valid
    if ((liq_symbol is not None) and (liq_number is not None)):
        raise TypeError('Use only one of "liq_symbol" and "liq_number".')
    if ((sol_symbol is not None) and (sol_number is not None)):
        raise TypeError('Use only one of "sol_symbol" and "sol_number".')
    if ((delta_t is not None) and (time_rescale_factor is not None)):
        raise TypeError('Use only one of "delta_t" and "time_rescale_factor".')
    if ((liq_symbol is None) and (liq_number is None) and (sol_symbol is None) and (sol_number is None)):
        raise TypeError('At least one of "liq_symbol", "liq_number", "sol_symbol" or "sol_number" must be specified!')

    # Parse filenames into an iterable sequence
    if type(filename) == str:
        filenames = (filename,)
    elif isinstance(filename, Iterable):
        filenames = filename
    else:
        filenames = (filename,)

    # Establish whether or not to attempt detecting timestep
    if delta_t is None:
        detect_timestep = True
        time_interval = None
        timestep_rescale = 1.0 if time_rescale_factor is None else float(time_rescale_factor)
    else:
        detect_timestep = False
        time_interval = abs(float(delta_t))

    # Iterate through files; for each file, iterate through frames
    cell_params = None
    current_timestep = None
    list_liq = list()
    list_sol = list()
    list_shifts = list()
    for name in filenames:
        traj = ase.io.iread(name, **kwargs)
        for atoms in traj:

            # Read cell_params if not yet initialized
            if cell_params is None:
                cell_params = length_rescale_factor * np.array(atoms.cell.cellpar()[0:3])
                
            # Read liquid particles
            if liq_symbol is not None:
                liq = length_rescale_factor * atoms.positions[atoms.symbols == liq_symbol]
            elif liq_number is not None:
                liq = length_rescale_factor * atoms.positions[atoms.numbers == liq_number]
            else:
                liq = np.zeros((0, 3), dtype=float)

            # Read solid particles
            if sol_symbol is not None:
                sol = length_rescale_factor * atoms.positions[atoms.symbols == sol_symbol]
            elif sol_number is not None:
                sol = length_rescale_factor * atoms.positions[atoms.numbers == sol_number]
            else:
                sol = np.zeros((0, 3), dtype=float)

            # Center coordinates appropriately
            liq, sol, shift = center_coordinates(cell_params, liq, sol)
            list_liq.append(liq)
            list_sol.append(sol)
            list_shifts.append(shift)

            # Attempt to detect timestep, and warn if inconsistent
            if (detect_timestep and ('timestep' in atoms.info)):
                if current_timestep is None:
                    current_timestep = atoms.info['timestep']
                else:
                    detected_step = atoms.info['timestep'] - current_timestep
                    if time_interval is None:
                        time_interval = detected_step
                    elif detected_step != time_interval:
                        warnings.warn('Inconsistent timestep across frames', RuntimeWarning)
                        time_interval = detected_step
                    current_timestep = atoms.info['timestep']

    # Rescale timestep if rescaling parameter is provided
    if (detect_timestep and (time_interval is not None)):
        time_interval *= timestep_rescale

    # Pack results into dictionary
    trajectory = dict()
    trajectory['cell_params'] = cell_params
    trajectory['liq'] = np.array(list_liq)
    trajectory['sol'] = np.array(list_sol)
    trajectory['shifts'] = np.array(list_shifts)
    if time_interval is not None:
        trajectory['delta_t'] = time_interval
    return trajectory

