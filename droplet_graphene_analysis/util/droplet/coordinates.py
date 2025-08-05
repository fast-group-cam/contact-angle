import warnings
import numpy as np
import ase
import ase.io
from pathlib import PurePath
from typing import IO, Iterable

#==================================================================================================

def center_coordinates(
        atoms: ase.Atoms,
        cell_params: np.ndarray,
        return_shift: bool = False
        ) -> tuple[np.ndarray, ...]:
    """Takes in a single frame of a trajectory of a droplet of water on graphene, and shifts the
    coordinates so that the centre-of-mass of the droplet is on the z-axis (at x = y = 0) with the
    graphene plane at z = 0.

    Parameters
    ----------
    atoms : ase.Atoms
        The frame to be analyzed; should only contain carbon, hydrogen, and oxygen atoms. It is
        permissible for `atoms` to contain no water molecules, but it must contain at least one
        carbon atom, and must contain exactly twice as many hydrogen atoms as oxygen atoms.
    cell_params : array_like
        The cell parameters, expressed as [cell_x, cell_y, cell_z].
    return_shift : bool, optional
        Whether or not to return the coordinate shift vector, which is useful for tracking the
        absolute motion of the droplet CoM; defaults to False for backwards compatibility.

    Returns
    -------
    waters : ndarray
        The Cartesian coordinates of the water molecules (taken as just the oxygen atoms), with
        shape (N_water, 3).
    carbons : ndarray
        The Cartesian coordinates of the carbon atoms, with shape (N_carbon, 3).
    hydrogens : ndarray
        The Cartesian coordinates of the hydrogen atoms, with shape (2 * N_water, 3).
    shift : ndarray, only if `return_shift` is true
        The vector of shape (3,) representing the shift of the coordinate system's origin with
        respect to the original coordinates.
    """

    # Split according to atom type
    carbons = atoms.positions[atoms.symbols=='C']
    oxygens = atoms.positions[atoms.symbols=='O']
    hydrogens = atoms.positions[atoms.symbols=='H']

    # Assert that only water molecules are contained
    N_water = oxygens.shape[0]
    if hydrogens.shape[0] != (2 * N_water):
        warnings.warn(f'Found {N_water} oxygens and {hydrogens.shape[0]} hydrogens, numbers do ' +
                      'not match!', RuntimeWarning)
        return ((oxygens, carbons, hydrogens, np.zeros(3)) if return_shift else
                (oxygens, carbons, hydrogens))

    shift = np.zeros(3)

    # If there are no carbons, return just the water (centred on unit cell)
    if carbons.shape[0] == 0:
        if N_water == 0:
            warnings.warn('Input trajectory is empty!', RuntimeWarning)
            return ((oxygens, carbons, hydrogens, shift) if return_shift else
                    (oxygens, carbons, hydrogens))
        CoM = np.mean(oxygens, axis=0)
        oxygens -= CoM
        hydrogens -= CoM
        shift += CoM
        cell_p = np.array(cell_params[0:3])
        for _ in range(3):
            oxygens -= cell_p * np.round(oxygens / cell_p)
            CoM = np.mean(oxygens, axis=0)
            oxygens -= CoM
            hydrogens -= CoM
            shift += CoM
        hydrogens -= cell_p * np.round(hydrogens / cell_p)
        return ((oxygens, carbons, hydrogens, shift) if return_shift else
                (oxygens, carbons, hydrogens))

    # Set middle of graphene sheet to z = 0
    cell_z = cell_params[2]
    for _ in range(3):
        carbons[:,2] -= cell_z * np.round(carbons[:,2] / cell_z)
        mean_carbon_z_coord = np.mean(carbons[:,2])
        carbons[:,2] -= mean_carbon_z_coord
        oxygens[:,2] -= mean_carbon_z_coord
        hydrogens[:,2] -= mean_carbon_z_coord
        shift[2] += mean_carbon_z_coord

    # If there are no water molecules, return just the carbons (centred on unit cell)
    if N_water == 0:
        cell_xy = np.array(cell_params[0:2])
        carbons[:,0:2] -= cell_xy * np.round(carbons[:,0:2] / cell_xy)
        return ((oxygens, carbons, hydrogens, shift) if return_shift else
                (oxygens, carbons, hydrogens))

    # Send all water molecules to the +z side of the graphene
    oxygens[:,2] = np.remainder(oxygens[:,2], cell_z)
    hydrogens[:,2] = np.remainder(hydrogens[:,2], cell_z)

    # First guess of droplet CoM without accounting for periodic boundaries
    CoM = np.mean(oxygens, axis=0)
    CoM[2] = 0
    oxygens -= CoM
    carbons -= CoM
    hydrogens -= CoM
    shift += CoM
    
    # Improve each guess of the CoM iteratively by:
    #   - Relative to the previous guess of the CoM, move all molecules to the same unit cell
    #   - Calculate a new guess (wrt previous guess) using this constrained droplet
    #   - Repeat 3 times, to undo all possible boundary crossings
    cell_xy = np.array(cell_params[0:2])
    for _ in range(3):
        oxygens[:,0:2] -= cell_xy * np.round(oxygens[:,0:2] / cell_xy)
        CoM = np.mean(oxygens, axis=0)
        CoM[2] = 0
        oxygens -= CoM
        carbons -= CoM
        hydrogens -= CoM
        shift += CoM

    # Centralize unit cell
    carbons[:,0:2] -= cell_xy * np.round(carbons[:,0:2] / cell_xy)
    hydrogens[:,0:2] -= cell_xy * np.round(hydrogens[:,0:2] / cell_xy)

    # Fixing a weird bug where the droplet is sometimes set to the wrong side of the unit cell
    CoM_z = np.mean(oxygens[:,2])
    oxygens[:,2] -= cell_z * np.round((oxygens[:,2] - CoM_z) / cell_z)
    hydrogens[:,2] -= cell_z * np.round((hydrogens[:,2] - CoM_z) / cell_z)

    return ((oxygens, carbons, hydrogens, shift) if return_shift else (oxygens, carbons, hydrogens))

#==================================================================================================

def read_droplet_trajectory(
        filename: str | PurePath | IO | Iterable,
        return_shift_trajectory: bool = False,
        **kwargs
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reads a file (or list of files) containing a NVT trajectory of a water droplet on graphene,
    and parses it automatically (including centering the coordinates via `center_coordinates`).

    Parameters
    ----------
    filename : file, str, list of files, or of str
        The trajectory file, or list of trajectory files, to read from.
    return_shift_trajectory : bool, optional
        Whether or not to return the trajectory of coordinate shift vectors, which is useful for
        tracking the absolute motion of the droplet CoM; defaults to False for backwards
        compatibility.
    **kwargs
        Extra arguments are passed directly to `ase.io.iread`.

    Returns
    -------
    cell_params : ndarray
        The system cell parameters, with shape (3,).
    waters : ndarray
        The time-evolution of the Cartesian coordinates of the water molecules (taken as just the
        oxygen atoms), with shape (N_frames, N_water, 3).
    carbons : ndarray
        The time-evolution of the Cartesian coordinates of the carbon atoms, with shape (N_frames,
        N_carbon, 3).
    hydrogens : ndarray
        The time-evolution of the Cartesian coordinates of the hydrogen atoms, with shape
        (N_frames, 2 * N_carbon, 3).
    shift_trajectory : ndarray, only if `return_shift_trajectory` is true
        The vector of shape (N_frames, 3) representing the shift of the coordinate system's origin
        with respect to the original coordinates.

    Notes
    -----
    The file(s) must either have parseable atomic species, such that `ase.io.read` correctly gives
    the atomic species back; or have atomic species specified by an internal index (e.g. in the
    LAMMPS style) which maps according to {1: 'C', 2: 'H', 3: 'O'}. In either case, the file must
    contain at least one carbon atom, and exactly twice as many hydrogen atoms as oxygen atoms
    (which may be zero).

    It is also assumed that the trajectory is fixed-volume with orthorhombic cell parameters, and
    fully periodic boundary conditions.
    """

    if type(filename) == str:
        filenames = (filename,)
    elif isinstance(filename, Iterable):
        filenames = filename
    else:
        filenames = (filename,)

    cell_params = None
    list_oxygens = list()
    list_carbons = list()
    list_hydrogens = list()
    list_shifts = list()

    for name in filenames:

        traj = ase.io.iread(name, **kwargs)
        need_to_reassign = None

        for atoms in traj:

            # Read cell parameters in
            if cell_params is None:
                cell_params = np.array(atoms.cell.cellpar()[0:3])
            
            # Check which ordering of elements is present...
            if need_to_reassign is None:
                elems = np.unique(atoms.numbers)
                if np.array_equal(elems, (1, 2, 3)) or np.array_equal(elems, (1,)):
                    need_to_reassign = True
                elif np.array_equal(elems, (1, 6, 8)) or np.array_equal(elems, (6,)):
                    need_to_reassign = False
                else:
                    raise RuntimeError('Elements not recognized!')
                
            # ...and reassign if necessary
            if need_to_reassign:
                atoms.numbers[atoms.numbers == 1] = 6
                atoms.numbers[atoms.numbers == 2] = 1
                atoms.numbers[atoms.numbers == 3] = 8

            oxygens, carbons, hydrogens, shift = center_coordinates(atoms, cell_params, return_shift=True)
            list_oxygens.append(oxygens)
            list_carbons.append(carbons)
            list_hydrogens.append(hydrogens)
            list_shifts.append(shift)

    return ((cell_params, np.array(list_oxygens), np.array(list_carbons), np.array(list_hydrogens),
             np.array(list_shifts)) if return_shift_trajectory else (cell_params,
             np.array(list_oxygens), np.array(list_carbons), np.array(list_hydrogens)))
