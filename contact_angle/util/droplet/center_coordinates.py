import warnings
import numpy as np
import ase
import ase.io
from pathlib import PurePath
from typing import IO

#==================================================================================================

def center_coordinates(
        atoms: ase.Atoms,
        cell_params: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function takes in a single frame of the trajectory, and shifts the coordinates so that
the centre-of-mass of the droplet is on the z-axis (at x = y = 0) with the graphene plane at z = 0;
it then returns the coordinates of the water molecules and carbon atoms. The inputs are:

    - `atoms`: The frame to be analyzed; should only contain carbon, hydrogen, and oxygen atoms.
    - `cell_params`: The cell parameters, expressed as [cell_x, cell_y, cell_z].

The output is a tuple of three np.NDArrays, the first array having shape (N_water, 3) for the 
Cartesian coordinates of the water molecules (taken as just oxygen atoms), the second array having
shape (N_carbon, 3) for the Cartesian coordinates of the carbon atoms, and the third array having
shape (2 * N_water, 3) for the Cartesian coordinates of the hydrogen atoms."""

    # Split according to atom type
    carbons = atoms.positions[atoms.symbols=='C']
    oxygens = atoms.positions[atoms.symbols=='O']
    hydrogens = atoms.positions[atoms.symbols=='H']

    # Assert that only water molecules are contained
    N_water = oxygens.shape[0]
    if hydrogens.shape[0] != (2 * N_water):
        warnings.warn(f'Found {N_water} oxygens and {hydrogens.shape[0]} hydrogens, numbers do ' +
                      'not match!', RuntimeWarning)
        return (oxygens, carbons, hydrogens)

    # Set middle of graphene sheet to z = 0
    cell_z = cell_params[2]
    carbons[:,2] -= cell_z * np.round(carbons[:,2] / cell_z)
    mean_carbon_z_coord = np.mean(carbons[:,2])
    carbons[:,2] -= mean_carbon_z_coord
    oxygens[:,2] -= mean_carbon_z_coord
    hydrogens[:,2] -= mean_carbon_z_coord

    # Send all water molecules to the +z side of the graphene
    oxygens[oxygens[:,2] < 0.0] += np.array((0, 0, cell_z))
    hydrogens[hydrogens[:,2] < 0.0] += np.array((0, 0, cell_z))

    # First guess of droplet CoM without accounting for periodic boundaries
    CoM = np.mean(oxygens, axis=0)
    CoM[2] = 0
    oxygens -= CoM
    carbons -= CoM
    hydrogens -= CoM
    
    # Improve each guess of the CoM iteratively by:
    #   - Relative to the previous guess of the CoM, move all molecules to the same unit cell
    #   - Calculate a new guess (wrt previous guess) using this constrained droplet
    #   - Repeat 3 times, to undo all possible boundary crossings
    cell_xy = np.array(cell_params[0:2])
    for i in range(3):
        oxygens[:,(0,1)] -= cell_xy * np.round(oxygens[:,(0,1)] / cell_xy)
        CoM = np.mean(oxygens, axis=0)
        CoM[2] = 0
        oxygens -= CoM
        carbons -= CoM
        hydrogens -= CoM

    # Centralize unit cell
    carbons[:,(0,1)] -= cell_xy * np.round(carbons[:,(0,1)] / cell_xy)
    hydrogens[:,(0,1)] -= cell_xy * np.round(hydrogens[:,(0,1)] / cell_xy)

    # Fixing a weird bug where the droplet is sometimes set to the wrong side of the unit cell
    oxygens[:,2] = np.remainder(oxygens[:,2], cell_z)
    hydrogens[:,2] = np.remainder(hydrogens[:,2], cell_z)
    CoM_z = np.mean(oxygens[:,2])
    oxygens[:,2] -= cell_z * np.round((oxygens[:,2] - CoM_z) / cell_z)
    hydrogens[:,2] -= cell_z * np.round((hydrogens[:,2] - CoM_z) / cell_z)

    return (oxygens, carbons, hydrogens)

#==================================================================================================

def read_droplet_trajectory(
        filename: str | PurePath | IO,
        **kwargs
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function reads a file containing a trajectory of a water droplet on graphene, and
parses it automatically (including centering the coordinates via `center_coordinates`). The input
arguments are passed directly to `ase.io.iread`.

Importantly, the file must either have parseable atomic species, or have atomic species specified
by an internal index (e.g. in the style of LAMMPS) which maps according to {1: carbon, 2: hydrogen,
3: oxygen}; and the file must contain at least one atom of each type.

The output is a tuple of four np.NDArrays: the first array has shape (3,) and gives the system cell
parameters; the second array has shape (N_frames, N_water, 3) for the time-evolution of the
Cartesian coordinates of the water molecules (taken as just oxygen atoms); the third array has
shape (N_frames, N_carbon, 3) for the carbon atoms; and the fourth array has shape (N_frames,
2 * N_water, 3) for the hydrogen atoms."""

    traj = ase.io.iread(filename, **kwargs)
    cell_params = None
    need_to_reassign = None

    list_oxygens = list()
    list_carbons = list()
    list_hydrogens = list()

    for atoms in traj:

        # Read cell parameters in
        if cell_params is None:
            cell_params = np.array(atoms.cell.cellpar()[0:3])
        
        # Check which ordering of elements is present...
        if need_to_reassign is None:
            elems = np.unique(atoms.numbers)
            if np.array_equal(elems, (1, 2, 3)):
                need_to_reassign = True
            elif np.array_equal(elems, (1, 6, 8)):
                need_to_reassign = False
            else:
                raise RuntimeError('Elements not recognized!')
            
        # ...and reassign if necessary
        if need_to_reassign:
            atoms.numbers[atoms.numbers == 1] = 6
            atoms.numbers[atoms.numbers == 2] = 1
            atoms.numbers[atoms.numbers == 3] = 8

        oxygens, carbons, hydrogens = center_coordinates(atoms, cell_params)
        list_oxygens.append(oxygens)
        list_carbons.append(carbons)
        list_hydrogens.append(hydrogens)

    return (cell_params, np.array(list_oxygens), np.array(list_carbons), np.array(list_hydrogens))
