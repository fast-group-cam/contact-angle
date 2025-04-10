import warnings
import numpy as np
import ase

def center_coordinates(atoms: ase.Atoms, cell_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """This function takes in a single frame of the trajectory, and shifts the coordinates so that
the centre-of-mass of the droplet is on the z-axis (at x = y = 0) with the graphene plane at z = 0;
it then returns the coordinates of the water molecules and carbon atoms. The inputs are:

    atoms:       The frame to be analyzed; should only contain carbon, hydrogen, and oxygen atoms.
    cell_params: The cell parameters, expressed as [cell_x, cell_y, cell_z].

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
    cell_xy = cell_params[0:2]
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

    return (oxygens, carbons, hydrogens)
