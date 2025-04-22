import numpy as np
import ase
from typing import Literal

#==================================================================================================
# Default parameters

C_C_DISTANCE = 1.426    # Interatomic C-C distance in graphene (in angstroms)

#==================================================================================================

def generate_sheet(
        max_cell_x: float,
        max_cell_y: float = None,
        interatomic_dist: float = C_C_DISTANCE,
        atomic_symbol: str = 'C',
        origin: Literal['corner', 'center'] = 'center'
        ) -> ase.Atoms:
    """This function generates a graphene sheet. The inputs are:

    - `max_cell_x`: The maximum dimension of cell_x in angstroms.

    - `max_cell_y`: The maximum dimension of cell_y in angstroms, if specified; otherwise taken to
    be equal to max_cell_x.

    - `interatomic_dist`: The interatomic distance to generate the sheet with.

    - `atomic_symbol`: The symbol of the atom used to generate the sheet.

    - `origin`: If set to 'corner', the sheet will be generated within the x-coordinate range of 0
    to cell_x, and y-coordinate range of 0 to cell_y; if set to 'center', the sheet will be
    generated within the x-coordinate range of -0.5 * cell_x to 0.5 * cell_x, and y-coordinate
    range of -0.5 * cell_y to 0.5 * cell_y. In either setting, the sheet will be aligned such that
    one atom is placed at (0, 0, 0).

The output is an ASE Atoms object, initialized with only information about chemical species and
position. The cell parameters [cell_x, cell_y, cell_z] are set such that cell_x and cell_y are the
largest possible integer multiples of the hexagonal grid (so that the sheet is correctly periodic)
that are lesser than or equal to max_cell_x and max_cell_y respectively, while cell_z is set to the
geometric mean of cell_x and cell_y."""
    
    max_cell_x = float(max_cell_x)
    if max_cell_y is None:
        max_cell_y = max_cell_x

    if origin.casefold() == 'center':
        sheet_centered = True
    elif origin.casefold() == 'corner':
        sheet_centered = False
    else:
        raise RuntimeError(f'Option undefined: {origin}')

    a_x = 1.5 * interatomic_dist
    a_y = interatomic_dist * np.sin(np.pi / 3)
    offset = np.array([interatomic_dist, 0, 0])
    #vec_a = np.array([a_x, a_y, 0])
    vec_b = np.array([a_x, -a_y, 0])

    tmp = max_cell_x / a_x
    units_x_min = (-int(np.floor(0.25 * tmp)) if sheet_centered else 0)
    units_x_max = (int(np.floor(0.25 * tmp)) if sheet_centered else int(np.floor(0.5 * tmp)))
    tmp = max_cell_y / a_y
    units_y_min = (-int(np.floor(0.25 * tmp)) if sheet_centered else 0)
    units_y_max = (int(np.floor(0.25 * tmp)) if sheet_centered else int(np.floor(0.5 * tmp)))

    positions = list()
    for i in range(units_x_min, units_x_max):
        for j in range(units_y_min, units_y_max):
            tmp = np.array([2 * i * a_x, 2 * j * a_y, 0])
            positions.append(tmp)
            positions.append(tmp + offset)
            positions.append(tmp + vec_b)
            positions.append(tmp + vec_b + offset)
    positions = np.array(positions)

    cell_x = (units_x_max - units_x_min) * 2 * a_x
    cell_y = (units_y_max - units_y_min) * 2 * a_y
    cell_z = np.sqrt(cell_x * cell_y)

    return ase.Atoms([atomic_symbol,] * positions.shape[0], positions=positions,
                     cell=[cell_x, cell_y, cell_z], pbc=True)
