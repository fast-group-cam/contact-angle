import numpy as np
from droplet_graphene_analysis.util import read_droplet_trajectory
from droplet_graphene_analysis.util.graphene import *

#==================================================================================================

def test():

    cell_params, _, carbons, _ = read_droplet_trajectory('tests/examples/graphene.xyz')

    interp = smooth_sheet(carbons[0], cell_params[0:2])
    assert np.allclose(interp, np.zeros_like(interp))

    angles = calc_inclination_angles(carbons[0], cell_params[0:2])
    assert np.allclose(angles, np.zeros_like(angles))

    _, fourier = calc_fourier_coefficients(carbons[0], cell_params[0:2])
    assert np.allclose(fourier, np.zeros_like(fourier))
