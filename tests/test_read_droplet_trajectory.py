import numpy as np
from droplet_graphene_analysis.util import read_droplet_trajectory

#==================================================================================================

def test_droplet():
    params, oxy, car, hyd = read_droplet_trajectory('tests/examples/small-droplet.xyz')
    assert np.allclose(params, (51.336, 49.4, 50))
    assert oxy.shape == (1, 130, 3)
    assert car.shape == (1, 960, 3)
    assert hyd.shape == (1, 260, 3)
    assert np.allclose(np.mean(car, axis=(0,1))[2], 0.0)
    assert np.allclose(np.mean(oxy, axis=(0,1))[0:2], (0.0, 0.0))

#==================================================================================================

def test_graphene():
    params, oxy, car, hyd = read_droplet_trajectory('tests/examples/graphene.xyz')
    assert np.allclose(params, (59.892, 59.27770684, 59.58406178))
    assert oxy.shape == (1, 0, 3)
    assert car.shape == (1, 1344, 3)
    assert hyd.shape == (1, 0, 3)
    assert np.allclose(np.mean(car, axis=(0,1))[2], 0.0)

#==================================================================================================

def test_trajectory():

    params, oxy, car, hyd = read_droplet_trajectory('tests/examples/droplet_part_0.xyz')
    assert np.allclose(params, (51.336, 49.4, 50))
    assert oxy.shape == (100, 130, 3)
    assert car.shape == (100, 960, 3)
    assert hyd.shape == (100, 260, 3)
    assert np.allclose(np.mean(car, axis=(0,1))[2], 0.0)
    assert np.allclose(np.mean(oxy, axis=(0,1))[0:2], (0.0, 0.0))

    params, oxy, car, hyd = read_droplet_trajectory('tests/examples/droplet_part_1.xyz')
    assert np.allclose(params, (51.336, 49.4, 50))
    assert oxy.shape == (100, 130, 3)
    assert car.shape == (100, 960, 3)
    assert hyd.shape == (100, 260, 3)
    assert np.allclose(np.mean(car, axis=(0,1))[2], 0.0)
    assert np.allclose(np.mean(oxy, axis=(0,1))[0:2], (0.0, 0.0))

    params, oxy, car, hyd = read_droplet_trajectory(['tests/examples/droplet_part_0.xyz',
                                                     'tests/examples/droplet_part_1.xyz'])
    assert np.allclose(params, (51.336, 49.4, 50))
    assert oxy.shape == (200, 130, 3)
    assert car.shape == (200, 960, 3)
    assert hyd.shape == (200, 260, 3)
    assert np.allclose(np.mean(car, axis=(0,1))[2], 0.0)
    assert np.allclose(np.mean(oxy, axis=(0,1))[0:2], (0.0, 0.0))
    