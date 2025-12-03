import numpy as np
import fast_contact_angle_2d.util.io as io

#==================================================================================================

def test_droplet():
    trajectory = io.read('tests/examples/small-droplet.xyz', liq_symbol='O', sol_symbol='C')
    params = trajectory['cell_params']
    liq = trajectory['liq']
    sol = trajectory['sol']
    assert np.allclose(params, (51.336, 49.4, 50))
    assert liq.shape == (1, 130, 3)
    assert sol.shape == (1, 960, 3)
    assert np.allclose(np.mean(sol, axis=(0,1))[2], 0.0, atol=0.0002)
    assert np.allclose(np.mean(liq, axis=(0,1))[0:2], (0.0, 0.0), atol=0.0002)

#==================================================================================================

def test_graphene():
    trajectory = io.read('tests/examples/graphene.xyz', sol_symbol='C')
    params = trajectory['cell_params']
    sol = trajectory['sol']
    assert np.allclose(params, (59.892, 59.27770684, 59.58406178))
    assert sol.shape == (1, 1344, 3)
    assert np.allclose(np.mean(sol, axis=(0,1))[2], 0.0, atol=0.0002)

#==================================================================================================

def test_trajectory():

    trajectory = io.read('tests/examples/droplet_part_0.xyz', liq_symbol='O', sol_symbol='C')
    params = trajectory['cell_params']
    liq = trajectory['liq']
    sol = trajectory['sol']
    assert np.allclose(params, (51.336, 49.4, 50))
    assert liq.shape == (100, 130, 3)
    assert sol.shape == (100, 960, 3)
    assert np.allclose(np.mean(sol, axis=(0,1))[2], 0.0, atol=0.0002)
    assert np.allclose(np.mean(liq, axis=(0,1))[0:2], (0.0, 0.0), atol=0.0002)

    trajectory = io.read('tests/examples/droplet_part_1.xyz', liq_symbol='O', sol_symbol='C')
    params = trajectory['cell_params']
    liq = trajectory['liq']
    sol = trajectory['sol']
    assert np.allclose(params, (51.336, 49.4, 50))
    assert liq.shape == (100, 130, 3)
    assert sol.shape == (100, 960, 3)
    assert np.allclose(np.mean(sol, axis=(0,1))[2], 0.0, atol=0.0002)
    assert np.allclose(np.mean(liq, axis=(0,1))[0:2], (0.0, 0.0), atol=0.0002)

    trajectory = io.read(['tests/examples/droplet_part_0.xyz',
                          'tests/examples/droplet_part_1.xyz'], liq_symbol='O', sol_symbol='C')
    params = trajectory['cell_params']
    liq = trajectory['liq']
    sol = trajectory['sol']
    assert np.allclose(params, (51.336, 49.4, 50))
    assert liq.shape == (200, 130, 3)
    assert sol.shape == (200, 960, 3)
    assert np.allclose(np.mean(sol, axis=(0,1))[2], 0.0, atol=0.0002)
    assert np.allclose(np.mean(liq, axis=(0,1))[0:2], (0.0, 0.0), atol=0.0002)
    
