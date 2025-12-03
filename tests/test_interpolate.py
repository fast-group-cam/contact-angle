import numpy as np
from fast_contact_angle_2d.interpolate import PeriodicGridInterpolator

#==================================================================================================

def test_2d():

    N_comps = 5
    N_pts = 100
    N_tests = 20
    N_batch = 10

    cell_params = np.ones(2) + (9.0 * np.random.random(2))
    Aij = np.random.random((2, N_comps))
    phi = 2.0 * np.pi * np.random.random((2, N_comps))
    freq = 2.0 * np.pi * np.arange(N_comps)[None,:] / cell_params[:,None]

    def func(x):
        phase = (freq * x[...,:,None]) + phi
        return np.prod(np.sum(Aij * np.cos(phase), axis=-1), axis=-1)
    
    dx = cell_params[0] / N_pts
    dy = cell_params[1] / N_pts
    xcoords = np.linspace((dx - cell_params[0]) / 2.0, (cell_params[0] - dx) / 2.0, N_pts)
    ycoords = np.linspace((dy - cell_params[1]) / 2.0, (cell_params[1] - dy) / 2.0, N_pts)
    coords = np.stack(np.meshgrid(xcoords, ycoords, indexing='ij'), axis=-1)
    values = func(coords)

    interp = PeriodicGridInterpolator(cell_params, values)
    epsilon = np.max(Aij * freq * cell_params[:,None]) / N_pts

    for _ in range(N_tests):
        testpts = 10.0 * np.random.random((N_batch, 2))
        assert np.allclose(interp(testpts), func(testpts), atol=epsilon)

#==================================================================================================

def test_2d_derivative():

    N_pts = 100
    N_tests = 20
    N_batch = 10

    cell_params = np.ones(2) + (9.0 * np.random.random(2))
    phi_x = 2.0 * np.pi * np.random.random()
    phi_y = 2.0 * np.pi * np.random.random()
    freq_x = 2.0 * np.pi / cell_params[0]
    freq_y = 2.0 * np.pi / cell_params[1]

    def func(x):
        return np.sin((freq_x * x[...,0]) + phi_x) + np.sin((freq_y * x[...,1]) + phi_y)
    def func_dx(x):
        return freq_x * np.cos((freq_x * x[...,0]) + phi_x)
    def func_dy(x):
        return freq_y * np.cos((freq_y * x[...,1]) + phi_y)
    
    dx = cell_params[0] / N_pts
    dy = cell_params[1] / N_pts
    xcoords = np.linspace((dx - cell_params[0]) / 2.0, (cell_params[0] - dx) / 2.0, N_pts)
    ycoords = np.linspace((dy - cell_params[1]) / 2.0, (cell_params[1] - dy) / 2.0, N_pts)
    coords = np.stack(np.meshgrid(xcoords, ycoords, indexing='ij'), axis=-1)
    values = func(coords)

    interp = PeriodicGridInterpolator(cell_params, values)
    interp_dx = interp.derivative(0)
    interp_dy = interp.derivative(1)
    epsilon = 2.0 * np.pi / N_pts
    epsilon_d = (2.0 * np.pi)**2 / N_pts

    for _ in range(N_tests):
        testpts = 10.0 * np.random.random((N_batch, 2))
        assert np.allclose(interp(testpts), func(testpts), atol=epsilon)
        assert np.allclose(interp_dx(testpts), func_dx(testpts), atol=epsilon_d)
        assert np.allclose(interp_dy(testpts), func_dy(testpts), atol=epsilon_d)

#==================================================================================================

def test_3d():

    N_comps = 4
    N_pts = 100
    N_tests = 20
    N_batch = 10

    cell_params = np.ones(3) + (9.0 * np.random.random(3))
    Aij = np.random.random((3, N_comps))
    phi = 2.0 * np.pi * np.random.random((3, N_comps))
    freq = 2.0 * np.pi * np.arange(N_comps)[None,:] / cell_params[:,None]

    def func(x):
        phase = (freq * x[...,:,None]) + phi
        return np.prod(np.sum(Aij * np.cos(phase), axis=-1), axis=-1)
    
    dx = cell_params[0] / N_pts
    dy = cell_params[1] / N_pts
    dz = cell_params[2] / N_pts
    xcoords = np.linspace((dx - cell_params[0]) / 2.0, (cell_params[0] - dx) / 2.0, N_pts)
    ycoords = np.linspace((dy - cell_params[1]) / 2.0, (cell_params[1] - dy) / 2.0, N_pts)
    zcoords = np.linspace((dz - cell_params[2]) / 2.0, (cell_params[2] - dz) / 2.0, N_pts)
    coords = np.stack(np.meshgrid(xcoords, ycoords, zcoords, indexing='ij'), axis=-1)
    values = func(coords)

    interp = PeriodicGridInterpolator(cell_params, values)
    epsilon = np.max(Aij * freq * cell_params[:,None]) / N_pts

    for _ in range(N_tests):
        testpts = 10.0 * np.random.random((N_batch, 3))
        assert np.allclose(interp(testpts), func(testpts), atol=epsilon)

#==================================================================================================

def test_2d_2d():

    N_comps = 5
    N_pts = 100
    N_tests = 20
    N_batch = 10

    cell_params = np.ones(2) + (9.0 * np.random.random(2))
    Aij = np.random.random((2, N_comps))
    phi = 2.0 * np.pi * np.random.random((2, N_comps))
    freq = 2.0 * np.pi * np.arange(N_comps)[None,:] / cell_params[:,None]

    def base_func(x):
        phase = (freq * x[...,:,None]) + phi
        return np.prod(np.sum(Aij * np.cos(phase), axis=-1), axis=-1)
    
    def func_2d(x):
        return np.stack((base_func(x), base_func(x + 1.0)), axis=-1)
    
    dx = cell_params[0] / N_pts
    dy = cell_params[1] / N_pts
    xcoords = np.linspace((dx - cell_params[0]) / 2.0, (cell_params[0] - dx) / 2.0, N_pts)
    ycoords = np.linspace((dy - cell_params[1]) / 2.0, (cell_params[1] - dy) / 2.0, N_pts)
    coords = np.stack(np.meshgrid(xcoords, ycoords, indexing='ij'), axis=-1)
    values = func_2d(coords)

    interp = PeriodicGridInterpolator(cell_params, values)
    epsilon = np.max(Aij * freq * cell_params[:,None]) / N_pts

    for _ in range(N_tests):
        testpts = 10.0 * np.random.random((N_batch, 2))
        assert np.allclose(interp(testpts), func_2d(testpts), atol=epsilon)

