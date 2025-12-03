import numpy as np
import fast_contact_angle_2d.util.io as io
import fast_contact_angle_2d.liquid.coarse_grain as cg

#==================================================================================================

def test_point():

    waters = np.array(((0.0, 0.0, 0.0),))
    prefactor = np.power(2 * np.pi, -1.5)
    one_std = prefactor * np.exp(-0.5)
    two_std = prefactor * np.exp(-2.0)

    density = cg.coarse_grained_density(np.array((0.0, 0.0, 0.0)), waters, coarse_grain_length=1.0)
    assert np.allclose(density, prefactor)

    N_tests = 50
    phi = np.random.random(N_tests) * 2 * np.pi
    cos = np.random.random(N_tests)
    sin = np.sqrt(1 - np.square(cos))
    axes = np.column_stack((np.cos(phi) * sin, np.sin(phi) * sin, cos))
    for axis in axes:
        density = cg.coarse_grained_density(axis, waters, coarse_grain_length=1.0)
        assert np.allclose(density, one_std)
        density = cg.coarse_grained_density(2.0 * axis, waters, coarse_grain_length=1.0)
        assert np.allclose(density, two_std)

    for axis in axes:
        inter_0, norm_0 = cg.find_interface(waters, (0.0, 0.0, 0.0), axis, tol=1e-08, max_dist=5.0,
                                            calc_normal=True, coarse_grain_length=1.0,
                                            cutoff_density=one_std, slicing_cutoff=None)
        inter_1, norm_1 = cg.find_interface(waters, 4.0 * axis, -axis, tol=1e-08, max_dist=5.0,
                                            calc_normal=True, coarse_grain_length=1.0,
                                            cutoff_density=one_std, slicing_cutoff=None,
                                            reverse_search=True)
        assert np.allclose(inter_0, axis)
        assert np.allclose(norm_0 , axis)
        assert np.allclose(inter_0, inter_1)
        assert np.allclose(norm_0 , norm_1)
        inter_0, norm_0 = cg.find_interface(waters, (0.0, 0.0, 0.0), axis, tol=1e-08, max_dist=5.0,
                                            calc_normal=True, coarse_grain_length=1.0,
                                            cutoff_density=two_std, slicing_cutoff=None)
        inter_1, norm_1 = cg.find_interface(waters, 4.0 * axis, -axis, tol=1e-08, max_dist=5.0,
                                            calc_normal=True, coarse_grain_length=1.0,
                                            cutoff_density=two_std, slicing_cutoff=None,
                                            reverse_search=True)
        assert np.allclose(inter_0, 2.0 * axis)
        assert np.allclose(norm_0 , axis)
        assert np.allclose(inter_0, inter_1)
        assert np.allclose(norm_0 , norm_1)

#==================================================================================================

def test_slab():

    trajectory = io.read('tests/examples/water-slab.xyz', liq_symbol='O')
    waters = trajectory['liq'][0]

    central_density = cg.coarse_grained_density(np.array((0.0, 0.0, 0.0)), waters)
    assert np.allclose(central_density, 0.035053560262109304)

    xx = np.linspace(-3, 3, 50)
    yy = np.linspace(-3, 3, 50)
    zz = np.linspace(-6, 6, 50)
    xx, yy, zz = np.meshgrid(xx, yy, zz)
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    densities = cg.coarse_grained_density(points, waters)
    assert np.allclose(np.mean(densities), 0.03249076838705669)

    inter_0, norm_0 = cg.find_interface(waters, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), calc_normal=True,
                                        tol=1e-08, slicing_cutoff=4)
    inter_1, norm_1 = cg.find_interface(waters, (0.0, 0.0, 15.0), (0.0, 0.0, -1.0), calc_normal=True,
                                        tol=1e-08, slicing_cutoff=4, reverse_search=True)
    assert np.allclose(inter_0, (0.0, 0.0, 12.77013225))
    assert np.allclose(norm_0 , (0.0, 0.0, 1.0), atol=0.06)
    assert np.allclose(inter_0, inter_1)
    assert np.allclose(norm_0, norm_1)

    inter_0, norm_0 = cg.find_interface(waters, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), calc_normal=True,
                                        tol=1e-08, slicing_cutoff=4)
    inter_1, norm_1 = cg.find_interface(waters, (0.0, 0.0, -15.0), (0.0, 0.0, 1.0), calc_normal=True,
                                        tol=1e-08, slicing_cutoff=4, reverse_search=True)
    assert np.allclose(inter_0, (0.0, 0.0, -12.66688555))
    assert np.allclose(norm_0 , (0.0, 0.0, -1.0), atol=0.06)
    assert np.allclose(inter_0, inter_1)
    assert np.allclose(norm_0, norm_1)

    N_tests = 50
    phi = np.random.random(N_tests) * 2 * np.pi
    cos = np.random.random(N_tests)
    sin = np.sqrt(1 - np.square(cos))
    axes = np.column_stack((np.cos(phi) * sin, np.sin(phi) * sin, cos))
    for axis in axes:
        inter_0, norm_0 = cg.find_interface(waters, (0.0, 0.0, 0.0), axis, calc_normal=True,
                                            tol=1e-08, slicing_cutoff=4)
        inter_1, norm_1 = cg.find_interface(waters, 15 * axis, -axis, calc_normal=True,
                                            tol=1e-08, slicing_cutoff=4, reverse_search=True)
        assert np.allclose(inter_0, inter_1)
        assert np.allclose(norm_0, norm_1)

#==================================================================================================

def test_droplet():

    trajectory = io.read('tests/examples/small-droplet.xyz', liq_symbol='O', sol_symbol='C')
    waters = trajectory['liq'][0]
    CoM = np.mean(waters, axis=0)

    assert np.allclose(CoM, (0, 0, 5.38673499))

    interface, norm = cg.find_interface(waters, CoM, (0, 0, 1), calc_normal=True,
                                        tol=1e-08, slicing_cutoff=4)
    assert np.allclose(interface, (0, 0, 11.7431722))
    assert np.allclose(norm, (-0.16674203, 0.08892512, 0.98198239))

    interface, norm = cg.find_interface(waters, CoM, (0, 0, -1), calc_normal=True,
                                        tol=1e-08, slicing_cutoff=4)
    assert np.allclose(interface, (0, 0, 0.829991896))
    assert np.allclose(norm, (0.11715618, 0.12325747, -0.98543494))

    interface, norm = cg.find_interface(waters, CoM, (1, 0, 0), calc_normal=True,
                                        tol=1e-08, slicing_cutoff=4)
    assert np.allclose(interface, (11.4934006, 0, 5.38673499))
    assert np.allclose(norm, (0.97003411, -0.0994569, 0.22168029))
