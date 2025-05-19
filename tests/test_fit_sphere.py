import numpy as np
from droplet_graphene_analysis.util import best_fit_sphere, best_fit_axial_sphere

#==================================================================================================

def test_2d():

    N_tests = 50
    N_pts = 100
    noise = 0.01
    c0 = (2.0 * np.random.random((N_tests, 2))) - 1.0
    a0 = 1.0 + (9.0 * np.random.random(N_tests))

    for n in range(N_tests):
        phi = 2 * np.pi * np.random.random(N_pts)
        r = a0[n] + (noise * (np.random.random(N_pts) - 0.5))
        pts = c0[n] + np.c_[r * np.cos(phi), r * np.sin(phi)]
        a, c = best_fit_sphere(pts, d=2)
        assert np.allclose(a, a0[n], atol=noise)
        assert np.allclose(c, c0[n], atol=noise)

#==================================================================================================

def test_3d():

    N_tests = 50
    N_pts = 100
    noise = 0.01
    c0 = (2.0 * np.random.random((N_tests, 3))) - 1.0
    a0 = 1.0 + (9.0 * np.random.random(N_tests))

    for n in range(N_tests):
        phi = 2 * np.pi * np.random.random(N_pts)
        cosine = (2.0 * np.random.random(N_pts)) - 1.0
        sine = np.sqrt(1 - (cosine**2))
        r = a0[n] + (noise * (np.random.random(N_pts) - 0.5))
        pts = c0[n] + np.c_[r * np.cos(phi) * sine, r * np.sin(phi) * sine, r * cosine]
        a, c = best_fit_sphere(pts, d=3)
        assert np.allclose(a, a0[n], atol=noise)
        assert np.allclose(c, c0[n], atol=noise)

#==================================================================================================

def test_3d_constrained():

    N_tests = 50
    N_pts = 100
    noise = 0.01
    c0 = (2.0 * np.random.random(N_tests)) - 1.0
    a0 = 1.0 + (9.0 * np.random.random(N_tests))

    for n in range(N_tests):
        phi = 2 * np.pi * np.random.random(N_pts)
        cosine = (2.0 * np.random.random(N_pts)) - 1.0
        sine = np.sqrt(1 - (cosine**2))
        r = a0[n] + (noise * (np.random.random(N_pts) - 0.5))
        pts = np.array([0.0, 0.0, c0[n]]) + np.c_[r * np.cos(phi) * sine, r * np.sin(phi) * sine, r * cosine]
        a, c = best_fit_axial_sphere(pts)
        assert np.allclose(a, a0[n], atol=noise)
        assert np.allclose(c, c0[n], atol=noise)