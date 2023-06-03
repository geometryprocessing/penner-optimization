import os, sys
file_dir = os.path.dirname(__file__)
module_dir = os.path.join(file_dir, '..')
sys.path.append(module_dir)
import optimize_impl.optimization as optimization
import optimization_py as opt
import optimize_impl.meshes as meshes
import numpy as np

def validate_generate_four_point_projection_with_symmetric_quad(r, s):
    # Generate quad for x position r
    v0 = np.array([-1,0])
    v1 = np.array([1,0])
    v2 = np.array([r,np.sqrt(1 - r * r)])
    v3 = np.array([r,-np.sqrt(1 - r * r)])
    A = opt.generate_four_point_projection(v0, v1, v2, v3)

    # Generate quad for x position s
    w0 = np.array([-1,0])
    w1 = np.array([1,0])
    w2 = np.array([s,np.sqrt(1 - s * s)])
    w3 = np.array([s,-np.sqrt(1 - s * s)])
    B = opt.generate_four_point_projection(w0, w1, w2, w3)
    
    # Map [r, 0, 1] from the r-quad to the s-quad
    P = B @ np.linalg.inv(A)
    w = P @ np.array([r,0,1]) 
    x = w[0] / w[2]

    # Check that the result is [s, 0, 1]
    assert(abs(x - s) < 1e-12)

def validate_generate_four_point_projection(vi, vj, vk, vl):
    # Compute output of generate_four_point_projection for standard projective basis
    A = opt.generate_four_point_projection(vi, vj, vk, vl)
    wi = A@np.array([1,0,0])
    wj = A@np.array([0,1,0])
    wk = A@np.array([0,0,1])
    wl = A@np.array([1,1,1])
    wi = wi[:2] / wi[2]
    wj = wj[:2] / wj[2]
    wk = wk[:2] / wk[2]
    wl = wl[:2] / wl[2]
    
    # Check that output matches analytic result
    assert(np.abs(np.max(vi - wi)) < 1e-12)
    assert(np.abs(np.max(vj - wj)) < 1e-12)
    assert(np.abs(np.max(vk - wk)) < 1e-12)
    assert(np.abs(np.max(vl - wl)) < 1e-12)

def validate_generate_four_point_projection_from_shear(sigma):
    # Create two triangle chart for shear
    c = np.exp(sigma / 2.0)
    r = (1 - c) / (1 + c)
    vi= np.array([-1,0])
    vj = np.array([1,0])
    vk = np.array([r,np.sqrt(1 - r * r)])
    vl = np.array([-r,-np.sqrt(1 - r * r)])

    # Compute output of generate_four_point_projection_from_shear for standard projective basis
    A = opt.generate_four_point_projection_from_shear(sigma)
    wi = A@np.array([1,0,0])
    wj = A@np.array([0,1,0])
    wk = A@np.array([0,0,1])
    wl = A@np.array([1,1,1])
    wi = wi[:2] / wi[2]
    wj = wj[:2] / wj[2]
    wk = wk[:2] / wk[2]
    wl = wl[:2] / wl[2]
    
    # Check that output matches analytic result
    assert(np.abs(np.max(vi - wi)) < 1e-12)
    assert(np.abs(np.max(vj - wj)) < 1e-12)
    assert(np.abs(np.max(vk - wk)) < 1e-12)
    assert(np.abs(np.max(vl - wl)) < 1e-12)

def validate_generate_four_point_projection_from_lengths(lij, lik, ljk, lil, ljl):
    # Create two triangle chart for lengths
    A = opt.generate_four_point_projection_from_lengths(lij, lik, ljk, lil, ljl)
    wi = A@np.array([1,0,0])
    wj = A@np.array([0,1,0])
    wk = A@np.array([0,0,1])
    wl = A@np.array([1,1,1])
    wi = wi[:2] / wi[2]
    wj = wj[:2] / wj[2]
    wk = wk[:2] / wk[2]
    wl = wl[:2] / wl[2]

    # Compute lengths of output
    Lij = np.linalg.norm(wi - wj)
    Lik = np.linalg.norm(wi - wk)
    Ljk = np.linalg.norm(wj - wk)
    Lil = np.linalg.norm(wi - wl)
    Ljl = np.linalg.norm(wj - wl)

    print(wi, wj, wk, wl)

    # Check the lengths of the triangles match the input
    assert(np.abs(np.max(lij - Lij)) < 1e-12)
    assert(np.abs(np.max(lik - Lik)) < 1e-12)
    assert(np.abs(np.max(ljk - Ljk)) < 1e-12)
    assert(np.abs(np.max(lil - Lil)) < 1e-12)
    assert(np.abs(np.max(ljl - Ljl)) < 1e-12)

    # Check the orientation of the triangles
    assert(wk[1] > 1e-12)
    assert(wl[1] < -1e-12)

    # Check if the matrix is degenerate
    assert(np.abs(np.linalg.det(A)) > 1e-12)


def validate_two_triangle_chart_coordinates_with_symmetric_quad(x, y, s):
    v, f, Th_hat = meshes.generate_symmetric_quad(x, y)
    C, vtx_reindex = opt.fv_to_double(v, f, Th_hat, False)
    lambdas = optimization.lambdas_from_mesh(C)
    proj, embed = opt.build_refl_proj(C)
    he2e, e2he = opt.build_edge_maps(C)
    proj = np.array(proj)
    he2e = np.array(he2e)
    lambdas_he = lambdas[proj[he2e]]
    

def test_generate_four_point_projection():
    validate_generate_four_point_projection(np.array([1,0]),
                                            np.array([0,0]),
                                            np.array([0,1]),
                                            np.array([1,1]))
    validate_generate_four_point_projection(np.array([1e3,0]),
                                            np.array([0,0]),
                                            np.array([0,1]),
                                            np.array([1,1]))
    validate_generate_four_point_projection(np.array([0,0]),
                                            np.array([1,0]),
                                            np.array([0,-1]),
                                            np.array([1,1]))


def test_generate_four_point_projection_with_symmetric_quad():
    validate_generate_four_point_projection_with_symmetric_quad(0,0)
    validate_generate_four_point_projection_with_symmetric_quad(0,0.5)
    validate_generate_four_point_projection_with_symmetric_quad(0.5,0)
    validate_generate_four_point_projection_with_symmetric_quad(0,-0.5)
    validate_generate_four_point_projection_with_symmetric_quad(-0.5,0)
    validate_generate_four_point_projection_with_symmetric_quad(-0.5,0.5)
    validate_generate_four_point_projection_with_symmetric_quad(0.1,0.6)

def test_generate_four_point_projection_from_shear():
    validate_generate_four_point_projection_from_shear(0)
    validate_generate_four_point_projection_from_shear(1)
    validate_generate_four_point_projection_from_shear(-1)
    validate_generate_four_point_projection_from_shear(25)
    validate_generate_four_point_projection_from_shear(-25)
    validate_generate_four_point_projection_from_shear(np.pi)
    
def test_generate_four_point_projection_from_lengths():
    validate_generate_four_point_projection_from_lengths(1.0, 1.0, 1.0, 1.0, 1.0)
    validate_generate_four_point_projection_from_lengths(0.3,
                                                         0.1666666666666667,
                                                         0.3726779962499649,
                                                         0.1666666666666667,
                                                         0.3726779962499649)
    # Known to be degenerate
    #validate_generate_four_point_projection_from_lengths(0.3333333333333333,
    #                                                     0.1666666666666667,
    #                                                     0.3726779962499649,
    #                                                     0.1666666666666667,
    #                                                     0.3726779962499649)