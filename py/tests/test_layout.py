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

def test_generate_four_point_projection_with_symmetric_quad():
    validate_generate_four_point_projection_with_symmetric_quad(0,0)
    validate_generate_four_point_projection_with_symmetric_quad(0,0.5)
    validate_generate_four_point_projection_with_symmetric_quad(0.5,0)
    validate_generate_four_point_projection_with_symmetric_quad(0,-0.5)
    validate_generate_four_point_projection_with_symmetric_quad(-0.5,0)
    validate_generate_four_point_projection_with_symmetric_quad(-0.5,0.5)
    validate_generate_four_point_projection_with_symmetric_quad(0.1,0.6)