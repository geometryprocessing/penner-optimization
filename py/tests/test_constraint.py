import os, sys
file_dir = os.path.dirname(__file__)
module_dir = os.path.join(file_dir, '..')
sys.path.append(module_dir)
import optimization_py as opt
import optimize_impl.optimization as optimization
import optimize_impl.meshes as meshes
import optimize_impl.targets as targets
import numpy as np

def validate_satisfies_triangle_inequality_triangle(ll):
    v, f, Th_hat = meshes.generate_tri()
    C, vtx_reindex = opt.fv_to_double(v, f, Th_hat, False)
    proj, embed = opt.build_refl_proj(C)
    he2e, e2he = opt.build_edge_maps(C)
    proj = np.array(proj)
    he2e = np.array(he2e)

    # Get lambdas with changed length
    lambdas = targets.lambdas_from_mesh(C)
    lambdas[0] = ll
    lambdas[proj[he2e[C.opp[0]]]] = ll
    lambdas_full = lambdas[proj]

    # Check result
    expected_result = (np.exp(ll/2) < 2*np.sqrt(2))
    assert(opt.satisfies_triangle_inequality(C, lambdas_full) == expected_result)


def test_satisfies_triangle_inequality_triangle():
    validate_satisfies_triangle_inequality_triangle(0)
    validate_satisfies_triangle_inequality_triangle(2)
    validate_satisfies_triangle_inequality_triangle(3)
    validate_satisfies_triangle_inequality_triangle(4)
    validate_satisfies_triangle_inequality_triangle(-100)
    validate_satisfies_triangle_inequality_triangle(100)