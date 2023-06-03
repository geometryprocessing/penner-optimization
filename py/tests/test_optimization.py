import os, sys
file_dir = os.path.dirname(__file__)
module_dir = os.path.join(file_dir, '..')
sys.path.append(module_dir)
import optimize_impl.optimization as optimization
import optimize_impl.meshes as meshes
import optimize_impl.targets as targets
import optimization_py as opt

def test_tri():
    v, f, Th_hat = meshes.generate_tri()
    C, vtx_reindex = opt.fv_to_double(v, f, Th_hat, False)
    lambdas_target = targets.lambdas_from_mesh(C)
    lambdas_init = lambdas_target.copy()
    _, lambdas = optimization.optimize_lambdas(C,
                                               lambdas_init,
                                               lambdas_target)

    assert(optimization.max_error(lambdas - lambdas_target) == 0.0)

