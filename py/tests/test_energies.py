import os, sys
file_dir = os.path.dirname(__file__)
module_dir = os.path.join(file_dir, '..')
sys.path.append(module_dir)
import optimize_impl.optimization as optimization
import optimize_impl.meshes as meshes
import optimize_impl.energies as energies
import optimize_impl.targets as targets
import optimization_py as opt
import finite_diff
import numpy as np

def validate_symmetric_dirichlet_energy(height, width):
    v, f, Th_hat = meshes.generate_right_triangle(2.0, 3.0)
    C, vtx_reindex = opt.fv_to_double(v, f, Th_hat, False)
    proj, embed = opt.build_refl_proj(C)
    he2e, e2he = opt.build_edge_maps(C)
    proj = np.array(proj)
    he2e = np.array(he2e)
    lambdas = targets.lambdas_from_mesh(C)
    lambdas_full = lambdas[proj]

    # Check result
    expected_result = height **2 + width **2 + 1.0 / (height ** 2) + 1.0 / (width ** 2)
    result = opt.symmetric_dirichlet_energy(
        C,
        lambdas_full,
        lambdas_full,
        False
    )[0][0]
    assert((result - expected_result) < 1e-12)

def validate_generate_opt_energy_funcs(C, energy_choice):
    opt_energy_funcs = energies.generate_opt_energy_funcs(
        C,
        energy_choice=energy_choice
    )
    opt_energy_func, g_opt_energy_func = opt_energy_funcs
    lambdas_target = targets.lambdas_from_mesh(C)
    lambdas = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=len(lambdas_target)
    )

    errors = finite_diff.validate_energy_gradient(
        C,
        lambdas_target,
        lambdas,
        opt_energy_func,
        g_opt_energy_func
    )

    assert (np.max(errors) < 1e-10)


def test_symmetric_dirichlet_energy():
    validate_symmetric_dirichlet_energy(1.0, 1.0)
    validate_symmetric_dirichlet_energy(2.0, 1.0)
    validate_symmetric_dirichlet_energy(1.0, 2.0)
    validate_symmetric_dirichlet_energy(2.0, 2.0)
    validate_symmetric_dirichlet_energy(0.5, 2.6)

def test_generate_opt_energy_funcs():
    v, f, Th_hat = meshes.generate_right_triangle(2.0, 3.0)
    C, vtx_reindex = opt.fv_to_double(v, f, Th_hat, False)
    validate_generate_opt_energy_funcs(C, "p_norm")
    validate_generate_opt_energy_funcs(C, "bd_norm")
    validate_generate_opt_energy_funcs(C, "surface_hencky_strain")
    validate_generate_opt_energy_funcs(C, "scale_distortion")

    v, f, Th_hat = meshes.generate_tet()
    C, vtx_reindex = opt.fv_to_double(v, f, Th_hat, False)
    validate_generate_opt_energy_funcs(C, "p_norm")
    validate_generate_opt_energy_funcs(C, "bd_norm")
    validate_generate_opt_energy_funcs(C, "surface_hencky_strain")
    validate_generate_opt_energy_funcs(C, "scale_distortion")
