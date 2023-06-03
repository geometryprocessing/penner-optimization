
import optimization_py as opt
import numpy as np
import scipy.sparse as sp

# *********************************************
# Below is deprecated code now rewritten in C++
# *********************************************

def compute_shear_change(
    C,
    lambdas_init,
    lambdas,
    use_original_triangulation=True
):
    # Generate embedding arrays
    proj, embed = opt.build_refl_proj(C)
    he2e, e2he = opt.build_edge_maps(C)
    proj = np.array(proj)
    embed = np.array(embed)
    he2e = np.array(he2e)
    e2he = np.array(e2he)

    if use_original_triangulation:
        # Get change in shear for original triangulation
        sigmas0 = opt.compute_shear(C, lambdas_init[proj[he2e]])
        sigmas = opt.compute_shear(C, lambdas[proj[he2e]])
        return sigmas - sigmas0
    else:
        # Make mesh delaunay
        C_del, lambdas_del, _, flip_seq = opt.make_delaunay_with_jacobian(C,
                                                                        lambdas[proj],
                                                                        False)
        lambdas_del = np.array(lambdas_del)

        # Duplicate flips with original metric
        _, lambdas_init_del_full = opt.flip_edges(C, lambdas_init[proj], flip_seq)
        lambdas_init_del_full = np.array(lambdas_init_del_full)

        # Get change in shear for final triangulation
        sigmas0 = opt.compute_shear(C_del, lambdas_init_del_full[he2e])
        sigmas = opt.compute_shear(C_del, lambdas_del[he2e])
        return sigmas - sigmas0


def as_symmetric_as_possible_translations(
    C,
    lambdas_init,
    lambdas,
    use_original_triangulation=True
):
    # Generate embedding arrays
    proj, embed = opt.build_refl_proj(C)
    he2e, e2he = opt.build_edge_maps(C)
    proj = np.array(proj)
    he2e = np.array(he2e)

    if use_original_triangulation:
        # Get change in shear for original triangulation
        sigmas0 = opt.compute_shear(C, lambdas_init[proj[he2e]])
        sigmas = opt.compute_shear(C, lambdas[proj[he2e]])
        delta_sigmas = sigmas - sigmas0

        # Generate Lagrangian system for the quadratic energy with redundant constraint removed
        M, b = opt.generate_translation_lagrangian_system(C, delta_sigmas)
        M = M[:-1,:-1]
        b = b[:-1]
    else:
        # Make mesh delaunay
        C_del, lambdas_del, _, flip_seq = opt.make_delaunay_with_jacobian(C,
                                                                        lambdas[proj],
                                                                        False)
        lambdas_del = np.array(lambdas_del)

        # Duplicate flips with original metric
        _, lambdas_init_del_full = opt.flip_edges(C, lambdas_init[proj], flip_seq)
        lambdas_init_del_full = np.array(lambdas_init_del_full)

        # Get change in shear for final triangulation
        sigmas0 = opt.compute_shear(C_del, lambdas_init_del_full[he2e])
        sigmas = opt.compute_shear(C_del, lambdas_del[he2e])
        delta_sigmas = sigmas - sigmas0

        # Generate Lagrangian system for the quadratic energy with redundant constraint removed
        M, b = opt.generate_translation_lagrangian_system(C_del, delta_sigmas)
        #M = M[:-1,:-1] Now removed directly on cpp end
        #b = b[:-1] Now removed directly on cpp end

	# Get translations
    return sp.linalg.spsolve(M, -b)[:he2e.size]

