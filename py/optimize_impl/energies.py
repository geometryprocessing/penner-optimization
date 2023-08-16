
import optimization_py as opt
import optimize_impl.targets as targets
import igl
import numpy as np


def lp_energy(v, w, p=2):
    """
    Compute the lp distance between v and w.
    
    params[in] np.arrays v, w: vectors to compare
    param[in] int p: order of the lp energy
    return float: lp distance between v and w
    """
    energy = np.linalg.norm(v - w, ord=p)
    return energy


def max_error(F):
    """
    Compute the max infinity norm error of F. Return 0 if F is empty
    
    params[in] np.array F: vector of (possibly negative) angle errors
    return float: max error
    """
    if len(F) == 0:
        return 0
    else:
        return np.max(np.abs(F))


def symmetric_stretches(lambdas, lambdas_target):
    """
    Compute the per edge symmetric stretch value for embedded log edge lengths lambdas 
    and target embedded log edge lengths lambdas_target. The symmetric stretch value is 
    the average of (1) the ratio of the actual length to the target length and (2) the
    ratio of the target length to the actual length.
    
    param[in] np.array lambdas: embedded log edge lengths 
    param[in] np.array lambdas_target: target embedded log edge lengths 
    return np.array: array of symmetric stretch values per edge
    """
    l = np.exp(lambdas/2)
    l_target = np.exp(lambdas_target/2)
    stretches = 0.5*(l/l_target + l_target/l)
    return stretches


def find_bd_and_int_edges(C, proj_maps):
    """
    Get the boundary and interior edge indices for mesh C. If C is a doubled mesh, these are the
    edges on the symmetry line. If C is a closed mesh, there is no boundary.

    param[in] Mesh C: (possibly symmetric) mesh
    param[in] tuple(np.array) proj_maps: tuple of he2e, e2he, proj, embed, and P maps
    return np.array: list of boundary edge indices
    return np.array: list of 
    """
    he2e, e2he, proj, embed, P = proj_maps
    bd_e = proj[np.where(np.arange(len(e2he)) == he2e[np.array(C.R)[e2he]])[0]]
    int_e = np.setdiff1d(np.arange(len(embed)), bd_e)

    return bd_e, int_e

def get_areas_per_edge(C, lambdas_full, proj_maps):
    """
    Get the dual areas associated with each edge of C.

    param[in] Mesh C: (possibly symmetric) mesh
    param[in] np.array lambdas_full: log edge lengths
    param[in] tuple(np.array) proj_maps: tuple of he2e, e2he, proj, embed, and P maps
    return np.array: list of per edge areas
    """
    he2areasq = np.array(opt.areas_squared_from_lambdas(C, lambdas_full))
    he2area = np.sqrt(he2areasq)
    opp = np.array(C.opp)
    he2e, e2he, proj, embed, P = proj_maps

    return (he2area[e2he] + he2area[opp[e2he]]) / 3.0


def generate_opt_energy_funcs(
    C,
    energy_choice='p_norm',
    p=2,
    bd_weight=1.0,
    reg_factor=0.0,
    use_log=True
):
    """
    Generate optimization energy and associated gradient functions for C for the energy specified
    by the energy_choice parameter. These functions are all of the form 
        E(lambdas_target, lambdas)
    Note that they may depend on C , so the functions defined for one function may not be defined
    for another mesh C'. A Lp norm regularization term may also be added to any energy. The energy
    options are as follows:

        p_norm: Lp norm of the log edge lengths
        bd_norm: Lp norm of the boundary log edge lengths
        p_norm_int: Integral Lp norm of log edge lengths weighted by edge area
        length_norm: Lp norm of the edge lengths (not log edge lengths)
        weighted_p_norm: Lp norm of log edge lengths with a weight for the boundary edges
        metric_distortion: L2 norm of the per face metric distortion
        area_distortion: L2 norm of the per face area distortion
        scale_distortion: L2 norm of the best fit scale factors

    param[in] Mesh C: (possibly symmetric) mesh
    param[in] string energy_choice: energy function to generate for optimization
    param[in] int p: Lp norm order for p norm energies
    param[in] float bd_weight: weight for weighted_p_norm energy
    param[in] float reg_factor: optional regularization factor
    param[in] bool use_log: use log edge lengths/penner coordinates instead of regular lengths/Penner coordinates
    return func(np.array, np.array)->float: energy function
    return func(np.array, np.array)->np.array: energy gradient function
    """
    # Build edge and projection maps
    he2e, e2he = opt.build_edge_maps(C)
    proj, embed = opt.build_refl_proj(C)
    he2e = np.array(he2e)
    e2he = np.array(e2he)
    proj = np.array(proj)
    embed = np.array(embed)
    P = opt.refl_matrix(C)
    proj_maps = (he2e, e2he, proj, embed, P)

    # Get boundary and interior edges
    bd_e, int_e = find_bd_and_int_edges(C, proj_maps)

    # Define energy function for the energy choice
    # Lp Norm: Lp norm of the log edge lengths
    if (energy_choice == 'p_norm'):
        def opt_energy_base(lambdas_target, lambdas):
            return (1/p)*(np.linalg.norm(lambdas - lambdas_target, ord=p)**p)
        def g_opt_energy_base(lambdas_target, lambdas):
            return (lambdas - lambdas_target)**(p-1)

    # Lp Norm Integral: Integral Lp norm of log edge lengths weighted by edge area
    elif (energy_choice == 'p_norm_int'):
        # Get fixed original area vector for (Euclidean) lengths for connectivity
        lambdas_eucl = targets.lambdas_from_mesh(C)
        e2area = get_areas_per_edge(C, lambdas_eucl[proj], proj_maps)

        # Normalize so average edge has weight 1
        weight = np.average(e2area)

        def opt_energy_base(lambdas_target, lambdas):
            p_norm_vector = (1/p)*(lambdas[proj] - lambdas_target[proj])**p
            return np.sum(p_norm_vector * e2area) / weight
        def g_opt_energy_base(lambdas_target, lambdas):
            J_p_norm = (lambdas[proj] - lambdas_target[proj])**(p-1)
            g = (J_p_norm * e2area) @ P
            return g / weight

    # Surface Henckey Strain: Generalized Hencky strain energy 
    elif (energy_choice == 'surface_hencky_strain'):
        # Generate weight matrix using orignal (fixed) Euclidean lengths
        lambdas_eucl = targets.lambdas_from_mesh(C)
        R = opt.generate_edge_to_face_he_matrix(C)
        M = opt.surface_hencky_strain_energy(C, lambdas_eucl[proj])
        W = R.T @ M @ R

        # Normalize so average face has weight 1
        he2areasq = np.array(opt.areas_squared_from_lambdas(C, lambdas_eucl[proj]))
        he2area = np.sqrt(he2areasq)
        weight = np.average(he2area)

        def opt_energy_base(lambdas_target, lambdas):
            delta_lambdas = (lambdas - lambdas_target)
            return 0.5 * delta_lambdas.T @ (W @ delta_lambdas) / weight
        def g_opt_energy_base(lambdas_target, lambdas):
            delta_lambdas = (lambdas - lambdas_target)
            g =  W @ delta_lambdas 
            return g / weight

    # Boundary Norm: Lp norm of the boundary log edge lengths
    elif (energy_choice == 'bd_norm'):
        def opt_energy_base(lambdas_target, lambdas):
            return (1/p)*(np.linalg.norm(lambdas[bd_e] - lambdas_target[bd_e], ord=p)**p)
        def g_opt_energy_base(lambdas_target, lambdas):
            g = np.zeros_like(lambdas)
            g[bd_e] = (lambdas[bd_e] - lambdas_target[bd_e])**(p-1)
            return g

    # Weighted Lp Norm: Lp norm of log edge lengths with a weight for the boundary edges
    elif (energy_choice == 'weighted_p_norm'):
        def opt_energy_base(lambdas_target, lambdas):
            int_norm = np.linalg.norm(lambdas[int_e] - lambdas_target[int_e], ord=p)
            bd_norm = np.linalg.norm(lambdas[bd_e] - lambdas_target[bd_e], ord=p)
            return (1/p)*(np.linalg.norm([int_norm, bd_weight * bd_norm], ord=p)**p)
        def g_opt_energy_base(lambdas_target, lambdas):
            g = (lambdas - lambdas_target)**(p-1)
            g[bd_e] = bd_weight * (lambdas[bd_e] - lambdas_target[bd_e])**(p-1)
            return g

    # Length Norm: Lp norm of the edge lengths (not log edge lengths)
    elif (energy_choice == 'length_norm'):
        def opt_energy_base(lambdas_target, lambdas):
            return (1/p)*(np.linalg.norm(np.exp(lambdas/2.0) - np.exp(lambdas_target/2.0), ord=p)**p)
        def g_opt_energy_base(lambdas_target, lambdas):
            return 0.5*(np.exp(lambdas/2))*((np.exp(lambdas/2.0) - np.exp(lambdas_target/2.0))**(p-1))

    # Exponential p Norm: Lp norm of the exponential of edge length differences (not log edge lengths)
    elif (energy_choice == 'exp_p_norm'):
        def opt_energy_base(lambdas_target, lambdas):
            return (1/p)*(np.linalg.norm(np.exp(lambdas - lambdas_target), ord=p)**p)
        def g_opt_energy_base(lambdas_target, lambdas):
            return np.exp(p*(lambdas - lambdas_target))
            
    # Scale Distortion: L2 norm of the best fit scale factors
    elif (energy_choice == 'scale_distortion'):
        def opt_energy_base(lambdas_target, lambdas):
            u = opt.best_fit_conformal(C, lambdas_target[proj], lambdas[proj])
            return 0.5*(np.linalg.norm(u)**2)
        def g_opt_energy_base(lambdas_target, lambdas):
            return opt.scale_distortion_direction(C, lambdas_target[proj], lambdas[proj]) * P

    # Metric Distortion: L2 norm of the per face metric distortion
    # WARNING: Generally unstable and uses Euclidean instead of target edge lengths
    elif (energy_choice == 'metric_distortion'):
        lambdas_eucl = targets.lambdas_from_mesh(C)
        def opt_energy_base(lambdas_target, lambdas):
            f2energy, _ = opt.metric_distortion_energy(C, lambdas_eucl[proj], lambdas[proj], False)
            return 0.5*(np.linalg.norm(f2energy)**2)
        def g_opt_energy_base(lambdas_target, lambdas):
            f2energy, J_f2energy = opt.metric_distortion_energy(
                C,
                lambdas_eucl[proj],
                lambdas[proj],
                True
            )
            return (f2energy.T * J_f2energy * P) / np.linalg.norm(f2energy)

    # Area Distortion: L2 norm of the per face area distortion 
    # WARNING: Generally unstable and uses Euclidean instead of target edge lengths
    elif (energy_choice == 'area_distortion'):
        lambdas_eucl = targets.lambdas_from_mesh(C)
        def opt_energy_base(lambdas_target, lambdas):
            f2energy, _ = opt.area_distortion_energy(C, lambdas_eucl[proj], lambdas[proj], False)
            return 0.5*(np.linalg.norm(f2energy)**2)
        def g_opt_energy_base(lambdas_target, lambdas):
            f2energy, J_f2energy = opt.area_distortion_energy(C,
                                                              lambdas_eucl[proj],
                                                              lambdas[proj],
                                                              True)
            return (f2energy.T * J_f2energy * P) / np.linalg.norm(f2energy)

    # Unweighted Symmetric Dirichlet: L2 norm of the symmetric dirichlet energy per face
    elif (energy_choice == 'unweighted_sym_dirichlet'):
        def opt_energy_base(lambdas_target, lambdas):
            f2energy, _ = opt.symmetric_dirichlet_energy(C, lambdas_target[proj], lambdas[proj], False)
            return 0.5*(np.linalg.norm(f2energy)**2)
        def g_opt_energy_base(lambdas_target, lambdas):
            f2energy, J_f2energy = opt.symmetric_dirichlet_energy(
                C,
                lambdas_target[proj],
                lambdas[proj],
                True
            )
            return (f2energy.T * J_f2energy * P)

    # Symmetric Dirichlet: Integral L2 norm of the symmetric dirichlet energy per face
    elif (energy_choice == 'sym_dirichlet'):
        # Normalize so average face has weight 1
        lambdas_eucl = targets.lambdas_from_mesh(C)
        he2areasq = np.array(opt.areas_squared_from_lambdas(C, lambdas_eucl[proj]))
        he2area = np.sqrt(he2areasq)
        f2area = he2area[C.h]
        average_face_area = np.average(f2area)
        f2area_normalized = f2area / average_face_area

        def opt_energy_base(lambdas_target, lambdas):
            f2energy, _ = opt.symmetric_dirichlet_energy(C, lambdas_target[proj], lambdas[proj], False)
            return 0.5*(np.linalg.norm(f2energy * f2area_normalized)**2)
        def g_opt_energy_base(lambdas_target, lambdas):
            f2energy, J_f2energy = opt.symmetric_dirichlet_energy(
                C,
                lambdas_target[proj],
                lambdas[proj],
                True
            )
            return ((f2area_normalized * f2energy).T * J_f2energy * P)

    # Default: 0
    else:
        def opt_energy_base(lambdas_target, lambdas):
            return 0
        def g_opt_energy_base(lambdas_target, lambdas):
            return np.zeros_like(lambdas)

    def opt_energy(lambdas_target, lambdas):
        reg_value = np.linalg.norm(lambdas - lambdas_target, ord=p)**p
        return opt_energy_base(lambdas_target, lambdas) + reg_factor * reg_value
    def g_opt_energy(lambdas_target, lambdas):
        # Regularize 
        g_k = g_opt_energy_base(lambdas_target, lambdas)
        g_k += reg_factor * (lambdas - lambdas_target)**(p-1)

        # Optionally use log coordinates
        if not use_log:
            J_l = opt.length_jacobian(lambdas)
            g_k = g_k * J_l

        return g_k

    return opt_energy, g_opt_energy


def first_invariant_vf(v, f, uv, fuv):
    """
    Compute the per face first metric invariant (sum of squared singular values) from
    initial and parameterization vertex positions

    param[in] np.array v: initial mesh vertex positions
    param[in] np.array f: mesh connectivity
    param[in] np.array uv: parameterization mesh vertex positions
    param[in] np.array fuv: parameterization mesh connectivity
    return np.array: per face first metric invariant
    """
    if uv.shape[1] < 3:
        uv_embed = np.zeros((len(uv), 3))
        uv_embed[:,:2] = uv[:,:2]
    else:
        uv_embed = uv[:,:3]
    double_area_0 = igl.doublearea(v, f)
    cot_alpha_0 = 2.0 * igl.cotmatrix_entries(v, f)
    l = igl.edge_lengths(uv_embed, fuv)
    return np.sum(cot_alpha_0 * l * l, axis=1) / double_area_0


def second_invariant_vf(v, f, uv, fuv):
    """
    Compute the per face second metric invariant (square root of the determinant) from
    initial and parameterization vertex positions

    param[in] np.array v: initial mesh vertex positions
    param[in] np.array f: mesh connectivity
    param[in] np.array uv: parameterization mesh vertex positions
    param[in] np.array fuv: parameterization mesh connectivity
    return np.array: per face second metric invariant
    """
    double_area_0 = igl.doublearea(v, f)
    double_area =igl.doublearea(uv, fuv)

    return double_area / double_area_0


def sym_dirichlet_vf(v, f, uv, fuv):
    """
    Compute the per face symmetric Dirichlet energy from initial and parameterization
    vertex positions

    param[in] np.array v: initial mesh vertex positions
    param[in] np.array f: mesh connectivity
    param[in] np.array uv: parameterization mesh vertex positions
    param[in] np.array fuv: parameterization mesh connectivity
    return np.array: per face symmetric Dirichlet energy
    """
    J1 = first_invariant_vf(v, f, uv, fuv)
    J2 = second_invariant_vf(v, f, uv, fuv)

    return J1 + J1 / (J2 * J2)


def dirichlet_vf(v, f, uv, fuv):
    """
    Compute the per face Dirichlet energy from initial and parameterization
    vertex positions

    param[in] np.array v: initial mesh vertex positions
    param[in] np.array f: mesh connectivity
    param[in] np.array uv: parameterization mesh vertex positions
    param[in] np.array fuv: parameterization mesh connectivity
    return np.array: per face Dirichlet energy
    """
    J1 = first_invariant_vf(v, f, uv, fuv)

    return J1


def mips_vf(v, f, uv, fuv):
    """
    Compute the per face MIPS energy from initial and parameterization
    vertex positions

    param[in] np.array v: initial mesh vertex positions
    param[in] np.array f: mesh connectivity
    param[in] np.array uv: parameterization mesh vertex positions
    param[in] np.array fuv: parameterization mesh connectivity
    return np.array: per face MIPS energy
    """
    J1 = first_invariant_vf(v, f, uv, fuv)
    J2 = second_invariant_vf(v, f, uv, fuv)

    return J1/J2


def amips_vf(v, f, uv, fuv):
    """
    Compute the per face AMIPS energy from initial and parameterization
    vertex positions

    param[in] np.array v: initial mesh vertex positions
    param[in] np.array f: mesh connectivity
    param[in] np.array uv: parameterization mesh vertex positions
    param[in] np.array fuv: parameterization mesh connectivity
    return np.array: per face AMIPS energy
    """
    J1 = first_invariant_vf(v, f, uv, fuv)
    J2 = second_invariant_vf(v, f, uv, fuv)

    return J1/J2 + J2 + 1/J2


def arap_vf(v, f, uv, fuv):
    """
    Compute the per face ARAP energy from initial and parameterization
    vertex positions

    param[in] np.array v: initial mesh vertex positions
    param[in] np.array f: mesh connectivity
    param[in] np.array uv: parameterization mesh vertex positions
    param[in] np.array fuv: parameterization mesh connectivity
    return np.array: per face ARAP energy
    """
    J1 = first_invariant_vf(v, f, uv, fuv)
    J2 = second_invariant_vf(v, f, uv, fuv)

    return J1 - 2.0 * np.sqrt(J1 + 2.0 * J2) + 2.0


def surface_hencky_strain_vf(v, f, uv, fuv):
    """
    Compute the per face ARAP energy from initial and parameterization
    vertex positions

    param[in] np.array v: initial mesh vertex positions
    param[in] np.array f: mesh connectivity
    param[in] np.array uv: parameterization mesh vertex positions
    param[in] np.array fuv: parameterization mesh connectivity
    return np.array: per face ARAP energy
    """
    area_0 = igl.doublearea(v, f) / 2.0
    cot_alpha_0 = 2.0 * igl.cotmatrix_entries(v, f)
    l_0 = igl.edge_lengths(v, f)
    l = igl.edge_lengths(uv, fuv)
    ll_0 = 2.0 * np.log(l_0)
    ll = 2.0 * np.log(l)
    weight = np.average(area_0)

    return opt.surface_hencky_strain_energy_vf(area_0, cot_alpha_0, l_0, ll - ll_0) / weight


def best_fit_conformal_vf(v, f, uv, fuv):
    Th_hat = np.zeros(len(v))

    print("Building original metric")
    print(v.shape)
    print(f.shape)
    print(uv.shape)
    print(fuv.shape)
    C_v, vtx_reindex = opt.fv_to_double(v, f, v, f, Th_hat, [], False)
    print("Building optimized metric")
    C_uv, vtx_reindex = opt.fv_to_double(v, f, uv, fuv, Th_hat, [], False)
    proj, embed = opt.build_refl_proj(C_v)
    he2e, e2he = opt.build_edge_maps(C_v)
    proj = np.array(proj)
    he2e = np.array(he2e)
    lambdas_target = targets.lambdas_from_mesh(C_v)
    lambdas = targets.lambdas_from_mesh(C_uv)
    print("Lambdas target length", lambdas_target.shape)
    print("Lambdas length", lambdas.shape)
    C_o = opt.add_overlay(C_v, lambdas_target)
    opt.make_tufted_overlay(C_o, v, f, Th_hat)
    r_perm = opt.best_fit_conformal(C_o._m, lambdas_target[proj], lambdas[proj])
    
    # Reindex energy
    r = np.zeros_like(r_perm)
    r[vtx_reindex] = np.array(r_perm)

    return r


def get_vertex_energy(
    v,
    f,
    uv,
    fuv,
    colormap,
    use_sqrt_scale=False,
    use_log_scale=False,
):
    # Ensure uv is #V x 3
    uv_embed = np.zeros((len(uv), 3))
    uv_embed[:,:2] = uv[:,:2]

    # Get energy
    print("Getting energy")
    print(v.shape)
    print(f.shape)
    print(uv.shape)
    print(fuv.shape)
    if (colormap == 'scale_factors'):
        energy = best_fit_conformal_vf(v, f, uv_embed, fuv)
    if (colormap == 'sym_dirichlet'):
        energy_f = get_face_energy(v, f, uv, fuv, colormap)
        energy_f_dup = np.array([energy_f, energy_f, energy_f]).T
        energy = igl.average_onto_vertices(v, f, energy_f_dup)[:,0]

    # Add sqrt or log scale
    if use_sqrt_scale:
        energy = np.sqrt(np.maximum(energy, 0))
    if use_log_scale:
        energy = np.log(np.maximum(energy + 1, 0))

    return energy

def get_face_energy(
    v,
    f,
    uv,
    fuv,
    colormap,
    use_face_weight=False,
    use_sqrt_scale=False,
    use_log_scale=False,
):
    # Ensure uv is #V x 3
    uv_embed = np.zeros((len(uv), 3))
    uv_embed[:,:2] = uv[:,:2]

    # Get energy
    energy = np.zeros(len(f))
    if (colormap == 'sym_dirichlet'):
        energy = sym_dirichlet_vf(v, f, uv_embed, fuv) - 4
    if (colormap == 'dirichlet'):
        energy = dirichlet_vf(v, f, uv_embed, fuv) - 2
    if (colormap == 'mips'):
        energy = mips_vf(v, f, uv_embed, fuv) - 2
    if (colormap == 'amips'):
        energy = amips_vf(v, f, uv_embed, fuv) - 4
    if (colormap == 'arap'):
        energy = arap_vf(v, f, uv_embed, fuv)
    if (colormap == 'surface_hencky_strain'):
        energy = surface_hencky_strain_vf(v, f, uv_embed, fuv)

    # Optionally use face weighting
    if (use_face_weight):
        print("Using face weights") # FIXME
        mesh_areas = 0.5 * igl.doublearea(v, f)
        energy = (mesh_areas * energy) / (np.sum(mesh_areas))


    # Add sqrt or log scale
    if use_sqrt_scale:
        print("Using sqrt scale") # FIXME
        energy = np.sqrt(np.maximum(energy, 0))
    if use_log_scale:
        print("Using log scale") # FIXME
        energy = np.log(np.maximum(energy + 1, 0))

    return energy

def get_interpolated_vertex_energy(
    v,
    f,
    uv,
    fuv,
    vn_to_v,
    endpoints,
    colormap,
    use_sqrt_scale=False,
    use_log_scale=False,
):
    # Ensure uv is #V x 3
    uv_embed = np.zeros((len(uv), 3))
    uv_embed[:,:2] = uv[:,:2]

    # Get energy
    if (colormap == 'inserted_vertices'):
        energy = inserted_vertices(v, vn_to_v, endpoints)

    # Add sqrt or log scale
    if use_sqrt_scale:
        energy = np.sqrt(np.maximum(energy, 0))
    if use_log_scale:
        energy = np.log(np.maximum(energy + 1, 0))

    return energy

def inserted_vertices(v, vn_to_v, endpoints):
    # Get vertices inserted from overlay
    r = np.zeros(len(v))
    r[endpoints[vn_to_v][:,0]>=0] = 1
    return r


