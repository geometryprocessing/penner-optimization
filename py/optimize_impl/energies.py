
import optimization_py as opt
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


def quadratic_sym_dirichlet_vf(v, f, uv, fuv):
    """
    Compute the per face quadratic symmetric dirichlet energy from initial and parameterization
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

    C_v = opt.generate_initial_mesh(v, f, v, f, Th_hat, [], [], False, True)
    C_uv = opt.generate_initial_mesh(v, f, uv, fuv, Th_hat, [], [], False, True)
    _, vtx_reindex = opt.fv_to_double(v, f, v, f, Th_hat, [], False)
    metric_coords = C_uv.get_metric_coordinates()
    r_perm = opt.best_fit_conformal(C_v, metric_coords)
    
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
    if (colormap == 'none'):
        energy = 0.25 * np.ones(len(v))
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
    if (colormap == 'quadratic_sym_dirichlet'):
        energy = quadratic_sym_dirichlet_vf(v, f, uv_embed, fuv)
    if (colormap == 'none'):
        energy = 0.25 * np.ones(len(f))

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
