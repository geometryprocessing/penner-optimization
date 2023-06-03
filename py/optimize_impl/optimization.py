import os, math
import scipy as sp
#import scipy.optimize
import numpy as np
import optimization_py as opt
import time
import logging
import multiprocessing
from optimize_impl.energies import lp_energy, max_error
import optimize_impl.energies as energies
import optimize_impl.targets as targets

def optimize_shear_basis_coordinates(
    C,
    reduced_metric_target,
    shear_basis_coords_init,
    scale_factors_init,
    shear_basis_matrix,
    proj_params,
    opt_params,
    method = 'L-BFGS-B'
):
    """
    Optimize a metric satisfying constraints in terms of a basis of the shear
    space orthogonal to the space of conformal scalings.

    param[in] C: mesh 
    param[in] reduced_metric_target: target metric for the optimization
    param[in] shear_basis_coords_init: initial shear coordinate basis coefficients
    param[in] shear_basis_coords_init: initial scale factors
    param[in] shear_basis_matrix: matrix with shear coordinate basis vectors as columns
    param[in] proj_params: parameters fro the projection to the constraint manifold
    param[in] opt_params: parameters for the optimization
    param[in] method: optimization method to use
    return optimized metric coordinates
    """
    reduction_maps = opt.ReductionMaps(C)

    # Get the optimization energy
    metric_target = reduced_metric_target[reduction_maps.proj]
    opt_energy = opt.EnergyFunctor(C, metric_target, opt_params)

    # Compute optimization domain
    scale_factor_basis_matrix = opt.conformal_scaling_matrix(C)
    optimization_domain_res = opt.compute_optimization_domain(
        C,
        shear_basis_coords_init,
        scale_factors_init,
        shear_basis_matrix,
        scale_factor_basis_matrix
    )
    domain_coords_init, constraint_domain_matrix, constraint_codomain_matrix = optimization_domain_res

    def fun(domain_coords):
        return opt.compute_domain_coordinate_energy(
            C,
            reduction_maps,
            opt_energy,
            domain_coords,
            constraint_domain_matrix,
            proj_params
        )

    def jac(domain_coords):
        energy_res = opt.compute_domain_coordinate_energy_with_gradient(
            C,
            reduction_maps,
            opt_energy,
            domain_coords,
            constraint_domain_matrix,
            constraint_codomain_matrix,
            proj_params
        )
        return energy_res[1]
    
    log = []
    def callback(domain_coords):
        energy = fun(domain_coords)
        print("Energy:", energy)
        log.append(energy)
    
    # Set bounds for the optimization to keep stable
    bounds = sp.optimize.Bounds(domain_coords_init-10, domain_coords_init+10)

    # Set solver options
    options = { 'maxiter': opt_params.num_iter,
                'disp': True }

    # Minimize energy
    res = sp.optimize.minimize(fun=fun,
                                  jac=jac,
                                  x0=domain_coords_init,
                                  method=method,
                                  bounds=bounds,
                                  options=options,
                                  callback=callback)

    # Get output metric coordinates
    domain_coords = np.array(res.x)
    reduced_metric_coords = opt.compute_domain_coordinate_metric(
        C,
        reduction_maps,
        domain_coords,
        constraint_domain_matrix,
        proj_params
    )

    return reduced_metric_coords, log, res

def incremental_projection(
    C,
    lambdas_init,
    lambdas_target,
    proj_params=opt.ProjectionParameters(),
    opt_params=opt.OptimizationParameters()
):
    """
    Optimize mesh C while linearly interpolating the target constraints in a discrete
    sequence of num_steps iterations.

    param[in] Mesh C: mesh to optimize energy over
    param[in] np.array lambdas_init: initial embedded log edge lengths for optimization
    param[in] np.array lambdas_target: target edge lengths used for energy definition
    param[in] ProjectionParameters proj_params: conformal projection parameters
    param[in] OptimizationParameters opt_params: optimization parameters
    return np.array: final embedded log edge lengths satisfying the angle constraints
    """
    logger = multiprocessing.get_logger()

    # Get the original and target angles
    Th_hat_init = targets.initial_angles(C, lambdas_init)
    Th_hat_target = np.array(C.Th_hat)

    # Iteratively develop the metric coordinates
    lambdas = lambdas_init
    for i, s in enumerate(np.linspace(0, 1, num_steps)):
        # Linearly interpolate constraint manifold from the original to the target
        logger.info("\n\nUsing interpolation values s={} in iteration {}".format(s, i))
        C.Th_hat = s * Th_hat_target + (1 - s) * Th_hat_init

        # Update lambdas and get log
        lambdas = opt.optimize_lambdas(
            C,
            lambdas_target,
            lambdas,
            proj_params,
            opt_params
        )

    return lambdas


# FIXME Need to clean code below

def slim_optimization(
    C,
    lambdas,
    v,
    f,
    Th_hat,
    energy_choice = 2,
    soft_penalty = 0,
    num_iter = 500
):
    """
    Use SLIM method to further optimize the layout generated by (C, lambdas) from initial
    mesh (v,f) with target angles Th_hat
    
    SLIM Energy choices:
    ARAP = 0,
    LOG_ARAP = 1,
    SYMMETRIC_DIRICHLET = 2,
    CONFORMAL = 3,
    EXP_CONFORMAL = 4,
    EXP_SYMMETRIC_DIRICHLET = 5
    
    param[in] Mesh C: Mesh
    param[in] np.array lambdas: log edge lengths for C
    param[in] np.array v: Original vertex positions for the mesh C
    param[in] np.array f: Original triangulation for the mesh C
    param[in] np.array Th_hat: Target angles for the mesh C
    param[in] int energy_choice: SLIM energy to optimize from the above choices
    param[in] float soft_penalty: SLIM soft constraint penalty
    param[in] int num_iter: maximum number of iterations for the SLIM optimization
    return float initial_energy: initial energy before SLIM optimization
    return float final_energy: final energy after SLIM optimization
    return np.array v_cut_o: overlay mesh vertices
    return np.array uv_cut_o: initial overlay parameterization points
    return np.array uv_slim: optimized parameterization points
    return np.array v_cut_o: common overlay mesh triangulation
    """
    # Get layout
    v_cut_o, ft_o, uv_cut_o = build_overlay_layout_FV(C, lambdas, v, f, Th_hat)
    
    # Generate SLIM optimizer using chosen energy
    b = []
    bc = np.zeros((0,2))
    slim_solver = igl.SLIM(v_cut_o,
                           ft_o,
                           uv_cut_o,
                           b,
                           bc,
                           energy_choice,
                           soft_penalty)
                           
    # Get initial SLIM energy
    initial_energy = slim_solver.energy()
    print("Initial energy: {}".format(initial_energy))

    # Optimize to get final energy and parameterization
    slim_solver.solve(num_iter)
    final_energy = slim_solver.energy()
    uv_slim = slim_solver.vertices()
    print("Final energy: {}".format(final_energy))

    return initial_energy, final_energy, v_cut_o, uv_cut_o, uv_slim, ft_o
            

# *********************************************
# Below is deprecated code now rewritten in C++
# *********************************************

class TriangleInequalityError(Exception):
    pass

def write_log(
    C,
    lambdas_target,
    lambdas,
    free_values,
    log,
    opt_params,
    opt_energy_func=None
):
    """
    Print the current status of mesh optimization and record it in log. The recorded information
    is determined by the verbosity as follows:

    verbosity 1:
        Maximum angle constraint value
        Optimization energy
        Maximum stretch (relative to the target lengths)
    verbosity 2:
        L2 energy
        L2 energy for the full doubled mesh
        Maximum metric distortion
        L2 norm of the metric distortion
        Maximum area distortion
        L2 norm of the area distortion
        Maximum best fit conformal scale factor
        L2 norm of the best fit conformal scale factors

    param[in] Mesh C: (possibly symmetric) mesh
    param[in] np.array lambdas_target: target embedded log edge lengths
    param[in] np.array lambdas: current embedded log edge lengths
    param[in] tuple(np.array) free_values: free and fixed edges and vertices for optimization
    param[in, out] dict log: log to write to
    param[in] OptimizationParmaters opt_params: parameters for optimizaiton (including verbosity)
    param[in] opt_energy_func: optimization energy function (only used for verbosity 2)
    """
    logger = multiprocessing.get_logger()

    proj, embed = opt.build_refl_proj(C)
    proj = np.array(proj)
    embed = np.array(embed)
    free_e, fixed_e, free_v, fixed_v = free_values

    # Log and print basic information
    if opt_params.verbosity >= 1:
        F, _, success = opt.F_with_jacobian(
            C,
            lambdas[proj],
            False,
            opt_params.use_edge_lengths
        )
        if not success:
            raise TriangleInequalityError
        F = np.array(F)[free_v]
        max_F = max_error(F)
        opt_energy = opt_energy_func(lambdas_target, lambdas)
        stretches = energies.symmetric_stretches(lambdas_target, lambdas)
        max_stretch = np.max(stretches)
        log['max_F'] = max_F
        log['opt_energy'] = opt_energy
        log['max_stretch'] = max_stretch
        logger.info("Max angle error: {}".format(max_F)) 
        logger.info("Optimization energy: {}".format(opt_energy))
        logger.info("Max stretch: {}\n".format(max_stretch))

    # Log and print more complete information
    if opt_params.verbosity >= 2:
        l2_energy = lp_energy(lambdas_target, lambdas, p=2)
        l2_energy_full = lp_energy(lambdas[proj], lambdas_target[proj], p=2)
        metric_distortion, _ = opt.metric_distortion_energy(C, lambdas_target[proj], lambdas[proj], False)
        area_distortion, _ = opt.area_distortion_energy(C, lambdas_target[proj], lambdas[proj], False)
        u_fit = opt.best_fit_conformal(C, lambdas_target[proj], lambdas[proj])
        log['l2_energy'] = l2_energy
        log['l2_energy_full'] = l2_energy_full
        log['max_metric_distortion'] = np.max(np.abs(metric_distortion))
        log['norm_metric_distortion'] = np.linalg.norm(metric_distortion)
        log['max_area_distortion'] = np.max(np.abs(area_distortion))
        log['norm_area_distortion'] = np.linalg.norm(area_distortion)
        log['max_conf_scale'] = np.max(np.abs(u_fit))
        log['norm_conf_scale'] = np.linalg.norm(u_fit)
        logger.debug("L2 energy: {}".format(l2_energy))
        logger.debug("Full mesh L2 energy: {}".format(l2_energy_full))
        logger.debug("Max metric distortion: {}".format(log['max_metric_distortion']))
        logger.debug("Norm metric distortion: {}".format(log['norm_metric_distortion']))
        logger.debug("Max area distortion: {}".format(log['max_area_distortion']))
        logger.debug("Norm area distortion: {}".format(log['norm_area_distortion']))
        logger.debug("Max conformal scale: {}".format(log['max_conf_scale']))
        logger.debug("Norm conformal scale: {}\n".format(log['norm_conf_scale']))


def get_descent_direction(
    lambdas_target,
    lambdas,
    proj_maps,
    free_values,
    g_opt_energy_func,
    direction_choice="gradient",
    max_grad_range=math.inf,
    g_prev=None,
    d_prev=None
):
    """
    Get descent direction from the target and current lambdas. The direction is chosen according
    to direction_choice.

    param[in] np.array lambdas_target: target embedded log edge lengths 
    param[in] np.array lambdas: embedded log edge lengths 
    param[in] tuple(np.array) proj_maps: tuple of he2e, e2he, proj, embed, and P maps
    param[in] tuple(np.array) free_values: free and fixed edges and vertices for optimization
    param[in] func(np.array, np.array)->np.array: opt energy gradient function
    param[in] string direction_choice: choice between 'gradient', 'conjugate_gradient', and 'unconstrained_minimizer' descent direction
    param[in] bool max_grad_range: half the descent direction until its values fall in this range
    param[in] np.array g_prev: previous gradient for conjugate gradient descent
    param[in] np.array d_prev: previous descent direction for conjugate gradient descent
    return np.array: gradient
    return np.array: unprojected descent direction
    """
    logger = multiprocessing.get_logger()

    # Unpack projection and free values information
    he2e, e2he, proj, embed, P = proj_maps
    free_e, fixed_e, free_v, fixed_v = free_values

    # Get gradient of optimization energy
    g = g_opt_energy_func(lambdas_target, lambdas)
    g = g[free_e]

    # Get gradient direction
    if (direction_choice == 'gradient'):
        d = -g
    # Get conjugate gradient direction
    # NOTE: Requires information g_prev and d_prev from previous iterations
    elif (direction_choice == 'conjugate_gradient'):
        if (g_prev is not None) and (d_prev is not None):
            gamma_PR = np.dot(g, g - g_prev) / np.dot(g_prev, g_prev)
            gamma = max(0, gamma_PR)
            d = -g + gamma * d_prev
        else:
            d = -g
    # Use unconstrained minimizing direction of -(lambdas - lambdas_target)
    elif (direction_choice == 'unconstrained_minimizer'):
        d = -(lambdas - lambdas_target)
    # Use 0 direction as default
    else:
        d = 0

    # Optionally reduce unprojected descent direction d_k to a given range
    while ((np.max(d) - np.min(d)) > max_grad_range):
        d /= 2

    return g, d


def project_descent_direction(
    C,
    lambdas,
    d,
    proj_maps,
    free_values,
    use_edge_lengths=True,
    use_log=True
):
    """
    Get projected descent direction by projecting the gradient or conjugate gradient descent
    to the tangent space to the constraint manifold. Return None if the projection fails.

    param[in] Mesh C: mesh
    param[in] np.array lambdas_target: target embedded log edge lengths 
    param[in] np.array lambdas: embedded log edge lengths 
    param[in] np.array d: unprojected descent direction
    param[in] tuple(np.array) proj_maps: tuple of he2e, e2he, proj, embed, and P maps
    param[in] tuple(np.array) free_values: free and fixed edges and vertices for optimization
    param[in] bool use_edge_lengths: use edge lengths instead of Penner coordinates iff true
    param[in] bool use_log: use log edge lengths/penner coordinates instead of regular lengths/Penner coordinates
    return np.array: projected descent direction
    """
    logger = multiprocessing.get_logger()

    # Unpack projection and free values information
    he2e, e2he, proj, embed, P = proj_maps
    free_e, fixed_e, free_v, fixed_v = free_values

    # Get initial constraints and Jacobian value

    F, J_F, success = opt.F_with_jacobian(
        C,
        lambdas[proj],
        True,
        use_edge_lengths=use_edge_lengths
    )
    if not success:
        raise TriangleInequalityError
        
    # Optional change of coordinates to regular length values
    if not use_log:
        J_l = opt.length_jacobian(lambdas[proj])
        J_F = J_F * J_l

    # Set degrees of freedom
    # FIXME This seems wrong. The constraint should be for the fixed v. The naming
    # is confusing if it refers to the v with variable scale factors
    F = np.array(F)[free_v]
    J_F = (J_F * P)[np.ix_(free_v, free_e)]

    # Get descent direction from constraint function and Jacobian
    if len(free_v) == 0:
        delta_lambdas = d
    else:
        delta_lambdas = opt.project_descent_direction(d, F, J_F)
    
    return np.array(delta_lambdas)

# TODO Make this a C++ function
def convert_penner_coordinates_to_log_edge_lengths(
    C,
    lambdas_proj,
    proj_maps
):
    he2e, e2he, proj, embed, P = proj_maps
    C_del, lambdas_full_del, _, flip_seq = opt.make_delaunay_with_jacobian(C, lambdas_proj[proj], False)
    flip_seq_eucl = (-np.array(flip_seq) - 1)
    mo = opt.add_overlay(C_del, np.array(lambdas_full_del)[embed])
    opt.flip_edges_overlay(mo, flip_seq_eucl[::-1])
    log_edge_lengths_he = 2.0 * np.log(mo._m.l)
    return log_edge_lengths_he[e2he[embed]]

def line_search(
    C,
    lambdas_target,
    lambdas,
    delta_lambdas,
    proj_maps,
    free_values,
    proj_params,
    opt_params,
    opt_energy_func,
    beta_0=1,
    use_log=True
):
    """
    Perform projected line search in the direction delta_lambdas.

    param[in] Mesh C: mesh
    param[in] np.array lambdas_target: target embedded log edge lengths 
    param[in] np.array lambdas: embedded log edge lengths 
    param[in] np.array delta_lambdas: descent direction for embedded log edge lengths 
    param[in] tuple(np.array) proj_maps: tuple of he2e, e2he, proj, embed, and P maps
    param[in] tuple(np.array) free_values: free and fixed edges and vertices for optimization
    param[in] ProjectionParameters proj_params: conformal projection parameters
    param[in] OptimizationParameters opt_params: optimization parameters
    param[in] func(np.array, np.array)->float opt_energy_func: optimization energy function
    param[in] float beta_0: initial line search step size
    param[in] bool use_log: use log edge lengths/penner coordinates instead of regular lengths/Penner coordinates
    return np.array: embedded log edge lengths after line step
    return dict: line serach log
    return float: step size used
    """
    logger = multiprocessing.get_logger()

    # Unpack projection and free values information
    he2e, e2he, proj, embed, P = proj_maps
    free_e, fixed_e, free_v, fixed_v = free_values

    # Get initial energy and angle constraint before line search
    opt_energy_init = opt_energy_func(lambdas_target, lambdas)
    F_init, _, success = opt.F_with_jacobian(
        C,
        lambdas[proj],
        False,
        opt_params.use_edge_lengths
    )
    if not success:
        raise TriangleInequalityError
    F_init = np.array(F_init)[free_v]
    max_F_init = max_error(F_init)

    # Perform line search
    ls_log = []
    beta = beta_0
    while True:
        # Stop if beta too small
        if (beta < 1e-16):
            logger.info("Beta {} too small to continue".format(beta))
            return lambdas, ls_log, beta

        # Record beta
        ls_log.append({})
        ls_log[-1]['beta'] = beta
        ls_log[-1]['post_step_status'] = {}
        ls_log[-1]['post_proj_status'] = {}
        logger.info("Line step with beta {}".format(beta))

        # Step 1: Line Step
        # Make tentative line step 
        lambdas_step = lambdas.copy()
        if use_log:
            lambdas_step[free_e] = lambdas[free_e] + beta * delta_lambdas
        else:
            lambdas_step[free_e] = 2 * np.log(np.exp(lambdas[free_e] / 2) + beta * delta_lambdas)
        lambdas_step[fixed_e] = lambdas_target[fixed_e]

        # Get energy and distortion measures after the step
        F_step, _, success = opt.F_with_jacobian(
            C,
            lambdas_step[proj],
            False,
            opt_params.use_edge_lengths
        )
        if not success:
            logger.debug("Reducing step size due to triangle inequality violation")
            beta = beta / 2
            continue
        F_step = np.array(F_step)[free_v]
        max_F_step = max_error(F_step)
        logger.debug("Status after line step")
        write_log(C,
                  lambdas_target,
                  lambdas_step,
                  free_values,
                  ls_log[-1]['post_step_status'],
                  opt_params,
                  opt_energy_func)

        # Continue line step if the resulting angle errors are too large
        if (max_F_step) > opt_params.max_angle:
            logger.info("Max angle error {} larger than bound {}".format(max_F_step, opt_params.max_angle))
            beta = beta / 2
            continue

        # Step 2: Projection
        # Project to constraint
        logger.info("Projecting to constraint")
        start_call = time.time()
        lambdas_proj, u = opt.project_to_constraint(C,
                                                    lambdas_step,
                                                    proj_params)
        lambdas_proj = np.array(lambdas_proj)

        logger.debug('Projecting to constraint took {} s'.format(time.time() - start_call))
        logger.debug('Range of conformal scale factors: [{}, {}]'.format(np.min(u), np.max(u)))

        # Get energy and distortion measures after the projection
        F_proj, _, success = opt.F_with_jacobian(
            C,
            lambdas_proj[proj],
            False,
            opt_params.use_edge_lengths
        )
        if not success:
            logger.debug("Reducing step size due to triangle inequality violation")
            beta = beta / 2
            continue
        F_proj = np.array(F_proj)[free_v]
        max_F_proj = max_error(F_proj)
        logger.info("Status after projection")
        write_log(C,
                  lambdas_target,
                  lambdas_proj,
                  free_values,
                  ls_log[-1]['post_proj_status'],
                  opt_params,
                  opt_energy_func)

        # Step 3: Check termination conditions
        # Optionally continue line search if energy did not decrease
        opt_energy_proj = opt_energy_func(lambdas_target, lambdas_proj)
        if (     (opt_params.require_energy_decr)
             and (opt_energy_func(lambdas_target, lambdas_proj) > opt_energy_init) ):
            beta = beta / 2
            continue
        
        # Optionally continue line search if angle error did not decrease enough
        if (max_F_proj >= max_F_init + opt_params.max_angle_incr):
            beta = beta / 2
            continue

        # Terminate line search
        return lambdas_proj, ls_log, min(opt_params.max_beta, 2*beta)


def optimize_lambdas(
    C,
    lambdas_init,
    lambdas_target,
    checkpoint_dir="",
    proj_params=opt.ProjectionParameters(),
    opt_params=opt.OptimizationParameters()
):
    """
    Use projected descent method to minimize an energy over the mesh using Penner coordinates
    while maintaining the angle constraints defined by C.Th_hat. 

    param[in] Mesh C: mesh to optimize energy over
    param[in] np.array lambdas_init: initial embedded log edge lengths for optimizaiton
    param[in] np.array lambdas_target: target edge lengths used for energy definition
    param[in] string checkpoint_dir: directory to store intermediate log edge length values
    param[in] ProjectionParameters proj_params: conformal projection parameters
    param[in] OptimizationParameters opt_params: optimization parameters
    return dict: debugging log of important values computed during the method
    return np.array: final embedded log edge lengths satisfying the angle constraints
    """
    log = []
    start = time.time()
    logger = multiprocessing.get_logger()

    # Build edge and projection maps to go between different representations for the log
    # edge lengths (i.e. embedded mesh edge lengths, full mesh edge lenghts, and full 
    # mesh halfedge lengths)
    he2e, e2he = opt.build_edge_maps(C)
    proj, embed = opt.build_refl_proj(C)
    he2e = np.array(he2e)
    e2he = np.array(e2he)
    proj = np.array(proj)
    embed = np.array(embed)
    P = opt.refl_matrix(C)
    proj_maps = (he2e, e2he, proj, embed, P)

    # Get boundary and interior vertices and edges
    fixed_v = np.where(C.fixed_dof)[0]
    free_v = np.where(np.logical_not(C.fixed_dof))[0]
    bd_e, int_e = energies.find_bd_and_int_edges(C, proj_maps)
    if (opt_params.fix_bd_lengths):
        fixed_e = bd_e
        free_e = int_e
    else:
        fixed_e = []
        free_e = np.arange(len(embed))
    free_values = (free_e, fixed_e, free_v, fixed_v)

    # Print warning if no constraint
    if len(free_v) == 0:
       logging.warning("Warning: no constraints.") 

    # Get energy functional
    opt_energy_funcs = energies.generate_opt_energy_funcs(
        C,
        energy_choice=opt_params.energy_choice,
        p=opt_params.p,
        bd_weight=opt_params.bd_weight,
        reg_factor=opt_params.reg_factor,
        use_log=opt_params.use_log
    )
    opt_energy_func, g_opt_energy_func = opt_energy_funcs

    # logging.info initial convergence information
    logger.info("Starting optimization of {} energy".format(opt_params.energy_choice))
    try:
        write_log(C, lambdas_target, lambdas_init, free_values, {}, opt_params, opt_energy_func)
    except TriangleInequalityError:
        logger.error("Triangle inequality error in initial mesh")
        return log, lambdas

    # Project to the constraint for the initial lengths
    logger.info("Performing initial conformal projection")
    lambdas, u = opt.project_to_constraint(C,
                                           lambdas_init,
                                           proj_params)
    lambdas = np.array(lambdas)
    logger.info("\nFirst projection complete")
    logger.info('Range of conformal scale factors: [{}, {}]'.format(np.min(u), np.max(u)))
    try:
        write_log(C, lambdas_target, lambdas, free_values, {}, opt_params, opt_energy_func)
    except TriangleInequalityError:
        logger.error("Triangle inequality error during initial conformal projection")
        return log, lambdas

    # Optionally save the initial lambdas to file
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        lambdas_init_file = os.path.join(checkpoint_dir, 'lambdas_init')
        np.savetxt(lambdas_init_file, lambdas_init)
        lambdas_conf_file = os.path.join(checkpoint_dir, 'lambdas_conf')
        np.savetxt(lambdas_conf_file, lambdas)

    # Perform optimization
    beta_k = opt_params.beta_0
    g_k = None
    d_k = None
    for k in np.arange(opt_params.num_iter):
        log_k = {}

        # Print start of iteration information
        logger.info("Iteration {}".format(k))
        logger.info("Optimization energy: {}\n".format(opt_energy_func(lambdas_target, lambdas)))

        # Get line search direction
        logger.debug("Getting line search direction")
        start_call = time.time()
        g_k, d_k = get_descent_direction(
            lambdas_target,
            lambdas,
            proj_maps,
            free_values,
            g_opt_energy_func,
            direction_choice=opt_params.direction_choice,
            max_grad_range=opt_params.max_grad_range,
            g_prev=g_k,
            d_prev=d_k
        )
        try:
            delta_lambdas_k = project_descent_direction(
                C,
                lambdas,
                d_k,
                proj_maps,
                free_values,
                opt_params.use_edge_lengths,
                use_log=opt_params.use_log
            )
        except TriangleInequalityError:
            logger.error("Triangle inequality error during descent direction projection")
            return log, lambdas
        delta_lambdas_norm_k = np.linalg.norm(delta_lambdas_k)
        log_k['delta_lambdas_norm'] = delta_lambdas_norm_k
        logger.debug('Finding direction took {} s'.format(time.time() - start_call))
        logger.debug('Delta lambdas norm {} in iteration {}\n'.format(delta_lambdas_norm_k, k))
        if math.isnan(delta_lambdas_norm_k):
            logger.error("NaN in line step")
            return log, lambdas

        # Perform line search
        try:
            lambdas_k, ls_log_k, beta_k = line_search(
                C,
                lambdas_target,
                lambdas,
                delta_lambdas_k,
                proj_maps,
                free_values,
                proj_params,
                opt_params,
                opt_energy_func,
                beta_0=beta_k,
                use_log=opt_params.use_log
            )
        except TriangleInequalityError:
            logger.error("Triangle inequality error during line search")
            return log, lambdas
        log_k['line_search'] = ls_log_k

        # Update log and print iteration output
        try:
            logger.info("Status at end of iteration {}".format(k))
            write_log(C, lambdas_target, lambdas_k, free_values, log_k, opt_params, opt_energy_func)
        except TriangleInequalityError:
            logger.error("Triangle inequality error after line search")
            return log, lambdas

        # Update log with additional stretch information
        max_stretch_init_proj_k = np.max(energies.symmetric_stretches(lambdas_k, lambdas_init))
        max_stretch_prev_proj_k = np.max(energies.symmetric_stretches(lambdas_k, lambdas))
        log_k['max_stretch_init_proj'] = max_stretch_init_proj_k
        log_k['max_stretch_prev_proj'] = max_stretch_prev_proj_k
        logger.debug("Max stretch relative to initial: {}".format(max_stretch_init_proj_k))
        logger.debug("Max stretch relative to previous: {}\n".format(max_stretch_prev_proj_k))

        # Update lambdas and g_{k-1}
        log.append(log_k)
        lambdas = lambdas_k
        g_km1 = g_k
        d_km1 = d_k

        # Optionally save the lambdas to file
        if checkpoint_dir:
            logger.info("Saving lambdas to file")
            os.makedirs(checkpoint_dir, exist_ok=True)
            lambdas_chkpt_file = os.path.join(checkpoint_dir, 'lambdas_chkpt')
            np.savetxt(lambdas_chkpt_file, lambdas)

        # Get ratio of the projected descent direction to the unprojected direction
        if (np.linalg.norm(d_k) > 0):
            conv_ratio = np.linalg.norm(delta_lambdas_k)/ np.linalg.norm(d_k)
        else:
            conv_ratio = np.linalg.norm(delta_lambdas_k)
        logger.info("Convergence ratio at end of iteration {}: {}\n\n".format(k, conv_ratio))

        # Check convergence conditions
        if (conv_ratio < opt_params.min_ratio):
            break
        if (beta_k < 1e-16):
            break

    # Final projection
    lambdas, u = opt.project_to_constraint(C,
                                           lambdas,
                                           proj_params)
    lambdas = np.array(lambdas)

    # Optionally save the lambdas to file
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        lambdas_file = os.path.join(checkpoint_dir, 'lambdas_opt')
        np.savetxt(lambdas_file, lambdas)

    return log, lambdas

def optimize_lambdas_vf(
    V,
    F,
    Th_hat,
    checkpoint_dir="",
    proj_params=opt.ProjectionParameters(),
    opt_params=opt.OptimizationParameters()
):
    """
    Generate connectivity with Penner coordinates for the VF mesh and optimize with Th_hat
    angle constraints.

    param[in] np.array V: input vertex positions
    param[in] np.array F: input connectivity
    param[in] np.array Th_hat: target angle constraints
    param[in] string checkpoint_dir: directory to store intermediate log edge length values
    param[in] ProjectionParameters proj_params: conformal projection parameters
    param[in] OptimizationParameters opt_params: optimization parameters
    return dict: debugging log of important values computed during the method
    return np.array: final embedded log edge lengths satisfying the angle constraints
    """
    C, vtx_reindex = opt.fv_to_double(V, F, V, F, Th_hat, [], False)
    lambdas_target, _ = targets.get_euclidean_target_lambdas(V, F, Th_hat)
    lambdas_init = lambdas_target.copy()
    log, lambdas = optimize_lambdas(
        C,
        lambdas_init,
        lambdas_target,
        checkpoint_dir=checkpoint_dir,
        proj_params=proj_params,
        opt_params=opt_params
    )

    return C, vtx_reindex, log, lambdas

# Deprecated
def line_search_min(
    C,
    lambdas_init,
    d,
    free_v,
    int_min=1.0,
    int_max=1e3,
    num_steps=10,
    use_max=False
):
    if use_max:
        def ls_func(x):
            return np.max(np.abs(x))
    else:
        def ls_func(x):
            return np.linalg.norm(x)

    a = int_min
    b = a
    c = a
    F_a = opt.F_with_jacobian(C, lambdas_init + a * d, False)[0]
    F_a = ls_func(np.array(F_a)[free_v])

    # Find b as large as possible
    while c < int_max:
        b = a + c
        lambdas = lambdas_init + b * d
        F_b = opt.F_with_jacobian(C, lambdas, False)[0]
        F_b = ls_func(np.array(F_b)[free_v])

        if (F_b <= F_a):
            c *= 2
        else:
            break

    for _ in np.arange(num_steps):
        c = (a + b) / 2.0
        lambdas = lambdas_init + c * d
        F_c = opt.F_with_jacobian(C, lambdas, False)[0]
        F_c = ls_func(np.array(F_c)[free_v])

        # Update a to c if this decreases the constraint
        if (F_c <= F_a):
            a = c
            F_a = F_c
        # Reduce b to c otherwise to shrink the search space
        else:
            b = c
            F_b = F_c

    return a


def scipy_minimize(
    C,
    lambdas_init,
    lambdas_target,
    method='SLSQP'
):
    # Use the projection of the initial lambdas as the initial guess
    # FIXME Make optional
    proj_params_struct = ProjectionParameters()
    lambdas_proj, u = project_to_constraint(C,
                                            lambdas_init,
                                            C.Th_hat,
                                            proj_params_struct)
    lambdas_proj = np.array(lambdas_proj)

    # Define objective function and Jacobian for target lambdas
    def objective_fun(lambdas):
        return 0.5 * (np.linalg.norm(lambdas - lambdas_target)**2)

    def objective_jac(lambdas):
        return lambdas - lambdas_target

    # Define constraint function and Jacobian for the mesh C
    def constraint_fun(lambdas):
        proj, embed = build_refl_proj(C)
        lambdas_full = lambdas[proj]
        F, _ = F_with_jacobian(C, lambdas_full, C.Th_hat, False)
        return 1 - np.array(F)

    def constraint_jac(lambdas):
        proj, embed = build_refl_proj(C)
        lambdas_full = lambdas[proj]
        _, J_F = F_with_jacobian(C, lambdas_full, C.Th_hat, True)
        P = refl_matrix(C)
        J_F = J_F * P
        return J_F.toarray()

    print(objective_fun(lambdas_proj))
    print(objective_jac(lambdas_proj).shape)
    print(constraint_fun(lambdas_proj).shape)
    print(constraint_jac(lambdas_proj).shape)

    # Set bounds for the optimization to keep stable
    # FIXME Set to something more reasonable
    bounds = scipy.optimize.Bounds(lambdas_proj-2, lambdas_proj+2)

    if method == 'trust-constr':
        constraints = scipy.optimize.NonlinearConstraint(fun=constraint_fun,
                                                         lb=0,
                                                         ub=0,
                                                         jac=constraint_jac)
    elif (method == 'COBYLA') or (method == 'SLSQP'):
        constraints = [ { 'type': 'ineq',
                          'fun':  constraint_fun,
                          'jac':  constraint_jac }, ]
    else:
        constraints = None

    # Set solver options
    options = { 'maxiter': 5,
                'disp': True }

    res = scipy.optimize.minimize(fun=objective_fun,
                                  x0=lambdas_proj,
                                  jac=objective_jac,
                                  method=method,
                                  bounds=bounds,
                                  constraints=constraints,
                                  options=options)

    return res

    