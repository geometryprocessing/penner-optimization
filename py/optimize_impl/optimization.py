
import numpy as np
import scipy as sp
import optimization_py as opt

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