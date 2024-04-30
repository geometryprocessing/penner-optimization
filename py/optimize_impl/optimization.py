
import numpy as np
import scipy as sp
import optimization_py as opt

def optimize_shear_basis_coordinates(
    C,
    opt_energy,
    shear_basis_matrix,
    proj_params,
    method = 'L-BFGS-B',
    num_iter=200
):
    """
    Optimize a metric satisfying constraints in terms of a basis of the shear
    space orthogonal to the space of conformal scalings.

    param[in] C: mesh 
    param[in] opt_energy: energy to optimize
    param[in] shear_basis_matrix: matrix with shear coordinate basis vectors as columns
    param[in] proj_params: parameters fro the projection to the constraint manifold
    param[in] method: optimization method to use
    param[in] num_iter: number of iterations to perform
    return optimized metric coordinates
    """
    reduction_maps = opt.ReductionMaps(C)

    # Compute optimization domain
    constraint_domain_matrix, constraint_codomain_matrix, init_domain_coords, init_codomain_coords = opt.compute_optimization_domain(
        C,
        shear_basis_matrix)

    def fun(domain_coords):
        return opt.compute_domain_coordinate_energy(
            C,
            opt_energy,
            constraint_domain_matrix,
            constraint_codomain_matrix,
            domain_coords,
            init_codomain_coords,
            proj_params
        )

    def jac(domain_coords):
        energy, gradient = opt.compute_domain_coordinate_energy_with_gradient(
            C,
            opt_energy,
            constraint_domain_matrix,
            constraint_codomain_matrix,
            domain_coords,
            init_codomain_coords,
            proj_params
        )
        return gradient
    
    log = []
    def callback(domain_coords):
        energy = fun(domain_coords)
        print("Energy:", energy)
        log.append(energy)
    
    # Set bounds for the optimization to keep stable
    bounds = sp.optimize.Bounds(init_domain_coords-10, init_domain_coords+10)

    # Set solver options
    options = { 'maxiter': num_iter,
                'disp': True }

    # Minimize energy
    res = sp.optimize.minimize(fun=fun,
                                  jac=jac,
                                  x0=init_domain_coords,
                                  method=method,
                                  bounds=bounds,
                                  options=options,
                                  callback=callback)

    # Get output metric coordinates
    domain_coords = np.array(res.x)
    reduced_metric_coords = opt.compute_domain_coordinate_metric(
        C,
        constraint_domain_matrix,
        constraint_codomain_matrix,
        domain_coords,
        init_codomain_coords,
        proj_params
    )

    return reduced_metric_coords, log, res