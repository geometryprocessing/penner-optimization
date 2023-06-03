import numpy as np
import optimization_py as opt
import random

def energy_finite_difference(
    energy_func,
    lambdas_target,
    lambdas,
    direction,
    h=0.01
):
    # Compute the central difference quotient approximation of J_F*v
    lambdas_plus = lambdas + (h * direction)
    lambdas_minus = lambdas - (h * direction) 
    energy_plus = energy_func(lambdas_target, lambdas_plus)
    energy_minus = energy_func(lambdas_target, lambdas_minus)
    energy_derivative = (energy_plus - energy_minus) / (2.0 * h)

    return energy_derivative

def validate_energy_gradient(
    C,
    lambdas_target,
    lambdas,
    energy_func,
    g_energy_func,
    n=3,
    h=0.01
):
    # Compute the initial value of the energy and the exact value of its gradient
    energy = energy_func(lambdas_target, lambdas)
    gradient = g_energy_func(lambdas_target, lambdas)

    errors = np.zeros(n)
    for index in np.arange(n):
        # Construct the ith elementary basis vector for random i
        i = random.randrange(len(gradient))
        e_i = np.zeros_like(lambdas)
        e_i[i] = 1.0

        # Compute the ith derivative of the energy
        g_i_approx = energy_finite_difference(
            energy_func,
            lambdas_target,
            lambdas,
            e_i,
            h
        )

        # Record the absolute value of the difference
        errors[index] = abs(gradient[i] - g_i_approx)
    
    return errors


def fin_diff_J_del(C, lambdas, Th_hat, v, h=0.01, float_type=float):
    # Compute the central difference quotient approximation of J_del*v
    he2e, e2he = build_edge_maps(C)
    e_lambdas = lambdas[e2he]
    e_lambdas_p = e_lambdas + (h * v)
    e_lambdas_m = e_lambdas - (h * v) 
    lambdas_p = e_lambdas_p[he2e]
    lambdas_m = e_lambdas_m[he2e]
    _, lambdas_del_p, _ = make_delaunay_with_jacobian(C, lambdas_p, float_type)
    _, lambdas_del_m, _ = make_delaunay_with_jacobian(C, lambdas_m, float_type)
    e_lambdas_del_p = lambdas_del_p[e2he]
    e_lambdas_del_m = lambdas_del_m[e2he]
    J_del_i_approx = (e_lambdas_del_p - e_lambdas_del_m) / (2*h)

    return J_del_i_approx


def fin_diff_J_F(C, lambdas, Th_hat, v, h=0.01, float_type=float):
    # Compute the central difference quotient approximation of J_F*v
    he2e, e2he = build_edge_maps(C)
    e_lambdas = lambdas[e2he]
    e_lambdas_p = e_lambdas + (h * v)
    e_lambdas_m = e_lambdas - (h * v) 
    lambdas_p = e_lambdas_p[he2e]
    lambdas_m = e_lambdas_m[he2e]
    F_p, _ = F_with_jacobian(C, lambdas_p, Th_hat, float_type, need_jacobian=False)
    F_m, _ = F_with_jacobian(C, lambdas_m, Th_hat, float_type, need_jacobian=False)
    # FIXME Normalize?
    #J_F_i_approx = (F_p - F_m) / (2*h*msqrt(np.dot(v,v)))
    J_F_i_approx = (F_p - F_m) / (2*h)

    return J_F_i_approx

def fin_diff_J_f(C, lambdas_target_full, lambdas_full, jacobian_func, v, h=0.01):
    # Compute the central difference quotient approximation of J_f*v
    lambdas_p = lambdas_full + (h * v)
    lambdas_m = lambdas_full - (h * v) 
    f_p, _ = jacobian_func(C, lambdas_target_full, lambdas_p, False)
    f_m, _ = jacobian_func(C, lambdas_target_full, lambdas_m, False)
    J_f_i_approx = (f_p - f_m) / (2*h)

    return J_f_i_approx
    
def fin_diff_J_f_sep(C, lambdas_target_full, lambdas_full, func, v, h=0.01):
    # Compute the central difference quotient approximation of J_f*v
    lambdas_p = lambdas_full + (h * v)
    lambdas_m = lambdas_full - (h * v) 
    f_p = func(C, lambdas_target_full, lambdas_p)
    f_m = func(C, lambdas_target_full, lambdas_m)
    J_f_i_approx = (f_p - f_m) / (2*h)

    return J_f_i_approx

def validate_J_del(C, lambdas, Th_hat, n=3, h=0.01, float_type=float):
    """
    Compare the exact solution of the Jacobian of del to an approximation computed
    with finite difference quotients.

    TODO
    """
    # Compute the initial value of lambdas_del and the exact value of J_del
    C_del, lambdas_del, J_del = make_delaunay_with_jacobian(C, lambdas, float_type)

    errors = np.zeros(n)
    for i in np.arange(n):
        # Construct the ith elementary basis vector
        e_i = np.array([float_type(0)]*int(len(lambdas)/2), dtype=float_type)
        e_i[i] = 1

        # Compute J_del*e_i numerically
        J_del_i_approx = fin_diff_J_del(C, lambdas, Th_hat, e_i, h, float_type)

        # Compute J_del*e_i exactly
        J_del_i_exact = (J_del * (e_i.T)).T

        # Record the squared norm of the difference of the two computations
        J_del_i_diff = J_del_i_exact - J_del_i_approx
        errors[i] = np.dot(J_del_i_diff, J_del_i_diff)

    return errors



def validate_J_F(C, lambdas, Th_hat, n=3, h=0.01, float_type=float):
    """
    Compare the exact solution of the Jacobian of F to an approximation computed
    with finite difference quotients.

    TODO
    """
    # Compute the initial value of F and the exact value of J_F
    F_0, J_F = F_with_jacobian(C, lambdas, Th_hat, float_type)

    errors = np.zeros(n)
    for i in np.arange(n):
        # Construct the ith elementary basis vector
        e_i = np.array([float_type(0)]*int(len(lambdas)/2), dtype=float_type)
        e_i[i] = 1

        # Compute J_F*e_i numerically
        J_F_i_approx = fin_diff_J_F(C, lambdas, Th_hat, e_i, h, float_type)

        # Compute J_F*e_i exactly
        J_F_i_exact = (J_F * (e_i.T)).T

        # Record the squared norm of the difference of the two computations
        J_F_i_diff = J_F_i_exact - J_F_i_approx
        errors[i] = np.dot(J_F_i_diff, J_F_i_diff)

    return errors

def validate_f(C,
               lambdas_target_full,
               lambdas_full,
               func,
               n=3,
               h=0.01):
    # Compute the initial value of f and the exact value of J_f
    f, J_f_exact = func(C, lambdas_target_full, lambdas_full, True)

    errors = np.zeros(n)
    for i in np.arange(n):
        # Construct the ith elementary basis vector
        e_i = np.zeros_like(lambdas_full, dtype=float)
        e_i[i] = 1

        # Compute J_f*e_i exactly
        J_f_i_exact = J_f_exact * e_i

        # Compute J_f*e_i numerically
        J_f_i_approx = fin_diff_J_f(C,
                                    lambdas_target_full,
                                    lambdas_full,
                                    func,
                                    e_i,
                                    h)

        # Record the squared norm of the difference of the two computations
        J_f_i_diff = J_f_i_exact - J_f_i_approx
        errors[i] = np.dot(J_f_i_diff, J_f_i_diff)
    
    return errors

def validate_f_sep(C,
                   lambdas_target_full,
                   lambdas_full,
                   func,
                   jacobian_func,
                   n=3,
                   h=0.01):
    # Compute the initial value of f and the exact value of J_f
    f = func(C, lambdas_target_full, lambdas_full)
    J_f_exact = jacobian_func(C, lambdas_target_full, lambdas_full)

    errors = np.zeros(n)
    for i in np.arange(n):
        # Construct the ith elementary basis vector
        e_i = np.zeros_like(lambdas_full, dtype=float)
        e_i[i] = 1

        # Compute J_f*e_i exactly
        J_f_i_exact = np.dot(J_f_exact, e_i)

        # Compute J_f*e_i numerically
        J_f_i_approx = fin_diff_J_f_sep(C,
                                        lambdas_target_full,
                                        lambdas_full,
                                        func,
                                        e_i,
                                        h)

        # Record the squared norm of the difference of the two computations
        J_f_i_diff = J_f_i_exact - J_f_i_approx
        errors[i] = np.dot(J_f_i_diff, J_f_i_diff)
    
    return errors

def validate_metric_distortion_energy(C,
                                      lambdas_target_full,
                                      lambdas_full,
                                      n=3,
                                      h=0.01):
    # Compute the initial value of f2energy and the exact value of J_f2energy
    f2energy, J_f2energy_exact = metric_distortion_energy(C, lambdas_target_full, lambdas_full, True)

    errors = np.zeros(n)
    for i in np.arange(n):
        # Construct the ith elementary basis vector
        e_i = np.array([float_type(0)]*int(len(lambdas)/2), dtype=float_type)
        e_i[i] = 1

        # Compute J_f2energy*e_i numericall
        f2energy_p = metric_distortion_energy(C, lambdas_target_full, lambdas_full, False)
        J_f2energy_i_diff = fin_diff_J_f(C,
                                         lambdas_target_full,
                                         lambdas_full,
                                         metric_distortion_energy,
                                         e_i,
                                         h)

        # Record the squared norm of the difference of the two computations
        J_f2energy_i_diff = J_f2energy_i_exact - J_F_i_approx
        errors[i] = np.dot(J_F_i_diff, J_F_i_diff)
    
    return errors
