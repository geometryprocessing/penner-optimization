import math
import copy
import os
import pickle
from importlib import reload
from collections import namedtuple
from multipledispatch import dispatch
from dataclasses import dataclass


#from math import exp,atan2,acos,sqrt
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpmath import mp,iv
import mpmath

from conformal_impl.halfedge import *
from conformal_impl.symmesh import *
from conformal_impl.overload_math import *
from conformal_impl.delaunay import *
from conformal_impl.ptolemy import *
from conformal_impl.valenceflips import *
from conformal_impl.geomops import *

# dataclass for a symmetic mesh (C,R) with an initial metric l, 
# conformal scale factors phi and target angles Theta_hat
@dataclass
class ConformalMesh:
    C:   Connectivity
    R:   Reflection 
    l:   np.ndarray
    phi: np.ndarray
    Th_hat: np.ndarray

def conf_energy_face(C, phi, l_init, Th_hat, angles, f):
    h0 = C.f2he[f]; h1 = C.next_he[h0]; h2 = C.next_he[h1]
    v0 = C.to[h1];   v1 = C.to[h2];     v2 = C.to[h0]
    phi_t = phi[[v0,v1,v2]]
    lambd = [2*mlog(l_init[h0]),2*mlog(l_init[h1]), 2*mlog(l_init[h2])]
    lambd_t = lambd 
    lambd_t[0] += phi_t[1]+phi_t[2]
    lambd_t[1] += phi_t[0]+phi_t[2]
    lambd_t[2] += phi_t[0]+phi_t[1]
    alpha = angles[[h0,h1,h2]]

    return 0.5*(alpha[0]*lambd_t[0] + alpha[1]*lambd_t[1] + alpha[2]*lambd_t[2] +
        mclsin(2,2*alpha[0]) + mclsin(2,2*alpha[1]) + mclsin(2,2*alpha[2])) \
        - 0.5*mpmath.pi*(phi_t[0]+phi_t[1]+phi_t[2])

def conf_energy(C,phi, l_init, angles, Th_hat):
    return np.sum([conf_energy_face(C, phi, l_init, Th_hat,angles, f) \
        for f in range(0,len(C.f2he)) if f not in C.bnd_loops]) + \
    0.5*np.dot(Th_hat,phi)


def f_angles_from_len(l):   # 3 angles from 3 edge lengths, returns tuple of angles opposite corresp. edge lengths
    return (ang_from_len(l[1],l[2],l[0]),ang_from_len(l[2],l[0],l[1]),ang_from_len(l[0],l[1],l[2]))

def f_cot_from_len(l):   # 3 angles from 3 edge lengths, returns tuple of angles opposite corresp. edge lengths
    return (cot_from_len(l[1],l[2],l[0]),cot_from_len(l[2],l[0],l[1]),cot_from_len(l[0],l[1],l[2]))


# angle opposite l3 from 3 edge lengths, using atan2
def vec_ang_from_len_atan(l):
    lp1 = l[:,[1,2,0]]
    lp2 = l[:,[2,0,1]]
    s = 0.5*np.sum(l,axis=1)
    s = np.tile(s,(3,1)).T
    return  np.arctan2(  4.0*np.sqrt(s*(s-lp1)*(s-lp2)*(s-l)), (lp1**2 + lp2**2 - l**2))

# angle opposite l3 from 3 edge lengths, using atan2, alternative
def vec_ang_from_len_atan_half(l):  
    lp1 = l[:,[1,2,0]]
    lp2 = l[:,[2,0,1]]
    t31 = lp1+lp2-l; t23 = lp1-lp2+l; t12 = -lp1+lp2+l; 
    l123 = np.sum(l,axis=1);
    l123 = np.tile(l123,(3,1)).T
    denom = np.sqrt(t12*t23*t31*l123);
    return 2*np.arctan2(t12*t23,denom);

vec_angles_from_len = vec_ang_from_len_atan_half

def vec_area_from_len(l):  # area from 3 edge lengths
    s = 0.5*np.sum(l,axis=1)
    return np.sqrt(s*(s-l[:,0])*(s-l[:,1])*(s-l[:,2]))

def vec_cot_from_len(l):  # cot of the angle opposite l3 from 3 edge length
    lp1 = l[:,[1,2,0]]
    lp2 = l[:,[2,0,1]]
    area = np.tile(vec_area_from_len(l),(3,1)).T
    return (lp1**2 + lp2**2 - l**2)/area/4.0


# TODO: there is a conflicting function in delaunay, per triangle
def check_tri_ineq(l1,l2,l3):
    return ( (np.array(l1+l2-l3) >=0)  &  (np.array(l1-l2+l3) >=0) &  (np.array(-l1+l2+l3) >=0)).all()


# first vertex left invariant by reflection
# we need to eliminate one dof in the Hessian to make system positive definite
def get_fixed_dof(R):
    return np.nonzero(R.refl==np.arange(0,len(R.refl)))[0][0]

# get subarray of phi for vertices with type D1 and S. Since symmetry implies doubled vertices must
# always have the same angle and thus phi values, all phi values can be inferred from these.
# Note that this is in some sense equivalent to working with a tufted double cover.
def full_phi_to_reduced(C,R,phi):
    indep_vtx = np.nonzero( np.isin(R.vtype,[LB.D1,LB.S]))[0]
    return phi[indep_vtx]

# inverse of full_phi_to_reduced
def reduced_phi_to_full(C,R,phi_r,float_type=float):
    indep_vtx = np.nonzero( np.isin(R.vtype, [LB.D1,LB.S]))[0]
    dep_vtx =   np.nonzero( R.vtype == LB.D2)[0]
    phi = np.array([float_type(0)]*len(C.out))
    phi[indep_vtx] = phi_r
    phi[dep_vtx] = phi_r[R.refl[dep_vtx]] 
    return phi


# computes gradient from phi, Th_hat, and mesh structure 
# C,R - symmetric mesh structure
# phi_r, l_init, and Th_hat are assumed to be in the right format corresponding to float_type 
#
# phi:  scale factors per vertex
# l_init: initial metric edge lengths, one per halfedge matching halfedges have the same value
# Th_hat: target total angles per vertex
# phi_r:  log scale factors per vertex,  assumed to be in a reduced form, i.e. only for vertices of type D1 and S 
# note this is (inconsistently ) not assumed for Th_hat
# float_type: type of floats to use, float64 or multiprecision math.mp.mpf both work
# however interval should also work 
# TODO: test interval -- mostly to detect when calculations of gradient are inaccurate
# returns tuple: 
#   (success (True or False), 
#   if success, the following values, if not replaced by None
#     grad  
#     H (if needHess, None otherwise)
#     angle array 
#    )
#  returned variables are in context format
def grad_and_Hess(C, R, l_init, Th_hat, phi_r, needHess=True, float_type=float):
    # expand phi to all mesh vertices; no real need but easier 
    # to compute the gradient and Hessian this way
    if R == None:
        phi = np.array([float_type(0)]*len(C.out))
        phi[:] = phi_r
    else:
        indep_vtx = np.nonzero( np.isin(R.vtype, [LB.D1,LB.S]))[0]
        dep_vtx =   np.nonzero( R.vtype == LB.D2)[0]
        phi = np.array([float_type(0)]*len(C.out))
        phi[indep_vtx] = phi_r
        phi[dep_vtx] = phi_r[R.refl[dep_vtx]]  
    
    # make sure the scaled lengths all satisifies delaunay condition
    # maintains mesh symmetry
    if R == None:
        flips = make_delaunay(C, l_init, phi=phi, float_type=float_type,length_fun=ptolemy_length_regular) 
    else:
        flips = make_sym_delaunay(C, R,l_init, LengthFun,phi) 

    # Return if error encountered while making the mesh Delaunay
    if flips == None:
        return (False, None, None, (), None)

    print('f'+str(flips)+' ',end='')
    # temporarily splitting quads in the symmetric mesh to triangles for Hessian/grad computation
    # breaks symmetry
    if R == None:
        Ctri = C 
    else:
        Ctri, Rtri = triangulate_quads(C,R,l_init)

    # more convenient to have f2he as matrix
    f2he =  np.reshape(Ctri.f2he,(len(Ctri.f2he),1)) 
    he2f = Ctri.he2f;  out = Ctri.out
    next_he = Ctri.next_he;  to = Ctri.to; opp = Ctri.opp;
   
    n_f  = len(f2he) # number of faces
    n_v  = len(out)  # number of vertices
    n_he = len(to)  # number of halfedges

    f_he = np.hstack([  f2he,  next_he[f2he],   next_he[next_he[f2he]] ]) # 3 indices of halfedges per face 
    f_v  = to[f_he] # 3 vertex indices per face 
    
    # compute conformally scaled edge lengths (3 per face, with average scale per face subtracted for
    # numerical stability   so each triangle's edges are rescaled by the average scale factor
    f_l_init = np.array([[l_init[f_he[i][0]],l_init[f_he[i][1]],l_init[f_he[i][2]]] for i in range(0,len(f_he))])
    f_scales = np.mean(phi[f_v],1)                                     # average scale factor per face 
    f_phi_rescaled = phi[f_v]-np.reshape(f_scales,(len(f_scales),1))  # 3 adjusted phis per face
    f_edge_scales =  np.hstack(  # edge scale factors 
       [ np.reshape( ((f_phi_rescaled[:,0]+f_phi_rescaled[:,2])/2.0),  (f_l_init.shape[0],1)), 
         np.reshape( ((f_phi_rescaled[:,1]+f_phi_rescaled[:,0])/2.0),  (f_l_init.shape[0],1)), 
         np.reshape( ((f_phi_rescaled[:,2]+f_phi_rescaled[:,1])/2.0),  (f_l_init.shape[0],1))
       ])
    if float_type == float:
        f_l_conf_rescaled = np.exp(f_edge_scales)
    else:
        f_l_conf_rescaled = np.array([[mexp(x) for x in y] for y in f_edge_scales])  
    f_l_conf_rescaled = f_l_conf_rescaled*f_l_init  # 3 conformally rescaled edge length per face x

    # if (happens only due to numerical issues) could not get a triangulation satisfying triangle 
    #  inequalities (the output of make_sym_delaunay should) bail out, 
    # this will result in step decrease in newton, and trying again
    #       
    if not check_tri_ineq(f_l_conf_rescaled[:,0], f_l_conf_rescaled[:,1], f_l_conf_rescaled[:,2]): 
        print("violating triangle inequality!")
        return (False, None, None, (), None)
    
    if float_type == float:
        f_angles = vec_angles_from_len(f_l_conf_rescaled)   
        f_cots = vec_cot_from_len(f_l_conf_rescaled)
    else:
        f_angles = np.apply_along_axis( f_angles_from_len,1,f_l_conf_rescaled)  # 3 angles per face
        f_cots   = np.apply_along_axis( f_cot_from_len   ,1,f_l_conf_rescaled)  # 3 cotangents per face
    
    # remap these to halfedges 
    he2angle = np.zeros(shape=(n_he,),dtype=float_type)   
    he2angle[np.reshape(f_he,(n_he,))] =  np.reshape(f_angles,(n_he,))   # angle opposite each halfedge
    he2cot = np.zeros(shape=(n_he,),dtype=float_type)
    he2cot[np.reshape(f_he,(n_he,))] =  np.reshape(f_cots,(n_he,))   # cot opposite each halfedge
    
    # assemble gradient
    if float_type == float:
        G_V = he2angle[next_he[next_he[range(0,len(to))]]]
        G_I = np.zeros(shape=len(to),dtype=int)
        G_J = to
        v_angle_sums = sp.coo_matrix((G_V,(G_I,G_J)),shape=(1,len(out)))
        v_angle_sums = v_angle_sums.toarray()[0]        
    else:
        v_angle_sums = np.zeros(shape=(len(out),),dtype=float_type)
        for v in range(0,len(out)):   # sum of incident angles per vertex
            nbrs = np.nonzero(to==v)
            v_angle_sums[v] = np.sum(he2angle[next_he[next_he[nbrs]]])
    v_grad = Th_hat-v_angle_sums

    if R == None:
        v_grad_r = np.zeros(shape=(len(C.out),),dtype=float_type)
        v_grad_r[:] = v_grad
    else:
        v_grad_r = np.zeros(shape=(len(indep_vtx),),dtype=float_type)
        v_grad_r[indep_vtx] = v_grad[indep_vtx]
        v_grad_r[R.refl[dep_vtx]] *= 2
               
    # assemble Hessian if requested, 
    if needHess:
        # The Hessian is the discrete cot Laplacian in the conformally scaled metric
        # two nonzero elements for each halfedge, + diagonal ones
        # off-diagonal ones are sums of cots of angles opposite to the edge
        # diagonal ones are computed so that the sum is zero
        H_I = [to[he] for he in range(0,n_he)]+ list(range(0,n_v))
        H_J = [to[opp[he]] for he in range(0,n_he)]+list(range(0,n_v))
        if float_type == float:
            S_I = np.zeros(shape=(len(to),),dtype=int)
            S_J = to
            S_V = 0.5*(he2cot[range(0,len(to))] + he2cot[opp[range(0,len(to))]])
            H_V = sp.coo_matrix((S_V,(S_I,S_J)),shape=(1,len(out)))
            H_V = np.append((-0.5*he2cot - 0.5*he2cot[opp[np.arange(0,n_he)]]), H_V.toarray()[0])
        else:
            H_V = (-0.5*he2cot - 0.5*he2cot[opp[np.arange(0,n_he)]]).tolist() + [
                  0.5*np.sum( he2cot[ 
                    np.argwhere(to==v).T[0].tolist() +  
                    opp[np.argwhere(to==v)].T[0].tolist()])
                    for v in range(0,n_v)]
        if R == None:
            Hr_I = H_I
            Hr_J = H_J
        else:
            # a map from all vertices to indepdenent ones 
            dep2indep = np.fmin(np.array(list(range(0,n_v))), np.array(R.refl[list(range(0,n_v))]))
            # for the reduced gradient/matrix, coefficients of dependent vertices are added to corresponding independent
            Hr_I = dep2indep[H_I]
            Hr_J = dep2indep[H_J]
        # reduced matrix using only indpendent degrees of freedom
        Hr = sp.coo_matrix((H_V,(Hr_I,Hr_J)),dtype=float_type)
        # returning angles for debugging
        return (True,v_grad_r, Hr,(flips,), he2angle)
    else: 
         return (True,v_grad_r, None, (flips,), he2angle)

# compute descent direction. The Newton direction is used unless it fails to be a descent direction
# i.e. if Hessian is not SPD for numerical reasons. In this case, a corrected matrix M = H + aI
# is used in place of the Hessian to generate the descent direction, which interpolates between the Newton
# direction and steepest descent.
# this computation is always done in float64 even if other computations
# are in multiprecision, because sparse solves do not work otherwise :-(
# but this should be ok as this only should affect convergence rate, not whether it converges
# returns descent direction d, guaranteed to have a neg. dot product with the gradient (up to numerical error)
# returns is in  the correct float format 
# H: Hessian matrix
# grad: gradient vector
# fixed_dof: index of degree of freedom to eliminate to make the Hessian positive definite
# a: correction parameter for the Newton descent direction
# float_type: type of floats to use, float64 or multiprecision math.mp.mpf both work
# use_gradient_only: if true, use stochastic gradient descent instead of (corrected) newton direction 
def get_descent_direction(H,grad,fixed_dof,a,float_type=float,use_gradient_only=False):
    gradflt = nparray_to_float64(grad) 
    gradflt[fixed_dof] = 0
    # remove degree of freedom from the Hessian
    if not use_gradient_only:
        Hf = H.astype(float,copy=True)
        Hflt = Hf.tocsr()
        nz_fixed_row = Hflt.getrow(fixed_dof).nonzero()[1]
        nz_fixed_col = Hflt.getcol(fixed_dof).nonzero()[1]
        Hflt[fixed_dof,nz_fixed_row] = 0
        Hflt[nz_fixed_col,fixed_dof] = 0       
        Hflt[fixed_dof,fixed_dof] = 1
    # use stochastic gradient descent with a random 20% of the gradient coefficients
    else:
        stochgrad = np.random.choice(len(grad), int(0.8*len(grad)),replace=False)
        gradflt[stochgrad] = 0

    while True:
         # sparse linear algebra in scipy cannot deal with arbitrary-precision numbers
        # so we always solve for the Newton direction in doubles 
        # we compute the matrix in arbitrary precision though, and then convert it to doubles
        Id = sp.eye(len(gradflt),dtype=float)
 #            M =  1./(1/+a)*Hflt+(a/(1.+a))*Id
        if not use_gradient_only:
            M =  Hflt+ a*Id
            d = sp.linalg.spsolve(M,-gradflt)
        else:   
            d = - gradflt.copy()
        newton_decr = np.dot(grad,d)
        
        # Use correction if the descent direction does not have a negative dot product
        # with the gradient
        if newton_decr < 0: #-1e-1*np.dot(grad,grad):
            a = 0.5*a # start from lower a on the next step
            break
        elif a == 0.:
            a = 1.  # we did not try correction yet, start from arbitr value 1
        else: 
            a = a*2  # correction was not enough, increase weigh of Id

    d = nparray_from_float64(d,float_type)
    return (d,a)


# simple line search along a descent direction
#
# special features: does _not_ use the function value only gradient; the reason for this is that
# in the situation of interest the function value is less sensitive to the changes of arg, so 
# e.g. function values can be indistinguishable along a search direction, but a min as zero crossing of the
# projection of the gradient can still be found
# As the function is assumed to be convex, and the projected gradient at the start is negative, 
# we halve the step until the projected gradient is negative; it is guaranteed to happen for small enough 
# step (worst case, we get to numerical zero)
# this takes us to a point where the function is guaranteed to decrease, assuming it is not constant along 
# lines which is unlikely
#
# C,R, l_init, Th_hat, float_type: as in get_grad_and_Hess
# d: descent direction, required to have neg. dot product with grad (computed using float_type arithmetics) 
# lambda0: starting value of step
# log_itr: list of log entries to add information to about this line_search step
# bound_norm: reject lambda if the line step increases the gradient norm
# bound_norm_thres: lambda threshold below which to drop the norm bound, even if bound_norm is True
# c1: value for armijo condition
def line_search(C,R, l_init, Th_hat, phi,
                d,
                lambda0,
                log_itr,
                float_type=float,
                bound_norm=True,
                bound_norm_thres=0,
                c1 = 1e-4):
    # Do not allow lambda to be smaller than machine precision
    if float_type == float:
        min_lambda = 1e-16
    else:
        min_lambda = 2**(-100)

    lm = lambda0
    ls_iter = 0
    log_itr['line_search'] = []

    # Copy initial state of the mesh to prevent error accumulation for large line steps
    C0 = copy.deepcopy(C)
    R0 = copy.deepcopy(R)
    l_init0 = copy.deepcopy(l_init)
    phi0 = copy.deepcopy(phi)

    # Compute the gradient, Hessian, and gradient norms
    (success, grad, dummy,flips_and_violations, angles) = grad_and_Hess(C,R, l_init, Th_hat,phi, False,float_type)
#    print( 'energy:',conf_energy(
#        C,reduced_phi_to_full(C,R,phi,float_type=float_type),
#        l_init,angles,Th_hat))
    max_grad_0 = np.max(abs(grad)) 
    l2_grad_sq_0  = np.dot(grad,grad)
    newton_decr = np.dot(grad, d)

    # FIXME To avoid nans/infs
    #while (np.max(lm*d) - np.min(lm*d) > 10):
    #    lm /= 2

    # Line search
    while True:  
        ls_iter = ls_iter + 1
        phi_lm = phi+lm*d 
        (success, grad, dummy,flips_and_violations, angles) = grad_and_Hess(C,R, l_init, Th_hat,phi_lm, False,float_type); 
#        print( 'energy:',conf_energy(C,reduced_phi_to_full(C,R,phi_lm,float_type=float_type), 
#        l_init, angles, Th_hat))

        # Record line search information in log
        if success: 
            max_grad = np.max(abs(grad))
            l2_grad_sq  = np.dot(grad,grad)
            log_rec = {'iter':ls_iter,
                       'lambda':lm,
                       'success':success,
                       'flips/viol':flips_and_violations,
                       'grad_norm_sq':np.dot(grad,grad),
                       'newton_decr':np.dot(grad,d)}
        else:
            log_rec = {'iter':ls_iter,
                       'lambda':lm,
                       'success':success,
                       'flips/viol':flips_and_violations,
                       'grad_norm_sq':np.nan,
                       'newton_decr':np.nan}
        proj_grad = np.dot(grad, d)
        log_itr['line_search'].append(log_rec)

        # Line search condition to ensure quadratic convergence
        if (    success # gradient computed successfully
            and (ls_iter == 2) # first backtrack
            and (l2_grad_sq <= l2_grad_sq_0 or not bound_norm)
            and (0.5 * (proj_grad + proj_grad_prev) <= c1 * newton_decr)):
            print("Using full step lm={} for quadratic convergence".format(2*lm))
            return (2*lm, (2*lm) >= min_lambda)


            

        # Return if line search conditions met, or reset the mesh and backtrack otherwise
        if (    (     success # gradient computed successfully
                  and np.dot(grad,d) <= 0 # negative projected gradient
                  and (l2_grad_sq <= l2_grad_sq_0 or not bound_norm) ) # bound norm conditions met
             or ( lm < min_lambda) ): # lambda too small for given precision
            return (lm, lm >= min_lambda)
        else:
            print('lm=',lm,end='')
            if success:
               print( 'newt_decr=',np.dot(grad,d),'D max_grad=',max_grad-max_grad_0, end=' ')

            # Restore the mesh to its state before th attempted line step
            copy_in_place_connectivity(C,C0)
            copy_in_place_reflection(R,R0)
            l_init[:] = l_init0
            phi[:] = phi0
            lm = 0.5*lm;         
            
            # Drop norm bound if lambda is below the threshold
            if ((bound_norm) and (lm < bound_norm_thres)):
                bound_norm = False
                print("Dropping norm bound")

        proj_grad_prev = proj_grad

# main Newton iteration 
# more or less standard line-search Newton with full Hessian solve 
# (but does not use function value and assumes convexity)
# C, R, l_init, Th_hat, float_type: as in get_grad_and_Hessian, but can be given as float64 even
#                                   for multiprecision solve  
# lambda0: starting value of lambda for line search
# bound_norm_thres: lambda value at which to stop requiring a decrease in norm during the line search
# error_eps: termination accuracy for the max(grad); for the function of interest this is exactly
#            the max angle error at a vertex
# max_iter: max number of iterations
# initial_ptolemy: if False, use Euclidean flips for the first make_delauany instead of Ptolemy flips
# reset_lambda: if False, use double the previous line search lambda for the current line search instead
#               of starting from lambda0
# gauss_bonnet_correction: if True, subtract the difference between the sum of the target angles Th_hat
#                          and the sum required to satisfy Gauss Bonnet from the first Th_hat value
# newton_decr_thres: if provided termination accuracy for newton decrement (otherwise inferred
#                    from error_eps)
# TODO:  currently needs to be negative, make positive 
# use_gradient_only: if True, use stochastic gradient descent instead of Newton's method
# log_dir: directory to store mesh and log pickle files each iteration in case the method is ended
#          before it terminates 
#
# All computations are done in float_type, _except_   descent direction computation 
# Hessian is always inverted in float64
#      
def newton( C, R, l_init, Th_hat, phi0, 
            float_type=float,
            lambda0=1.0,
            bound_norm_thres=1e-10,
            error_eps=1e-14,
            max_itr=1000,
            initial_ptolemy=False,
            reset_lambda=True,
            gauss_bonnet_correction=False,
            newton_decr_thres=None,
            use_gradient_only=False,
            log_dir=None):
    log = []
    a = 0. # initial factor in the Hessian SPD correction H + a*I
    phi    = nparray_from_float64(phi0  ,float_type)
    Th_hat = nparray_from_float64(Th_hat,float_type)
    l_init = nparray_from_float64(l_init,float_type)
    itr = 0
    lambda_prev = lambda0

    # Set default newton_decr_thres from error_eps if not provided
    if newton_decr_thres is None: 
        newton_decr_thres = -0.01*error_eps*error_eps

    # Prevent grad norm from increasing if above bound_norm_thres
    if (lambda0 <= bound_norm_thres):
        bound_norm = False
    else:
        bound_norm = True
        print("Using norm bound")

    # FIXME Only implemented in C++
    if gauss_bonnet_correction:
        if float_type == float:
            conversion_marcel(C, R, l_init, phi, Th_hat, gauss_bonnet_correction=True)
        else:
            conversion_marcel_mpf(C, R, l_init, phi, Th_hat, MPFR_PREC=mp.prec, gauss_bonnet_correction=True)
    if not initial_ptolemy:
        if float_type == float:
            make_delaunay_marcel(C, R, l_init, phi, Th_hat, initial_ptolemy=False)
        else:
            make_delaunay_marcel_mpf(C, R, l_init, phi, Th_hat, MPFR_PREC=mp.prec, initial_ptolemy=False)

    # ** main Newton loop             
    while True: 
        print("itr "+str(itr)+" ",end='')
        log.append(dict())
        
        # grad computation assumes that initial values phi0 and l_init produce scaled lengths 
        # satisfying triangle inequality 
        # on subsequent iterations, line search ensures this
        (success, grad, H, flips_and_violations,angles) = grad_and_Hess(C, R, l_init, Th_hat, phi,
                                                                        not use_gradient_only,
                                                                        float_type);
        log[itr]['flips/viol'] = flips_and_violations

        # should not happen but just in case
        if not success:
            log[itr]['term_reason'] = 'triangle inequality violation'
            log[itr]['term_data'] = flips_and_violations
            break
        
        # grad vector entries are errors of total angles at vertices vs target Theta_hat, 
        # stopping 1: if max error  is small enough terminate
        if np.max(abs(grad)) <  error_eps:
            if itr > 0:
                log[itr]['line_search'] = log[itr-1]['line_search']
            log[itr]['max_grad'] = np.max(abs(grad))
            log[itr]['grad_norm_sq'] = np.dot(grad,grad)
            log[itr]['term_reason'] = 'Max error below threshold'; log[itr]['term_data'] = (np.max(abs(grad)))
            break
        log[itr]['max_grad'] = np.max(abs(grad))
        
        # compute descent direction -- always done in floats as sparse solve does not work with mupltiprecision
        if R == None:
            fixed_dof = 0
        else:
            fixed_dof = get_fixed_dof(R)
        d,a = get_descent_direction(H,grad,fixed_dof,a,float_type=float_type,use_gradient_only=use_gradient_only)

        # comptue newton (or other descent direction) decrement
        # stopping 2: if newton decrement is small enough (or positive) terminate
        newton_decr = np.dot(grad,d)
        log[itr]['newton_decr'] = newton_decr; log[itr]['grad_norm_sq'] = np.dot(grad,grad)
        print('newton_decr', newton_decr, 'grad_norm_sq',  np.dot(grad,grad), 'max_grad',np.max(abs(grad)) )    
        if newton_decr >=  0:  # should never happen but just in case
            log[itr]['term_reason'] = 'Postive newton decrement'; log[itr]['term_data'] = (newton_decr)
            break
        if  newton_decr  > newton_decr_thres:
            log[itr]['term_reason'] = 'Newton decrement abs below threshold'; log[itr]['term_data'] = (newton_decr)
            break   
        # stopping 3: if maximal iteration reach terminate
        if itr == max_itr:
            log[itr]['term_reason'] = 'Maximal iteration reached'
            break

        # Set initial lambda for line search
        if reset_lambda:
            lm = lambda0
        else:
            lm = min(1.0,lambda_prev*2)

        # reset lambda when it goes above norm bound threshold
        if (lm > bound_norm_thres and not bound_norm):
            bound_norm = True
            lm = lambda0
            print("Using norm bound")

        # Perform line search
        (lm,success) = line_search(C, R,l_init, Th_hat,phi,d,lm,
                                   log[itr],float_type, bound_norm=bound_norm)

        # stopping 4: if line search failed due to too small of a lambda terminate
        if not success: 
            log[itr]['term_reason'] = 'Line search failed to find large enough step'; log[itr]['term_data'] = lm          
            break
        else:
            # advance phi
            phi = phi + lm*d

        # Update bound_norm if lambda dropped below the threshold during the line search
        lambda_prev = lm
        print('lambda used:',lm)
        if lm <= bound_norm_thres:
            bound_norm = False


        # Save log and checkpoint data to file if log_dir provided
        if log_dir:
            log_file = os.path.join(log_dir, 'log.p')
            model_file = os.path.join(log_dir, 'model.p')
            with open(log_file, 'wb') as f:
                pickle.dump(log,f)
            with open(model_file, 'wb') as f:
                pickle.dump((C,R,l_init,Th_hat,phi),f)

        # FIXME Moved here from an incorrect location
        itr = itr + 1

    # Print termination information
    if grad is not None:
        print('\nfinal max error:',  np.max(abs(grad)))
    print(log[itr]['term_reason'])

    return {'l_init_final':l_init,'phi':phi,'log':log}


# expand target angles to a double mesh C,R from values Th_hat_in on a mesh C_in 
def double_Theta(C,R,Th_hat_in,Cin,float_type=float):
    Th_hat = np.array([float_type(0.0)]*len(C.out), dtype=float_type)
    for v in range(0,len(C.out)):
        if R.vtype[v] == LB.S: 
            Th_hat[v] = 2*Th_hat_in[Cin.fr[C.out[v]]]
       #     print(Th_hat[v])
        elif v < len(Th_hat_in):
            Th_hat[v] = Th_hat_in[Cin.fr[C.out[v]]]
        else:
            assert R.refl[v] < len(Th_hat_in),'both the vertex and its reflection have indices > len of input Th hat'
            Th_hat[v] = Th_hat_in[Cin.fr[C.out[R.refl[v]]]]
    return Th_hat

# load a triangle mesh connnectivity, and per-vertex angles 
# edge lengths are currently ignored and all set to 1
# input:
# modeldir: directory where models are stored
# model:  name w/o obj: assumes that modeldir contains failes models/<model>_simp.obj and Theta_hat/
#Theta_hat_<model>
# obj file contains a triangle mesh with disk topology 
# Theta_hat contains angles per vertex of the triangle mesh 
# returns:
# ConformalMesh dataclass, containing
# doubled symmetric mesh  (C,R) obtained from input by creating a copy and 
# gluing along the boundary; R encodes the symmetry map, see symmmesh.py
# associated per-vetex angles  Th_hat for the doubled mesh, edge lengths (all 1)
# and phi (all zero)

def load_mesh(modeldir, model,improve_val=False):
    v,f = igl.read_triangle_mesh(modeldir+'models/'+model+'_simp.obj')
    next_he,opp,bnd_loops,vtx_reindex =FV_to_NOB(f)
    C = NOB_to_connectivity(next_he,opp,bnd_loops)
    Th_hat_in = np.loadtxt(modeldir+'Theta_hat/'+'Theta_hat_'+model,dtype=float)
    Th_hat_in = Th_hat_in[vtx_reindex]
    if improve_val:
        improve_valence(C,Th_hat_in)
    Cd,Rd = NOB_to_double(C.next_he,C.opp,C.bnd_loops)
    Th_hat = double_Theta(Cd,Rd,Th_hat_in,C)
    l_init = np.ones((len(Cd.next_he),))
    phi0 = np.zeros((len(Cd.out),))
    return ConformalMesh(C=Cd,R=Rd, phi=phi0, l=l_init,Th_hat=Th_hat)

# main function:  
# input:  CM, an ConformalMesh structure 
# output: CM at the end of optimization, and a log
# all other parameters are passed directly to Newton
def conformal_map(CM, 
                  float_type=float,
                  lambda0=1.0,
                  bound_norm_thres=1e-10,
                  error_eps=1e-14,
                  max_itr=1000,
                  initial_ptolemy=False,
                  reset_lambda=True,
                  gauss_bonnet_correction=False,
                  newton_decr_thres=None,
                  use_gradient_only=False,
                  log_dir=None):
    # leaving input intact
    CMw = copy.deepcopy(CM)
    # TODO connectivity of CMw is modified in-place 
    # on the other hand l_init and phi are not -- fix
    if CMw.R == None:
        phi_r = CMw.phi
    else:
        phi_r = full_phi_to_reduced(CMw.C,CMw.R,CMw.phi)
    newton_out = newton(CMw.C, CMw.R, CMw.l,CMw.Th_hat, phi_r,
                        float_type=float_type,
                        lambda0=lambda0,
                        bound_norm_thres=bound_norm_thres,
                        error_eps=error_eps, 
                        max_itr=max_itr,
                        initial_ptolemy=initial_ptolemy,
                        reset_lambda=reset_lambda,
                        gauss_bonnet_correction=gauss_bonnet_correction,
                        newton_decr_thres=newton_decr_thres,
                        use_gradient_only=use_gradient_only,
                        log_dir=log_dir)
    CMw.l   = newton_out['l_init_final']
    if CMw.R == None:
        CMw.phi = newton_out['phi']
    else:
        CMw.phi = reduced_phi_to_full(CMw.C,CMw.R,newton_out['phi'],float_type=float_type)
    return CMw, newton_out
