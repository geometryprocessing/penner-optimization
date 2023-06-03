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

from conformal import *
from halfedge import *
from symmesh import *
from overload_math import *
from conformal_py import *

# Convert a python conformal mesh to a pybind11 C++ conformal mesh object
# with lambda0, bound_norm_thres, and pt parameters
def python_to_cpp_CM(CM,float_type=float,lambda0=1,bound_norm_thres=1,error_eps=1e-14,
                     pt_fids=[], pt_bcs=[], return_pt=False, MPFR_PREC=300):
    mp.prec = MPFR_PREC
    C = CM.C
    R = CM.R
    elen = nparray_from_float64(CM.l, float_type)
    phi = nparray_from_float64(CM.phi, float_type)
    Th_hat = nparray_from_float64(CM.Th_hat, float_type)

    n_he = len(C.next_he)
    n_v = len(C.out)
    n_f = len(C.f2he)

    # Nonsymmetric Case
    if R is None:
        # Get the original connectivity structure
        _n = C.next_he
        _to = C.to
        _f = C.he2f
        _h = C.f2he
        _out = C.out
        _opp = C.opp

        # Create trivial reflection information
        _etype = np.zeros(n_he, dtype=int)
        _R = np.zeros(n_he, dtype=int)
    # Symmetric case
    else:
        # Triangulate the mesh and label virtual quad diagonals
        Ctri, Rtri = triangulate_quads(C,R,elen)

        # Get the triangulated connectivity structure
        _n = Ctri.next_he
        _to = Ctri.to
        _f = Ctri.he2f
        _h = Ctri.f2he
        _out = Ctri.out
        _opp = Ctri.opp

        # Get the triangulated halfedge reflection structure and labels
        _R = Rtri.refl_he
        _etype = Rtri.etype

    # Create CM C++ data object
    if float_type==float:
        elen_in = elen
        phi_in = phi
        Th_hat_in = Th_hat
        return ConformalMetricDelaunayDouble(_n,_to,_f,_h,_out,_opp,
                                            _etype,
                                            _R,
                                            elen_in,
                                            phi_in,
                                            Th_hat_in,
                                            pt_fids,
                                            pt_bcs,
                                            lambda0,
                                            bound_norm_thres,
                                            error_eps,
                                            MPFR_PREC)
    else:
        # Convert mpf values to string for passing to pybind
        vnstr = np.vectorize(lambda a:str(repr(a))[5:-2])
        elen_in = vnstr(elen)
        phi_in = vnstr(phi)
        Th_hat_in = vnstr(Th_hat)
        return ConformalMetricDelaunayMpf(_n,_to,_f,_h,_out,_opp,
                                          _etype,
                                          _R,
                                          elen_in,
                                          phi_in,
                                          Th_hat_in,
                                          pt_fids,
                                          pt_bcs,
                                          lambda0,
                                          bound_norm_thres,
                                          error_eps,
                                          MPFR_PREC)

# Inverse of python_to_cpp_CM 
def cpp_to_python_CM(CM,float_type=float):
    # Extract mesh information from the C++ CM data object
    n,to,f,h,out,opp,etype,R_out = CM.get_m()
    l = CM.get_l()
    u = CM.get_u()
    Th_hat = CM.get_Theta_hat()
    n_he = len(n)
    n_v = len(out)
    n_f = len(h)
    n_e = int(n_he/2)

    # Update C with the new mesh information
    next_he = np.array(n)
    to = np.array(to)
    he2f = np.array(f)
    f2he = np.array(h)
    out = np.array(out)
    opp = np.array(opp)
    fr = to[opp]
    prev_he = np.arange(0,n_he)

    # Initialize default quad info
    quad_info  =(-1)*np.ones((n_f,3),dtype=int)
    quad_info[:] = np.array(quad_info,dtype=int) 

    C = Connectivity(next_he=next_he, prev_he=prev_he, opp=opp, to=to, fr=fr, he2f=he2f, out=out, f2he=f2he,
                 bnd_loops=[], quad_info=quad_info)

    # Update R if symmesh
    if np.max(R_out) > 0:
        # Get half edge reflection map and labels
        refl_he = np.array(R_out)
        etype = np.array(etype)

        # Get face reflection map and labels (for triangulated mesh)
        refl_f = C.he2f[refl_he[C.f2he]]
        ftype = np.arange(n_f)
        for he in range(0,n_he):
            if etype[he] in {LB.D1,LB.D2}:
                ftype[C.he2f[he]] = etype[he]
        ftype[(refl_f == np.arange(0,n_f))] = LB.T

        # Replace triangulated quads with actual quads and update face reflection maps
        R = Reflection(refl=np.arange(n_v),vtype=np.zeros(n_v,dtype=int),
                       refl_f=refl_f,refl_he=refl_he,ftype=ftype,etype=etype)
        create_quads(C,R)

        # Get vertex reflection map and labels
        # Note that this is constructed after the quads to prevent issues
        # with C.out being a quad diagonal
        R.refl[:] = C.to[R.refl_he[C.out]]
        for he in range(0,n_he):
            if R.etype[he] in {LB.D1,LB.D2}:
                R.vtype[C.to[he]] = R.etype[he]
        R.vtype[(R.refl == np.arange(0,n_v))] = LB.S


    # Update elen and phi
    if float_type==float:
        elen = np.array(l)
        phi = np.array(u)
        Th_hat = np.array(Th_hat)
    else:
        elen = np.array([mp.mpf(li) for li in l])
        phi = np.array([mp.mpf(ui) for ui in u])
        Th_hat = np.array([mp.mpf(Thi) for Thi in Th_hat])
    
    return ConformalMesh(C=C,phi=phi,R=R,Th_hat=Th_hat,l=elen)


# Line search using a C++ Conformal Mesh with descent direction d
def line_search_cpp(CM, d, lambda0,log_itr,float_type=float,bound_norm=True):
    lm = lambda0
    ls_iter =0
    min_lambda = 1e-16
    log_itr['line_search'] = []
    CM.MakeDelaunay(True)
    CM.compute_angles()
    grad = nparray_from_float64(np.array(CM.get_g()), float_type)
    max_grad_0 = np.max(abs(grad))
    l2_grad_sq_0  = np.dot(grad,grad)
    phi = nparray_from_float64(np.array(CM.get_u()), float_type)
    while True:
        ls_iter = ls_iter + 1

        # Set the new phi value
        phi_lm = phi+0.99*lm*d
        if float_type != float:
            vnstr = np.vectorize(lambda a:str(repr(a))[5:-2])
            phi_lm = vnstr(phi_lm)
        CM.set_u(phi_lm)

        # Calculate the gradient
        CM.MakeDelaunay(True)
        CM.compute_angles()
        grad = nparray_from_float64(np.array(CM.get_g()), float_type)
        max_grad = np.max(abs(grad))
        l2_grad_sq  = np.dot(grad,grad)
        log_rec = {'iter':ls_iter,'lambda':lm, 'grad_norm_sq':np.dot(grad,grad),'newton_decr':np.dot(grad,d)}
        log_itr['line_search'].append(log_rec)
        if (np.dot(grad,d) < 0 and (l2_grad_sq < l2_grad_sq_0 or not bound_norm))  or lm < min_lambda:
            return (lm, lm >= min_lambda)
        else:
            print('lm=',lm,end='')
            print( 'newt_decr=',np.dot(grad,d),'D max_grad=',max_grad-max_grad_0, end=' ')
            lm = 0.5*lm;

# Newton's method using a C++ Conformal Mesh
def newton_cpp( CM,
                float_type=float,
                lambda0=1.0,
                newton_decr_thres=None,eps=1e-12, max_iter=1000,bound_norm=True):
    log = []
    itr = 0
    lambda_prev = lambda0

    if newton_decr_thres is None:
        newton_decr_thres = -0.01*eps*eps

    # Make mesh Delaunay (line search ensures this is true in later iterations)
    CM.MakeDelaunay(True)
    CM.compute_angles()

    while True:
        print("itr "+str(itr)+" ",end='')
        log.append(dict())

        # grad computation assumes that initial values phi0 and l_init produce scaled lengths 
        # satisfying triangle inequality 
        # on subsequent iterations, line search ensures this

        grad = nparray_from_float64(np.array(CM.get_g()), float_type)
        # grad vector entries are errors of total angles at vertices vs target Theta_hat, 
        # stopping 1: if max error  is small enough terminate
        if np.max(abs(grad)) <  eps:
            if itr > 0:
                log[itr]['line_search'] = log[itr-1]['line_search']
            log[itr]['max_grad'] = np.max(abs(grad))
            log[itr]['grad_norm_sq'] = np.dot(grad,grad)
            log[itr]['term_reason'] = 'Max error below threshold'; log[itr]['term_data'] = (np.max(abs(grad)))
            break
        log[itr]['max_grad'] = np.max(abs(grad))

        # compute descent direction       
        d = nparray_from_float64(np.array(CM.descent_direction_py()), float_type)
        newton_decr = np.dot(grad,d)

        log[itr]['newton_decr'] = newton_decr; log[itr]['grad_norm_sq'] = np.dot(grad,grad)
        print('newton_decr', newton_decr, 'grad_norm_sq',  np.dot(grad,grad), 'max_grad',np.max(abs(grad)) )

        if newton_decr >=  0:  # should never happen but just in case
            log[itr]['term_reason'] = 'Postive newton decrement'; log[itr]['term_data'] = (newton_decr)
            break
        # remaining stopping criteria:      
        # stopping 2
        if  newton_decr  > newton_decr_thres:
            log[itr]['term_reason'] = 'Newton decrement abs below threshold'; log[itr]['term_data'] = (newton_decr)
            break
        (lm,success) = line_search_cpp(CM,d,min(1.0,lambda_prev*2),
                                   log[itr],float_type,bound_norm=bound_norm)
        lambda_prev = lm
        print('lambda used:',lm)
        # stopping 3
        if not success:
            log[itr]['term_reason'] = 'Line search failed to find large enough step'; log[itr]['term_data'] = lm
            break
        # stopping 4   
        if itr == max_iter:
            log[itr]['term_reason'] = 'Maximal iteration reached'
            break
        itr = itr + 1

    if grad is not None:
        print('\nfinal max error:',  np.max(abs(grad)))
    print(log[itr]['term_reason'])
    return log

# Create C++ Conformal Mesh object and run Newton
def conformal_map_cpp(CM,
                      max_iter=1500,
                      eps=1e-12,
                      float_type=float,
                      newton_decr_thres=None,
                      lambda0=0.1,
                      log_dir=None,
                      use_gradient_only=False,
                      bound_norm=True):
    CMw = python_to_cpp_CM(CM,float_type)
    log = newton_cpp(CMw, float_type=float_type, eps=eps, lambda0=lambda0,
                     newton_decr_thres=newton_decr_thres, max_iter=max_iter,
                     bound_norm=bound_norm)
    CM_out = cpp_to_python_CM(CMw,float_type=float_type)
    return CM_out, log


