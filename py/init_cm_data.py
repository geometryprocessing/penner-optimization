#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:04:26 2021

@author: rcapouel
"""
import igl
from conformal_impl.conformal import *
from conformal_impl.delaunay import *

# Compute edge lengths of a mesh with reindexed vertices
def connectivity_l(C, vtx_reindex, v):
    return np.linalg.norm(v[vtx_reindex[C.to]] - v[vtx_reindex[C.fr]],axis=1)

# Extend the initial lengths of a mesh to the doubled mesh
def double_l(l_in, R, C):
    in_bnd_loop = np.isin(C.he2f, C.bnd_loops)
    l = l_in[np.logical_not(in_bnd_loop)]
    return np.concatenate((l,np.flip(l)))

# load data for conformal map
# separate two patches into two sub-meshes and store the data inside CM
def prepare_cm_data(v, f, Th_hat_in,
                    unit_lengths=False,
                    float_type=float,
                    use_valance_flips=False,
                    return_index=False,
                    tufted_cover=False):

    bds = igl.all_boundary_loop(f)
    is_bd = np.array([False]*len(v))
    for bd in bds:
        for vt in bd:
            is_bd[vt] = True

    # handle each patch separately
    compo = igl.face_components(f)
    n_compo = np.max(compo)+1
    nf_compo = [0]*n_compo
    for fid in range(len(f)):
        nf_compo[compo[fid]] += 1
    if n_compo == 1:
        order = [0]
    else:
        if nf_compo[0] > nf_compo[1]: # handle the small patch first
            order = [1, 0]
        else:
            order = [0, 1]
    
    CMs = []; Is = []; Js = []

    for i in order:
        fi = np.array([f[j] for j in range(len(f)) if compo[j] == i])
        vi, fi, I, J = igl.remove_unreferenced(v, fi)
        bd = igl.boundary_loop(fi)
        Is.append(I); Js.append(J)
        Th_hat_i = Th_hat_in[J]
        next_he,opp,bnd_loops,vtx_reindex = FV_to_NOB(fi)
        C = NOB_to_connectivity(next_he,opp,bnd_loops)
        
        Th_hat_i = Th_hat_i[vtx_reindex]
        if use_valance_flips:
            improve_valence(C, Th_hat_i)
            #improve_valence_vertex_greedy(C, Th_hat_i)
            #C, Th_hat_i = improve_valence_vertex_split(C,Th_hat_i)
        
        Cd,Rd = NOB_to_double(C.next_he,C.opp,C.bnd_loops)

        # Identify the vertices of the doubled and original mesh
        if tufted_cover:
            # Create map from original and doubled vertices to new identified vertices 
            indep_vtx = np.nonzero( np.isin(Rd.vtype, [LB.D1,LB.S]))[0]
            dep_vtx =   np.nonzero( Rd.vtype == LB.D2)[0]
            v_map = np.arange(len(Cd.out))
            v_map[indep_vtx] = np.arange(len(indep_vtx))
            v_map[dep_vtx] = v_map[Rd.refl[dep_vtx]]

            # Create new connectivity and reflection objects with identified vertices
            Cd = Connectivity(to=v_map[Cd.to], fr=v_map[Cd.fr], out=Cd.out[indep_vtx],
                              next_he=Cd.next_he, prev_he=Cd.prev_he, opp=Cd.opp,
                              he2f=Cd.he2f, f2he=Cd.f2he, bnd_loops=Cd.bnd_loops,
                              quad_info=Cd.quad_info)
            Rd = Reflection(refl=np.arange(len(indep_vtx)),vtype=Rd.vtype[indep_vtx],
                            refl_f=Rd.refl_f, refl_he=Rd.refl_he,
                            ftype=Rd.ftype, etype=Rd.etype)
            Th_hat = 2*Th_hat_i[C.fr[Cd.out]]
        else:
            Th_hat = double_Theta(Cd,Rd,Th_hat_i,C,float_type=float_type)
        phi0 = np.zeros((len(Cd.out),))
        phi0 = nparray_from_float64(phi0, float_type)

        # get lengths, either from mesh or unit
        if unit_lengths:
            l_init = np.ones((len(Cd.next_he),),dtype=float_type)
        else:
            l_init = connectivity_l(C,vtx_reindex,vi)
            l_init = double_l(l_init,Rd,C)
        l_init = nparray_from_float64(l_init, float_type)

        CMs.append(ConformalMesh(C=Cd, R=Rd, phi=phi0, l=l_init, Th_hat=Th_hat))
    if return_index:
        if tufted_cover:
            vtx_reindex = v_map[vtx_reindex]
        return CMs, Is, Js, vtx_reindex
    else:   
        return CMs, Is, Js

# load data for conformal map for uncut mesh
def prepare_cm_uncut_data(v, f, Th_hat_in, unit_lengths=False, float_type=float, use_valance_flips=False, return_index=False):
    next_he,opp,bnd_loops,vtx_reindex = FV_to_NOB(f)
    C = NOB_to_connectivity(next_he,opp,bnd_loops)

    n_v = len(v)
    n_f = len(C.f2he)
    n_he = len(C.next_he)
    refl_v = np.arange(n_v)
    refl_f = np.arange(n_f)
    refl_he = np.arange(n_he)
    vtype = np.full(n_v,LB.D1)
    ftype = np.full(n_f,LB.D1)
    etype = np.full(n_he,LB.D1)
    #R = Reflection(refl=refl_v,refl_f=refl_f,refl_he=refl_he,vtype=vtype,ftype=ftype,etype=etype) 
    R = None
    
    Th_hat = Th_hat_in[vtx_reindex]

    if use_valance_flips:
        #C, Th_hat = improve_valence_vertex_split(C,Th_hat)
        #improve_valence_vertex_greedy(C, Th_hat)
        improve_valence(C, Th_hat)
    if unit_lengths:
        l_init = np.ones((len(C.next_he),))
    else:
        l_init = connectivity_l(C,vtx_reindex,v)
    phi0 = np.zeros((len(C.out),))
    l_init = nparray_from_float64(l_init, float_type)
    phi0 = nparray_from_float64(phi0, float_type)
    
    CMs = []; 

    CMs.append(ConformalMesh(C=C, R=R, phi=phi0, l=l_init, Th_hat=Th_hat))
    if return_index:
        return CMs, vtx_reindex
    else:
        return CMs
