import math
import numpy as np
from collections import namedtuple
from conformal_impl.halfedge import *
from conformal_impl.symmesh import *
from conformal_impl.ptolemy import *
#from conformal_py import *

#  cosine theorem: also works for hyperbolic metric Delaunay criterion, 
# even with the triangle inequality violated (still "cos(a)" + "cos(b)", 
# although the quantities may exceed 1 in abs value

def Delaunay_ind_T(ld,la,lb): 
#    return (la**2 + lb**2 -ld**2)/(2*la*lb)
    return (la/lb + lb/la -(ld/la)*(ld/lb))/2

# lq = quad diagonal
# derivation: (lq^2 + lp1^2 - ld^2)/(2*lp1*lq)  = 
# (lp1*lp2 + ld^2 + lp1^2 - ld^2)/(2*lp1*lq) 
# = (lp1 + lp2)/(2*lq)

def Delaunay_ind_Q_side(lr1,lr2,lq):
#   print('using Q side for', lr1,lr2,lq)
    return (lr2 + lr1)/(2*lq)

def Delaunay_ind_Q_base(ls,lr,lq):
#    print('using Q base for s,r,q', ls,lr,lq)
    return (ls/lq + lq/ls - (lr/ls)*(lr/lq))/2

# for each triple set scale factors so that the diag edge 
# is not scaled. Suboptimal (would be better e.g. to take average
# of scales for the whole triple) 

def scaled_length(l, phi_fr, phi_to, phi0_ref, phi1_ref):
 # 0.5*(phi[C.fr[h_ref]] + phi[C.to[h_ref]])
    return l*mexp(0.5*((phi_fr+phi_to)-(phi0_ref + phi1_ref)))

# triangle inequality test
def check_tri_ineq(l1,l2,l3):
    return ( l1+l2 - l3 >=0  and  l2+l3 >= l1 and  l3+l1 >= l2) 

def Delaunay_ind_face(C,R,lc,h,phi=None):        
    f = C.he2f[h]
    if f in C.bnd_loops:
        return None
    if R.ftype[f] != LB.Q:
        hd = h; hb = C.next_he[hd]; ha = C.next_he[hb]
        ld =lc[hd]
        if phi is not None:
            la = scaled_length(lc[ha],*phi[[C.fr[ha],C.to[ha],C.fr[hd],C.to[hd]]])
            lb = scaled_length(lc[hb],*phi[[C.fr[hb],C.to[hb],C.fr[hd],C.to[hd]]])
        else: 
            la = lc[ha]; lb = lc[hb]
#       return Delaunay_ind_T(lc[hd], lc[ha],lc[hb])
        return Delaunay_ind_T(ld, la,lb)
    elif R.etype[h] == LB.P:
        hr1 = h; hs2 = C.next_he[h]; hr2 = C.next_he[hs2]; hs1 = C.next_he[hr2]
        hq = C.quad_info[f,1]
        lr = lc[hr1]
        if phi is not None:
            ls = scaled_length(lc[hs1],*phi[[C.fr[hs1],C.to[hs1],C.fr[hr1],C.to[hr1]]])
            # diagonals do not have valid to/fr, using other edges
            lq = scaled_length(lc[hq ],*phi[[C.fr[hr1],C.to[hs2],C.fr[hr1],C.to[hr1]]])
        else: 
            lr = lc[hr1];  ls = lc[hs1]; lq = lc[hq]
        return Delaunay_ind_Q_base(ls,lr,lq)
#       return Delaunay_ind_Q_base(lc[hs1],lc[hr1],lq)
    else:  # h is D1 or D2
        hs1 = h; hr1 = C.next_he[h]; hs2 = C.next_he[hr1]; hr2 = C.next_he[hs2]
        hq = C.quad_info[f,1]
        if phi is not None:
            lr1 = scaled_length(lc[hr1],*phi[[C.fr[hr1],C.to[hr1],C.fr[hs1],C.to[hs1]]])
            lr2 = scaled_length(lc[hr2],*phi[[C.fr[hr2],C.to[hr2],C.fr[hs1],C.to[hs1]]])
            lq =  scaled_length(lc[hq ],*phi[[C.fr[hs1],C.to[hr1],C.fr[hs1],C.to[hs1]]])
        else: 
            lr1 = lc[hr1]; lr2 =lc[hr2]; lq = lc[hq]
        return Delaunay_ind_Q_side(lr1,lr2,lq)        
   #    return Delaunay_ind_Q_side(lc[hr1],lc[hr2],lq)

# is the pair of faces corresponding to h Delauany
# this does not rely on responsiblity and symmetric triple normalization
# to provide additional validation
def Delaunay_ind(C,R,lc,h,phi=None):
    ind1 = Delaunay_ind_face(C,R,lc,h,phi)
    ind2 = Delaunay_ind_face(C,R,lc,C.opp[h],phi)
    if ind1 == None or ind2 == None:
        return None
    else:
        return ind1 + ind2

def Delaunay_test(C,R,lc,h,phi=None):
    ind = Delaunay_ind(C,R,lc,h,phi)
    return ind, (ind != None and ind > 0)

# simplified version working only when both faces are triangles, and no scaling 
# mostly for sanity checks
def Delaunay_test_tri(ld,la,lb,lao,lbo):
    ind = Delaunay_ind_T(ld,la,lb) +  Delaunay_ind_T(ld,lao,lbo)
    return ind, ind > 0


def Delaunay_ind_tri_face(C, lc,h,phi=None):        
    f = C.he2f[h]
    if f in C.bnd_loops:
        return None
    hd = h; hb = C.next_he[hd]; ha = C.next_he[hb]
    ld =lc[hd]
    if phi is not None:
        la = scaled_length(lc[ha],*phi[[C.fr[ha],C.to[ha],C.fr[hd],C.to[hd]]])
        lb = scaled_length(lc[hb],*phi[[C.fr[hb],C.to[hb],C.fr[hd],C.to[hd]]])
    else: 
        la = lc[ha]; lb = lc[hb]
    return Delaunay_ind_T(ld, la,lb)

def Delaunay_ind_tri(C,lc,h,phi=None):
    ind1 = Delaunay_ind_tri_face(C,lc,h,phi)
    ind2 = Delaunay_ind_tri_face(C,lc,C.opp[h],phi)
    if ind1 == None or ind2 == None:
        return None
    else:
        return ind1 + ind2

def Delaunay_test_tri_phi(C,lc,h,phi=None):
    ind = Delaunay_ind_tri(C,lc,h,phi)
    return ind, (ind != None and ind > 0)

def is_sym_Delaunay(C,R,lc,phi=None):
    ok = True
    for h in range(0,len(C.next_he)):
        ind,is_Del = Delaunay_test(C,R,lc,h,phi) 
        if ind is not None and not is_Del: 
            ok = False
            f0,f1 = C.he2f[h], C.he2f[C.opp[h]]
            h_r,triple = get_central_he_and_triple_type(C,R,h)
            print('non-Delaunay h',[ h,f0,f1], 'resp',h_r,triple,  
                  TripleLabel[R.ftype[f0]]+TripleLabel[R.etype[h]]+TripleLabel[R.ftype[f1]] )
    return ok

def is_Delaunay(C,lc,phi=None):
    ok = True
    for h in range(0,len(C.next_he)):
        ind,is_Del = Delaunay_test_tri_phi(C,lc,h,phi) 
        if ind is not None and not is_Del: 
            ok = False
    return ok

def  sym_edge_lengths(C,R,indep_ind, indep_len):
    lc = np.zeros((len(C.next_he),),dtype=type(indep_len[0]))
    lc[indep_ind] = indep_len
    lc[C.opp[indep_ind]] = indep_len
    lc[R.refl_he[indep_ind]] = indep_len
    lc[R.refl_he[C.opp[indep_ind]]] = indep_len
    assert((lc != -1).all())
    return lc

def validate_sym_lengths(C,R,elen): 
    mismatch_refl = np.argwhere(elen != elen[R.refl_he])
    ok = True
    if len(mismatch_refl) > 0: 
        ok = False
        print('symmetric halfedges have different lengths', mismatch_refl.T, R.refl_he[mismatch_refl.T], elen[mismatch_refl.T])
    mismatch_opp = np.argwhere(elen != elen[C.opp]);
    if len(mismatch_opp) > 0: 
        ok = False
        print('opp halfedges have different lengths', mismatch_opp.T, C.opp[mismatch_opp.T], elen[mismatch_opp.T])
    return ok


def check_tri_ineq_all(C,elen):
    ok = True
    for f in range(0,len(C.f2he)):
        h0 = C.f2he[f]; h1 = C.next_he[h0]; h2 = C.next_he[h1]
        assert(C.next_he[h2] == h0)
        if not check_tri_ineq(elen[h0],elen[h1],elen[h2]):
            print('triangle ineq violation', f, elen[[h0,h1,h2]])
            ok = False
    return ok

def is_Delaunay_trimesh(C,elen,phi=None):
    for hd in range(0,len(C.next_he)):
        if  (C.he2f[hd] not in C.bnd_loops) and (C.he2f[C.opp[hd]] not in C.bnd_loops):
            hb = C.next_he[hd]; ha = C.next_he[hb];
            hdo = C.opp[hd]
            hbo = C.next_he[hdo]; hao = C.next_he[hbo]; 
            ld = elen[hd] 
            if phi is not None: 
                 la  = scaled_length(elen[ha ],*phi[[C.fr[ha ], C.to[ha ], C.fr[hd],C.o[hd]]])
                 lb  = scaled_length(elen[hb ],*phi[[C.fr[hb ], C.to[hb ], C.fr[hd],C.o[hd]]])
                 lao = scaled_length(elen[hao],*phi[[C.fr[hao], C.to[hao], C.fr[hd],C.o[hd]]])
                 lbo = scaled_length(elen[hbo],*phi[[C.fr[hbo], C.to[hbo], C.fr[hd],C.o[hd]]])
            else: 
                la = elen[ha]; lb = elen[hb]; lao = elen[hao]; lbo = elen[hbo]

#            dind = Delaunay_ind_T(ld,la,lb) +  Delaunay_ind_T(ld,lao,lbo)

            dind,is_Del=  Delaunay_test_tri(ld,la,lb,lao,lbo)
            if not is_Del:
                print('non-Delaunay triangle pair', hd, C.he2f[hd],C.he2f[hdo], 'Del ind = ', dind)
                print(elen)
                return False 
    return True

# Symmetric Delaunay
#
# input: symmetric tri/quad mesh  as defined in symmmesh.py with 
#  connectivity C and reflection structure R, array of lengths invariant with respect to R 
#
# result: in-place modification of C and R: a symmetric Delaunay tri/quad mesh, with lengths
#     computed in decorated hyperbolic metric described in Springborn 2018, using Ptolemy flips 
#      
#   

def make_sym_delaunay(C,R,elen, length_funs, phi=None): 
    flips = 0
    stk = []
    mark = [False for h in range(0,len(C.next_he))]
    for h in range(0,len(C.next_he)):
        if is_responsible(C,R,h):
            stk.append(h)
            mark[h] = True
    while stk:
        h = stk.pop(-1)
        mark[h] = False
        # TODO should not check sign directly here 
        # to accomodate intervals etc encapsulate in a function 
        # returning boolean; should have the property that if 
        # it returns true for a configuration, for flipped it returns false
        ind,is_Del = Delaunay_test(C,R,elen,h,phi)
        if  not is_Del:
            h_r,triple = get_central_he_and_triple_type(C,R,h)
            invalid_he =[]
            if h_r is not None:
                invalid_he = FlipFun[triple](C,R,h_r,elen,length_funs[triple])
                ind_post,is_Del_post =  Delaunay_test(C,R,elen,h_r,phi)
                if not is_Del_post:
                    print('flipping a ND edge resulted in ND', ind,ind_post,h_r,triple)
                    print(phi[C.to[h_r]],phi[C.fr[h_r]],phi[C.to[invalid_he]],  elen[h_r],elen[invalid_he])  
                    if(mp.isnan(elen[h_r])):
                        print('nan encountered while making mesh Delaunay')
                        return None
                flips +=1
                if flips % 100000 == 0: 
                    print('flips ',flips, get_triple_counts(C,R))
            for he in invalid_he:
                for hef in [he,C.opp[he],R.refl_he[he],R.refl_he[C.opp[he]]]:
                    if is_responsible(C,R,hef) and not mark[hef]:
                        stk.append(hef)
                        mark[hef] = True
    return flips

# given connectivity associated with edge lengths per halfedge
# do edge flips until the mesh is delaunay
# the new edge lengths is computed under Euclidean metric
# and C, elen is updated in place
def check_flap(C, hd):
    hb = C.next_he[hd]; ha = C.next_he[hb];
    hdo = C.opp[hd]; hao = C.next_he[hdo]; hbo = C.next_he[hao]; 
    if (hb == hd):
        return False
    if (ha == hd):
        return False
    if (ha == hb):
        return False
    if (hbo == hdo):
        return False
    if (hao == hdo):
        return False
    if (hao == hbo):
        return False
    if (C.next_he[ha] != hd):
        return False
    if (C.next_he[hbo] != hdo):
        return False
    if (C.he2f[hd] != C.he2f[ha]):
        return False
    if (C.he2f[ha] != C.he2f[hb]):
        return False
    if (C.he2f[hdo] != C.he2f[hao]):
        return False
    if (C.he2f[hao] != C.he2f[hbo]):
        return False

    return True

def make_delaunay(C, elen, phi=None, float_type=float,length_fun=euclidean_length): 
    flips = 0
    stk = []
    mark = [False for h in range(0,len(C.next_he))]
    for h in range(0,len(C.next_he)):
      if C.he2f[h] in C.bnd_loops or C.he2f[C.opp[h]] in C.bnd_loops:
        continue
      if h < C.opp[h]: # pick the smaller id halfedge
        stk.append(h)
        mark[h] = True
    while stk:
        hd = stk.pop(-1); ld = elen[hd]
        mark[hd] = False
        hb = C.next_he[hd]; ha = C.next_he[hb];
        hdo = C.opp[hd]; hao = C.next_he[hdo]; hbo = C.next_he[hao]; 
        la = elen[ha]; lb = elen[hb]; lao = elen[hao]; lbo = elen[hbo]
        ind, is_Del = Delaunay_test_tri_phi(C,elen,hd,phi)

        # Self adjacent triangles are always Delaunay
        if (hdo == ha) or (hdo == hb):
            is_Del = True

        if not is_Del:
            # FIXME
            if (C.next_he[hbo] != hdo):
                print(C.next_he[hbo], C.next_he[C.next_he[hbo]], hdo, hbo, hao)
                print("Valid?", is_valid_connectivity(C))
                assert False
       #     print( 'is Del before',  Delaunay_test_tri(elen[hd], elen[ha], elen[hb],  elen[hao],elen[hbo]))
       #     print( 'before:',elen[hd], elen[ha], elen[hb],  elen[hao],elen[hbo])
            # FIXME 
            assert check_flap(C, hd)
            regular_flip(C, hd, elen, length_fun=length_fun)
            assert check_flap(C, hd)
                #print(hd, ha, hb, hdo, hbo, hao)
                #print(ind)
                #print(elen[[hd, ha, hb, hdo, hbo, hao]])
                #la = scaled_length(elen[ha],*phi[[C.fr[ha],C.to[ha],C.fr[hd],C.to[hd]]])
                #lb = scaled_length(elen[hb],*phi[[C.fr[hb],C.to[hb],C.fr[hd],C.to[hd]]])
                #print(la, lb)
       #     print('after:',elen[hd], elen[ha], elen[hb],  elen[hao],elen[hbo])
       #     print( 'is Del after',  Delaunay_test_tri(elen[hd], elen[ha], elen[hao],  elen[hb],elen[hbo]))
            flips +=1
            ind_post, is_Del_post =  Delaunay_test_tri_phi(C,elen,hd,phi)
            if not is_Del_post:
                print("Delaunay test failed with pre and post inds:", ind, ind_post)
 #           assert check_tri_ineq(elen[hd], elen[ha], elen[hao])
 #           assert check_tri_ineq(elen[hd], elen[hb], elen[hbo])
            for hef in [ha, hb, hao, hbo]:
                if C.he2f[hef] in C.bnd_loops or C.he2f[C.opp[hef]] in C.bnd_loops:
                    continue
                if (hef < C.opp[hef] and not mark[hef]):
                    stk.append(hef)
                    mark[hef] = True
                elif (hef > C.opp[hef] and not mark[C.opp[hef]]):
                    stk.append(C.opp[hef])
                    mark[C.opp[hef]] = True
    return flips

def test_make_delaunay():
    mp.dps = 50
    next_he = np.array([1,2,0,4,5,3]);
    opp = np.array([-1,3,-1,1,-1,-1]);
    n_he = len(next_he)
    next_he, opp = build_boundary_loops(next_he, opp)
    bnd_loops = bnd_face_list(next_he, n_he)
    C = NOB_to_connectivity(next_he,opp,bnd_loops)
    l = np.array([1, msqrt(3.0), 1, msqrt(3.0), 1, 1, 1, 1, 1, 1], dtype=mp.mpf)
    l = nparray_from_float64(l, mp.mpf)
    for idx in range(len(l)):
      if l[idx] != 0.0:
        l[C.opp[idx]] = l[idx]
    make_delaunay(C, l, mp.mpf)
    for idx in range(len(l)):
      print(idx, l[idx])
