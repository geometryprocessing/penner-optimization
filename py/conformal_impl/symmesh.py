# Symmetry structure on a halfedge mesh 
#
# Functions in this file deal with connectivity only, with the execption of flips that 
#  call functions recomputing lengths defined (currently) in delaunay.py 
#  (these functions should be passed in as args instead)
#
#  Main functions:
#
#  NOB_to_double(next_he_in,opp_in,bnd_loops_in)  from NOB, create a halfedge for a double and reflection 
#          map structure  as defined below 
# 
#  is_valid_symmetry(C,R)   check if a reflection map structure is valid as defined below
#
#  Flip functions for all configurations: 
#  regular_flip(C,R,h, elen=None, swap=False)
#  flip_111(C,R,h,elen=None)
#  flip_TPT(C,R, h,elen=None)
#  flip_11T(C,R, h, elen=None)
#  flip_TPQ(C,R, h, elen=None)
#  flip_11Q(C,R, h,elen=None)
#  flip_QPQ(C,R, h,elen=None)
#
#  get_central_he_and_triple_type(C,R,he)   for a symmetric config defined by he, determine a unique central he, among he R(he), O(he) and R(O(he))
#   in a way  the flip operations assume (defined precisely in function description)
#   this definition is triple-dependent, i.e. if the triple type defined by the 
#   half-edge changes it may change
#
#  is_responsible(C,R,he)    is a halfedge responsible for the symmetric config 
#       this definition only depends on the halfedge label and index 
#       not on the triple unlike central halfedge above; 
#       this is used in the Delaunay function to ensure that edges are not 
#       placed on stack twice
#
#   Example of the difference of central and responsible: 
#   Suppose an edge h corresponds to a triple 111; then there is no way 
#    to distinguish  (locally)  h or opp(h) using labels and reflections.
#   So an arbitrary choice has to be made. 
#   As a result of flipping other edges, the triple may become 11T
#   In this case, there is a canonical choice defined by the 
#   get_central_he_and_triple_type function, which may be different
#   While is_responsible does not change, whether the edge is central or not does.
#   For this reason, different functions are used for orienting a triple 
#   canonically and using halfedge as a unique triple id in Delaunay.. 
#
#  random_flips(C,R,N)    do N random sym flips on a mesh 
#
#  get_triple_counts(C,R)
#
#  triangulate_quads(C,R,lc) for a symmetric mesh convert all quads into triangles breaking symmetry
#
#  create_quads(C,R) inverse of triangulate_quads
# 
#  load_double(fname)  load a mesh from a file and create a symmetric doubled meshes
#
#
# Auxiliary functions 
#
# extend_labels_to_boundary_loops(C,R)
# reindex_vertices(C,R,new_index)
# 
#

# Reflectional symmetry map 
# Validity for all reflection maps on halfedges, faces and vertices: these need to be involutions.
#
# refl_he: (n_he): involution on the set of halfedges. Validity:  reverses N, i.e., P(R(h)) = R(N(h)), and 
# commutes with O:  O(R(h)) = R(O(h)).  This implies that it maps orbits of N and O to orbits, i.e., is 
# well-defined on edges and faces.  
#
# For vertices, the situation is a bit more complicated. Because the sequence of halfedges is reversed for each 
# face, if halfedge h points to vertex v, i.e., is in the orbit of C = P \circ O corresponding to v, 
# then R(h) points away from v, so O(h) points to v. From this, we see that the orbit corresponding to a vertex 
# R(v) is obtained as the set of edges O(R(h)), equivalently, R(O(h)) 
#
# The maps on vertices and faces are inferred from refl_he:
#
# refl (TODO rename refl_v): (n_v)  symmetry map on vertices. 
# Validity:  h \in v, where v is the orbit associated with v, iff O(R(h)) is in R(v); automatic, if the mesh 
# is valid, and refl_he is valid
# 
# refl_f: (n-f) symmetry map on faces. 
# Validity: h \in f  iff R(h) is in R(f)
#
# Labels
#
# The reflection maps implies (uniquely on connected components, but I did not write a proof) a partition of 
# the mesh into two components D1 and D2. For any connected component M_i of the mesh, D1 \cap M_i is connected.
# These components need to be represented explicitly for various operations

# D1, D2: vertices,halfedges,faces belonging to exactly one domain with the symmetry map R mapping D1 <-> D2
# for an edge on the boundary between D1 and D2, one halfedge gets label D1 and the other D2
# T:   triangular face f split between D1, D2, R(f) = f 
# Q:   quad face f split between D1 and D2  R(f) = f 
# P:   halfedge h "perpendicular" to the fixed set of the symmetry (i.e., boundary of M), R(h) = h, 
#      endpoints swapped (note: endpoints may coincide, but in a geometric realization two 
#      halves of the edge exchange places
# S:   vertex in the fixed set of the symmetry map

# TODO: maybe code to infer these from refl_he would be useful; currently these are assigned in the special case 
# treatment of mesh doubling, and are maintained by flip operaitons 
#
# TODO:  labels may be unnecessary as these are fully inferred from the symmetry map
#  it is impossible to tell D1 from D2 locally, but distinguishing the two for identifying the 
#  flip type is not necessary (need to verify).  These are only needed in the end to separate the two parts 
#  but even then, one can probably get away with local cuts on the symmetric faces, which creates a boundary 
#  between two components. 
#  
# etype: halfedge label, one of D1, D2, P, A.  
#   Validity: 
#     label(h) = Di <=> refl_he[h] != h 
#     label(h) = P ("perpendicular" to D1-D2 boundary) <=> refl_he[h] = h
# ftype: (n_f) face label, one of D1, D2, T, Q.   Note that boundary loops also have labels
#   Validity: 
#     label(f) = Di <=> refl_f(f) != f and label(h)=Di for all halfedges
#     label(f) = T  <=> refl_f[f]  = f, the face is a triangle and refl_he[h] in f for all halfedges
#     label(f) = Q  <=> same as T, but the face is a quad 
# vtype: (n_v) vertex labels, one of D1, D2, VS.
#    Validity: 
#     label(v) = Di <=> refl_v[v] != v; and for any halfedge h in v, label(h) = Di
#     label(v) = S  <=> refl_v[v]  = v, and for any halfedge h in v, opp[refl_he[h]] is in v.
#


import math
import random
import numpy as np
from collections import namedtuple
import igl
from conformal_impl.halfedge import *
from conformal_impl.overload_math import *


Reflection   = namedtuple('Reflection', 'refl refl_he refl_f vtype etype ftype')

# Labels for symmetric meshes;

LB_Type = namedtuple('LB_Type',['D1','D2','P','Q','T','S','R'])
# the order is used, so assigning numerical values
LB = LB_Type(1,2,3,4,5,6,7)
TripleLabel = ['A','1','2','P','Q','T','S','R']  # in a hacky way, 
# 0 is used to encode edges = (halfedge pairs) which do not have the same label for both 
# this is only used in the is_responsible function 

TriplePairs = { '111':'111','1A2':'TPT','TPT':'1A2','11T':'TPQ','11Tr':'QPT','TPQ':'11T','QPT':'11Tr','11Q':'QPQ','QPQ':'11Q'}
ValidTriples =  ['111','1A2','TPT','11T','11Tr','TPQ','QPT','11Q','QPQ']


def extend_labels_to_boundary_loops(C,R):
    # vertices are already labeled, only need halfedges and faces
    refl_he = np.append(R.refl_he,  np.zeros((len(C.next_he)-len(R.refl_he),),dtype=int))
    refl_f  = np.append(R.refl_f,   np.zeros((len(C.f2he   )-len(R.refl_f ),),dtype=int))
    etype   = np.append(R.etype,    np.zeros((len(C.next_he)-len(R.refl_he),),dtype=int))
    ftype   = np.append(R.ftype,    np.zeros((len(C.f2he   )-len(R.refl_f ),),dtype=int))

    faces = build_orbits(C.next_he)
    bnd_he = np.array([he for i in C.bnd_loops for he in faces[i]],dtype=int)
    # use the reflection of opp halfedge which is interior so has to be defined
    refl_he[bnd_he]     = C.opp[refl_he[C.opp[bnd_he]]]
    # the reflected face is the face of any of the reflected edges
    refl_f[C.bnd_loops] = C.he2f[refl_he[C.f2he[C.bnd_loops]]]
    #  types of 2 paired halfedges are the same
    etype[bnd_he]       = etype[C.opp[bnd_he]]
    ident_f = np.arange(0,len(C.f2he))
    # boundary loop labels bit tricky (although largely irrelevant - type of boundary loops is never used)
    # setting mostly to simplify validity tests
    # determine which boundary loops are symmetric 
    sym_bnd_faces = np.argwhere( (C.he2f[refl_he[C.f2he]] == ident_f)  & np.isin(ident_f, C.bnd_loops))
    sym_bnd_faces.reshape((len(sym_bnd_faces,)))
    non_sym_bnd_faces = np.setdiff1d(np.array(C.bnd_loops),sym_bnd_faces)
    ftype[sym_bnd_faces] = LB.T # !!!!!  arbitrary assignment, would be better to have separate label or perhaps use
    # for non-symmetric should be possible to infer from the halfedges
    # halfedges can be either Di or A, but an A halfedge cannot be a boundary one in 
    # a valid sym. mesh
    ftype[non_sym_bnd_faces] = etype[C.f2he[non_sym_bnd_faces]] 
    return Reflection(refl=R.refl,vtype=R.vtype,refl_f=refl_f,refl_he=refl_he,ftype=ftype,etype=etype)

#  Validation of a reflectional symmetry map  and element labels on halfedges, faces and vertices
#
# refl_he: (n_he): involution on the set of halfedges. Validity:  reverses N, i.e., P(R(h)) = R(N(h)), and 
# commutes with O:  O(R(h)) = R(O(h)).
#
# refl: (n_v)  symmetry map on vertices. 
# Validity:  h \in v, where v is the orbit associated with v, iff O(R(h)) is in R(v);
# 
# refl_f: (n_f) symmetry map on faces. 
# Validity: h \in f  iff R(h) is in R(f)
#
# Labels
#
# etype: halfedge label, one of D1, D2, P, A.  
#   Validity: 
#     label(h) = D1 or D2 <=> refl_he[h] != h
#     label(h) = P ("perpendicular" to D1-D2 boundary) <=> refl_he[h] = h
# ftype: (n_f) face label, one of D1, D2, T, Q.   Note that boundary loops also have labels
#   Validity: 
#     label(f) = D1 or D2 <=> refl_f(f) != f and label(h)=Di matching the face for all halfedges
#     label(f) = T  <=> refl_f[f] = f, the face is a triangle and refl_he[h] in f for all halfedges
#     label(f) = Q  <=> same as T, but the face is a quad 
# vtype: (n_v) vertex labels, one of D1, D2, S.
#    Validity: 
#     label(v) = Di => refl_v[v] != v; and for any halfedge h in v, label(h) = Di 
#     label(v) = S  => refl_v[v] = v, and for any halfedge h in v, opp[refl_he[h]] is in v
def is_valid_symmetry(C,R):
    ok = True
    ident_he = np.arange(0,len(C.next_he))
    ident_f  = np.arange(0,len(C.f2he))
    ident_v  = np.arange(0,len(C.out))
    # reversing N
    ok = ok and check_cond( (C.prev_he[R.refl_he] == R.refl_he[C.next_he]).all(),'P(R(h)) = R(N(h)) failed')
    # commuting with O
    ok = ok and check_cond( (C.opp[R.refl_he] == R.refl_he[C.opp]).all() , 'O(R(h)) = R(O(h)) failed')
    # h in v =>  O(R(h)) in R(v) 
    to_or_he = C.to[C.opp[R.refl_he]]
    ok = ok and check_cond( (R.refl[C.to[C.to!=-1]] == to_or_he[to_or_he != -1]).all(), 'h in v =>  O(R(h)) in R(v) failed')
    # h in f =>  R(h) in R(f)
    ok = ok and check_cond( (R.refl_f[C.he2f] == C.he2f[R.refl_he]).all(), 'h in f =>  R(h) in R(f) failed')
    # boundary loops symmetric to boundary loops
    ok = ok and check_cond( (np.isin(ident_f,C.bnd_loops) == np.isin(R.refl_f,C.bnd_loops)).all(), 
                           'f in bnd_loops => R(f) in bnd_loops failed') 
    # labels
    # below implications in the from  a => b rewritten as !a or b
    # labels of edges 
    # D1 and D2 are mapped to each other
    ok = ok and check_cond( (( R.etype == LB.D1) ==  (R.etype[R.refl_he] == LB.D2) ).all(),
                           'R(h) in D2 for h in D1 failed') 
    # obsolete: for halfedges with label A,  R(h) = opp(h)
    # ok = ok and check_cond( ((R.etype != LB.A ) | (R.refl_he == C.opp)).all(),'h in EA => R(h)=O(h) failed')
    # for halfedges with label P,  R(h) = h
    ok = ok and check_cond(((R.etype != LB.P ) | (R.refl_he == ident_he)).all(),'h in EP => R(h) =h failed')
    # labels of faces 
    # D1 and D2 are mapped to each other
    ok = ok and check_cond( ((R.ftype == LB.D1) ==  (R.ftype[R.refl_f] == LB.D2)).all(),
                           'f in D1 => R(f) in D2 failed')
    # T and Q are mapped to themselves
    ok = ok and check_cond(((R.ftype != LB.T ) | (R.refl_f  == ident_f)).all(), 'f in FT => R(f)=f failed')
    ok = ok and check_cond(((R.ftype != LB.Q ) | (R.refl_f  == ident_f)).all(), 'f in FQ => R(f)=f failed')
    
    # labels of vertices 
    # D1 and D2 are mapped to each other
    ok = ok and check_cond(((R.vtype == LB.D1) ==  (R.vtype[R.refl] == LB.D2)).all(),
                           'v in D1 => R(v) in D2 failed')
    # S are mapped to themselves
    ok = ok and check_cond(((R.vtype != LB.S ) | (R.refl  == ident_v)).all(),'v in FT => R(v)=v failed')    
    # halfedge-face consistency
    # obsolete: halfedges of faces of type Di are of type Di or A
#    ok = ok and check_cond(((R.ftype[C.he2f] != LB.D1) | np.isin(R.etype,[LB.D1,LB.A])).all(),
#    'he of a face in D1 is not in D1 or EA') 
#    ok = ok and check_cond(((R.ftype[C.he2f] != LB.D2) | np.isin(R.etype,[LB.D2,LB.A])).all(),
#    'he of a face in D2 is not in D2 or EA')
   
    # halfedges of faces of type Di are of type Di 
    ok = ok and check_cond(((R.ftype[C.he2f] != LB.D1) | (R.etype == LB.D1)).all(),
    'he of a face in D1 is not in D1') 
    ok = ok and check_cond(((R.ftype[C.he2f] != LB.D2) | (R.etype == LB.D2)).all(),
    'he of a face in D2 is not in D2')
    
    # halfedges of faces of type T or Q have symmetric halfedges in the same face
    ok = ok and check_cond(((R.ftype[C.he2f] != LB.T ) | (C.he2f[R.refl_he] == C.he2f) ).all(),
                           'for h in f in ET, R(h) is not in f')
    ok = ok and check_cond(((R.ftype[C.he2f] != LB.Q ) | (C.he2f[R.refl_he] == C.he2f) ).all(),
                           'for h in f, f in EQ, R(h) is not in f')
        
    # halfedge-vertex consistency
    ok = ok and check_cond(((R.vtype[C.to[C.to!=-1]] != LB.D1) | (np.isin(R.etype[C.to!=-1], [LB.D1,LB.P]))).all(),
            'he of a vertex in D1 is not in D1 or EP') 
    ok = ok and check_cond(((R.vtype[C.to[C.to!=-1]] != LB.D2) | (np.isin(R.etype[C.to!=-1], [LB.D2,LB.P]))).all(),
            'he of a vertex in D2 is not in D2 or EP')
    to_or_he = C.to[C.opp[R.refl_he]]
    ok = ok and check_cond(((R.vtype[C.to[C.to !=-1]] != LB.S)  | (to_or_he[to_or_he!=-1] == C.to[C.to !=-1])).all(),
            'for h in v, v in S, O(R(h)) is not in v')
    return ok
                 

# change vertex indices in a Connectivity and Reflection structures, update out, to, fr, refl_v, vtype
def reindex_vertices(C,R,new_index):
    out = C.out.copy()
    new_index = np.array(new_index,dtype=int)
    out[new_index] = C.out
    to = C.to.copy(); fr = C.fr.copy()
    to[C.to != -1] = new_index[C.to[C.to !=-1]]
    fr[C.fr != -1] = new_index[C.fr[C.fr !=-1]]
    refl_v = R.refl.copy()
    refl_v[new_index] = new_index[R.refl]
    vtype = R.vtype.copy()
    vtype[new_index] = R.vtype
    Cnew =  Connectivity(to=to, fr=fr, out=out,
            next_he=C.next_he, prev_he=C.prev_he, opp=C.opp, he2f=C.he2f,  f2he=C.f2he, bnd_loops=C.bnd_loops, 
                         quad_info=C.quad_info)
    Rnew = Reflection(refl=refl_v,vtype=vtype,
                refl_f=R.refl_f, refl_he=R.refl_he, ftype=R.ftype, etype=R.etype)
    return (Cnew, Rnew)

# Creates a doubled mesh with reflectional symmetry map and corresponding labels froma   mesh connectivity with (possibly empty) boundary, specified by next_he, opp, bnd_loops
# !!! assumes that the boundary loop halfedges are at the end of indices so boundary loops are at the end 
# without this assumption need more complex renumbering of halfedges, not worth the trouble for now
# as long as the input is generated by one of to_NOB functions, should be ok
# construct a mesh with double number of faces and glued to the original mesh along the boundary
# in:     next_he, opp, bnd_loops 
# return:  Connectivity and Reflection structures for the double mesh 

def NOB_to_double(next_he_in,opp_in,bnd_loops_in):
    
    # making sure that our assumption on the input is satisfied
    assert (np.array(bnd_loops_in) >= bnd_loops_in[0]).all(), "the input assumes that all boundary faces are at the end"
    n_f = bnd_loops_in[0]
    
    faces_in = build_orbits(next_he_in)

    bnd_loop_he = np.array([he for l in bnd_loops_in for he in faces_in[l]])
    # number of halfedges belonging to faces = number of the first boundary loop halfedge
    n_he_nb = np.min(bnd_loop_he)
    
    # truncated array excluding boundary loop halfedges, we do not need these for the double
    opp_nb   = opp_in[0:n_he_nb].copy()
    faces_nb = faces_in[0:bnd_loops_in[0]].copy()

    # the symmetry map simply reverses order of halfedges
    refl_he = np.array(list(range(2*n_he_nb-1,-1,-1)),dtype=int)
                     
    #  the faces as a consequence of the above symmetry map definition also reverse order
    faces_new = [[refl_he[faces_nb[n_f-i-1][len(faces_nb[n_f-i-1])-j-1]] 
                  for j in range(0,len(faces_nb[n_f-i-1]))] for i in range(0,n_f)] 
    faces_double = faces_nb + faces_new 
    # h.edges enumerated following face seq.
    he_double = np.array([he for f in faces_double for he in f],dtype=int) 
    next_he_double_faces  = [ f[1:]+[f[0]] for f in faces_double]
    next_he_double = (-1)*np.ones((2*n_he_nb,),dtype=int)
    next_he_double[he_double]  = np.array([he for f in next_he_double_faces for he in f])
    # rebuild faces in canonical order inferred from faces; it could not be determined until next_he is built 
    ordered_faces_double = build_orbits(next_he_double)
    # halfedges of faces enumerated sequentially
    he_double_ordered = np.array([he for f in ordered_faces_double for he in f],dtype=int) 
    # he2f is needed to use refl_he to define refl_f
    # for halfedges enumerated in face order, replace them by respective face indices
    face_he2f     = np.array([i for i,f in enumerate(ordered_faces_double) for he in f],dtype=int)
    he2f = face_he2f.copy()  
    # now build the he2f map
    he2f[he_double_ordered]  = face_he2f
    # reconstruct refl_f from refl_he
    refl_f = np.array([he2f[refl_he[f[0]]] for f in ordered_faces_double ],dtype=int)
    # 

    # *** opp for double  *** 
    bnd_he  =  np.argwhere(opp_nb >= n_he_nb)    # boundary face halfedges = paired with boundary loop halfedges
    inter_he = np.argwhere(opp_nb <  n_he_nb)    # interior halfedges = rest
    bnd_he_new = refl_he[bnd_he]                 # boundary face halfedges of the second half of double
   
    opp_nb[bnd_he] = bnd_he_new           # update op to link two sides of double
    # using  #opp(R(h)) = R(opp(h))
    opp_new = np.array( [refl_he[opp_nb[refl_he[he]]] for he in range(n_he_nb,2*n_he_nb)],dtype=int)
    opp_double = np.concatenate([opp_nb, opp_new])
    
    # there is no boundary in double 
    bnd_loops_double =[]
    
    # construct next_he, prev_he,  he2f, to, fr, f2he, out
    C = NOB_to_connectivity(next_he_double,opp_double,bnd_loops_double)
    
    n_v = len(C.out)    
    # set reflection map for vertices based on halfedges
    refl_v  = np.array(C.to[refl_he[C.out]])
    
    # ** face, edge and vertex labels
    ftype = np.array( [LB.D1]*n_f + [LB.D2]*n_f ) 
    
    etype = np.zeros((2*n_he_nb,),dtype=int)
    etype[0:n_he_nb] = LB.D1
    etype[n_he_nb:2*n_he_nb] = LB.D2
    #bnd_he = (etype[C.opp] != etype) # this includes halfedges on the boundary between two halfes of the double
    #etype[bnd_he] = LB.A
    etype[bnd_he] = LB.D1
    etype[bnd_he_new] = LB.D2

    vtype = np.zeros((n_v,),dtype=int)        
    # if a vertex is of type LB.Di, it has at least one halfedge of 
    # type LB.Di pointing at it and none of type Dj, j != i.  But it may also be of type S
    # initially all vertices are assigned type D1 or D2, if they happen to have an incident edge of this type; 
    # and then the ones that are mapped to themselves are set to S
    for he in range(0,2*n_he_nb):
        if etype[he] in {LB.D1,LB.D2}:
            vtype[C.to[he]] = etype[he] 
    vtype[(refl_v == np.arange(0,n_v))] = LB.S
    
    R = Reflection(refl=refl_v,refl_f=refl_f,refl_he=refl_he,vtype=vtype,ftype=ftype,etype=etype) 
    return (C,R)


# A flip for a single triangle pair sharing the halfedge h
# works only for 1,1,1 and 2,2,2 triples,  no labels or reflection maps need adjustment
# swap argument is needed to maintain validity of reflection maps when a pair of flips on symmetric triangle pairs 
# is done; there are two ways to assign indices to new halfedges and faces after the flip, swap selects one 
# done in place!!

def regular_flip(C,h, elen=None, swap=False,length_fun=None): 
    # notation
    hd  = h;        hb  = C.next_he[hd ]; ha  = C.next_he[hb ]
    hdo = C.opp[h]; hbo = C.prev_he[hdo]; hao = C.prev_he[hbo]
    va = C.to[ha];  vb = C.to[hd]; vg = C.to[hb]; vgo = C.fr[hbo]
    f = C.he2f[hd];  fo = C.he2f[hdo]
    # flip
    if swap:
        hd,hdo = hdo,hd
        f,fo   = fo,f
    build_face(C,[hd, ha, hao],[vg, va,vgo],f)
    build_face(C,[hdo,hbo,hb ],[vgo,vb,vg ],fo)
    if elen is not None:
        elen[hd] = elen[hdo] =length_fun(*elen[[hd,ha,hb,hao,hbo]])
    return [ha,hb,hao,hbo]



# perform flips on two symmetic pairs of triangles of types 111 and 222 
# done in place!!

def flip_111(C,R,h,elen=None,length_fun=None):
    hd1 = h
    f1 = C.he2f[hd1];  f1o = C.he2f[C.opp[hd1]]
    hd2 = C.opp[R.refl_he[hd1]]
    f2 = C.he2f[hd2];  f2o = C.he2f[C.opp[hd2]]
    assert R.etype[hd1] == LB.D1 and  R.ftype[f1] == LB.D1 and R.ftype[f1] == LB.D1,"triple is not 111-222"
    assert R.etype[hd2] == LB.D2 and  R.ftype[f2] == LB.D2 and R.ftype[f2] == LB.D2,"triple is not 111-222"
    inv1 = regular_flip(C,hd1,elen, length_fun=length_fun)
    inv2 = regular_flip(C,hd2,elen,swap=True,length_fun=length_fun) 
    if elen is not None:
      len_avg = (elen[hd1] + elen[hd2])/2
      elen[hd1] = len_avg; elen[C.opp[hd1]] = len_avg
      elen[hd2] = len_avg; elen[C.opp[hd2]] = len_avg
    return inv1+inv2

# perform a flip on a pair of triangles of types 1A2 (inverse flip_TPT)
# done in place!!

def flip_1A2(C,R, h,elen=None,length_fun=None):
    # notation
    hd1 =       h ; hb1 = C.next_he[hd1]; ha1 = C.next_he[hb1]
    hd2 = C.opp[h]; hb2 = C.prev_he[hd2]; ha2 = C.prev_he[hb2]
    va  = C.to[ha1]; vg2 = C.to[ha2]; vb = C.to[hb2]; vg1 = C.to[hb1]
    f0 = C.he2f[hd1]; f1 = C.he2f[hd2]
    assert R.etype[hd1] != R.etype[hd2] and R.ftype[f0] == LB.D1 and R.ftype[f1] == LB.D2, "triple not 1A2"

    # flip
    build_face(C,[hd1,ha1,ha2],[vg1,va,vg2],f0) 
    build_face(C,[hd2,hb2,hb1],[vg2,vb,vg1],f1)
    
    R.refl_he[hd1] = hd1;  R.refl_he[hd2] = hd2 
    R.etype  [hd1] = LB.P; R.etype  [hd2] = LB.P
    R.refl_f  [f0] = f0;   R.refl_f  [f1] = f1     
    R.ftype   [f0] = LB.T; R.ftype   [f1] = LB.T  
    if elen is not None:
        elen[hd1] = elen[C.opp[hd1]]  = length_fun(*elen[ [hd1,ha1,hb1,ha2,hb2]])
    return [ha1,hb1,ha2,hb2]



# perform a flip on a pair of triangles of types TPT (inverse flip_1A2)
# done in place!!

def flip_TPT(C,R, h,elen=None,length_fun=None):  
    # notation
    hd1 =       h ; ha1 = C.next_he[hd1]; ha2 = C.prev_he[hd1] #f0,
    hd2 = C.opp[h]; hb1 = C.prev_he[hd2]; hb2 = C.next_he[hd2] #f1 
    va  = C.to[ha1]; vg2 = C.to[ha2]; vb = C.to[hb2]; vg1 = C.to[hb1] 
    f0 = C.he2f[hd1]; f1 = C.he2f[hd2]
    assert R.etype[hd1]==LB.P and R.ftype[f0] == LB.T and R.ftype[f1] == LB.T, "triple not TPT"
    assert R.etype[ha1] == LB.D1,"wrong triple orientation  ha1 type"+TripleLabel[R.etype[ha1]]
    # flip
    build_face(C,[hd1,hb1,ha1],[vb ,vg1,va],f0) 
    build_face(C,[hd2,ha2,hb2],[va ,vg2,vb],f1)
    
    R.refl_he[hd1] = hd2;   R.refl_he[hd2] = hd1 
    R.etype  [hd1] = LB.D1; R.etype  [hd2] = LB.D2
    R.refl_f  [f0] = f1;    R.refl_f  [f1] = f0
    R.ftype   [f0] = LB.D1; R.ftype   [f1] = LB.D2; 
    if elen is not None: 
        elen[hd1] = elen[C.opp[hd1]]  = length_fun(*elen[[hd1,ha1,hb1,ha2,hb2]])
    return [ha1,hb1,ha2,hb2] 

# perform a flip on a pair of sym triangles D1,D2 + quad, type 11T (inverse flip_TPQ)
# done in place!!
#
# This handles two configurations, 11T and 11Tr; if D1 is up, D2 is down, 
# then 11T has triangle on the left and 11Tr on the right
# We rotate it in this case by switching to R(h) so that D2 is up, 
# and triangle is always on the left.

def flip_11T(C,R, h, elen=None,length_fun=None):
    if R.etype[C.next_he[C.opp[h]]] == LB.P:
        h = R.refl_he[h]
    # notation
    hd1 = h; hd2 = R.refl_he[h]
    assert (R.etype[hd1]== R.ftype[C.he2f[hd1]] and  R.ftype[C.he2f[C.opp[hd1]]] == LB.T), "triple not 11T"
    lb1 = R.etype[hd1]
    lb2 = 3- lb1
    hb1 = C.next_he[hd1]; ha1 = C.next_he[hb1]  #f0
    hb2 = C.prev_he[hd2]; ha2 = C.prev_he[hb2]  #f1
    hd1o = C.opp[hd1];  hd2o = C.opp[hd2];    hp2 = C.next_he[hd2o] #f2
    va  = C.to[ha1]; vg2 = C.to[ha2]; vb2 = C.to[hb2]; vb1 = C.to[hp2]; vg1 = C.to[hb1]
    f0 = C.he2f[hd1]; f1 = C.he2f[hd2]; f2 = C.he2f[hd1o]
    # reserved
    build_face(C,[hd2,hd2o],[-1,-1],f2) 
    R.refl_he[hd2] = hd2;  R.refl_he[hd2o] = hd2o;   
    R.refl_f[f2] = f2;     R.ftype[f2] = LB.T 
    # flip
    build_face(C,[hd1,ha1,ha2]     ,[vg1,va,vg2]     ,f0) 
    build_face(C,[hd1o,hb2,hp2,hb1],[vg2,vb2,vb1,vg1],f1) #quad
    C.quad_info[f1,:] = [f2,hd2,hd2o]
   
    R.refl_he[hd1] = hd1;  R.refl_he[hd1o] = hd1o;  
    R.etype  [hd1] = LB.P; R.etype  [hd1o] = LB.P
    R.etype  [hd2] = LB.P; R.etype  [hd2o] = LB.P 
    R.refl_f  [f0] = f0;   R.refl_f   [f1] = f1;
    R.ftype   [f0] = LB.T; R.ftype    [f1] = LB.Q;  
    if elen is not None:
        elen[hd1],elen[hd2]  = length_fun(*elen[[hd1,ha1,hb1,hp2]])
        elen[C.opp[hd1]] = elen[hd1];  elen[C.opp[hd2]] = elen[hd2]
    return [ha1,hb1,ha2,hb2,hp2]


# perform a flip on a pair of a triangle and quad type TPQ (inverse flip_11T)
# done in place!!
#
# This case requires special handling because there are actually two possible configurations,
# if D1 triangle is up, and D2 is down, T may be on the left of Q or on the right. 
# in this case, we switch the starting halfedge, which is equivalent to rotating
# the configuraton so that D2 is up, and triangle is always on the left.


def flip_TPQ(C,R, h, elen=None,length_fun=None):
    # orients the configuration so that trianle is "on the left"
    if R.ftype[C.he2f[h]] == LB.Q:
        h = C.opp[h]
    # notation
    hd1 = h; ha1 = C.next_he[hd1]; ha2 = C.next_he[ha1] #f0
    hd1o = C.opp[h]; hb1 = C.prev_he[hd1o]; hb2 = C.next_he[hd1o]; hp2 = C.next_he[hb2] #f1
    va  = C.to[ha1]; vg2 = C.to[ha2]; vb2 = C.to[hb2]; vb1 = C.to[hp2]; vg1 = C.to[hb1]
    f0 = C.he2f[hd1]; f1 = C.he2f[hd1o]
    f2,hd2,hd2o = C.quad_info[f1,:]
    C.quad_info[f1,:] = [-1,-1,-1] 
    lb1 = R.etype[ha1] 
    lb2 = 3 - lb1
    assert R.etype[hd1]==LB.P and R.ftype[f0] == LB.T and R.ftype[f1] == LB.Q, "triple not TPQ"
    
    # flip
    build_face(C,[hd1, hb1,ha1], [vb1,vg1,va ],f0) 
    build_face(C,[hd2, ha2,hb2], [va ,vg2,vb2],f1)
    build_face(C,[hd2o,hp2,hd1o],[vb2,vb1,va ],f2)
    
    R.refl_he [hd1]  = hd2;  R.refl_he[hd1o] = hd2o;  
    R.etype   [hd1]  = lb1;  R.etype  [hd1o] = lb1;
    R.refl_he[hd2o]  = hd1o; R.refl_he [hd2] = hd1;
    R.etype   [hd2]  = lb2;  R.etype  [hd2o] = lb2;
    R.refl_f   [f0]  = f1;   R.refl_f   [f1] = f0;   R.refl_f[f2] = f2
    R.ftype    [f0]  = lb1;  R.ftype    [f1] = lb2;  R.ftype [f2]  = LB.T
    
    if elen is not None: 
        hq = hd2
        elen[hd1] = elen[hd2]  = length_fun(*elen[[hd1,ha1,hb1,hp2,hq]])
        elen[C.opp[hd1]] = elen[hd1]; elen[C.opp[hd2]] = elen[hd2]
    return [ha1,hb1,ha2,hb2,hp2] 



# perform a flip on two triangles of types D1,D2 and a quad, type 11Q (inverse flip_QPQ)
# done in place!!

def flip_11Q(C,R, h,elen=None,length_fun=None):
    # notation
    hd1 =           h;   hb1 = C.next_he[hd1]; ha1 = C.next_he[hb1] #f0 triangle
    hd2 = R.refl_he[h];  hb2 = C.prev_he[hd2]; ha2 = C.prev_he[hb2] #f1 triangle
    hd1o = C.opp[hd1];   hd2o = C.opp[hd2];    hp2 = C.next_he[hd2o]; hp1 = C.next_he[hd1o]  #f2 quad
    va1  = C.to[ha1]; va2 = C.to[hp1]; vg2 = C.to[ha2]; vb2 = C.to[hb2]; vb1 = C.to[hp2]; vg1 = C.to[hb1]
    
    f0 = C.he2f[hd1]; f1 = C.he2f[hd2]; f2 = C.he2f[hd1o]
    
    assert R.etype[hd1]==LB.D1 and R.ftype[f0] == LB.D1 and R.ftype[f2] == LB.Q, "triple not 11Q"
    
    # reserved face and edge, keeping maps consistent
    # needs to be done before the rest to make sure that disconn. edges are not used for out
    build_face(C,[hd2,hd2o],[-1,-1],f2)
    R.refl_he[hd2] = hd2;  R.refl_he[hd2o] = hd2o;  
    R.etype[hd2]   = LB.P; R.etype[hd2o]   = LB.P 
    R.ftype[f2] = LB.T
    # flip
    build_face(C,[hd1 ,ha1,hp1,ha2],[vg1,va1,va2,vg2],f0)  # quad
    C.quad_info[f0,:] = [f2,hd2,hd2o]
    build_face(C,[hd1o,hb2,hp2,hb1],[vg2,vb2,vb1,vg1],f1)  # quad
    C.quad_info[f1,:] = C.quad_info[f2,:]
    hq = C.quad_info[f1,1]
    C.quad_info[f2,:] = [-1,-1,-1]
    
    R.refl_he[hd1] = hd1;  R.refl_he[hd1o] = hd1o;  
    R.etype  [hd1] = LB.P; R.etype  [hd1o] = LB.P
    R.refl_f  [f0] = f0;   R.refl_f   [f1] = f1;   R.refl_f[f2] = f2;
    R.ftype   [f0] = LB.Q; R.ftype    [f1] = LB.Q; R.ftype[ f2]  = LB.T 
    if elen is not None: 
        elen[hd1],elen[hd2],elen[hq]  = length_fun(*elen[[hd1,ha1,hb1,hp1,hp2,hq]])
        elen[C.opp[hd1]] = elen[hd1];  elen[C.opp[hd2]] = elen[hd2]; elen[C.opp[hq]] = elen[hq]
    return  [ha1,hb1,ha2,hb2,hp1,hp2]

# perform a flip on two quads, type QPQ (inverse flip_11Q)
# done in place!!

def flip_QPQ(C,R, h,elen=None,length_fun=None):
    # notation
    assert R.etype[C.next_he[h]] == LB.D1,"wrong orient "+TripleLabel[R.etype[h]]
    hd1 =         h; ha1 = C.next_he[hd1];  hp1 = C.next_he[ha1];  ha2 = C.next_he[hp1]   #f0 quad
    hd1o = C.opp[h]; hb1 = C.prev_he[hd1o]; hp2 = C.prev_he[hb1];  hb2 = C.prev_he[hp2];  #f1 quad
    va1  = C.to[ha1]; va2 = C.to[hp1]; vg2 = C.to[ha2]; vb2 = C.to[hb2]; vb1 = C.to[hp2]; vg1 = C.to[hb1]
    f0 = C.he2f[hd1]; f1 = C.he2f[hd1o]
    assert R.etype[hd1]==LB.P and R.ftype[f0] == LB.Q and R.ftype[f1] == LB.Q, "triple not QPQ"
    # retrieve reserved face and edge for adding a face
    f2,hd2,hd2o = C.quad_info[f0,:]; 
    C.quad_info[f0,:] = [-1,-1,-1] 
    # flips
    build_face(C,[hd1, hb1,ha1],     [vb1,vg1,va1]    ,f0) 
    build_face(C,[hd2, ha2,hb2],     [va2,vg2,vb2]    ,f1)
    build_face(C,[hd2o,hp2,hd1o,hp1],[vb2,vb1,va1,va2],f2)  # quad
    C.quad_info[f2,:] = C.quad_info[f1,:]
    hq = C.quad_info[f1,1]
    C.quad_info[f1,:] = [-1,-1,-1]
    lb1 = LB.D1 
    lb2 = LB.D2
    # symmetry and labels
    R.refl_he[hd1] = hd2;   R.refl_he[hd1o] = hd2o;   
    R.etype  [hd1] = lb1;   R.etype  [hd1o] = lb1;
    R.refl_he[hd2] = hd1;   R.refl_he[hd2o] = hd1o;
    R.etype  [hd2] = lb2;   R.etype  [hd2o] = lb2;
    R.refl_f  [f0] = f1;    R.refl_f   [f1] = f0;   R.refl_f[f2] = f2
    R.ftype   [f0] = lb1;   R.ftype    [f1] = lb2;  R.ftype [f2] = LB.Q
    
    if elen is not None: 
        elen[hd1],elen[hq]  = length_fun(*elen[[hd1,ha1,hb1,hp1,hp2,hd2,hq]])
        elen[C.opp[hd1]] = elen[hd1]; 
        elen[hd2] = elen[hd1];  elen[C.opp[hd2]] = elen[hd1];
        elen[C.opp[hq]] = elen[hq]
    return  [ha1,hb1,ha2,hb2,hp1,hp2]


# put it all together, map valid triple type to corresponding flip function
# flip_TPQ handles both TPQ and QPT, flip_1TT handles both 1TT and 1TTr
# the entries here for 11Tr and QPT are only for test cases
# get_central_and_triple function always reduces these to 11T and TPQ

FlipFun = {'111':flip_111, '1A2':flip_1A2, 'TPT':flip_TPT, '11T':flip_11T, '11Tr':flip_11T,'TPQ': flip_TPQ, 'QPT': flip_TPQ, '11Q':flip_11Q, 'QPQ':flip_QPQ}


#  Each symmetric flip configuration contains two (1A2, TPT, TPQ,QPQ) or four (111/222, 11T, 11Q)
#  diagonal halfedges, i.e. halfedges with faces on both sides included in the 
#  triple
#  Given any one of them, h,  the rest are recovered as O(h), R(h), R(O(h))
#  (if there are two, O(h) = R(h))
#  Flip functions are written with the assumption that a specific halfedge in this set is chosen, to extract correctly the local configuration without dealing 
#  with all possible reflections and label switches  
#
#  This function makes the following choices: 
#  Any invalid (nonflippable) configuration:  returns None
#  111+222:  returns one of the diagonal D1 halfedges of 111 
#  1A2, 11T,11Q:  returns the D1 diag halfedge belonging to face of type D1
#  TPT, TPQ,QPT, QPQ: returns the P diag halfedge pointing to D1, i.e. N(h) is D1
#  Note: For the 11T/TPQ case, there are in fact two pairs 
#  11T/TPQ and 11Tr/TPQ where 11Tr stands for 11T with D2 and D1 labels exchanged
#  or equivalently, with face T "pointing right" or "pointing left" if D1 is 
#  "up".
#
#  input: connectivity and reflection structure, halfedge
#  output: central halfedge  + triple label
#
def get_central_he_and_triple_type(C,R,he):
    # boundary edges are never flipped
    is_bnd= (C.he2f[he] in C.bnd_loops) and (C.he2f[C.opp[he]] not in C.bnd_loops)
    if is_bnd: 
        return None,None
    # pairs consisting of 1 face need not be flipped -- always Delauanay; TODO: update writeup
    if C.he2f[he] == C.he2f[C.opp[he]]: 
        return None,None

    # edge is of type D1 or D2:
    if R.etype[he]  <= LB.D2:
        f0label = R.ftype[C.he2f[he]];  f1label  = R.ftype[C.he2f[C.opp[he]]]
        #  XiY is always Delanuay if X,Y are T or Q
        if min(f0label,f1label) > LB.D2:
            return None,None
        #  This leaves one face Di halfedge Di, i=1,2 and the remaining face any     
        if R.etype[he] == LB.D2:
            he = R.refl_he[he]
        # now face and halfedge are D1, remaining face any
        # switch faces so that D1 is first; the result is one of 111, 1A2,11T,11Q   
        if R.ftype[C.he2f[he]] > R.ftype[C.he2f[C.opp[he]]]:
            he =  C.opp[he]
    
    # edge is of type P: can only happen if faces are of types T or Q  
    if R.etype[he] == LB.P:
        if R.etype[C.next_he[he]] == LB.D2:
            he = C.opp[he]
        # only for QPT
        if R.ftype[C.he2f[he]] > R.ftype[C.he2f[C.opp[he]]]:
            he = C.opp[he]

    # special handling of labels of edges in EA
    l_e  = R.etype[he] if (R.etype[he] == R.etype[C.opp[he]]) else 0
    triple = TripleLabel[R.ftype[C.he2f[he]]]+ TripleLabel[l_e]+ TripleLabel[R.ftype[C.he2f[C.opp[he]]]] 
    assert triple in ValidTriples, 'invalid triple label produced '+triple
    return he,triple
    

def is_responsible(C,R,he):
    return (R.etype[he] == LB.D1 and  R.etype[C.opp[he]] == LB.D2 )  or \
           (R.etype[he] == LB.D1 and  R.etype[C.opp[he]] == LB.D1   and he < C.opp[he])  or \
           (R.etype[he] == LB.P  and he < C.opp[he])

# perform N random symmetric flips on a symmetric mesh 
def random_flips(C,R,N):
    random.seed(91)
    for i in range(0,N):
        he = random.randrange(0,len(C.next_he))
        he_r,triple = get_central_he_and_triple_type(C,R,he)
        if he_r and triple in {'111','1A2','TPT','11T', 'TPQ','11Q','QPQ'}:
            FlipFun[triple](C,R,he_r)
        if not is_manifold(C) or not is_valid_symmetry(C,R):
            print('failed',i,h_r,triple)
        else:
            print(N,' flips preserve manifoldness/symmetry')
            return

# count symmetric triples of different types in a mesh 
def get_triple_counts(C,R):
    def he_to_label(C,R,h): 
        if R.etype[h] == R.etype[C.opp[h]]:
            return R.etype[h]
        else:
            return 0
    triple_types = []
    def norm_triple(h):
        fl1, fl2 = R.ftype[C.he2f[h]], R.ftype[C.he2f[C.opp[h]]]
        el = he_to_label(C,R,h)
        if min(fl1,fl2) == 2: 
            fl1 = 1 if fl1 == 2 else fl1 
            fl2 = 1 if fl2 == 2 else fl2
            el = 1 if el == 2 else el
        if min(fl1,fl2) >= 3:
            el = 1 if el == 2 else el
        return TripleLabel[min(fl1,fl2)]+TripleLabel[el] +TripleLabel[max(fl1,fl2)]
    triple_types = np.array([ norm_triple(h) for h in range(0,len(C.next_he))])
    return np.unique(triple_types,return_counts=True)

# convert all quads in a symmetric mesh to triangles, breaking symmetry
# as all quads are symmetric trapezoids, their diagonals are equal so both splits are equivalent
# this is used to compute the 
# this is *not* in place, creates a copy of the mesh 
def triangulate_quads(C,R,lc):
    Ctri = copy_namedtuple(C)
    Rtri = copy_namedtuple(R)
    for f in range(0,len(Ctri.f2he)):
        if R.ftype[f] == LB.Q: 
            fnew = Ctri.quad_info[f,0]
            dnew,dnewo = Ctri.quad_info[f,1:]
            Ctri.quad_info[f,:] = [-1,-1,-1]

            # Triangulated quad diagonals are labeled with Q instead of P
            assert(Rtri.etype[dnew] == LB.P)
            assert(Rtri.etype[dnewo] == LB.P)
            Rtri.etype[dnew] = LB.Q;
            Rtri.etype[dnewo] = LB.Q;

            h0 = Ctri.f2he[f]; h1 = Ctri.next_he[h0]; h2 = Ctri.next_he[h1]; h3 = Ctri.next_he[h2]
            v0 = Ctri.to[h0];  v1 = Ctri.to[h1];      v2 = Ctri.to[h2];      v3 = Ctri.to[h3]
            assert(Ctri.next_he[h3] == h0)    
            build_face(Ctri,[dnew, h0,h1],[v3,v0,v1], f)
            build_face(Ctri,[dnewo,h2,h3],[v1,v2,v3], fnew)
    return Ctri, Rtri

# Convert all triangulated quads in a symmetric mesh to actual quads. This is
# the inverse  of triangulate_quads, but unlike that method it is in place. 
def create_quads(C,R):
    # Update connectivity information for each quad
    for he in np.where(R.etype == 4)[0]:
        if C.opp[he] < he:
            continue
        hd1 = he; hd2 = C.opp[he]
        hb1 = C.next_he[hd1]; ha1 = C.next_he[hb1]  #f0
        hb2 = C.prev_he[hd2]; ha2 = C.prev_he[hb2]  #f1
        va1  = C.to[ha1]; va2 = C.to[ha2]; vb2 = C.to[hb2]; vb1 = C.to[hb1];
        f0 = C.he2f[hd2]; f1 = C.he2f[hd1];
        build_face(C,[hd1,hd2],[-1,-1],f0)
        build_face(C,[hb1,ha1,ha2,hb2], [vb1,va1,va2,vb2]     ,f1)
        C.quad_info[f1,:] = [f0,hd1,hd2]

        # Update face reflection map and label info
        R.ftype[f0] = LB.T
        R.ftype[f1] = LB.Q
        R.refl_f[f0] = C.he2f[R.refl_he[C.f2he[f0]]]
        R.refl_f[f1] = C.he2f[R.refl_he[C.f2he[f1]]]
        
        # Update diagonal label info
        R.etype[hd1] = LB.P
        R.etype[hd2] = LB.P
    return

#  Load a mesh from a file and 
#  create a doubled symmetric mesh
# 
def load_double(fname):
    v,f = igl.read_triangle_mesh(fname)
    next_he,opp,bnd_loops,vtx_reindex =FV_to_NOB(f)
    C = NOB_to_connectivity(next_he,opp,bnd_loops)
    return NOB_to_double(next_he,opp,bnd_loops)

# split function: given C and R build half of mesh (quads are also split to triangles)
# input: old C and old R - closed double mesh
# output: next_he_new, opp_new, bnd_loops: NOB of new mesh 
#         I: map from old halfedges to new halfedges
#         J: map from T and Q faces ids of new halfedges 
def build_split_NOB(C, R):
    n_he = len(C.he2f)
    n_f = len(C.f2he)
    
    # assign a new label for all halfedges
    I = [-1]*n_he
    reindex_he = 0 # total number of old halfedges
    for i in range(0, n_he):
        if C.to[i] == -1 or C.he2f[i] in C.bnd_loops: # by pass reserved edges and boundary loop edges
            continue
        if R.etype[i] == LB.D1 or R.etype[i] == LB.P:
            I[i] = reindex_he
            reindex_he += 1
    J = [[]]*n_f
    for f in range(0, n_f):
        # bypass the reserved faces
        if C.to[C.f2he[f]] == -1 or f in C.bnd_loops: 
            continue
        if R.ftype[f] == LB.Q: # quads have one new boundary halfedges and two new diagonal halfedges
            J[f] = [reindex_he, reindex_he+1, reindex_he+2]
            reindex_he += 3
        elif R.ftype[f] == LB.T:
            J[f] = [reindex_he]
            reindex_he += 1

    opp_new     = np.array([-1]*reindex_he, dtype=int)
    next_he_new = np.array([-1]*reindex_he, dtype=int)
    
    # set opp for all old edges's with new map
    for i in range(len(I)):
        if I[i] != -1:
            opp_new[I[i]] = I[C.opp[i]]
    
    # traverse the faces and assign opp_new and next_he_new
    # note: in this pass we only handle the old halfedges (D1 and P)
    for f in range(0, n_f):
        if R.ftype[f] == LB.D1:
            h0   = C.f2he[f]; h1   = C.next_he[h0]; h2   = C.next_he[h1]
            h0_o = C.opp[h0]; h1_o = C.opp[h1]    ; h2_o = C.opp[h2]
            next_he_new[I[h0]] = I[h1]; next_he_new[I[h1]] = I[h2]; next_he_new[I[h2]] = I[h0]
            
    # in this pass we handle the new halfedges (in T and Q)
    for f in range(0, n_f):
        # bypass the reserved faces
        if C.to[C.f2he[f]] == -1 or f in C.bnd_loops: 
            continue
        elif R.ftype[f] == LB.T or R.ftype[f] == LB.Q:
            # find a P edge
            hp = C.f2he[f]
            while True:
                if R.etype[hp] == LB.P:
                    break
                hp = C.next_he[hp]
            
            hm = J[f][0]; opp_new[hm] = -1; # new boundary edge has opp as -1
            if R.ftype[f] == LB.T:
                h_next = C.next_he[hp]; h_prev = C.prev_he[hp]; 
                if R.etype[h_next] == LB.D1:      # T = (P, D1, D2)
                    next_he_new[I[hp]]     = I[h_next]
                    next_he_new[I[h_next]] = hm
                    next_he_new[hm]        = I[hp]
                    opp_new[I[hp]]         = I[C.opp[hp]]
                    opp_new[I[h_next]]     = I[C.opp[h_next]]
                else: # T = (P, D2, D1)
                    next_he_new[I[hp]]     = hm
                    next_he_new[hm]        = I[h_prev]
                    next_he_new[I[h_prev]] = I[hp]
                    opp_new[I[hp]]         = I[C.opp[hp]]
                    opp_new[I[h_prev]]     = I[C.opp[h_prev]]
                
            elif R.ftype[f] == LB.Q:
                hg1 = J[f][1]; hg2 = J[f][2] # two diagonal halfedges - next_he[hm] = hg1
                h_next = C.next_he[hp]; h_prev = C.prev_he[hp]; ho = C.next_he[h_next]
                opp_new[hg1] = hg2; opp_new[hg2] = hg1
                if R.etype[h_next] == LB.D1:      # T = (P, D1, P', D2)
                    next_he_new[I[hp]]     = I[h_next]
                    next_he_new[I[h_next]] = hg2
                    next_he_new[hg2]       = I[hp]
                    next_he_new[hm]        = hg1
                    next_he_new[hg1]       = I[ho]
                    next_he_new[I[ho]]     = hm
                    opp_new[I[hp]]         = I[C.opp[hp]]
                    opp_new[I[ho]]         = I[C.opp[ho]]
                    opp_new[I[h_next]]     = I[C.opp[h_next]]
                    
                else:                             # T = (P, D2, P', D1)
                    next_he_new[I[hp]]     = hm
                    next_he_new[hm]        = hg1
                    next_he_new[hg1]       = I[hp]
                    next_he_new[hg2]       = I[ho]
                    next_he_new[I[ho]]     = I[h_prev]
                    next_he_new[I[h_prev]] = hg2
                    opp_new[I[hp]]         = I[C.opp[hp]]
                    opp_new[I[ho]]         = I[C.opp[ho]]
                    opp_new[I[h_prev]]     = I[C.opp[h_prev]]
    
    n_he = len(opp_new)
    next_he_new, opp_new = build_boundary_loops(next_he_new, opp_new)
    
    assert(is_valid_NO(next_he_new, opp_new))
    bnd_loops = bnd_face_list(next_he_new, n_he)

    return next_he_new, opp_new, bnd_loops, I, J

def split_symmesh(C, R, l, phi, Th_hat, label_local=[], s2f_local=[], float_type=float):
    next_he, opp, boundary_loops, I, J = build_split_NOB(C, R)
    F = build_orbits(next_he)
    n_tri = 0; n_quad = 0; n_other = 0
    for f in range(0, len(F)):
        if len(F[f]) == 3:
            n_tri += 1
        elif len(F[f]) == 4:
            n_quad += 1
        else:
            n_other += 1
    print("T: ", n_tri, ", Q: ", n_quad, ", Other", n_other)
    C_new = NOB_to_connectivity(next_he, opp, boundary_loops)
    print("genus and bd:", genus_and_boundaries_from_connectivity(C_new))
    print("is manifold: ",is_manifold(C_new))
    # update phis and Th_hat
    # - for all D1 edges, we copy both phis and Th_hat for both of its endpoints
    # - then for the new vertices (added on halfedge hp)
    #   if hp was on a boundary - Th_hat = PI/2
    #   if hp was not on a boundary - Th_hat = PI
    #   phi values should copy from phi[to[hp]]
    
    n_v = len(C_new.out)
    phi_new = np.array([float_type(0.0)]*n_v)
    Th_hat_new = np.array([float_type(0.0)]*n_v)
    label_local_new = np.array([-1]*n_v,dtype=int)
    is_P_vertex = np.array([False]*n_v,dtype=bool)
    s2f_local_new = np.array([-1]*n_v,dtype=int)
    
    for h in range(len(C.he2f)):
        # bypass reserved halfedges and boundary loop edges
        if C.to[h] == -1 or C.he2f[h] in C.bnd_loops: 
            continue
        elif R.etype[h] == LB.D1:
            v0 = C.fr[h]; v1 = C.to[h]
            phi_new[C_new.to[I[h]]] = phi[v1]
            phi_new[C_new.fr[I[h]]] = phi[v0]
            if len(label_local) != 0:
                label_local_new[C_new.to[I[h]]] = label_local[v1]
                label_local_new[C_new.fr[I[h]]] = label_local[v0]
            if len(s2f_local) != 0:
                s2f_local_new[C_new.to[I[h]]] = s2f_local[v1]
                s2f_local_new[C_new.fr[I[h]]] = s2f_local[v0]
            if R.vtype[v1] == LB.S:
                Th_hat_new[C_new.to[I[h]]] = Th_hat[v1]/2
            else:
                Th_hat_new[C_new.to[I[h]]] = Th_hat[v1]
            if R.vtype[v0] == LB.S: 
                Th_hat_new[C_new.fr[I[h]]] = Th_hat[v0]/2
            else:
                Th_hat_new[C_new.fr[I[h]]] = Th_hat[v0]
        elif R.etype[h] == LB.P:
            f = C.he2f[h]; hm = J[f][0]
            hp = h; hd = C.next_he[h]
            is_border = (C.he2f[C.opp[hp]] in C.bnd_loops) # hp is already on border in C
            if R.etype[hd] == LB.D1:   # T = (Ph, D1, D2); Q = (Ph, D1, P, D2)
                phi_new[C_new.to[hm]] = phi[C.to[hp]]
                is_P_vertex[C_new.to[hm]] = True
                if len(label_local) != 0:
                    label_local_new[C_new.to[hm]] = 0
                if len(s2f_local) != 0:
                    s2f_local_new[C_new.to[hm]] = -1
                if is_border:
                    Th_hat_new[C_new.to[hm]] = mpi(float_type)/2
                else:
                    Th_hat_new[C_new.to[hm]] = mpi(float_type)
            elif R.etype[hd] == LB.D2: # T = (Ph, D2, D1); Q = (Ph, D2, P, D1)
                phi_new[C_new.fr[hm]] = phi[C.to[hp]]
                is_P_vertex[C_new.fr[hm]] = True
                if len(label_local) != 0:
                  label_local_new[C_new.fr[hm]] = 0
                if len(s2f_local) != 0:
                    s2f_local_new[C_new.fr[hm]] = -1
                if is_border:
                    Th_hat_new[C_new.fr[hm]] = mpi(float_type)/2
                else:
                    Th_hat_new[C_new.fr[hm]] = mpi(float_type)

    # update lengths
    n_h = len(C_new.he2f)
    l_new = np.array([float_type(0.0)]*n_h)
    for i in range(0, len(C.next_he)):
        if I[i] != -1:
            if R.etype[i] == LB.P:  # assign half of length for 'P' edges got splitted
                l_new[I[i]] = l[i]/2
                l_new[C_new.opp[I[i]]] = l[i]/2
            else:                   # otherwise copy original value
                l_new[I[i]] = l[i]
                l_new[C_new.opp[I[i]]] = l[i]
    for f in range(0, len(C.f2he)):
        if C.to[C.f2he[f]] == -1 or f in C.bnd_loops:
            continue
        if R.ftype[f] == LB.T:
            hm = J[f][0]; hd1 = C_new.prev_he[hm]; hpn = C_new.next_he[hm]
            phi_va  = phi_new[C_new.to[hd1]]
            phi_vm  = phi_new[C_new.to[hm]]
            phi_vb1 = phi_new[C_new.to[hpn]]
            l_hd1 = l_new[hd1]#  * exp((phi_va + phi_vb1)/2)
            l_hpn = l_new[hpn] * mexp((phi_vm - phi_va)/2)
            if l_hpn > l_hd1: 
              l_new[hm] = msqrt(l_hpn**2-l_hd1**2)/mexp((phi_vm - phi_vb1)/2)
            else:
              l_new[hm] = msqrt(-l_hpn**2+l_hd1**2)/mexp((phi_vm - phi_vb1)/2)
            l_new[C_new.opp[hm]] = l_new[hm]
        elif R.ftype[f] == LB.Q:
            hm = J[f][0]; hg1 = J[f][1]; hg2 = J[f][2] 
            assert hg1 == C_new.next_he[hm]
            hp1n = C_new.next_he[hg1]; hd1 = C_new.prev_he[hg2]; hp2n = C_new.next_he[hg2]
            phi_vm1 = phi_new[C_new.fr[hm]]
            phi_vm2 = phi_new[C_new.to[hm]]
            phi_vb1 = phi_new[C_new.to[hp2n]]
            phi_va1 = phi_new[C_new.to[hg1]]
            assert phi_vm1 == phi_va1 and phi_vm2 == phi_vb1, "expect phi for midpoint equals to endpoints"
            lc_hp1n = l_new[hp1n] * mexp((phi_vm1-phi_vb1)/2)
            lc_hp2n = l_new[hp2n] * mexp((phi_vm2-phi_va1)/2)
            lc_hd1  = l_new[hd1] # * exp((phi_va1+phi_vb1)/2)
            lc_hm_sq  = lc_hd1**2-(lc_hp2n-lc_hp1n)**2
            lc_hm     = msqrt(lc_hm_sq)
            l_new[hm] = lc_hm /mexp((phi_vm1+phi_vm2-phi_vb1-phi_va1)/2)
            l_new[C_new.opp[hm]] = l_new[hm]
            l_new[hg1] = msqrt(lc_hm_sq + lc_hp1n**2)/mexp((phi_vm2-phi_vb1)/2)
            l_new[hg2] = l_new[hg1]
    return C_new, l_new, phi_new, Th_hat_new, label_local_new, is_P_vertex, s2f_local_new

def make_opp_sequential(C, l):
    # list of lists of face halfedges
    faces = build_orbits(C.next_he)
    # extract boundary halfedges
    bnd_he = [h for f in C.bnd_loops for h in faces[f]]
    # set faces for these to -1
    he2f = C.he2f.copy()
    he2f[bnd_he] = -1

    # renumbering faces
    fnew_to_old = np.array([-1]*len(faces))
    reindex = 0
    for f in range(len(faces)):
        if f not in C.bnd_loops:
            fnew_to_old[f] = reindex
            reindex += 1
#     print(fnew_to_old)
    
    # list of pairs of opp halfedges
    he_pairs = build_orbits(C.opp)
    # renumber sequentially, guarantees that  opp is sequential
    new_to_old = np.array([h for pair in he_pairs for h in pair], dtype=int)
    # inverse map
    old_to_new = new_to_old.copy()
    old_to_new[new_to_old] = np.arange(0,len(C.next_he))
    # update next, to, he2f, out, f2he
    next_he_new = old_to_new[C.next_he[new_to_old]]
    to_new = C.to[new_to_old]
    
    he2f_new = he2f.copy()
    he2f_new[old_to_new] = fnew_to_old[he2f]
    
    out_new = old_to_new[C.out]
    f2he_new = old_to_new[C.f2he]
    
    # exclude boundary loop faces
    f2he_new = np.array([f2he_new[f] for f in range(len(f2he_new)) if fnew_to_old[f] != -1])
    
    # per halfedge
    l_new = l[new_to_old]
    # skip every other one  -- corresponding to every other halfedge
    l_new = np.array(l_new[0:: 2])
    return next_he_new, to_new, he2f_new, out_new, f2he_new, l_new

# Performs an in place deep copy of B into A for a connectivity object 
def copy_in_place_connectivity(A,B): 
    if A == None or B == None: 
        return 
    A.next_he[:] = B.next_he 
    A.prev_he[:] = B.prev_he 
    A.opp[:] = B.opp 
    A.to[:] = B.to 
    A.fr[:] = B.fr 
    A.he2f[:] = B.he2f 
    A.out[:] = B.out 
    A.f2he[:] = B.f2he 
    A.bnd_loops[:] = B.bnd_loops[:] 
    A.quad_info[:] = B.quad_info[:] 
 
# Performs an in place deep copy of B into A for a reflection 
def copy_in_place_reflection(A,B): 
    if A == None or B == None: 
        return 
    A.refl[:] = B.refl 
    A.refl_he[:] = B.refl_he 
    A.refl_f[:] = B.refl_f 
    A.vtype[:] = B.vtype 
    A.etype[:] = B.etype 
    A.ftype[:] = B.ftype 

