# Connectivity structure (halfedge, fixed-size mesh) 
#   NOB = (N,O, boundary loops) minimal representation of halfedge described below
#
# Main functions:   
#  
# FV_to_NOB(F)  convert a list of faces, face = list of vertex ids, to  NOB structure
#
# NOB_to_FV(next_he,opp,bnd_loops,reindex=[], fr=[])) inverse (when possible, not for every NOB structure)
# 
# FE_to_NOB(F) convert a list of faces, face = list of edge ids, to NOB
# NOB_to_FE(next_he,opp, bnd_loops)  inverse, works always
#
# NOB_to_connectivity(next_he,opp,bnd_loops,quad_info=None)   build a complete halfedge structure (below) from NOB
#  
# NOB_connected_components(C)   find connected components of a mesh
#
# genus_and_boundaries_from_connectivity(C)  for each connected component, determine genus and number of boundaries
#
# is_valid_connectivity(C)    tests if a connectivity structure is valid, as defined below
#
#  Auxiliary functions
#
# general purpose: 
# build_orbits(perm)  build orbits of a permutation  
# compare_namedtuples(N1,N2)  compare namedtuples consisting of numpy arrrays
# copy_namedtuple(N)   copy namedtuples
#
# helpers for other functions
# build_boundary_loops(next_he,opp)
# bnd_face_list(next_he,n_he)
# normalize_f2he_and_out(C)
# is_valid_vert(next_he,opp,to,out,fr)
# is_valid_faces_and_bnd(next_he,opp,he2f,f2he,bnd_loops)#
# build_face(C,he_list,to_list, f)
#
#
# The only reason for (yet) another implementation is because many 
# existing implementtions such as cgal seem to impose restrictions that make it impossible to deal with arbitrary 
# topological tesselations of surfaces, e.g., faces glued to themselves. These are needed for Delaunay on 
# arbitrary PL surfaces. It is also easier to validate. 
#
# It is not designed to support any connectivity-changing operations that do not preserve
# the numbers of vertices faces (counting boundary loops as faces) and edges except in a very limited way
# described below. 
# 
#
# The conditions below describe validity fully (modulo any mistakes in the proof)
# i.e., any choice of index assignments to the arrays meeting these conditions produces a valid structure corresponding
# to an oriented manifold mesh (with a small technical exception related to "reserved" faces and edges explained below), 
# and conversely, any partition of an oriented manifold into disk-topology topological polygons can be represented in this way.
#
# n_he: number of haldgedges, including halfedges corresponding to boundary loops; n_he = 2*number of edges
# n_v:  number of vertices, needs to match the number of orbits in the circulator  P \circ O (below)
# n_f:  number of faces, needs to match the number of orbits in the next-halfedge permutation N
# for each field, its size is indicated in parentheses 
#
# The following fields define the connectivity uniquely and can be chosen arbitrarily subject to the following condition
# The rest can be reconstructed from these.
# next_he: (n_he) next halfedge map N; the only requirement is bijectivity of N  on 0..n_he-1
# opp:     (n_he) opposite-halfedge map O; bijective involution O^2 = I, with all orbits of length exactly 2; 
#                 in other words, paritions the set of halfedges into pairs
# bnd_loops: (size < n_f) list of face indices that correspond to boundary loops, to support meshes with boundaries
#           Validity: opposite halfeges cannot both belong to boundary loops, indices < n_f
#           This uses face indices inferred from next_he 
#
# The following fields are for indexing data associated with faces and vertices
# to:      (n_he) head vertex of a halfedge;  
#          Validity: 
#          (1) vertex indices are required to be continuous, i.e., if v_max is the maximal index, any v < v_max 
#          is presentas "to" for some halfedge. This is necessary for defining the "out" array
#          (2) all halfedges h in an orbit of P \circ O have the same value of to[h]
#          Note: if the connectivity is created from face=list of vertex indices representation, a reindexing array is 
#          produced, mapping new indices to old; continuity is not required for indices used as input. 
#          Ugly exception: for "reserved" halfedges, set to -1
#
# he2f:    (n_he) face of the halfedge. Validity similar to vertices:
#          (1) Face indices are required to form a continuous range 0..n_f-1, necessary for f2he
#          (2) all halfedges in an orbit of N have the same index
#
# out:     (n_v) a halfedge for which the vertex is the tail (using tail vs head for historical compatibility)
#          Can be any halfedge in the orbit of  P \circ O corresponding to the vertex
#          Validity: to[out[v]] = v
# f2he:    (n_f) a halfedge belonging to a face or boundary loop, which are represented as faces, can be any 
#          halfedge of the orbit corresponding to the face
#
# Convenience fields:
#
# fr:      (n_he) tail vertex of a halfedge  fr[h] = to[opp[h]]
# prev_he: (n_he) prev halfedge; inverse of N.
#
# Reserved faces and halfedges
# 
# (possibly should be moved to symmeshe)
# The structure is not suitable for general operations requiring adding/removing faces and edges
# However, maintaining valid Delaunay tesselation for reflectionally symmetric meshes (defined in symmesh.py 
# requires quad faces in restricted situations. While it is possible to represent these as pairs of triangles
# this makes managing symmetry maps more complicated and error-prone, because there are 
# inherently nonsymmetric edges in the structure.  
#
# The following hack is used, that keeps everything valid, as defined above,  except out/to arrays, for 
# which validity definition needs to be modified. 
#
# The only operations supported are:  
#  (1) two triangle faces -> one quad; one face and one halfedge pair are disconnected from the mesh
#  (2) one quad -> two triangle faces; one face and one halfedge pair are connected back to the mesh 
#  Note that this preserves the number of faces and halfedges
#  To maintain validity of the structure, the disconnected face and halfedges are connected 
#  into a separate 2-gon, glued to itself (i.e., a topological sphere)
#  This changes the number of components in the mesh, and would require creating new vertices for each such 
#  new component; to avoid this we allow (for reserved edges) setting 'to' to -1, i.e. no vertex
#
# Because halfedges and faces are removed/added only when a quad is created from/converted to a triangle
# we keep the disconnected faces and edges associated with the quads using an additional array
#
# quad_info: (n_f x 3)  -1's for triangular faces, (f,he1,he2) indices of associated disconnected face and two halfedges for quads
#
# operations destroying quads, create two triangles instead, one uses the face index of the quad, 
# and the other reserved index stored in the quad.  The new halfedges between two triangles gets the reserved halfedge
#

import math
import numpy as np
import scipy.sparse as sp
from collections import namedtuple

Connectivity = namedtuple('Connectivity','next_he prev_he opp to fr he2f out f2he bnd_loops quad_info')

def check_cond(cond,message):
    if not cond:
        print(message)
        return False
    else:
        return True
        
# in: perm a permutation  given by a list of non-repeating integers in the range 0..len(perm)
# returns: a list of lists, each list represents a cycle of perm
# the order of lists is in the order of the smallest halfedge index in each cycle, this is used in several places
def build_orbits(perm): 
    visited = [False]*(max(perm)+1)
    cycles = []
    for i in range(0,max(perm)+1):
        if not visited[i]:
            cycles.append([])
            i_it = i
            while True: 
                visited[i_it] = True
                cycles[-1].append(i_it)
                i_it = perm[i_it]
                if i_it == i:
                    break
    return cycles        

# in: opp: opp half-edge map, for boundary halfedges -1
#     next_he: next-halfedge map same length as opp
# returns: 
#     opp_ext:  opp with an extra coupled halfedge added at the end for each boundary halfedge
#               all halfedges have a pair 
#     next_he: next_halfedge map, same length as opp_ext; newly added halfedges are linked into boundary loops
def build_boundary_loops(next_he,opp):
    bnd_he = np.argwhere(opp == -1)
    bnd_he = bnd_he.reshape((len(bnd_he,)))
    n_bnd_he = len(bnd_he)
    n_he = len(opp)
    
    opp_ext = np.concatenate([opp, np.zeros((n_bnd_he,),dtype=int)])
    opp_ext = opp_ext.reshape((len(opp_ext),))
    opp_ext[bnd_he] = np.arange(n_he,n_he+n_bnd_he)
    opp_ext[np.arange(n_he,n_he+n_bnd_he)] = bnd_he

    next_he_ext = np.concatenate([next_he,(-1)*np.ones((n_bnd_he,),dtype=int)])
    for he in bnd_he: 
        he_it = next_he[he] 
        while opp[he_it] != -1: 
            he_it = next_he[opp[he_it]]
        next_he_ext[opp_ext[he_it]] = opp_ext[he]
    return(next_he_ext,opp_ext)

def bnd_face_list(next_he,n_he):
    faces = build_orbits(next_he)
    return [i for i in range(0,len(faces)) if faces[i][0] >= n_he]

# in:  Fv list of faces, face = list of vertices of a face, not including boundary loops
#         requirements: 
#         (1) no vertex is listed twice sequentially in a face (ambigous connectivity)
#         (2) a sequential pair (v1,v2) in a face is present no more than once anywhere.
#             
#
# returns: 
#      next_he:  list of lists of halfedges per face, including boundary loops
#      opp:  opposite halfedge, i.e. if opp[i] = j, and i = (v1,v2) in a face then  j = (v2,v1)
#            defined for all halfedges using extra boundary loop halfedges
#      bnd_loops: indices of boundary-loop faces
#      vtx_reindex: map from orbits of the circulator map of halfedges to the original vertex indices
#  TODO: this checks input validity, but I did not verify if fully


def FV_to_NOB(F): 
    # list of indices of halfedges per face, sequentially numbered, not including boundary-loop faces
    Fhe   = np.cumsum(np.array([len(F[i]) for i in range(0,len(F))],dtype=int))
    Fhe   = [list(range(0,Fhe[0]))] + [list(range(Fhe[i-1],Fhe[i])) for i in range(1,len(Fhe))] 

    # next-halfedge map (no boundary-loop halfedges yet) ***
    next_he =  [ [ Fhe[j][i] for i in range(1,len(Fhe[j]))  ] + [Fhe[j][0 ]] for j in range(0,len(F))]
    next_he =  np.array([ he  for f in next_he for he in f],dtype=int)
    n_he = len(next_he)    
    # adjacency matrix to build opp 
    # indices of tail and head vertices of halfedges
    tail  = np.array([ he  for f in F for he in f],dtype=int)
    head  = [ [ F[j][i] for i in range(1,len(F[j]))  ] + [F[j][0 ]] for j in range(0,len(F))]
    head =  np.array([ he  for f in head for he in f],dtype=int)
    
    n_he = len(tail)
    he_index = range(1,n_he+1) # shifting indices by 1 to distinguish from zero entries in the matrix
    # adjacency matrix (tail, head) -> halfedge index+1
    vv2he = sp.coo_matrix( (he_index,(tail,head)),dtype=int).tocsr()

    # validity checks
    # no repeated pairs of vertices
    M = vv2he.copy()
    M.sum_duplicates()
    assert M.nnz == vv2he.nnz, "duplicated halfedges, cannot construct adjacency"
    # nothing on the diagonal
    assert np.linalg.norm(M.diagonal()) == 0.0, "a halfedge connects vertex to itself, cannot construct adjacency"
    
    opp = np.zeros((n_he,),dtype=int)
    for he in range(0,n_he):
        opp[he] = vv2he[head[he],tail[he]]-1  # -1 to get correct index back, -1 = 0 in 
        # the matrix, i.e. no pair, boundary halfege

    # now add boundary loop halfedges
    next_he,opp = build_boundary_loops(next_he,opp)
    prev_he = (-1)*np.ones((len(next_he),),dtype=int)
    prev_he[next_he] = np.arange(0,len(next_he))
    circ = prev_he[opp]
    vert = build_orbits(circ)
    # look up head (vertex index from input array) based on the halfedge index of the first halfedge of the orbit
    # maps orbit indices to indices of input vertices
    vtx_reindex = np.array([ head[vert[i][0]] for i in range(0,len(vert))])
    # check for vertex non-manifoldness (multiple orbits get the same vertex)
    assert len(np.unique(vtx_reindex)) == len(vert), "multiple rings glued to a single vertex"
    bnd_loops = bnd_face_list(next_he,n_he)
    return (next_he,opp,bnd_loops,vtx_reindex)

# convert (next_he, opp, bnd_loops) mesh definition to a list of faces, each represented as a list of vertices 
# no checking is done here, so the result for e.g.,
# meshes with multiple edges between two vertices may not be wel-defined
#
# in: next_he, opp, bnd_loops: as defined above, components of Connectivity
#     (optional) reindexing of vertices to use, the indices in the order of orbits of P\circ O mapped to some other orde
#     (optional) fr: if we have this array, can pass it in, otherwise it is inferred
#  returns:  list of lists of vertices for each face


def NOB_to_FV(next_he,opp,bnd_loops,reindex=[], fr=[]):
    faces = build_orbits(next_he)
    if len(fr)==0:
        prev_he =(-1)*np.ones((len(next_he),),dtype=int)
        prev_he[next_he] = np.arange(0,len(next_he))
        verts = build_orbits(prev_he[opp])
        hind = [opp[he] for v in verts for he in v]
        vind = [ i  for  i in range(0,len(verts)) for j in verts[i]]
        fr = (-1)*np.ones(len(next_he,),dtype=int)
        fr[hind] = vind
    else:
        faces = [f for f in faces if (fr[f[0]] != -1)] # drop faces formed by reserved edges
    if len(reindex)>0: 
        return [reindex[fr[he]] for i in range(0,len(faces)) if (not i in bnd_loops) for he in faces[i]]
    else: 
         return [fr[he] for  i in range(0,len(faces)) if (not i in bnd_loops) for he in faces[i]]

# convert a mesh represented as a list of lists of edge ids for each faces to (next_he, opp, bnd_loops)
# unlike face = list-of-vertices representation, this can be used to store compactly any valid 2d oriented manifold
# polygonal complex
#
# in:  F: list of faces, face = list of edges of a face, not including boundary loops
#         each edge may be present a total of no more than twice (may be in the same face or a different one)
# returns: next_he, opp, bnd_loops, as defined in Connectivity
#
# TODO: Not tested much!! add reindexing as for vertices?
# 
def FE_to_NOB(F): 
    # list of indices of halfedges per face, sequentially numbered, not including boundary-loop faces
    Fhe   = np.cumsum(np.array([len(F[i]) for i in range(0,len(F))],dtype=int))
    Fhe   = [list(range(0,Fhe[0]))] + [list(range(Fhe[i-1],Fhe[i])) for i in range(1,len(Fhe))] 

    # next-halfedge map (no boundary-loop halfedges yet) ***
    next_he =  [ [ Fhe[j][i] for i in range(1,len(Fhe[j]))  ] + [Fhe[j][0 ]] for j in range(0,len(F))]
    next_he =  np.array([ he  for f in next_he for he in f],dtype=int)
    n_he = len(next_he)
    e2he = (-1)*np.ones((np.max(np.array(F)+1),2),dtype=int)
    FE_flat = np.array(F,dtype=int).flatten()
    # build an array  e -> (he1,he2), where he2 = -1, for edge ocurring once
    for he in range(0,n_he):
        if e2he[FE_flat[he],0] == -1:
            e2he[FE_flat[he],0] = he
        elif e2he[FE_flat[he],1] == -1:
            e2he[FE_flat[he],1] = he
        else:
            assert False, "3 or more ocurrences of the same edge"    
    # build opp with -1 for unpaired halfedges  
    opp = np.ones((n_he,),dtype=int)
    for e in range(0,len(e2he)):
        opp[e2he[e,0]] = e2he[e,1]
        if e2he[e,1] !=-1:
            opp[e2he[e,1]] = e2he[e,0]
            
    # now add boundary loop halfedges
    next_he,opp = build_boundary_loops(next_he,opp)
    
    faces = build_orbits(next_he)
    bnd_loops = [i for i in range(0,len(faces)) if faces[i][0] >= n_he]
 #   bnd_loops =[]    
    return(next_he,opp,bnd_loops)

# extract a face = list-of-edges representation from NOB; inverse of the FE_to_NOB
# in: next_he,opp, bnd_loops, as defined in Connectivity 
# returns: list of lists of edges for each face 
# TODO: not tested much !!
# 
def NOB_to_FE(next_he,opp, bnd_loops):
    faces = build_orbits(next_he)
    # indices of edges in the output are the indices of halfedges with the lower index in each pair
    reindex = (-1)*np.ones((len(next_he),),dtype=int)
    edge_cnt = 0
    for i in range(0,len(next_he)):
        if reindex[i] == -1:
            reindex[i] = edge_cnt
            reindex[opp[i]] = edge_cnt
            edge_cnt += 1
    return [ [reindex[h] if opp[h] > h else reindex[opp[h]] for h in faces[i]] for i in range(0,len(faces)) if not (i in bnd_loops)]

# extract connected components from a mesh 
# in: Connectivity structure
# returns: list of lists of indices of faces, one per component 
def NOB_connected_components(C):
    visited = [False]*len(C.f2he)
    component_faces = []
    reserved_f  = np.array([False]*len(C.f2he)) 
    reserved_f[C.quad_info[C.quad_info[:,0] != -1,0]] = True  
    reserved_f = set(reserved_f)
    for f in range(0,len(C.f2he)):
        if  not visited[f] and not f in reserved_f:
            stk = [f]
            component_faces.append([])
            while len(stk) > 0:
                fc = stk.pop(); 
                if not visited[fc]:
                    visited[fc] = True
                    component_faces[-1].append(fc)
                    he = C.f2he[fc]; he_it = he
                    while True: 
                        fn = C.he2f[C.opp[he_it]]
                        if not visited[fn]:
                            stk.append(fn)
                        he_it = C.next_he[he_it]
                        if he_it == he:
                            break
    return component_faces

# Compute the genus and the number of boundary loops for each connected component of the mesh
# in: C Connectivity structure 
# returns:  list of pairs (genus, number of boundary loops) for each component

def genus_and_boundaries_from_connectivity(C):
    faces = build_orbits(C.next_he)
    comp_faces = NOB_connected_components(C) # excludes reserved faces
    comp_halfedges = [ [he for f in c for he in faces[f]] for c in comp_faces]
    comp_vertices  = [ np.unique([C.to[he] for f in c for he in faces[f]]).tolist() for c in comp_faces]    
    comp_n_f =    [len(f) for f in comp_faces]
    comp_n_e =    [len(he)/2 for he in comp_halfedges]
    comp_n_v = [len(v) for v in comp_vertices]
    comp_bnd = [len(set(f).intersection(C.bnd_loops)) for f in comp_faces]
    return [ (int(1-(comp_n_f[i]+ comp_n_v[i] - comp_n_e[i])/2),comp_bnd[i]) for i in range(0,len(comp_faces))] 

# number of vertices of each valence in the mesh
def valence_counts(C):
    unique,counts = np.unique(
     np.array([len(np.nonzero(C.to==[i])[0]) for i in range(0,len(C.out))]),
     return_counts=True
    )
    return(np.vstack((unique,counts)).T)

# in: 
#      next_he, opp, bnd_loops as defined in Connectivity
# returns: 
#   Connectivity structure C, with fields above plus inferred from these: 
#   * he2f:    index of face for each halfedge (faces ordered as next_he orbits, i.e., by minimal halfedge index
#   * prev_he: inverse of next_he
#   * to:      index of the vertex halfedge is pointing to (vertices ordered as circulator orbits, i.e., by minimal index 
#            of the halfedge pointing from v
#   * fr:      tails of halfedges
#   * f2he:    an index of a halfedge of f (initially set to the min index halfedge in the face)
#   * out:     an index of a halfedge pointing from v (initially set to min index)


def NOB_to_connectivity(next_he,opp,bnd_loops,quad_info=None):    
    n_he = len(next_he)
    prev_he = (-1)*np.ones((n_he,),dtype=int)
    prev_he[next_he] = np.arange(0,n_he)
    faces = build_orbits(next_he) # including boundary loops 
    
    # maps face -> attached halfedges and faces
    if quad_info is None:
        quad_info  =(-1)*np.ones((len(faces),3),dtype=int)
    quad_info = np.array(quad_info,dtype=int) 
    reserved_he = np.array( [False]*n_he,dtype=bool)
    reserved_he[quad_info[quad_info[:,1] != -1,1]] = True  
    reserved_he[quad_info[quad_info[:,2] != -1,2]] = True  
    
    f2he = np.array([f[0] for f in faces],dtype=int)    
    he2f = np.array( [i for i in range(0,len(faces)) for j in faces[i]],dtype=int)
    Fhe  = np.array([he for f in faces for he in f],dtype=int)
    he2f[Fhe] = he2f.copy()
    
    circ = prev_he[opp]
    vert_all = build_orbits(circ)     # lists of halfedge pointing to each vertex  
    vert = [v for v in vert_all if not reserved_he[v[0]]] # skip vertices originating from reserved haldedges
    out  = np.array([next_he[v[0]] for v in vert],dtype=int) 
    to   = (-1)*np.ones((n_he,),dtype=int)
    vind   = np.array( [i for i in range(0,len(vert)) for j in vert[i]],dtype=int)
    vhe  = np.array([he for v in vert for he in v], dtype=int)
    to[vhe] = vind
    # tail (fr) vertex of a halfedge =  head vertex of opposite; works for boundaries because we have boundary loops
    fr = np.array([to[opp[i]] for i in range(0,n_he)])
    C = Connectivity(next_he=next_he, prev_he=prev_he, opp=opp, to=to, fr=fr, he2f=he2f, out=out, f2he=f2he, 
                     bnd_loops=bnd_loops, quad_info=quad_info)

    return C

#sets f2he and out to min halfedge indices of the face or vertex respectively
# in: C Connectivity
# result None
# Warning: modifies input in place
# 
def normalize_f2he_and_out(C):
    for i,he in enumerate(C.f2he):
        he_it = he 
        while True:
            he_it = C.next_he[he_it]
            if he_it < C.f2he[i]:
                C.f2he[i] = he_it 
            elif he_it == he:
                break
    for i,he in enumerate(C.out):
        he_it = he
        while True:
            he_it = C.next_he[C.opp[he_it]]
            if he_it < C.out[i]:
                C.out[i] = he_it 
            elif he_it == he:
                break        

# helper function for flips, modify connectivity to create a new face from existing halfedges indices
# and assign it an existing index
# in: 
#  C Connectivity structure
#  he_list:  a list of halfedge indices to form a face, so that next_he is cyclic on he_list
#  to_list:  the list of endpoints to set for each halfedge 
#  f: face index to use
# !!!! no attempt is made to verify that the operation produces a valid mesh 

def build_face(C,he_list,to_list, f):
    he_list_next = he_list[1:]+[he_list[0]]
    he_list_prev = [he_list[-1]]+he_list[0:-1]
    C.next_he[he_list] = he_list_next
    C.prev_he[he_list] = he_list_prev
    C.to[he_list] = to_list
    C.fr[he_list] = [to_list[-1]] + to_list[:-1]
    C.he2f[he_list] = f
    loc_next = he_list[1:]+[he_list[0]]
    C.out[[v for v in to_list if v != -1]] = [loc_next[i] for i,v in enumerate(to_list) if v != -1]    
    C.f2he[f] = he_list[0]


# check validity of the (next_he, opp) any valid combination defines an oriented manifold surface 
# next_he: (n_he) next halfedge map N; the only requirement is bijectivity of N  on 0..n_he-1
# opp:     (n_he) opposite-halfedge map O; bijective involution O^2 = I, with all orbits of length exactly 2; 
#                 in other words, paritions the set of halfedges into pairs
# in: next_he, opp 
# returns  True iff valid

def is_valid_NO(next_he,opp):
    prev_he = (-1)*np.ones((len(next_he),),dtype=int)
    prev_he[next_he] = np.arange(0,len(next_he))
    # all entries of prev_he were filled = next_he has len(next_he) distinct values in the range (0..n_he)
    ok = check_cond( (prev_he != -1).all(), 'not all halfedges are present in next_he')
    if not ok:
        print(next_he,opp,prev_he)
    # opp should be an involution this implies bijectivity
    ident = np.arange(0,len(opp))
    ok = ok and check_cond(len(np.argwhere(opp[opp] != ident)) == 0, 'opp^2 != id')
    # opp cannot have fixed points 
    ok = ok and check_cond(len(np.argwhere(opp == ident)) == 0, 'opp[h]=h for some h')
    return ok

# Validity of the vertex-related fields to,out,fr in Connectivity
# 
# to:      (n_he) head vertex of a halfedge;  
#          Validity: 
#          (1) vertex indices are required to be 0... n_v-1, with all indices used in 'to'.
#          This is necessary for defining the "out" array
#          (2) all halfedges in an orbit of O have the same index 
#          Note: if the connectivity is created from face=list of vertex indices representation, a reindexing array is 
#          produced, mapping new indices to old; continuity is not required for indices used as input. 
#          Ugly exception: for "reserved" halfedges, set to -1
# out:     (n_v) a halfedge for which the vertex is the tail (using tail vs head for historical compatibility)
#          Can be any halfedge in the orbit of  P \circ O corresponding to the vertex
#          Validity: to[out[v]] = v
# fr:      (n_he) tail vertex of a halfedge  fr[h] = to[opp[h]]
# 
# in: next_he, opp, to, out, fr 
# returns  True iff valid


def is_valid_vert(next_he,opp,to,out,fr):
    prev_he = (-1)*np.ones((len(next_he),),dtype=int)
    prev_he[next_he] = np.arange(0,len(next_he))
    circ = prev_he[opp]
    verts = build_orbits(circ)
    # skip vertices if disconnected halfedges
    verts = [vo for vo in verts if to[vo[0]] != -1]
    n_v = len(verts)
    # indices of vertices are  continous 0.. n_v
    # in to, skip -1's that are associated with disconnected edges in reserved_he
    ok = check_cond(np.max(to[to != -1]) == n_v-1 and len(np.unique(to[to != -1])) == n_v,
                    'number of vertices in "to" does not match the number of orbits')
    # verify there is a single index per set
    to_index_set_sizes = np.array([len(np.unique(to[v])) for v in verts])
    ok = ok and check_cond( (to_index_set_sizes == 1).all(),'multiple vertices per orbit of circulator')
    # check  out 
    ok = ok and check_cond( len(out) == n_v, '# of circulator orbits does not match number of vertices')
    ok = ok and check_cond( (to[opp[out]] == np.arange(0,n_v)).all(), 'out-halfedge tail is not vertex')
    if not ok:
        print(to[opp[out]])
    # check fr
    ok = ok and check_cond( (fr == to[opp]).all(), 'fr does not agree with to')
    return ok

# Verifies validity of face-relaged fields in Connectivity, he2f, f2he, bnd_loops 
#
# he2f:    (n_he) face of the halfedge. Validity similar to vertices:
#          (1) Face indices are required to form a continuous range 0..n_f-1, necessary for f2he
#          (2) all halfedges in an orbit of N have the same index
#
# f2he:    (n_f) a halfedge belonging to a face or boundary loop, which are represented as faces, can be any 
#          halfedge of the orbit corresponding to the face
#
# bnd_loops: (size < n_f) list of face indices that correspond to boundary loops, to support meshes with boundaries
#           Validity: opposite halfeges cannot both belong to boundary loops, indices < n_f
#
# returns True iff valid

def is_valid_faces_and_bnd(next_he,opp,he2f,f2he,bnd_loops):
    faces = build_orbits(next_he)
    n_f = len(faces)
    # indices of faces are  continous 0.. n_f
    ok = check_cond( np.max(he2f) == n_f-1 and len(np.unique(he2f)) == n_f, 'face indices are not continuous 0..n_f')
    # verify there is a single index per set
    he2f_index_set_sizes = np.array([len(np.unique(he2f[f])) for f in faces])
    ok = ok and check_cond( (he2f_index_set_sizes == 1).all(), 'more than one face per orbit of next_he')
    # check consistency between he2f and f2he 
    ok = ok and check_cond(len(f2he) == n_f,'number of faces does not match the number of orbits')
    ok = ok and check_cond( (he2f[f2he] == np.arange(0,n_f)).all(), 'f2he halfedge does not belong to the face')
    if len(bnd_loops) > 0:
        ok = ok and check_cond( max(bnd_loops) < n_f and len(bnd_loops) < n_f, 'boundary loop indices out of range or too many')
        # check if any two linked halfedges both belong to boundary loops
        ok = ok and check_cond( (~( np.isin(he2f,bnd_loops)  &  np.isin(he2f[opp],bnd_loops))).all(), 'two boundary loops share an edge')
    return ok

# Validates all components of a Connectivity structure using functions above
def is_valid_connectivity(C):
    ok = is_valid_NO(C.next_he,C.opp)
    ok = ok and is_valid_vert(C.next_he,C.opp,C.to,C.out,C.fr)
    ok = ok and  is_valid_faces_and_bnd(C.next_he,C.opp,C.he2f,C.f2he,C.bnd_loops)
    return ok

# alias;  any C defining a manifold is a valid connectivity structure and the other way around 
is_manifold = is_valid_connectivity

# compare two namedtuples with all components numerical arrays or lists
# for verification of flips
def compare_namedtuples(N1,N2): 
    if len(N1) != len(N2):
        return False
    else:
        return np.array([(np.array(N1[i]) == np.array(N2[i])).all() for i in range(0,len(N1))]).all()

# assumes copy defined for components of namedtuple
def copy_namedtuple(N): 
    return type(N)(*[x.copy() for x in N])

# input:  TT, TTi, F
# output: connectivity structure and map from new vertices to old
def TT_to_Connectivity(TT, TTi, F, L, float_type=float):
  nf = len(F)
  opp = np.array([-1]*3*nf, dtype=int)
  for f in range(nf):
      for i in range(3):
          if TT[f][i] != -1:
            opp[3*f+i] = 3*TT[f][i]+TTi[f][i]
  l = np.array([float_type(L[f][i]) for f in range(nf) for i in range(3)], dtype=float_type)
  # is_cut_h = np.array([cut_mask[f][i] for f in range(nf) for i in range(3)], dtype=bool)

  # build next_he
  Fsize = np.array([3]*nf, dtype=int)
  Fend = np.cumsum(Fsize)
  next_he = [ list(range(1,Fend[0]))+[0]]
  next_he = next_he + [ list(range(Fend[i-1]+1,Fend[i]))+[Fend[i-1]] for i in range(1,len(Fend))]
  next_he = np.array([he for f in next_he for he in f],dtype=int)

  n_he = len(next_he)
  next_he, opp = build_boundary_loops(next_he, opp)
  assert is_valid_NO(next_he, opp)

  bnd_loops = bnd_face_list(next_he, n_he)
  C = NOB_to_connectivity(next_he, opp, bnd_loops)

  nl = len(l)
  l = np.concatenate([l, np.zeros(len(C.next_he)-nl)])
  for i in range(len(C.next_he)-nl):
      idx = nl+i
      l[idx] = l[C.opp[idx]]
  
  #  halfedge to old vertex index of the tip
  to_old = np.array([F[f][(i+1)%3] for f in range(nf) for i in range(3)], dtype=int)
  assert(is_valid_connectivity(C))
  reindex = np.array(to_old[C.prev_he[C.out]], dtype=int)
  # print(genus_and_boundaries_from_connectivity(C))
  return C, reindex, l
