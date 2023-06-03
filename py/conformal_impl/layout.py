from conformal_impl.conformal import *
import conformal_py as smpy
import igl
import csv
import sys
import pickle

def perp(a, float_type=float):
    b = np.array([float_type(0.0)]*2)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# top-level function given metric, produce u, v pairs assigned to endpoints of each halfedge
def layout(C, l, phi_0, is_cut_h=[], float_type=float, start_h=-1, neg_u=False):
    
    phi_0 = nparray_from_float64(phi_0, float_type)
    l     = nparray_from_float64(l,     float_type)
    
    n_h = len(C.next_he)
    n_v = len(C.out)
    n_f = len(C.f2he)
    
    u = np.array([float_type(0.0)]*n_h)
    v = np.array([float_type(0.0)]*n_h)

    xi = np.array([phi_0[C.to[i]]-phi_0[C.to[C.opp[i]]] for i in range(len(C.next_he))], dtype=float_type)
    
    cut_given = False
    cut_final = np.copy(is_cut_h)
    is_cut_h_gen = [False] * n_h # if no cut is given, layout function will produce a cut

    if len(is_cut_h) != 0:
        cut_given = True

    phi = np.array([float_type(0.0)]*n_h) # scale factor of the vertex each halfedge pointing to
    
    # set starting point - use a boundary edge
    if start_h == -1:
      for i in range(n_h):
          if C.he2f[C.opp[i]] in C.bnd_loops:
              h = C.next_he[C.next_he[i]]; 
              break
    else:
        h = C.prev_he[start_h]
    
    # set first endpoint
    u[h] = 0.0; v[h] = 0.0;
    phi[h] = 0.0;
    
    # set second endpoint
    h = C.next_he[h]
    assert(C.he2f[h] not in C.bnd_loops)
    phi[h] = xi[h];
    if neg_u:
        u[h] = -l[h]*mexp(phi[h]/2); 
    else:
        u[h] =  l[h]*mexp(phi[h]/2);
    v[h] = 0.0;
    done = [False] * n_f
    Q = [h] # queue
    done[C.he2f[h]] = True

    while Q:
        h = Q.pop(0)
        hn = C.next_he[h]; hp = C.next_he[hn];
        phi[hn] = phi[h] + xi[hn]
        p1 = np.array([float_type(u[hp]), float_type(v[hp])])
        p2 = np.array([float_type(u[h]) , float_type(v[h])])
        assert(l[h] != 0.0)
        l0 = float_type(1.0)
        l1 = mexp((phi[hn]-phi[hp])/2) * (l[hn]/l[h])
        l2 = mexp((phi[hn]-phi[h]) /2) * (l[hp]/l[h])
        pn = p1 + (p2-p1)*(1+(l2/l0)**2-(l1/l0)**2)/2 + perp(p2-p1,mp.mpf)*2*area_from_len(1,l1/l0,l2/l0)
 
        u[hn] = pn[0]; v[hn] = pn[1]
        hno = C.opp[hn]; hpo = C.opp[hp]; ho = C.opp[h]
        
        if C.he2f[hno] not in C.bnd_loops and not done[C.he2f[hno]] and not (cut_given and is_cut_h[hn]):
            done[C.he2f[hno]] = True;
            phi[hno] = phi[h];
            phi[C.next_he[C.next_he[hno]]] = phi[hn];
            u[hno] = u[h];
            v[hno] = v[h];
            u[C.next_he[C.next_he[hno]]] = u[hn];
            v[C.next_he[C.next_he[hno]]] = v[hn];
            Q.append(hno);
        else:
            is_cut_h_gen[hn] = True; is_cut_h_gen[C.opp[hn]] = True

        if C.he2f[hpo] not in C.bnd_loops and not done[C.he2f[hpo]] and not (cut_given and is_cut_h[hp]):
            done[C.he2f[hpo]] = True;
            phi[hpo] = phi[hn];
            phi[C.next_he[C.next_he[hpo]]] = phi[hp];
            u[hpo] = u[hn];
            v[hpo] = v[hn];
            u[C.next_he[C.next_he[hpo]]] = u[hp];
            v[C.next_he[C.next_he[hpo]]] = v[hp];
            Q.append(hpo);
        else:
            is_cut_h_gen[hp] = True; is_cut_h_gen[C.opp[hp]] = True
        
        if C.he2f[ho] not in C.bnd_loops and not done[C.he2f[ho]] and not (cut_given and is_cut_h[ho]):
            done[C.he2f[ho]] = True;
            phi[ho] = phi[hp];
            phi[C.next_he[C.next_he[ho]]] = phi[h];
            u[ho] = u[hp];
            v[ho] = v[hp];
            u[C.next_he[C.next_he[ho]]] = u[h];
            v[C.next_he[C.next_he[ho]]] = v[h];
            Q.append(ho);
        # else:
        #     is_cut_h_gen[ho] = True; is_cut_h_gen[C.opp[ho]] = True

    if not cut_given:
        cut_final = is_cut_h_gen
    return u, v, cut_final

# given mesh produces a bfs spanning tree with singularites at leaves
def bfs(C, Theta_hat, avoid_vt=[], non_branch_vt=[], float_type=float):
    pi = mpi(float_type)

    n_v = len(C.out)
    n_h = len(C.next_he)
    
    is_cut_h = [False] * n_h
    
    FE = build_orbits(C.next_he)
    bnd_v = np.array([C.to[h] for fi in C.bnd_loops for h in FE[fi]])
    # find a boundary vertex that's not a corner
    v = -1
    for i in range(len(bnd_v)):
        if Theta_hat[bnd_v[i]] != pi/2 and bnd_v[i] not in avoid_vt:
            v = bnd_v[i]
    assert(v != -1)
    root = v # need this info for the trim operation
    non_branch_vt = np.append(non_branch_vt,root)

    # propogate to all non-visited neighbors
    done = [False] * n_v
    Q = [v]; done[v] = True;
    while Q:
        vn = Q.pop(0)
        hn = C.out[vn];
        if vn in bnd_v:
            # if vn is a boundary vertex then we need to
            # further make sure opp[hn] is boundary edge
            while C.he2f[C.opp[hn]] not in C.bnd_loops:
                hn = C.next_he[C.opp[hn]]
        h0 = hn;
        n_children = 0
        while True:
            if vn in non_branch_vt and n_children == 1: # make sure the root is not branch point
              break
            vm = C.to[hn];
            if not done[vm]:
                done[vm] = True;
                if not vm in bnd_v: # do not connect to boundary
                    Q.append(vm)
                    n_children += 1
                    is_cut_h[hn] = True; is_cut_h[C.opp[hn]] = True;
            hn = C.opp[C.next_he[C.next_he[hn]]]
            if hn == h0 or C.he2f[hn] in C.bnd_loops:
                break
    # check whether all interior vertices are visited
    for i in range(len(C.out)):
        if i not in bnd_v:
            assert(done[i] == True)
    return is_cut_h, root

# given tree-like cutgraph - try to remove any degree-1 vertices that's not a cone/root
def trim_cuts(C, Th_hat_new, is_cut_h, root, float_type=float):
    pi = mpi(float_type)

    # get collection of cones
    bnd_v = np.array([C.to[h] for h in range(len(is_cut_h)) if C.he2f[h] in C.bnd_loops])
    cones = np.array([i for i in range(len(C.out)) if np.abs(Th_hat_new[i]-2*pi) > 1e-10 and i not in bnd_v])
    
    # repeatedly find non-singularity vertices with degree 1 on the cutgraph and remove the edge
    while True:
        degree = [0]*(2*len(C.out)) # this factor 2 is added in case for tufted cover
        for i in range(len(C.he2f)):
            v0 = C.to[i]; v1 = C.to[C.opp[i]]
            if v0 > v1 or not is_cut_h[i]: 
                continue;
            degree[v0] += 1; degree[v1] += 1
        found_trim = False
        for i in range(len(C.he2f)):
            v0 = C.to[i]; v1 = C.to[C.opp[i]]
            is_leaf_v0 = (degree[v0] == 1 and v0 not in cones and v0 != root)
            is_leaf_v1 = (degree[v1] == 1 and v1 not in cones and v1 != root)
            if is_cut_h[i] and (is_leaf_v0 or is_leaf_v1):
                degree[v1] -= 1; degree[v0] -= 1
                is_cut_h[i] = False; is_cut_h[C.opp[i]] = False
                found_trim = True
        if not found_trim:
            break

def apply_cuts(C, is_cut_h, u, v, ee_map = [], float_type=float):

    n_f = len(C.f2he)
    f_he = np.array([[C.f2he[i], C.next_he[C.f2he[i]], C.next_he[C.next_he[C.f2he[i]]]] \
                     for i in range(n_f) if i not in C.bnd_loops],dtype=int)
    F = np.array(C.to[f_he], dtype=int) # f*3 matrix with vertex ids
    is_cut = np.array(is_cut_h)
    cuts = np.array(is_cut[f_he], dtype=int) # change is_cut_h to matrix format
    cuts = np.array(cuts[:,[1,2,0]], order='C')
    
    EE = np.array([], dtype=int)

    n_v = len(C.out)

    # apply cut to F
    # prepare a dummy vertex matrix
    _V = np.array([[float(0.0)]*3]*n_v)
    F0 = F # save connectivity before cut
    print(_V.shape)
    print(F.max())
    TT, TTi = igl.triangle_triangle_adjacency(F0)
    _V, F = igl.cut_mesh(_V, F, cuts)
    print(_V.shape)
    print(F.max())
    uf = u[f_he]; vf = v[f_he]
    uv = np.array([[float_type(0.0)]*2]*len(_V))
    Vn_to_V = np.array([int(0)]*len(_V))
    # assign correct uv position for all vertices
    for i in range(len(F)):
        for k in range(3):
            uv[F[i,k]] = [uf[i,k], vf[i,k]]
            Vn_to_V[F[i,k]] = F0[i,k]
            he = f_he[i, k]; assert C.to[he] == F0[i,k]
            # if edge is cut_to_sing, we simply look up the vertex id in F
            if cuts[i][k] != 0: 
              v0 = F[i,k]; v1 = F[i,(k+1)%3];
              v2 = F[TT[i,k], TTi[i,k]]; v3 = F[TT[i,k], (TTi[i][k]+1)%3]
              if len(EE) == 0:
                EE = np.array([[v0, v1, v2, v3, -1, -1]], dtype=int)
              else:
                EE = np.concatenate([EE, np.array([[v0, v1, v2, v3, -1, -1]], dtype=int)])
            
            # if edge is straight cut, we cannot use TT, TTi (they are -1)
            if len(ee_map) != 0 and ee_map[he] != -1:
              v0 = F[i,(k+2)%3]; v1 = F[i,k]
              ho = ee_map[he]; f1 = C.he2f[ho]; k1 = -1
              # search for it in the f_he matrix
              for k in range(3):
                if f_he[f1][k] == ho:
                  k1 = k; break
              assert k1 != -1
              v2 = F[f1, (k1+2)%3]; v3 = F[f1, k1]
              if len(EE) == 0:
                EE = np.array([[v0, v1, v2, v3, 999, 999]], dtype=int)
              else:
                EE = np.concatenate([EE, np.array([[v0, v1, v2, v3, 999, 999]], dtype=int)])

    return uv, F, Vn_to_V, TT, TTi, cuts, EE
