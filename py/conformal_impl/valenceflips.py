                 
from conformal_impl.halfedge import * 
from conformal_impl.symmesh import *


def flip_if_improves_valence(C,h,valences,trg_valences,max_val):    
    if C.opp[h] < h or C.he2f[h] in C.bnd_loops or C.he2f[C.opp[h]] in C.bnd_loops:
        return False
    if C.he2f[h] == C.he2f[C.opp[h]]:
        return False
    hl  = C.next_he[h]
    hru = C.next_he[C.opp[h]]
    vtx = [C.to[h], C.to[C.opp[h]], C.to[hl], C.to[hru]]

    tail = C.to[C.opp[h]]; tip = C.to[h]
    
    valence_flipped = valences.copy()
    valence_flipped[vtx]  = valences[vtx] + np.array([-1,-1,1,1])
    
    val_diff =         valences[vtx]        - trg_valences[vtx]
    val_diff_flipped = valence_flipped[vtx] - trg_valences[vtx]  
    if np.sum( val_diff*val_diff) > np.sum(val_diff_flipped*val_diff_flipped)  and (valence_flipped >= 1).all() and (valence_flipped <= max_val).all():
        inv = regular_flip(C,h)
        valences[:] = valence_flipped
        return True
    else:
        return False

def improve_valence(C, Th_hat): 
    valences = np.array([len(np.nonzero(C.to==[i])[0]) for i in range(0,len(C.out))])
    trg_valences = Th_hat/math.pi*3.0  
    val_diff = valences-trg_valences
    max_val = max(valences)
    print('init energy',np.sum( val_diff*val_diff), 'max_diff ',np.max(abs(val_diff)))
    n_he = len(C.to)
    iter = 0
    tot_flips = 0
    while iter < 10000:
        iter +=1
        num_flips = 0
        for h in range(0,n_he): 
                if flip_if_improves_valence(C,h,valences,trg_valences,max_val):
                    num_flips += 1
                    tot_flips +=1 
        if num_flips == 0: 
            print("No more flips to be done")
            break
    print(tot_flips)
    valences = np.array([len(np.nonzero(C.to==[i])[0]) for i in range(0,len(C.out))])
    val_diff = valences-trg_valences
    print('final energy',np.sum( val_diff*val_diff), 'max_diff ',np.max(abs(val_diff)))

# Find the outgoing halfedges from a vertex v of C, skipping boundary edges
def outgoing_edges(C, v):
    h = C.out[v]
    outgoing = []
    while True:
        if C.he2f[h] not in C.bnd_loops and C.he2f[C.opp[h]] not in C.bnd_loops and C.he2f[h] != C.he2f[C.opp[h]]:
            outgoing.append(h)
        h = C.next_he[C.opp[h]]
        if h == C.out[v]:
            return np.array(outgoing, dtype=int)

# Flip the halfedge h and update the valence information
def flip(C,h,valences):    
    if C.he2f[h] in C.bnd_loops or C.he2f[C.opp[h]] in C.bnd_loops or C.he2f[h] == C.he2f[C.opp[h]]:
        return
    hl  = C.next_he[h]
    hru = C.next_he[C.opp[h]]
    vtx = [C.to[h], C.to[C.opp[h]], C.to[hl], C.to[hru]]
    valences[vtx] = valences[vtx] + np.array([-1,-1,1,1])
    regular_flip(C,h)

# This method greedily tries to force down the vertices with a large error in
# the valence, forcing the error to zero and distributing the additional edges
# to neighboring vertices
def improve_valence_vertex_greedy(C, Th_hat):
    valences = np.array([len(np.nonzero(C.to==[i])[0]) for i in range(0,len(C.out))])
    trg_valences = Th_hat/math.pi*3.0  
    val_diff = valences-trg_valences
    max_val = max(valences)
    print('init energy',np.sum( val_diff*val_diff), 'max_diff ',np.max(abs(val_diff)))
    n_v = len(C.out)
    n_he = len(C.to)
    tot_flips = 0
    for _ in range(10):
        iter = 0
        while iter < 100:
            iter += 1    
            num_flips = 0
            
            # Iterate through all vertices sorted by the valence error
            val_diff = valences-trg_valences
            order = np.argsort(-val_diff)[:200]
            for v in order:
                v_edges = outgoing_edges(C, v)
                local_diffs = val_diff[C.to[v_edges]]

                # Greedily find edges to flip
                k = int(trg_valences[v])
                if k > len(local_diffs):
                    continue
                edges_to_flip = v_edges[np.argsort(local_diffs)[k:]]
                for h in edges_to_flip:
                    #flip_if_improves_valence(C,h,valences,trg_valences,max_val)
                    flip(C,h,valences)
        iter = 0
        tot_flips = 0
        max_val = max(valences)
        while iter < 100:
            iter +=1
            num_flips = 0
            for h in range(0,n_he): 
                    if flip_if_improves_valence(C,h,valences,trg_valences,max_val):
                        num_flips += 1
                        tot_flips +=1 
            if num_flips == 0: 
                break
        valences = np.array([len(np.nonzero(C.to==[i])[0]) for i in range(0,len(C.out))])
        val_diff = valences-trg_valences
        print('final energy',np.sum( val_diff*val_diff), 'max_diff ',np.max(abs(val_diff)))

# This method performs edge flips while also creating new vertices to distribute out the
# valence differences across more vertices
def improve_valence_vertex_split(C, Th_hat):
    valences = np.array([len(np.nonzero(C.to==[i])[0]) for i in range(0,len(C.out))])
    trg_valences = Th_hat/math.pi*3.0  
    val_diff = valences-trg_valences
    max_val = max(valences)
    print('init energy',np.sum( val_diff*val_diff), 'max_diff ',np.max(abs(val_diff)))
    n_v = len(C.out)
    n_he = len(C.to)
    tot_flips = 0
    for _ in range(3):
        iter = 0
        while iter < 5:
            iter += 1    
            num_flips = 0
            
            # Iterate through all vertices sorted by the valence error
            val_diff = valences-trg_valences
            order = np.argsort(-val_diff)[:5]
            for v in order:
                print(v)
                v_edges = outgoing_edges(C, v)
                local_diffs = val_diff[C.to[v_edges]]

                # Greedily find edges to flip
                k = int(trg_valences[v])
                if k > len(local_diffs):
                    continue
                edges_to_flip = v_edges[np.argsort(local_diffs)[k:]]
                for h in edges_to_flip:
                    #flip_if_improves_valence(C,h,valences,trg_valences,max_val)
                    #flip(C,h,valences)
                    Th_hat,C = edge_split(Th_hat,C,h)
                valences = np.array([len(np.nonzero(C.to==[i])[0]) for i in range(0,len(C.out))])
                trg_valences = Th_hat/math.pi*3.0  
                val_diff = valences - trg_valences
        iter = 0
        tot_flips = 0
        max_val = max(valences)
        while iter < 100:
            iter +=1
            num_flips = 0
            for h in range(0,n_he): 
                    if flip_if_improves_valence(C,h,valences,trg_valences,max_val):
                        num_flips += 1
                        tot_flips +=1 
            if num_flips == 0: 
                break
        valences = np.array([len(np.nonzero(C.to==[i])[0]) for i in range(0,len(C.out))])
        val_diff = valences-trg_valences
        print('final energy',np.sum( val_diff*val_diff), 'max_diff ',np.max(abs(val_diff)))
        print('Num vertices:', len(C.out))
    return C, Th_hat


def edge_split(v,C,he):
    n_v = v.shape[0]
    n_he = C.next_he.shape[0]
    n_f = C.f2he.shape[0]
    v1 = C.fr[he]
    v2 = C.to[he]
    #print('vertices', v1,v2)
    midpoint = 0.5*v[v1] + 0.5*v[v2]
    v = np.append(v,2*math.pi)
    # Get two edge flap triangles
    heo = C.opp[he]
    assert(C.opp[heo]==he)
    he1 = C.next_he[he]
    he2 = C.next_he[he1]
    heo1 = C.next_he[heo]
    heo2 = C.next_he[heo1]
    assert(C.next_he[he2] == he)
    assert(C.next_he[heo2] == heo)
    assert(C.prev_he[he] == he2)
    assert(C.prev_he[heo] == heo2)
    w = C.to[he1]
    wo = C.to[heo1]
    # Create new next_he and prev_he arrays
    next_he = np.concatenate((C.next_he,np.zeros((6,),dtype=int)))
    prev_he = np.concatenate((C.prev_he,np.zeros((6,),dtype=int)))
    # First triangle
    next_he[he] = prev_he[he2] = n_he
    next_he[n_he] = prev_he[he] = he2
    next_he[he2] = prev_he[n_he] = he
    # Second triangle
    next_he[n_he+1] = prev_he[he1] = n_he+2
    next_he[n_he+2] = prev_he[n_he+1] = he1
    next_he[he1] = prev_he[n_he+2] = n_he+1
    # Third triangle
    next_he[n_he+3] = prev_he[heo2] = n_he+4
    next_he[n_he+4] = prev_he[n_he+3] = heo2
    next_he[heo2] = prev_he[n_he+4] = n_he+3
    # Fourth triangle
    next_he[n_he+5] = prev_he[heo1] = heo
    next_he[heo] = prev_he[n_he+5] = heo1
    next_he[heo1] = prev_he[heo] = n_he+5
    # Create new opp array
    new_opp = n_he + np.array([1,0,3,2,5,4],dtype=int)
    opp = np.concatenate((C.opp,new_opp))
    # Create new to and fr arrays
    to = np.concatenate((C.to,np.zeros((6,),dtype=int)))
    fr = np.concatenate((C.fr,np.zeros((6,),dtype=int)))
    # First triangle
    fr[he] = to[he2] = v1
    fr[n_he] = to[he] = n_v
    fr[he2] = to[n_he] = w
    # Second triangle
    fr[n_he+1] = to[he1] = w
    fr[n_he+2] = to[n_he+1] = n_v
    fr[he1] = to[n_he+2] = v2
    # Third triangle
    fr[n_he+3] = to[heo2] = v2
    fr[n_he+4] = to[n_he+3] = n_v
    fr[heo2] = to[n_he+4] = wo
    # Fourth triangle
    fr[n_he+5] = to[heo1] = wo
    fr[heo] = to[n_he+5] = n_v
    fr[heo1] = to[heo] = v1
    f = C.he2f[he]
    assert(he in [C.f2he[f], C.next_he[C.f2he[f]], C.next_he[C.next_he[C.f2he[f]]]])
    fo = C.he2f[heo]
    assert(heo in [C.f2he[fo], C.next_he[C.f2he[fo]], C.next_he[C.next_he[C.f2he[fo]]]])
    new_he2f = np.array([f,n_f,n_f,n_f+1,n_f+1,fo],dtype=int)
    he2f = np.concatenate((C.he2f,new_he2f))
    he2f[he1] = n_f
    he2f[heo2] = n_f+1
    new_f2he = np.array([he1,heo2],dtype=int)
    #print(C.f2he)
    f2he = np.concatenate((C.f2he,new_f2he))
    f2he[f] = he
    f2he[fo] = heo
    #print(f2he,f,fo,he,heo)
    #print("cycle",next_he[f2he],next_he[next_he[f2he]])
    out = np.append(C.out,heo)
    out[v1] = he
    out[v2] = he1
    out[w] = he2
    out[wo] = heo2
    bnd_loops = C.bnd_loops
    quad_info  =(-1)*np.ones((len(f2he),3),dtype=int) 
    quad_info = np.array(quad_info,dtype=int)  
    C = Connectivity(next_he,
                     prev_he,
                     opp,
                     to,
                     fr,
                     he2f,
                     out,
                     f2he,
                     bnd_loops,
                     quad_info)
    return v,C
