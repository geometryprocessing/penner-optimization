import scipy
import scipy.sparse as sp
from conformal_impl.conformal import *
from conformal_impl.delaunay import *
from conformal_py import *
from optimization_py import *
import time

# *********************************************
# Below is deprecated code now rewritten in C++
# *********************************************

# FIXME There are conflicting definitions of this function
def check_tri_ineq(l1,l2,l3):
    return ( (np.array(l1+l2-l3) >=0)  &  (np.array(l1-l2+l3) >=0) &  (np.array(-l1+l2+l3) >=0)).all()

def log_length_regular(lld, lla, llb, llao, llbo):
    """
    TODO
    Compute the log length (i.e. Penner coordinate) of diagonal edge d after a regular flip.
    """
    a = (lla + llbo - lld)/2
    b = (llb + llao - lld)/2
    c = (a + b)/2  # subtract the average for numerical stability
    return 2*(c + mlog(mexp(a - c) + mexp(b - c)))

def Delaunay_ind_tri_face_lambdas(C, lambdas, h):
    """
    TODO
    """
    f = C.he2f[h]
    if f in C.bnd_loops:
        return None
    hd = h; hb = C.next_he[hd]; ha = C.next_he[hb]

    # Compute the edge lengths, scaled so that the diagonal edge is 1
    scale = lambdas[hd]/2
    ld = mexp(lambdas[hd]/2 - scale)
    la = mexp(lambdas[ha]/2 - scale)
    lb = mexp(lambdas[hb]/2 - scale)

    return Delaunay_ind_T(ld, la,lb)

def Delaunay_ind_tri_lambdas(C, lambdas, h):
    """
    TODO
    """
    ind1 = Delaunay_ind_tri_face_lambdas(C, lambdas, h)
    ind2 = Delaunay_ind_tri_face_lambdas(C, lambdas, C.opp[h])
    if ind1 == None or ind2 == None:
        return None
    else:
        return ind1 + ind2

def Delaunay_test_tri_lambdas(C, lambdas, h):
    """
    TODO
    """
    ind = Delaunay_ind_tri_lambdas(C, lambdas, h)
    return ind, (ind != None and ind > 0)

def build_edge_maps(C):
    """
    TODO
    """
    # Build map from edges to halfedges as the list of lower index halfedges
    e2he = np.where(np.arange(len(C.opp)) < C.opp)[0]

    # Construct map from halfedges to edges
    he2e = np.arange(len(C.opp))
    he2e[e2he] = np.arange(len(e2he))
    he2e[C.opp[e2he]] = np.arange(len(e2he))

    # FIXME Validity check
    assert (np.max(np.abs(he2e - he2e[C.opp])) == 0)

    return he2e, e2he

def update_jacobian_del(C_del, lambdas_del, he2e, J_del, hd, float_type=float):
    """
    Given connectivity and Penner coordinates, update the Jacobian of the delaunay function
    for a flip of halfedge hd.

    Modifies: J_del
    """
    hb = C_del.next_he[hd]; ha = C_del.next_he[hb];
    hdo = C_del.opp[hd]; hao = C_del.next_he[hdo]; hbo = C_del.next_he[hao]; 

    # Get edges corresponding to halfedges
    ed = he2e[hd]; eb = he2e[hb]; ea = he2e[ha]; eao = he2e[hao]; ebo = he2e[hbo];

    # Compute the shear for the edge ed
    lla = lambdas_del[ha]; llb = lambdas_del[hb];
    llao = lambdas_del[hao]; llbo = lambdas_del[hbo];
    x = mexp((lla + llbo - llb - llao)/2)

    # Update rows J_del_d 
    J_del_d = J_del.getrow(ed)
    J_del_a = J_del.getrow(ea)
    J_del_b = J_del.getrow(eb)
    J_del_ao = J_del.getrow(eao)
    J_del_bo = J_del.getrow(ebo)
    # FIXME Off by factor of two in tests
    #D_d = -2
    #D_a = D_bo = (2 * x) / (1 + x)
    #D_b = D_ao = 2 / (1 + x)
    D_d = -1
    D_a = D_bo = (1 * x) / (1 + x)
    D_b = D_ao = 1 / (1 + x)
    J_del[ed,:] = (   D_d  * J_del_d
                    + D_a  * J_del_a
                    + D_b  * J_del_b
                    + D_ao * J_del_ao
                    + D_bo * J_del_bo )

def make_delaunay_with_jacobian(C,
                                lambdas, 
                                float_type=float,
                                need_jacobian=True):
    """
    Given connectivity and current log edge lengths, make the mesh Delaunay with respect
    to the Penner coordinates lambdas, and compute the Jacobian of this map with respect to
    these coordinates. The Delaunay mesh is created without modifying the initial connectivity.
    
    NOTE: This method is deprecated for a C++ alternative

    Return: Delaunay mesh, Penner coordinates for the Delaunay mesh, Jacobian of Delaunay map
    """
    C_del = copy_namedtuple(C)
    lambdas_del = np.copy(lambdas)
    flips = 0

    # Get edge maps
    he2e, e2he = build_edge_maps(C_del)

    # Initialize Jacobian to the identity if needed
    n_e = len(e2he)
    if need_jacobian:
        J_del_V = [float_type(1)]*n_e
        J_del_I = np.arange(n_e)
        J_del_J = np.arange(n_e)
        J_del = sp.coo_matrix((J_del_V,(J_del_I,J_del_J)),dtype=float_type).tolil()

    # Mark and add all interior edges to stack, using the corresponding halfedge with lower
    # index as a representative
    stk = []
    mark = [False for h in range(0,len(C.next_he))]
    for h in range(0,len(C_del.next_he)):
      if C_del.he2f[h] in C_del.bnd_loops or C_del.he2f[C_del.opp[h]] in C_del.bnd_loops:
        continue
      if h < C_del.opp[h]: # pick the smaller id halfedge
        stk.append(h)
        mark[h] = True

    # Flip edges until the mesh is Delaunay
    while stk:
        hd = stk.pop(-1); lld = lambdas_del[hd];
        mark[hd] = False
        hb = C_del.next_he[hd]; ha = C_del.next_he[hb];
        hdo = C_del.opp[hd]; hao = C_del.next_he[hdo]; hbo = C_del.next_he[hao]; 
        ind, is_Del = Delaunay_test_tri_lambdas(C_del, lambdas_del, hd)

        # Self adjacent triangles are always Delaunay
        if (hdo == ha) or (hdo == hb):
            is_del = True

        # Flip edges that fail the Delaunay condition and update the Jacobian
        if not is_Del:
            assert C_del.next_he[hbo] == hdo
            regular_flip(C_del, hd, lambdas_del, length_fun=log_length_regular)
            if need_jacobian:
                update_jacobian_del(C_del, lambdas_del, he2e, J_del, hd, float_type)

            ind_post, is_Del_post =  Delaunay_test_tri_lambdas(C_del, lambdas_del, hd)
            flips += 1
            if not is_Del_post:
                print("Delaunay test failed on {} with pre in {} and post ind {}:".format(hd, ind, ind_post))

            # Mark and add edges that might be invalid after the flip to the stack, again using
            # a representative halfedge
            for hef in [ha, hb, hao, hbo]:
                if C_del.he2f[hef] in C_del.bnd_loops or C_del.he2f[C_del.opp[hef]] in C_del.bnd_loops:
                    continue
                if (hef < C_del.opp[hef] and not mark[hef]):
                    stk.append(hef)
                    mark[hef] = True
                elif (hef > C_del.opp[hef] and not mark[C_del.opp[hef]]):
                    stk.append(C_del.opp[hef])
                    mark[C_del.opp[hef]] = True

    # FIXME Remove
    print("Flips: ", flips)
    if need_jacobian:
        J_del = J_del.tocsr() 
        return C_del, lambdas_del, J_del
    else:
        return C_del, lambdas_del, None


def alphas_with_jacobian(C_del,
                         lambdas_del,
                         float_type=float,
                         need_jacobian=True):
    """
    FIXME Change to be S*J_alphas
    Given Delaunay connectivity and Penner coordinates, compute the 3|F| vector alpha of angles per
    triangle corner (indexed by halfedge) and the Jacobian matrix. For simplicity, the Jacobian
    is computed as a 3|F|x3|F| matrix indexed by halfedge, where the column i corresponding to
    halfedge he_i is the partial derivative of alpha with respect to the Penner coordinate for 
    the Delaunay triangulation for the edge e = {he_i, opp[he_i]}.

    NOTE: This method is deprecated for a C++ alternative

    Return: vector of angles alpha, Jacobian J_apha
    """
    # more convenient to have f2he as matrix
    f2he =  np.reshape(C_del.f2he,(len(C_del.f2he),1)) 
    he2f = C_del.he2f;  out = C_del.out
    next_he = C_del.next_he;  to = C_del.to; opp = C_del.opp;
   
    n_f  = len(f2he) # number of faces
    n_v  = len(out)  # number of vertices
    n_he = len(to)  # number of halfedges

    # Get edge maps
    he2e, e2he = build_edge_maps(C_del)

    f_he = np.hstack([  f2he,  next_he[f2he],   next_he[next_he[f2he]] ]) # 3 indices of halfedges per face 
    f_v  = to[f_he] # 3 vertex indices per face 
    
    # compute euclidean edge lengths (3 per face, with average log length per face subtracted for
    # numerical stability so each triangle's log edge length have mean 0)
    f_lambdas_del = lambdas_del[f_he]
    f_mean = np.mean(f_lambdas_del, 1) # average log length per face
    f_lambdas_del_rescaled = f_lambdas_del - np.reshape(f_mean, (len(f_mean), 1))
    if float_type == float:
        f_l_del_rescaled = np.exp(f_lambdas_del_rescaled/2)
    else:
        f_l_del_rescaled = np.array([[mexp(x)/2 for x in y] for y in f_lambdas_del_rescaled])  

    # if (happens only due to numerical issues) could not get a triangulation satisfying triangle 
    #  inequalities (the output of make_sym_delaunay should) bail out, 
    # this will result in step decrease in newton, and trying again
    #       
    if not check_tri_ineq(f_l_del_rescaled[:,0], f_l_del_rescaled[:,1], f_l_del_rescaled[:,2]): 
        print("violating triangle inequality!")
        return (False, None, None)
    
    # Compute angles and angle cotangents from the lengths
    if float_type == float:
        f_angles = vec_angles_from_len(f_l_del_rescaled)   
        f_cots = vec_cot_from_len(f_l_del_rescaled)
    else:
        f_angles = np.apply_along_axis( f_angles_from_len,1,f_l_del_rescaled)  # 3 angles per face
        f_cots   = np.apply_along_axis( f_cot_from_len   ,1,f_l_del_rescaled)  # 3 cotangents per face
    
    # Remap angles and cotangents to halfedges 
    he2angle = np.zeros(shape=(n_he,),dtype=float_type)   
    he2angle[np.reshape(f_he,(n_he,))] =  np.reshape(f_angles,(n_he,))   # angle opposite each halfedge
    he2cot = np.zeros(shape=(n_he,),dtype=float_type)
    he2cot[np.reshape(f_he,(n_he,))] =  np.reshape(f_cots,(n_he,))   # cot opposite each halfedge

    # FIXME Print the min and max angle
    #print("Triangle angle min:", np.min(he2angle))
    #print("Triangle angle max:", np.max(he2angle))

    # Sum up angles at vertices
    if float_type == float:
        G_V = he2angle[next_he[next_he]]
        G_I = np.zeros(shape=len(to),dtype=int)
        G_J = to
        alphas = sp.coo_matrix((G_V,(G_I,G_J)),shape=(1,len(out)))
        alphas = alphas.toarray()[0]        
    else:
        alphas = np.zeros(shape=(len(out),),dtype=float_type)
        for v in range(0,len(out)):   # sum of incident angles per vertex
            nbrs = np.nonzero(to==v)
            alphas[v] = np.sum(he2angle[next_he[next_he[nbrs]]])

    # Return if Jacobian not needed
    if (not need_jacobian):
        return (True, alphas, None)
    
    # assemble Jacobian
    if float_type == float:
        # FIXME Add factor of 2?
        #G_V = np.concatenate(( -he2cot[next_he],
        #                        he2cot[next_he] + he2cot[next_he[next_he]],
        #                       -he2cot[next_he[next_he]] ))
        G_V = np.concatenate(( -0.5*he2cot[next_he],
                                0.5*he2cot[next_he] + 0.5*he2cot[next_he[next_he]],
                               -0.5*he2cot[next_he[next_he]] ))
        G_I = np.concatenate(( to[next_he],
                               to[next_he],
                               to[next_he] ))
        G_J = np.concatenate(( he2e[next_he[next_he]],
                               he2e,
                               he2e[next_he] ))
        J_alphas = sp.coo_matrix((G_V, (G_I,G_J)), dtype=float_type).tocsr()
    else:
        # TODO Implement based on Hessian
        J_alphas = None

    return (True, alphas, J_alphas)


def F_with_jacobian(C,
                    lambdas,
                    Th_hat,
                    float_type=float,
                    need_jacobian=True):
    """
    Given connectivity and Penner coordinates, compute the angle constraint function F as well
    as the Jacobian at the given Penner coordinates

    NOTE: This method is deprecated for a C++ alternative

    Return: Jacobian of the angle constraint function
    """
    # Compute make_delaunay and alphas with associated Jacobians for the input
    C_del, lambdas_del, J_del = make_delaunay_with_jacobian(C,
                                                            lambdas,
                                                            float_type=float_type,
                                                            need_jacobian=need_jacobian)
    success, alphas, J_alphas = alphas_with_jacobian(C_del,
                                                     lambdas_del,
                                                     float_type,
                                                     need_jacobian=need_jacobian)

    # Return F and J_F (if needed)
    F = alphas - Th_hat
    if need_jacobian:
        J_F = J_alphas * J_del
        return F, J_F
    else:
        return F, None

   
def line_search_direction(C,
                          lambdas_init,
                          lambdas_k,
                          Th_hat,
                          float_type=float,
                          p=2):
    """
    Compute descent direction for projected gradient descent with Lp energy 
    E(lambdas) = (1/p)*||lambdas - lambdas_init||_p^p 

    Modifies: lambdas
    """
    # Initialize log for iteration
    log_k = {}

    # Get angle constraint function F with Jacobian
    F_k, J_F_k = F_with_jacobian(C, lambdas_k, Th_hat, float_type)

    # Log the max and norm of F_k
    log_k['F_k_norm'] = msqrt(np.dot(F_k, F_k))
    log_k['F_k_max'] = np.max(np.abs(F_k))
    print("F_k norm sq:", log_k['F_k_norm'])
    print("F_k max:", log_k['F_k_max'])

    # Eliminate the last angle constraint from F_k as it is redundant due to Gauss-Bonnet
    F_k_red = F_k[:-1]
    J_F_k_red = J_F_k[:-1,:]

    # Get the per edge lambdas_k array from the per halfedge array
    he2e, e2he = build_edge_maps(C)
    e_lambdas_k = lambdas_k[e2he]
    e_lambdas_init = lambdas_init[e2he]

    # Solve for correction vector mu
    L_k = J_F_k_red * J_F_k_red.T
    e_lambdas_diff_k = e_lambdas_k - e_lambdas_init
    # FIXME Might want to scale differently
    #lp_norm_k = np.linalg.norm(e_lambdas_diff_k, ord=p)
    g_k = e_lambdas_diff_k * np.abs(e_lambdas_diff_k)**(p-2)
    # FIXME Unclear if + or - F_k
    v_k = J_F_k_red * g_k.T - F_k_red
    mu_k = sp.linalg.spsolve(L_k, -v_k)

    # Check the norm of the residual of L_k * mu_k = -v_k
    res = L_k * mu_k + v_k
    log_k['mu_k_res_norm'] = msqrt(np.dot(res, res))
    print("mu_k res norm:", log_k['mu_k_res_norm'])

    # Compute lambda line search direction
    delta_e_lambdas_k = -g_k - (J_F_k_red.T * mu_k)

    # Check that delta lambdas is in the kernel of the full J_F_k
    res = J_F_k * delta_e_lambdas_k.T
    log_k['delta_e_lambdas_residual_norm'] = msqrt(np.dot(res, res))
    print("Delta e_lambdas residual norm:", log_k['delta_e_lambdas_residual_norm'])

    # Extend edge based delta lambdas to halfedge based delta lambdas
    delta_lambdas_k = delta_e_lambdas_k[he2e]

    return delta_lambdas_k, log_k

# Project to the constraint
def project_to_constraint(C,
                          lambdas,
                          Th_hat,
                          float_type=float,
                          bound_norm_thres=1.0,
                          proj_params={},
                          R=None):
    """
    TODO
    """
    # Prepare Conformal mesh data object
    phi0 = np.array([float_type(0)]*len(C.out))
    if float_type == float:
        elen = np.exp(lambdas/2)
    else:
        elen = np.array([mexp(he)/2 for he in lambdas])
    CM = ConformalMesh(C=C, R=R, phi=phi0, l=elen, Th_hat=Th_hat)

    # Get conformal parameters
    max_itr = proj_params['max_itr'] if 'max_itr' in proj_params else 1000

    # Run conformal projection method
    CM_proj, newton_out = conformal_map(CM,
                                        float_type=float_type,
                                        initial_ptolemy=True,
                                        bound_norm_thres=bound_norm_thres,
                                        max_itr=max_itr,
                                        error_eps=1e-8)

    # Update lambdas with phi values
    phi = nparray_from_float64(CM_proj.phi, float_type)
    lambdas_proj = lambdas + (phi[C.to] + phi[C.to[C.opp]])

    return lambdas_proj


