
import optimize_impl.optimization as optimization
import optimization_py as opt
import meshplot as plot
import igl
from matplotlib import cm
import numpy as np

def gen_checkers(
    n_checkers_x=50,
    n_checkers_y=50,
    width=1000,
    height=1000
):
    '''
    Generate checkerboard pattern for the meshplot viewer.

    TODO Parameters
    '''
    # tex dims need to be power of two.
    array = np.ones((width, height, 3), dtype='float32')

    # width in texels of each checker
    checker_w = width / n_checkers_x
    checker_h = height / n_checkers_y

    for y in range(height):
        for x in range(width):
            color_key = int(x / checker_w) + int(y / checker_h)
            if color_key % 2 == 0:
                array[x, y, :] = [ 1., 1.0, 1.0 ]
            else:
                array[x, y, :] = [ 0.5, 0.5, 0.5 ]
    return array

# ***********************
# Below is spaghetti code
# ***********************

def generate_vf_colormap_viewer(
    C,
    lambdas,
    lambdas_target,
    v,
    f,
    Th_hat,
    width=1000,
    height=1000
):
    proj, embed = build_refl_proj(C)
    he2e, e2he = build_edge_maps(C)
    proj = np.array(proj)
    he2e = np.array(he2e)

    # Get per vertex stretch function
    stretches = symmetric_stretches(lambdas, lambdas_target)
    c = halfedge_function_to_vertices(C, stretches[proj[he2e]])

    vw = plot.Viewer(dict(width=width, height=height))
    vw.add_mesh(v, f, c,
                shading={"wireframe": True, "flat": False})

    return vw, c

def generate_best_fit_conformal_viewer(C,
                                       lambdas,
                                       lambdas_target,
                                       v,
                                       f,
                                       Th_hat,
                                       width=1000,
                                       height=1000):
    proj, embed = build_refl_proj(C)
    proj = np.array(proj)

    # Get per vertex stretch function
    stretches = symmetric_stretches(lambdas, lambdas_target)
    c = best_fit_conformal(C, lambdas_target[proj], lambdas[proj])
    c = np.array(c)

    vw = plot.Viewer(dict(width=width, height=height))
    vw.add_mesh(v, f, c,
                shading={"wireframe": True, "flat": False})

    return vw, c


def build_overlay_FV(v, f, Th_hat, C, lambdas):
    C_o = build_overlay_mesh(C, lambdas, [], False)
    v_int = np.array(interpolate_3d(v, f, Th_hat, C, lambdas)).T
    return v_int[np.array(C_o.to)]
#    return v_int[np.array(C_o.to)[C_o.n]]

def generate_layout_viewers(v, f, Th_hat, C, lambdas,width=1000,height=1000):
    uv, f_h = build_layout_FV(v, f, Th_hat, C, lambdas)
    v_h = build_overlay_FV(v, f, Th_hat, C, lambdas)

    # Get texture
    tex = gen_checkers(width=width,height=height)
    # Generate mesh with texture
    vw_mesh = plot.Viewer(dict(width=width, height=height))
    vw_mesh.add_mesh(v_h,
                     f_h,
                     uv=uv,
                     shading={"wireframe": True, "flat": True},
                     texture_data=tex)

    # Generate layout
    vw_layout = plot.Viewer(dict(width=width, height=height))
    vw_layout.add_mesh(uv,
                       f_h,
                       uv=uv,
                       shading={"wireframe": True, "flat": True},
                       texture_data=tex)

    return vw_mesh, vw_layout

def generate_delaunay_overlay(v, f, Th_hat, C, lambdas, vtx_reindex,
                              initial_ptolemy=True):
    proj, embed = build_refl_proj(C)
    he2e, e2he = build_edge_maps(C)
    proj = np.array(proj)
    embed = np.array(embed)
    he2e = np.array(he2e)
    e2he = np.array(e2he)
    
    # Get overlay mesh with original lengths
    lambdas_0 = lambdas_from_mesh(C)
    C_o = add_overlay(C, lambdas_0)

    # Make original mesh Delaunay and duplicate flips for the final coordinates with Ptolemy flips
    flip_seq_init = make_delaunay_overlay(C_o, initial_ptolemy)
    _, lambdas_full_flip = opt.flip_edges(C,
                                            lambdas[proj],
                                            flip_seq_init)
    print("Flips to make input Delaunay: {}".format(len(flip_seq_init)))

    # Remap barycentric coordinates if Euclidean flips were used
    if not initial_ptolemy:
        bc_original_to_eq_overlay(C_o)

    # Make final mesh Delaunay and duplicate flips in the overlay mesh
    C_o, lambdas_full_del, _, flip_seq = opt.make_delaunay_with_jacobian_overlay(C_o, lambdas_full_flip, False)
    #C_o, lambdas_full_del, _, flip_seq = opt.make_delaunay_with_jacobian(C_o._m, lambdas_full_flip, False)
    #C_del, lambdas_full_del, _, _ = opt.make_delaunay_with_jacobian(C, lambdas[proj], False)
    #flip_edges_overlay(C_o, flip_seq)
    #C_o._m = C_del # FIXME Sketchy
    lambdas_del_he = np.array(lambdas_full_del)[he2e]
    print("Flips to make output Delaunay: {}".format(len(flip_seq)))

    # Interpolate points in the original mesh
    v_reindex = np.zeros((3,len(C_o._m.out)))
    v_reindex[:,:len(vtx_reindex)] = (v.T)[:,vtx_reindex]
    flip_seq_full = flip_seq_init + flip_seq
    v_overlay = interpolate_3d(C_o, flip_seq_full, v_reindex, False)

    # Make mesh tufted
    if (len(igl.boundary_loop(f)) > 0):
        print("Making tufted")
        make_tufted_overlay(C_o, v, f, Th_hat)

    # Change lengths to target values
    l_del = np.exp(lambdas_del_he / 2.0)
    opt.change_lengths_overlay(C_o, l_del)

    return C_o

def mesh_parametrization(v, f, Th_hat, C, lambdas_init, lambdas, vtx_reindex,
                         initial_ptolemy=True,
                         conformal_interpolation=False,
                         use_new_code=False):
    # Get cones and bd for the mesh
    cones, bd = get_cones_and_bd(v, f, Th_hat, vtx_reindex)

    if conformal_interpolation:
        alg_params = conf.AlgorithmParameters()
        alg_params.initial_ptolemy = False
        alg_params.error_eps = 1e-8

        ls_params = conf.LineSearchParameters()
        ls_params.bound_norm_thres = 1
        ls_params.do_reduction = True
        stats_params = conf.StatsParameters()
        C_o, u, _, _, _, v_overlay = conf.conformal_metric_double(v, f, Th_hat, [], [], alg_params, ls_params, stats_params)
        pass
        print(np.max(C_o.seg_bcs))
        #bc_eq_to_scaled(C_o, C_o._m, lambdas_del_he)
    else:
        #u = np.zeros_like(C.Th_hat)
        sigmas0 = opt.compute_shear(C, lambdas_init[proj[he2e]])
        sigmas = opt.compute_shear(C, lambdas[proj[he2e]])
        print("Shear change", np.max(np.abs(sigmas-sigmas0)))
        opt.bc_reparametrize_eq(C_o, (sigmas-sigmas0)/2)
        #bc_eq_to_scaled(C_o, C_o._m, lambdas_del_he)
        print(np.max(C_o.seg_bcs))

    # Generate layout
    if use_new_code:
        u_scalar, v_scalar, u_int, v_int, u_o, v_o, is_cut_o = opt.generate_layout_overlay_lambdas(C_o, lambdas_del_he, bd, cones, False, -1)
    else:
        _, _, _, u_o, v_o, is_cut_o = opt.get_layout(C_o, u, bd, cones, False)
        u_scalar = u_o
        v_scalar = v_o
        u_int = u_o 
        v_int = v_o
    if np.any(np.isnan(u_scalar)) or np.any(np.isnan(v_scalar)):
        print("NaN encountered while generating layout")
    if np.any(np.isnan(u_int)) or np.any(np.isnan(v_int)):
        print("NaN encountered while interpolating layout")
    if np.any(np.isnan(u_o)) or np.any(np.isnan(v_o)):
        print("NaN encountered while generating overlay layout")
    v3d, u_o_out, v_o_out, f_out, ft_out = get_FV_FTVT(C_o, is_cut_o, v_overlay, u_o, v_o)
    if np.any(np.isnan(u_o_out)) or np.any(np.isnan(v_o_out)):
        print("NaN encountered while combining faces")

    # Reindex vertex positions
    u_o_out_copy = np.array(u_o_out)
    v_o_out_copy = np.array(v_o_out)
    v3d_out_copy = np.array(v3d).T
    u_o_out = u_o_out_copy.copy()
    v_o_out = v_o_out_copy.copy()
    v3d_out = v3d_out_copy.copy()
    n_v = len(vtx_reindex)
    u_o_out[vtx_reindex] = u_o_out_copy[:n_v]
    v_o_out[vtx_reindex] = v_o_out_copy[:n_v]
    v3d_out[vtx_reindex] = v3d_out_copy[:n_v]

    # Reindex faces
    f_out = reindex_F(f_out, vtx_reindex)
    ft_out = reindex_F(ft_out, vtx_reindex)
    f_out = np.array(f_out)
    ft_out = np.array(ft_out)

    return v3d_out, f_out, u_scalar, v_scalar, u_o_out, v_o_out, ft_out, C_o, v_overlay


def mesh_parametrization_deprecated(v, f, Th_hat, lambdas, flip_seq_init=[], initial_ptolemy=True, interpolate_from_original=False, conformal_interpolation=False, use_linear_interpolation=False):
    # Build mesh
    C, vtx_reindex = fv_to_double(v, f, v, f, Th_hat, [], False)
    C0, vtx_reindex = fv_to_double(v, f, v, f, Th_hat, [], False)
    #lambdas_init = lambdas_from_mesh(C)
    lambdas_init, _ = get_euclidean_target_lambdas(v, f, Th_hat)
    proj, embed = build_refl_proj(C)
    he2e, e2he = build_edge_maps(C)
    proj = np.array(proj)
    embed = np.array(embed)
    he2e = np.array(he2e)
    e2he = np.array(e2he)
    
    # Get overlay mesh with original lengths and optimized lengths
    C0_o = add_overlay(C0, lambdas_init)
    print(np.max(C0_o.seg_bcs))

    # Get cones and bd for the mesh
    cones, bd = get_cones_and_bd(v, f, Th_hat, vtx_reindex)

    # Make original mesh Delaunay with Euclidean flips and duplicate in optimized mesh with Ptolemy flips
    if initial_ptolemy:
        print("Ptolemy")
        C_o = add_overlay(C, lambdas)
        flip_seq_0 = []
    else:
        print("Not ptolemy")
        flip_seq_0 = make_delaunay_overlay(C0_o, False)
        bc_original_to_eq_overlay(C0_o)

        C_o = add_overlay(C, lambdas_init)
        make_delaunay_overlay(C_o, False)
        C_o_ptolemy = add_overlay(C, lambdas)
        flip_seq_ptolemy = -np.array(flip_seq_0) - 1
        flip_edges_overlay(C_o_ptolemy, flip_seq_ptolemy)
        change_lengths_overlay(C_o, C_o_ptolemy._m.l)
        bc_original_to_eq_overlay(C_o)

    # Make optimized mesh Delaunay with Ptolemy flips and duplicate in original mesh
    flip_seq_1 = make_delaunay_overlay(C_o, True)
    print(np.max(C0_o.seg_bcs))
    flip_edges_overlay(C0_o, flip_seq_1)
    print("Overlay length", np.max(C_o._m.l))

    # Interpolate points in the original mesh
    v_reindex = np.zeros((3,len(C0_o._m.out)))
    v_reindex[:,:len(vtx_reindex)] = (v.T)[:,vtx_reindex]
    flip_seq_full = flip_seq_0 + flip_seq_1
    if interpolate_from_original:
        v_overlay = interpolate_3d(C0_o, flip_seq_full, v_reindex, False)
    else:
        v_overlay = interpolate_3d(C_o, flip_seq_full, v_reindex, False)
    print("overlay max", np.max(v_overlay))
    # Make meshes tufted
    make_tufted_overlay(C_o, v, f, Th_hat)
    make_tufted_overlay(C0_o, v, f, Th_hat)

    # Get layout
    lambdas_full = lambdas[proj]
    C_del, lambdas_full_del, _ = opt.make_delaunay_with_jacobian(C, lambdas_full, False)
    lambdas_del_he = np.array(lambdas_full_del)[he2e]
    if conformal_interpolation:
        #u = np.zeros(len(v))
        #u_scalar, v_scalar, _, u_o, v_o, is_cut_o = get_layout_overlay(C_o, u, bd, cones, False, -1)
        #bc_eq_to_scaled(C_o, C_o._m, lambdas_del_he)
        #print(C_o.seg_bcs)
        #print("Conformal Interpolation")
        #u_scalar, v_scalar, u_o, v_o, is_cut_o = opt.generate_layout_overlay_lambdas(C_o, lambdas_del_he, bd, cones, False, -1)
        print(np.max(C0_o.seg_bcs))
        bc_eq_to_scaled(C0_o, C0_o._m, lambdas_del_he)
        print(np.max(C0_o.seg_bcs))
        #print(C0_o.seg_bcs)
        print("Conformal Interpolation")
        u_scalar, v_scalar, u_o, v_o, is_cut_o = opt.generate_layout_overlay_lambdas(C0_o, lambdas_del_he, bd, cones, False, -1)
        print(np.max(u_scalar), np.max(v_scalar), np.max(u_o), np.max(v_o))
    else:
        sigmas0 = opt.compute_shear(C, lambdas_init[proj[he2e]])
        sigmas = opt.compute_shear(C, lambdas[proj[he2e]])
        print(sigmas)
        print(sigmas0)
        print("Shear change", np.max(np.abs(sigmas-sigmas0)))
        opt.bc_reparametrize_eq(C0_o, (sigmas-sigmas0)/2)
        bc_eq_to_scaled(C0_o, C0_o._m, lambdas_del_he)
        #bc_eq_to_two_triangle_chart(C0_o, C0_o._m)
        #bc_two_triangle_chart_to_scaled(C0_o, C0_o._m, lambdas_del_he, use_linear_interpolation)
        #print(C0_o.seg_bcs)
        #bc_eq_to_scaled(C_o, C_o._m, lambdas_del_he)
        print("Projective Interpolation")
        u_scalar, v_scalar, u_o, v_o, is_cut_o = opt.generate_layout_overlay_lambdas(C0_o, lambdas_del_he, bd, cones, False, -1)
        print(np.max(u_scalar), np.max(v_scalar), np.max(u_o), np.max(v_o))
    v3d, u_o_out, v_o_out, f_out, ft_out = get_FV_FTVT(C_o, is_cut_o, v_overlay, u_o, v_o)

    # Reindex vertex positions
    u_o_out_copy = np.array(u_o_out)
    v_o_out_copy = np.array(v_o_out)
    v3d_out_copy = np.array(v3d).T
    u_o_out = u_o_out_copy.copy()
    v_o_out = v_o_out_copy.copy()
    v3d_out = v3d_out_copy.copy()
    n_v = len(vtx_reindex)
    u_o_out[vtx_reindex] = u_o_out_copy[:n_v]
    v_o_out[vtx_reindex] = v_o_out_copy[:n_v]
    v3d_out[vtx_reindex] = v3d_out_copy[:n_v]

    # Reindex faces
    f_out = reindex_F(f_out, vtx_reindex)
    ft_out = reindex_F(ft_out, vtx_reindex)
    f_out = np.array(f_out)
    ft_out = np.array(ft_out)

    return v3d_out, f_out, u_scalar, v_scalar, u_o_out, v_o_out, ft_out, C_o, v_overlay

def render_layout(m,
                  v,
                  f,
                  uv,
                  c,
                  show_lines,
                  lighting_factor,
                  height,
                  width,
                  output_dir,
                  average_per_vertex):
    os.makedirs(output_dir, exist_ok=True)

    # Average ambient shading for face or colormap for vertex functions
    ao = generate_ambient_occlusion(v, f) 
    if (not average_per_vertex):
        ao = np.average(ao[f], axis=1)

    # Generate mesh viewer with shading for original mesh
    viewer = opt.generate_mesh_viewer(v, f, show_lines)
    opt.add_shading_to_mesh(viewer, c, ao, lighting_factor)

    # Save mesh viewer to file
    image_path = os.path.join(output_dir, m+'_mesh.png')
    opt.save_mesh_screen_capture(viewer, image_path, height, width)

    # Generate mesh viewer with shading for layout
    viewer = opt.generate_mesh_viewer(uv, f, show_lines)
    opt.add_shading_to_mesh(viewer, c, ao, lighting_factor)

    # Save mesh viewer to file
    image_path = os.path.join(output_dir, m+'_layout.png')
    opt.save_mesh_screen_capture(viewer, image_path, height, width)

def build_overlay_FV(C,
                     lambdas,
                     v,
                     f,
                     Th_hat):
    '''
    TODO Update with new methods
    Get the FV representation of the triangulated overlay mesh from C with log lenghts lambdas
    and initial state given by (v,f) with target angles Th_hat.

    param[in] Mesh C: Mesh to get overlay mesh for
    param[in] np.array lambdas: log edge lengths for C
    param[in] np.array v: Original vertex positions for the mesh C
    param[in] np.array f: Original triangulation for the mesh C
    param[in] np.array Th_hat: Target angles for the mesh C
    return np.array v_o: vertices for the original mesh embedding with overlay triangulation
    return np.array f_o: connectivity for the original mesh embedding with overlay triangulation
    '''
    v_o, f_o, _, _, _, _, _ = mesh_parametrization(v, f, Th_hat, lambdas)
    
    v_o = np.array(v_o)
    f_o = np.array(f_o)

    return v_o, f_o

def parametrize_mesh_fv(v, f, Th_hat,
                        lambdas, tau_init, tau, tau_post,
                        initial_ptolemy=False, flip_in_original_metric=True):
    # TODO Update
    parametrize_res = opt.parametrize_mesh(v, f, Th_hat,
                                           lambdas, tau_init, tau, tau_post,
                                           [], [],
                                           initial_ptolemy, flip_in_original_metric)

    v_o, f_o, u_param_o, v_param_o, ft_o, _, _, _ = parametrize_res
    v_o = np.array(v_o)
    f_o = np.array(f_o)
    ft_o = np.array(ft_o)

    # Combine uv coordinates
    uv_o = np.array([u_param_o, v_param_o]).T

    # Get the vertices for the cut mesh
    vt_o = np.zeros((len(uv_o),3), dtype=np.float64)
    vt_o[ft_o] = v_o[f_o]

    return vt_o, ft_o, uv_o


def build_overlay_layout_FV(v,
                            f,
                            Th_hat,
                            C,
                            lambdas_init,
                            lambdas,
                            vtx_reindex,
                            initial_ptolemy=True,
                            use_python_code=False,
                            conformal_interpolation=False):
    '''
    TODO Update
    Get the FV representation of the triangulated overlay mesh layout from C with log lenghts
    lambdas and initial state given by (v,f) with target angles Th_hat. Note that the mesh here
    has a different connectivity than build_overlay_FV due to additional cuts necessary to layout
    the mesh.

    param[in] Mesh C: Mesh to get overlay layout mesh for
    param[in] np.array lambdas: log edge lengths for C
    param[in] np.array v: Original vertex positions for the mesh C
    param[in] np.array f: Original triangulation for the mesh C
    param[in] np.array Th_hat: Target angles for the mesh C
    return np.array v_cut_o: vertices for the original mesh embedding with cut overlay triangulation
    return np.array ft_o: connectivity for the original mesh embedding with cut overlay triangulation
    return np.array uv_cut_o: vertices for the layout with overlay triangulation
    '''
    # Get cut overlay mesh and layout
    if use_python_code:
        v_o, f_o, _, _, u_param, v_param, ft_o, _, _ = mesh_parametrization(v,
                                                                            f,
                                                                            Th_hat,
                                                                            C,
                                                                            lambdas_init,
                                                                            lambdas,
                                                                            vtx_reindex,
                                                                            initial_ptolemy=initial_ptolemy,
                                                                            conformal_interpolation=conformal_interpolation)
    else:
        v3d_out, f_out, u_o_out, v_o_out, ft_out =opt.parametrize_mesh(v, f, Th_hat, lambdas, True)
    #v_o, f_o, _, _, u_param, v_param, ft_o = layout_lambdas(v, f, Th_hat, C, lambdas)
    v_o = np.array(v_o)
    f_o = np.array(f_o)
    ft_o = np.array(ft_o)

    # Combine uv coordinates
    uv_cut_o = np.array([u_param, v_param]).T

    # Get the vertices for the cut mesh
    v_cut_o = np.zeros((len(uv_cut_o),3),dtype=np.float64)
    v_cut_o[ft_o] = v_o[f_o]

    return v_cut_o, ft_o, uv_cut_o

def build_layout_FV(v,
                            f,
                            Th_hat,
                            C,
                            lambdas_init,
                            lambdas,
                            vtx_reindex,
                            initial_ptolemy=True,
                            use_python_code=False,
                            conformal_interpolation=False):
    # Get cut overlay mesh and layout
    if use_python_code:
        v_o, f_o, u_param, v_param, _, _, ft_o, C_o, _ = mesh_parametrization(v,
                                                                            f,
                                                                            Th_hat,
                                                                            C,
                                                                            lambdas_init,
                                                                            lambdas,
                                                                            vtx_reindex,
                                                                            initial_ptolemy=initial_ptolemy,
                                                                            conformal_interpolation=conformal_interpolation)
    else:
        v_o, f_o, u_param, v_param, _, _, ft_o, C_o, _ = mesh_parametrization_VL(v, f, Th_hat, C, lambdas)
    #v_o, f_o, _, _, u_param, v_param, ft_o = layout_lambdas(v, f, Th_hat, C, lambdas)
    v_o = np.array(v_o)
    f_o = np.array(f_o)
    ft_o = np.array(ft_o)

    # Combine uv coordinates
    uv_layout = np.array([u_param, v_param]).T

    n = np.array(C_o._m.n)
    f_layout = np.array([n, n[n], n[n[n]]]).T

    return uv_layout, f_layout

def generate_overlay_viewer(C,
                            lambdas,
                            v,
                            f,
                            Th_hat,
                            width=1000,
                            height=1000,
                            view_original_edges=True,
                            initial_ptolemy=False,
                            interpolate_from_original=False,
                            use_python_code=False):
    '''
    Generate meshplot viewers for the mesh with checkerboard patterns and the layout from C with
    log lengths lambdas and initial state given by (v,f) with target angles Th_hat.

    param[in] Mesh C: Mesh to generate viewers for
    param[in] np.array lambdas: log edge lengths for C
    param[in] np.array v: Original vertex positions for the mesh C
    param[in] np.array f: Original triangulation for the mesh C
    param[in] np.array Th_hat: Target angles for the mesh C
    params[in] int width, height: dimensions for the meshplot viewer
    param[in] bool view_original_edges: If True, view the original edges on the overlay mesh
    return meshplot.Viewer vw_mesh: meshplot viewer for the original mesh with uv coordinates
    return meshplot.Viewer vw_layout: meshplot viewer for the layout
    '''
    # Build overlay mesh and layout
    v_cut_o, ft_o, uv_cut_o = build_overlay_layout_FV(C,
                                                      lambdas,
                                                      v,
                                                      f,
                                                      Th_hat,
                                                      initial_ptolemy=initial_ptolemy,
                                                      interpolate_from_original=interpolate_from_original,
                                                      use_python_code=use_python_code)

    # Get texture and boundary vertices
    tex = gen_checkers(width=width,height=height)
    bd_v = igl.boundary_loop(ft_o)

    # Generate mesh with texture
    vw_mesh = plot.Viewer(dict(width=width, height=height))
    vw_mesh.add_mesh(v_cut_o,
                     ft_o,
                     uv=uv_cut_o,
                     shading={"wireframe": view_original_edges, "flat": True},
                     texture_data=tex)
    if (len(bd_v) > 0):
        vw_mesh.add_lines(v_cut_o[bd_v[:-1]],
                          v_cut_o[bd_v[1:]],
                          shading={"line_color": "red", "line_width": 20})
        vw_mesh.add_lines(v_cut_o[bd_v[-1]],
                          v_cut_o[bd_v[0]],
                          shading={"line_color": "red", "line_width": 20})
    if view_original_edges:
        he2v = get_edges(v, f, Th_hat, C, lambdas)
        he2v = np.array(he2v)
        he2v = he2v[np.logical_and((he2v[:,0] != -1), (he2v[:,1] != -1))]
        vw_mesh.add_lines(v_cut_o[he2v[:,0]],
                          v_cut_o[he2v[:,1]],
                          shading={"line_color": "red", "line_width": 20}) 

    # Generate layout
    vw_layout = plot.Viewer(dict(width=width, height=height))
    vw_layout.add_mesh(uv_cut_o,
                       ft_o,
                       uv=uv_cut_o,
                       shading={"wireframe": view_original_edges, "flat": True},
                       texture_data=tex)
    if (len(bd_v) > 0):
        vw_layout.add_lines(uv_cut_o[bd_v[:-1]],
                            uv_cut_o[bd_v[1:]],
                            shading={"line_color": "red", "line_width": 20})
        vw_layout.add_lines(uv_cut_o[bd_v[-1]],
                            uv_cut_o[bd_v[0]],
                            shading={"line_color": "red", "line_width": 20})
    if view_original_edges:
        vw_layout.add_lines(uv_cut_o[he2v[:,0]],
                            uv_cut_o[he2v[:,1]],
                            shading={"line_color": "red", "line_width": 20}) 

    return vw_mesh, vw_layout