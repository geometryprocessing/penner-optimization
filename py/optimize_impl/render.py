import igl
import numpy as np
from conformal_impl.overload_math import *
from tqdm import trange
import optimization_py as opt
from matplotlib import cm, colors
import optimize_impl.energies as energies
import os


def count_valence(n, opp, h0, is_cut):
# Get the number of cut-edges touching to[h0]
  hi = opp[n[n[h0]]]
  if is_cut[h0]:
    valence = 1
  else:
    valence = 0
  while hi != h0:
    if is_cut[hi]:
      valence += 1
    hi = opp[n[n[hi]]]
  return valence

def trim_cuts(n, opp, to, cones, is_cut_h):
# Given tree-like cutgraph, try to remove any degree-1 vertices that's not a cone
    any_trimmed = True
    while any_trimmed: # repeatedly trim degree-1 cuts
      any_trimmed = False
      for hi in range(len(opp)):
        v0 = to[hi]; v1 = to[opp[hi]]
        valence0 = count_valence(n, opp, opp[hi], is_cut_h)
        valence1 = count_valence(n, opp, hi,      is_cut_h)
        if is_cut_h[hi] and ((valence0 == 1 and v0 not in cones) or (valence1 == 1 and v1 not in cones)):
            is_cut_h[hi] = False
            is_cut_h[opp[hi]] = False
            any_trimmed = True
    
def add_cut_to_sin(n, opp, to, cones, edge_labels, is_cut_h, reindex, is_mesh_doubled):
    trim_cuts(n, opp, to, cones, is_cut_h)

    cut_to_sin_list = [];
    cnt_cut = 0

    for he in range(len(is_cut_h)):
        if (is_cut_h[he] == True) and (not is_mesh_doubled or (is_mesh_doubled and edge_labels[he] == '\x01')):
            vid_from = to[opp[he]]
            vid_to = to[he]
            cut_to_sin_list.append([reindex[vid_from], reindex[vid_to]])
            cnt_cut += 1
    return cut_to_sin_list

def add_shading(color_rgb, v3d, f, fid_mat_input, bc_mat_input, view, proj, flat_shading=False):
    #compute normals (per face)
    normals = igl.per_face_normals(v3d,f,np.array([1.0,1.0,1.0]))
    pv_normals = igl.per_vertex_normals(v3d, f)
    pv_normals4 = np.zeros((pv_normals.shape[0], 4))
    pv_normals4[:pv_normals.shape[0],:3] = pv_normals
    normals4 = np.zeros((normals.shape[0],4))
    normals4[:normals.shape[0],:3] = normals

    ao = igl.ambient_occlusion(v3d, f, v3d, pv_normals,500)

    # normal transformation matrix
    norm_trans = np.linalg.inv(view).transpose()

    light_eye = np.array([0.0, 0.3, 0.0])
    (H, W, _) = color_rgb.shape
    for i in trange(H):
        for j in range(W):
            fid = fid_mat_input[i][j]
            bc = bc_mat_input[i][j]
            if fid > -1:
                diff = color_rgb[i,j,:]
                amb = 0.2 * diff
                spec = 0.3 + 0.1 * (diff - 0.3)
                ao_factor = ao[f[fid, 0]] * bc[0] + ao[f[fid, 1]] * bc[1] + ao[f[fid, 2]] * bc[2]

                pos = v3d[f[fid, 0]] * bc[0] + v3d[f[fid, 1]] * bc[1] + v3d[f[fid, 2]] * bc[2]
                pos4 = np.ones(4); pos4[0:3] = pos
                pos_eye = np.dot(view, pos4)[0:3]
                if flat_shading:
                    norm4 = normals4[fid]
                else:
                    norm4 = pv_normals4[f[fid, 0]] * bc[0] + pv_normals4[f[fid, 1]] * bc[1] + pv_normals4[f[fid, 2]] * bc[2]
                norm_eye = np.dot(norm_trans, norm4)[0:3]
                norm_eye = norm_eye / np.linalg.norm(norm_eye)

                # diff color
                vec_to_light_eye = light_eye - pos_eye
                dir_to_light_eye = vec_to_light_eye / np.linalg.norm(vec_to_light_eye)
                clamped_dot_prod = max(np.dot(dir_to_light_eye, norm_eye), 0)
                color_diff = clamped_dot_prod * diff

                # spec color
                proj_to_norm = np.dot(-dir_to_light_eye, norm_eye) * norm_eye
                refl_eye = proj_to_norm - (-dir_to_light_eye - proj_to_norm)
                surf_to_view_eye = - pos_eye / np.linalg.norm(pos_eye)
                clamped_dot_prod = max(0, np.dot(refl_eye, surf_to_view_eye))
                spec_factor = pow(clamped_dot_prod, 35)
                color_spec = spec_factor * spec

                color_new = amb + 1.2 * color_diff + color_spec
                for k in range(3):
                    color_new[k] = max(0, min(1, color_new[k]))
                color_rgb[i,j,:] = color_new * (0.5 + (1-ao_factor)*0.5)
    return color_rgb


def color_mesh_with_grid(fid_mat, bc_mat, h, n, to, u, v, r, H, W, colormap, norm, N_bw = 15, thick = 0.1, uv_scale=0):
    # Get uv units
    if (uv_scale == 0):
        uv_scale = max((np.max(u) - np.min(u)), (np.max(v) - np.min(v)))
    u = u/uv_scale
    v = v/uv_scale
    u_min = float(np.nanmin(u))
    u_max = float(np.nanmax(u))
    v_min = float(np.nanmin(v))
    v_max = float(np.nanmax(v))
    u_unit = 1 / N_bw
    v_unit = u_unit
    u_thick = thick * u_unit 
    v_thick = thick * v_unit

    # Generate color grid
    color_rgb_gd = np.zeros((H, W, 3))
    for i in trange(H):
        for j in range(W):
            if fid_mat[i][j] > -1:
                # Get grid point information
                fid = fid_mat[i][j]
                bc = bc_mat[i][j]
                e0 = h[fid]
                e1 = n[e0]
                e2 = n[e1]

                # Interpolate color
                r0 = float(r[to[e0]])
                r1 = float(r[to[e1]])
                r2 = float(r[to[e2]])
                r_pt = float(r0 * bc[1] + r1 * bc[2] + r2 * bc[0])

                # Interpolate uv coordinate
                u_pt = float(u[e0]) * bc[1] + float(u[e1]) * bc[2] + float(u[e2]) * bc[0]
                v_pt = float(v[e0]) * bc[1] + float(v[e1]) * bc[2] + float(v[e2]) * bc[0]

                # Color according to r in interior of grid cells
                if  u_thick < ((u_pt - u_min)%u_unit) <= (u_unit - u_thick) and  v_thick < ((v_pt - v_min)%v_unit) <= (v_unit-v_thick):
                    color_rgb_gd[i,j,:] = np.array(colormap(norm(r_pt))[:3])
                # Shade grid lines darker
                else:
                    color_rgb_gd[i,j,:] = 0.55 * np.array(colormap(norm(r_pt))[:3])

            # Color additional features
            elif fid_mat[i][j] == -1:
                color_rgb_gd[i,j,:] = np.array([1.0,1.0,1.0])
            elif fid_mat[i][j] == -2: # Red sphere
                color_rgb_gd[i,j,:] = np.array([0.7,0.1,0.2])
                #color_rgb_gd[i,j,:] = np.array([1.0,0.0, 0.75])
            elif fid_mat[i][j] == -3: # Blue sphere
                color_rgb_gd[i,j,:] = np.array([0.5,0.7,0.35])
                #color_rgb_gd[i,j,:] = np.array([0.0,0.75,1.0])
            elif fid_mat[i][j] == -4:
                color_rgb_gd[i,j,:] = np.array([0,0,0])
            elif fid_mat[i][j] == -5:
                color_rgb_gd[i,j,:] = np.array([1,0.1,0.1])

    return color_rgb_gd


def get_corner_uv(n, h, to, f, fuv, uv):
    u = np.zeros(len(n))
    v = np.zeros(len(n))

    # Get per corner uv
    for i in range(len(fuv)):
        hh = h[i]
        if (to[hh] != f[i,1]):
            print("error")
        u[hh] = uv[fuv[i,1],0]
        v[hh] = uv[fuv[i,1],1]   
        u[n[hh]] = uv[fuv[i,2],0]
        v[n[hh]] = uv[fuv[i,2],1]
        u[n[n[hh]]] = uv[fuv[i,0],0]
        v[n[n[hh]]]= uv[fuv[i,0],1]

    return u, v

def generate_ambient_occlusion(v, f, n_rays=500):
    """
    Compute ambient occlusion values for the mesh (v,f)

    param[in] np.array v: vertex positions for the mesh
    param[in] np.array f: triangulation for the mesh
    param[in] int n_rays: number of rays for occulsion computation
    return np.array: per vertex ambient occlusion values
    """
    # Compute per vertex normals
    face_normals = igl.per_face_normals(v,f,np.array([1.0,1.0,1.0]))
    vertex_normals = igl.per_vertex_normals(v, f)
    vertex_normals4 = np.zeros((vertex_normals.shape[0], 4))
    vertex_normals4[:vertex_normals.shape[0],:3] = vertex_normals

    # Compute ambient occlusion
    ao = igl.ambient_occlusion(v, f, v, vertex_normals, n_rays)

    return 1.0 - ao

def cprs_arr(x):
    zeros = np.zeros_like(x)
    ones = np.ones_like(x)
    x = np.maximum(zeros,np.minimum(ones, x))
    return np.maximum(0, np.minimum(ones, 3 * x * x - 2 * x * x * x))


def generate_colormap(x,
                      shift=0,
                      scale=None,
                      clamp=True):
    """
    Generate a color map from an array of function values x.

    param[in] np.array x: function to generate color values for
    param[in] float shift: translate x by this value before generating color map
    param[in] float scale: scale x by this value (after translation) before generating color
            map. If None is used, the average value of x (after translation) is used instead
    param[in] clamp: If true, clamp linear interpolation instead of using arctan
    return np.array: colormap giving an RGB value for each element of x
    """
    # Shift function
    c = x - shift

    if not clamp:
        c = c / scale
        # Clamp colormap to range [0,1]
        #c = np.maximum(c, 0)
        #c = np.minimum(c, 1)
        # Map (-infty,infty) -> (0,1) with arctan followed by a linear map
        c = (np.arctan(c) / (np.pi/2))

    if (scale > 0):
        norm = colors.CenteredNorm(scale*0.5, scale*0.6)
    else:
        norm = colors.CenteredNorm(0, 1)

    print(np.max(c), np.average(c)) # FIXME
    # Use the coolwarm color scheme
    return np.array(cm.coolwarm(norm(c))[:,:3])


def get_layout_colormap(
    v,
    f,
    uv,
    fuv,
    colormap,
    scale = 1.0,
    use_sqrt_scale=False,
    use_log_scale=False,
    average_per_vertex=False
):
    # Get energy
    energy = energies.get_face_energy(v, f, uv, fuv, colormap, use_sqrt_scale=use_sqrt_scale, use_log_scale=use_log_scale)
    print(np.max(energy), np.average(energy))

    # Generate colormap
    c = generate_colormap(energy, shift=0, scale=scale)
    if (average_per_vertex):
        c = igl.average_onto_vertices(v, f, c)

    return c

    
def render_layout(
    v,
    f,
    uv,
    c,
    show_lines,
    lighting_factor,
    average_per_vertex
):
    # Average ambient shading for face or colormap for vertex functions
    ao = generate_ambient_occlusion(v, f) 
    if (not average_per_vertex):
        ao = np.average(ao[f], axis=1)

    # Generate mesh viewer with shading for layout
    viewer = opt.generate_mesh_viewer(uv, f, show_lines)
    opt.add_shading_to_mesh(viewer, c, ao, lighting_factor)

    return viewer
