# Methods to obtain target angle and edge length values from a mesh.

import igl, math
import numpy as np
import optimization_py as opt

def initial_angles(C, lambdas):
    """
    Generate vertex angles for the mesh C with embedded log edge lengths lambdas.
    
    param[in] Mesh C: (possibly symmetric) mesh
    param[in] np.array lambdas: embedded log edge lengths
    return np.array: per vertex angles for C
    """
    proj, embed = opt.build_refl_proj(C)
    lambdas_full = lambdas[proj]
    C_del, lambdas_full_del, _, _ = opt.make_delaunay_with_jacobian(C, lambdas_full, False)
    angles = np.array(opt.alphas_with_jacobian(C_del, lambdas_full_del, False)[0])
    return angles

def map_to_disk(v, f):
    bd = igl.boundary_loop(f)
    Th_hat = 2 * math.pi * np.ones(len(v))
    Th_hat[bd] = math.pi * (len(bd) - 2) / len(bd)
    return Th_hat

def map_to_rectangle(v, f):
    bd = igl.boundary_loop(f)
    Th_hat = 2 * math.pi * np.ones(len(v))
    Th_hat[bd] = math.pi
    n = len(bd)
    corners = [0, int(n/4), int(2*n/4), int(3*n/4)]
    Th_hat[bd[corners]] = math.pi/2
    return Th_hat

def map_to_sphere(v, f):
    global_curvature = 4 * np.pi
    average_curvature = global_curvature / len(v)
    Th_hat = (2 * math.pi - average_curvature) * np.ones(len(v))
    return Th_hat

def generate_tutte_param(v, f):
    """
    Generate harmonic convex Tutte embedding for mesh VF with circular boundary.

    param[in] np.array v: vertex positions
    param[in] np.array f: mesh faces
    return np.array: parametric coordinates
    """
    bd = igl.boundary_loop(f)
    bd_uv = igl.map_vertices_to_circle(v, bd)
    uv = igl.harmonic_weights(v, f, bd, bd_uv, 1)
    uv_embed = np.zeros((len(uv), 3))
    uv_embed[:,:2] = uv[:,:2]

    return uv_embed

def lambdas_from_mesh(C):
    """
    Generate log edge lengths for the embedded original mesh in C.
    
    param[in] Mesh C: (possibly symmetric) mesh
    return np.array: embedded log edge lengths for C
    """
    reduction_maps = opt.ReductionMaps(C)
    embed = np.array(reduction_maps.embed)
    e2he = np.array(reduction_maps.e2he)

    lambdas = 2*np.log(C.l)[np.array(e2he)[embed]]
    return lambdas
    
# *********************************************
# Below is deprecated code now rewritten in C++
# *********************************************

def lengths_from_vf(v, f):
    l = np.zeros(f.shape)
    for face_index, face in enumerate(f):
        for i in np.arange(3):
            vj = face[(i + 1)%3]
            vk = face[(i + 2)%3]
            pj = v[vj]
            pk = v[vk]
            length_vector = pk - pj
            l[face_index, i] = length_vector.norm()

    return l


def get_euclidean_target_lambdas(v, f, Th_hat):
    # Get mesh and the embedding for the undoubled mesh
    C, _ = opt.fv_to_double(v, f, v, f, Th_hat, [], False)
    proj, embed = opt.build_refl_proj(C)
    proj = np.array(proj)
    embed = np.array(embed)

    # Get target lengths after Euclidean flips and flip sequence
    flip_seq = opt.make_delaunay(C, False)
    he2e, e2he = opt.build_edge_maps(C) 
    lambdas_target_flipped = 2*np.log(C.l)[e2he]
    print("num halfedges:", len(C.l))
    print("lambdas size:", len(lambdas_target_flipped))
    
    # Convert Euclidean flip sequence to Ptolemy sequence
    flip_seq_ptolemy = (-np.array(flip_seq) - 1)
    print(lambdas_target_flipped)
    print(flip_seq_ptolemy)

    # Flip edges with Ptolemy flips to get the lengths for the original connectivity
    # Warning: not quite correct; need to flip three times to undo ccw flip
    _, lambdas_target_full = opt.flip_edges(C,
                                            lambdas_target_flipped,
                                            flip_seq_ptolemy[::-1])
    
    # Return the embedded lambda lengths
    return np.array(lambdas_target_full)[embed], flip_seq_ptolemy

