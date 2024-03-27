# Methods to obtain target angle and edge length values from a mesh.

import igl, math
import numpy as np
import optimization_py as opt


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

    
