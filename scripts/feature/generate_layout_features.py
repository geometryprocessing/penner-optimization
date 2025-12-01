import os, sys
base_dir = os.path.dirname(__file__)
module_dir = os.path.join(base_dir, '..', '..', 'py')
sys.path.append(module_dir)
import numpy as np
import pandas as pd
import penner
import igl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir")
args = vars(parser.parse_args())

files = os.listdir(args['input_dir'])
suffix='_features'
suffix='_refined_with_uv_out.obj'
suffix='_refined_with_uv_out_connected.obj'
models = [f[:-(len(suffix))] for f in files if f.endswith(suffix)]
output_dir = './'
for model in models:
    # load mesh and feature edges
    try:
        uv_path = os.path.join(args['input_dir'], model + suffix)
        V, uv, _, F, FT, _ = igl.read_obj(uv_path)
        feature_edges = penner.load_mesh_edges(uv_path)
    except:
        print(f"Could not load mesh at {uv_path}")
        continue
    
    # convert feature edges to uv
    print(feature_edges)
    feature_edges = [[e[0], e[1]] for e in feature_edges]
    reverse_feature_edges = [[e[1], e[0]] for e in feature_edges]
    feature_edges = feature_edges + reverse_feature_edges
    print(feature_edges)
    feature_corners = penner.compute_edge_corners(feature_edges, F)
    uv_feature_edges = penner.compute_corner_edges(feature_corners, FT)

    # save features
    feature_path = os.path.join(output_dir, model + '_layout_features.obj')
    with open(feature_path, "w") as uv_mesh_file:
        for uv in uv:
            uv_mesh_file.write("v {} {} {}\n".format(uv[0], uv[1], 0))
        for feature_edge in uv_feature_edges:
            vi = feature_edge[0]
            vj = feature_edge[1]
            uv_mesh_file.write("l {} {}\n".format(vi + 1, vj + 1))