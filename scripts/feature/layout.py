# Script to project a marked metric to holonomy constraints with feature alignment

import os, sys
base_dir = os.path.dirname(__file__)
module_dir = os.path.join(base_dir, '..', 'py')
sys.path.append(module_dir)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts')
sys.path.append(script_dir)
import numpy as np
import pandas as pd
import penner
import igl
import optimization_scripts.script_util as script_util

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def compute_seamless_error(F, uv, FT):
    uv_length_error, uv_angle_error, uv_length, uv_angle = penner.compute_seamless_error(F, uv, FT)
    return max(np.max(uv_length_error), np.max(uv_angle_error))

def is_seamless(F, uv, FT, thres=1e-10):
    return (compute_seamless_error(F, uv, FT) < thres)

def run_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']

    # create output directory
    if (args['layout_dir'] == ""):
        args['layout_dir'] = os.path.join(args['output_dir'], 'connected')
    os.makedirs(args['layout_dir'], exist_ok=True)

    # get output directory for the mesh to load data
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # load mesh layout
    try:
        uv_path = os.path.join(output_dir, name + ".obj")
        V, uv, _, F, FT, _ = igl.read_obj(uv_path)
        feature_edges = penner.load_mesh_edges(uv_path)
    except:
        print(f"Could not load mesh at {uv_path}")
        return {}

    # check if input is valid
    initial_seamless_error = compute_seamless_error(F, uv, FT)
    if (initial_seamless_error > 1e-7):
        print(f"Input mesh is not seamless with error {initial_seamless_error}")
        return

    # attempt to make connected domain
    uv_c, FT_c = penner.generate_connected_parameterization(V, F, uv, FT)

    # try multiprecision if fails
    seamless_error = compute_seamless_error(F, uv_c, FT_c)
    print(f"connected mesh is not seamless with error {seamless_error}; fallback to multiprecision")
    if (seamless_error > 1e-10):
        print(f"original mesh had seamless error {initial_seamless_error}")
        print(f"connected mesh is not seamless with error {seamless_error}; fallback to multiprecision")
        uv_c, FT_c = penner.generate_connected_parameterization_mpfr(V, F, uv, FT)

        # report if multiprecision fails as well
        seamless_error = compute_seamless_error(F, uv_c, FT_c)
        if (seamless_error > 1e-8):
            print(f"connected mesh is not seamless even with multiprecision; error is {seamless_error}")

    # rescale uv coordinates to approximate unit quad size
    do_rescale = True
    if do_rescale:
        num_tris = len(F)
        bb_diag = igl.bounding_box_diagonal(V)
        scale = (np.sqrt(num_tris / 2) / bb_diag)
        uv_c *= scale

    # write connected meesh
    uv_mesh_path = os.path.join(args['layout_dir'], name + '_connected.obj')
    penner.write_obj_with_uv(uv_mesh_path, V, F, uv_c, FT_c)

    # copy feature edges 
    with open(uv_mesh_path, "a") as uv_mesh_file:
        for feature_edge in feature_edges:
            vi = feature_edge[0]
            vj = feature_edge[1]
            uv_mesh_file.write("l {} {}\n".format(vi + 1, vj + 1))

def run_many(args):
    script_util.run_many(run_one, args)

def add_arguments(parser):
    alg_params = penner.NewtonParameters()
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")
    parser.add_argument("--layout_dir",
                        help="directory for connected layouts", default="")
    parser.add_argument(
        "--suffix",
        help="suffix for output files",
        default=""
    )

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Layout parameterization in single component")
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel method
    run_many(args)
