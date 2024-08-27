# Script to generate an overlay from a marked mesh

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import holonomy_py as holonomy
import optimization_py as opt 
import pickle, math, logging
import igl
import optimization_scripts.script_util as script_util
import optimize_impl.render as render

def similarity_overlay_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']

    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # Skip meshes that are already processed
    try:
        uv_mesh_path = os.path.join(output_dir, name + '_refined_with_uv.obj')
        if os.path.isfile(uv_mesh_path):
            V, F = igl.read_triangle_mesh(uv_mesh_path)
            if (len(V) > 0):
                print("Skipping processed mesh")
                return
    except:
        pass

    # Get logger
    log_path = os.path.join(output_dir, name+'_convert_to_vf.log')
    logger = script_util.get_logger(log_path)
    logger.info("Converting {} to vf".format(name))

    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['input_dir'], fname))
        Th_hat = np.loadtxt(os.path.join(output_dir, m + '_Th_hat'), dtype=float)
        rotation_form = np.loadtxt(os.path.join(output_dir, m + '_kappa_hat'), dtype=float)
    except:
        logger.info("Could not open mesh data at {}", args['input_dir'])
        return

    # Get final optimized lambdas
    try:
        metric_coords_path = os.path.join(output_dir, name + "_metric_coords")
        logger.info("Loading metric coordinates from {}".format(metric_coords_path))
        reduced_metric_coords = np.loadtxt(metric_coords_path)
    except:
        logger.error('Could not load metric')
        return

    # Get mesh information
    is_bd = igl.is_border_vertex(V, F)
    build_double = (np.sum(is_bd) != 0)
    _, vtx_reindex = opt.fv_to_double(V, F, V, F, Th_hat, [], False)

    # Get cones
    cones = np.array([id for id in range(len(Th_hat)) if np.abs(Th_hat[id]-2*math.pi) > 1e-15 and not is_bd[id]], dtype=int)
    cones = [idx for idx in range(len(vtx_reindex)) if vtx_reindex[idx] in cones]

    # Get flip sequence
    flip_seq = np.array([])
    try:
        flip_seq = np.loadtxt(os.path.join(output_dir, name + "_flip_seq"), dtype=int)
    except:
        logger.error('Could not load flip sequence')

    # Generate initial similarity metric
    free_cones = []
    marked_metric_params = holonomy.MarkedMetricParameters()
    marked_metric, _ = holonomy.generate_marked_metric(V, F, V, F, Th_hat, rotation_form, free_cones, marked_metric_params)

    # Make overlay
    cut_h = []
    #_, V_o, F_o, uv_o, FT_o, is_cut_h, _, fn_to_f, endpoints = holonomy.generate_VF_mesh_from_marked_metric(V, F, Th_hat, marked_metric, cut_h)
    #vf_res = opt.generate_VF_mesh_from_metric(V, F, Th_hat, marked_metric, marked_metric.get_metric_coordinates(), cut_h, False)
    vf_res = opt.generate_VF_mesh_from_metric(V, F, Th_hat, marked_metric, reduced_metric_coords, cut_h, False)
    _, V_o, F_o, uv_o, FT_o, is_cut_h, _, fn_to_f_o, endpoints_o = vf_res

    # Save new meshes
    uv_mesh_path = os.path.join(output_dir, name + '_overlay_with_uv.obj')
    logger.info("Saving uv mesh at {}".format(uv_mesh_path))
    opt.write_obj_with_uv(uv_mesh_path, V_o, F_o, uv_o, FT_o)

    # Refine original mesh using overlay
    logger.info("Running refinement")
    refinement_mesh = opt.RefinementMesh(V_o, F_o, uv_o, FT_o, fn_to_f_o, endpoints_o)
    V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r = refinement_mesh.get_VF_mesh()

    # Save cut information
    simp_path = os.path.join(output_dir, name + '_is_cut_h')
    logger.info("Saving cut information at {}".format(simp_path))
    np.savetxt(simp_path, is_cut_h)

    # Save cut to singularity information
    # TODO Generate this from file data instead of pickle
    cut_to_sin_list = render.add_cut_to_sin(marked_metric.n, marked_metric.opp, marked_metric.to, cones, marked_metric.type, is_cut_h, vtx_reindex, build_double)
    simp_path = os.path.join(output_dir, name + '_cut_to_sin_list.pickle')
    logger.info("Saving cut to singularity information at {}".format(simp_path))
    with open(simp_path, 'wb') as file:
        pickle.dump(cut_to_sin_list, file)
    simp_path = os.path.join(output_dir, name + '_overlay_with_uv_cut_to_sin_list.pickle')
    logger.info("Saving cut to singularity information at {}".format(simp_path))
    with open(simp_path, 'wb') as file:
        pickle.dump(cut_to_sin_list, file)
    simp_path = os.path.join(output_dir, name + '_refined_with_uv_cut_to_sin_list.pickle')
    logger.info("Saving cut to singularity information at {}".format(simp_path))
    with open(simp_path, 'wb') as file:
        pickle.dump(cut_to_sin_list, file)

    # Write fn_to_f to file
    face_map_path = os.path.join(output_dir, name + '_fn_to_f')
    logger.info("Saving new to old face map at {}".format(face_map_path))
    np.savetxt(face_map_path, fn_to_f_o, fmt='%i')

    # Write vn_to_v to file
    vertex_map_path = os.path.join(output_dir, name + '_vn_to_v')
    logger.info("Saving trivial new to old vertex map at {}".format(vertex_map_path))
    vn_to_v = np.arange(len(uv_o))
    np.savetxt(vertex_map_path, vn_to_v, fmt='%i')

    # Write endpoints to file
    endpoints_path = os.path.join(output_dir, name + '_endpoints')
    logger.info("Saving endpoints at {}".format(endpoints_path))
    np.savetxt(endpoints_path, endpoints_o, fmt='%i')

    # Write combined refined mesh with uv
    uv_mesh_path = os.path.join(output_dir, name + '_refined_with_uv.obj')
    logger.info("Saving refined uv mesh at {}".format(uv_mesh_path))
    opt.write_obj_with_uv(uv_mesh_path, V_r, F_r, uv_r, FT_r)

def similarity_overlay_many(args):
    script_util.run_many(similarity_overlay_one, args)

def add_similarity_overlay_arguments(parser):
    alg_params = opt.AlgorithmParameters()
    ls_params = opt.LineSearchParameters()
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("--fit_field",      help="fit intrinsic cross field for rotation form",
                                                     action="store_true")
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")
    parser.add_argument(
        "--suffix",
        help="suffix for output files",
        default=""
    )

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Run optimization method")
    add_similarity_overlay_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel optimization method
    similarity_overlay_many(args)
