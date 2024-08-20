# Script to project a marked metric to holonomy and fixed boundary constraints

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import penner
import igl
import pickle, math, logging
import optimization_scripts.script_util as script_util
import optimize_impl.render as render

def run_one(args, fname):
    # get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    name = m

    # create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # get logger
    log_path = os.path.join(output_dir, name+'_optimize_fixed_boundary.log')
    logger = script_util.get_logger(log_path)
    logger.info("Projecting {} to constraints".format(name))

    # open mesh
    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['input_dir'], fname))
    except:
        logger.info("Could not open mesh data")
        return

    # build common data structures
    free_cones = []
    Th_hat_flat = 2 * np.pi * np.ones(len(V))
    rotation_form = []
    m, vtx_reindex = penner.generate_mesh(V, F, V, F, Th_hat_flat, free_cones)
    marked_metric_params = penner.MarkedMetricParameters()
    boundary_constraint_generator = penner.BoundaryConstraintGenerator(m)

    # build isometric boundary constraints
    if (args['boundary_type'] == 'isometric'):
        logger.info("Building isometric boundary metric")
        Th_hat = Th_hat_flat
        free_cones = penner.find_boundary_vertices(m, vtx_reindex)
    # build polygon boundary constraints
    if (args['boundary_type'] == 'polygon'):
        logger.info("Building polygon boundary metric")
        Th_hat = penner.generate_polygon_cones(m, vtx_reindex, args['num_corners'], False)
        boundary_constraint_generator.mark_cones_as_junctions();
        boundary_constraint_generator.set_uniform_feature_lengths(0.);

    # build angle constraint metric
    logger.info("Building marked metric")
    halfedge_matrix, ell = boundary_constraint_generator.build_boundary_constraint_system();
    marked_metric, _ = penner.generate_marked_metric(V, F, V, F, Th_hat, rotation_form, free_cones, marked_metric_params)

    # generate boundary paths
    logger.info("Building boundary paths")
    boundary_paths, boundary_map = penner.build_boundary_paths(m)
    boundary_constraint_system = halfedge_matrix * boundary_map

    # generate dirichlet constraint mesh
    logger.info("Building dirichlet metric")
    dirichlet_metric = penner.DirichletPennerConeMetric(
        marked_metric, 
        boundary_paths,
        boundary_constraint_system,
        ell)

    # Initialize parameters
    alg_params = penner.NewtonParameters()
    alg_params.error_eps = args['conf_error_eps']
    alg_params.max_itr = args['conf_max_itr']
    alg_params.do_reduction = args['do_reduction']
    alg_params.reset_lambda = args['reset_lambda']
    alg_params.lambda0 = args['lambda_init']
    alg_params.max_time = args['max_time']
    alg_params.solver = 'ldlt'
    alg_params.output_dir = output_dir
    alg_params.log_level = 6
    alg_params.error_log = True

    # Project to constraint, undoing flips to restore initial connectivity
    logger.info("Optimizing metric")
    optimized_metric = penner.optimize_metric_angles(dirichlet_metric, alg_params)

    # Return if no output needed
    if args['skip_output']:
        return

    # Save metric coordinate information
    output_path = os.path.join(output_dir, name + '_metric_coords')
    logger.info("Saving metric coordinates at {}".format(output_path))
    reduced_metric_coords = optimized_metric.get_reduced_metric_coordinates()
    np.savetxt(output_path, reduced_metric_coords)

    # Make overlay
    logger.info("Making overlay")
    cut_h = []
    vf_res = penner.generate_VF_mesh_from_metric(V, F, Th_hat, marked_metric, reduced_metric_coords, cut_h, False)
    _, V_o, F_o, uv_o, FT_o, is_cut_h, _, fn_to_f_o, endpoints_o = vf_res

    # Save new meshes
    uv_mesh_path = os.path.join(output_dir, name + '_overlay_with_uv.obj')
    logger.info("Saving uv mesh at {}".format(uv_mesh_path))
    penner.write_obj_with_uv(uv_mesh_path, V_o, F_o, uv_o, FT_o)

    # Refine original mesh using overlay
    logger.info("Running refinement")
    refinement_mesh = penner.RefinementMesh(V_o, F_o, uv_o, FT_o, fn_to_f_o, endpoints_o)
    V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r = refinement_mesh.get_VF_mesh()

    # Write combined refined mesh with uv
    uv_mesh_path = os.path.join(output_dir, name + '_refined_with_uv.obj')
    logger.info("Saving refined uv mesh at {}".format(uv_mesh_path))
    penner.write_obj_with_uv(uv_mesh_path, V_r, F_r, uv_r, FT_r)

    # Save cut information
    simp_path = os.path.join(output_dir, name + '_is_cut_h')
    logger.info("Saving cut information at {}".format(simp_path))
    np.savetxt(simp_path, is_cut_h)

    # Save cut to singularity information
    # TODO Generate this from file data instead of pickle
    cones = []
    build_double=True
    cut_to_sin_list = render.add_cut_to_sin(marked_metric.n, marked_metric.opp, marked_metric.to, cones, marked_metric.type, is_cut_h, vtx_reindex, build_double)
    simp_path = os.path.join(output_dir, name + '_cut_to_sin_list.pickle')
    logger.info("Saving cut to singularity information at {}".format(simp_path))
    with open(simp_path, 'wb') as file:
        pickle.dump(cut_to_sin_list, file)


def run_many(args):
    script_util.run_many(run_one, args)

def add_arguments(parser):
    alg_params = penner.AlgorithmParameters()
    ls_params = penner.LineSearchParameters()
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("--boundary_type",      help="type of boundary constraint to use",
                                                     default='isometric')
    parser.add_argument("--conf_error_eps",      help="maximum error for conformal projection",
                                                     type=float, default=alg_params.error_eps)
    parser.add_argument("--max_time",      help="maximum time for projection",
                                                     type=float, default=1e10)
    parser.add_argument("-m", "--conf_max_itr",   help="maximum number of iterations for the conformal method",
                                                     type=int, default=alg_params.max_itr)
    parser.add_argument("--do_reduction",      help="do reduction for conformal step",
                                                     type=bool, default=ls_params.do_reduction)
    parser.add_argument("--reset_lambda",      help="reset lambda for each conformal step",
                                                     type=bool, default=ls_params.reset_lambda)
    parser.add_argument("--num_corners",      help="number of corners for polygon boundary",
                                                     type=int, default=0)
    parser.add_argument("--lambda_init",      help="initial lambda",
                                                     type=bool, default=ls_params.lambda0)
    parser.add_argument("--skip_output",      help="don't write metric output if true",
                                                     action="store_true")
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Run optimization method")
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel optimization method
    run_many(args)
