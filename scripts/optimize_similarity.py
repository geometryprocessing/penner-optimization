# Script to optimize a similarity metric with holonomy constraints

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import penner
import pickle, math
import igl
import optimization_scripts.script_util as script_util
import optimize_impl.render as render

def optimize_similarity_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index] 
    name = m
    V, F = igl.read_triangle_mesh(os.path.join(args['input_dir'], fname))
    Th_hat = np.loadtxt(os.path.join(args['input_dir'], name + "_Th_hat"), dtype=float)
    rotation_form = np.loadtxt(os.path.join(args['input_dir'], name + "_kappa_hat"), dtype=float)
    
    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, name+'_convert_to_vf.log')
    logger = script_util.get_logger(log_path)
    logger.info("Converting {} to vf".format(name))

    # Generate initial similarity metric
    free_cones = []
    fix_boundary = False
    set_holonomy_constraints = True
    similarity_metric = penner.generate_similarity_mesh(V, F, V, F, Th_hat, rotation_form, free_cones, fix_boundary, set_holonomy_constraints)

    # Get mesh information
    is_bd = igl.is_border_vertex(V, F)
    build_double = (np.sum(is_bd) != 0)
    _, vtx_reindex = opt.fv_to_double(V, F, V, F, Th_hat, [], False)

    # Get cones
    cones = np.array([id for id in range(len(Th_hat)) if np.abs(Th_hat[id]-2*math.pi) > 1e-15 and not is_bd[id]], dtype=int)
    cones = [idx for idx in range(len(vtx_reindex)) if vtx_reindex[idx] in cones]

    # Build energies
    energy_choice = args['similarity_energy_choice']
    if (energy_choice == "integrated"):
        opt_energy = penner.IntegratedEnergy(similarity_metric)
    elif (energy_choice == "coordinate"):
        num_coords = len(similarity_metric.get_reduced_metric_coordinates())
        num_form_coords = similarity_metric.n_homology_basis_loops()

        if (num_form_coords == 0):
            logger.error("Cannot optimize jump coordinates for genus 0")
            return

        coordinates = np.arange(num_coords - num_form_coords, num_coords)
        opt_energy = penner.CoordinateEnergy(similarity_metric, coordinates)
    else:
        logger.error("No valid energy selected")
        return

    # Perform optimization
    proj_params, opt_params = script_util.generate_parameters(args)
    opt_params.output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    if (args['optimization_method'] == 'metric'):
        optimized_similarity_metric = opt.optimize_metric(similarity_metric, opt_energy, proj_params, opt_params)
    if (args['optimization_method'] == 'shear'):
        shear_basis_matrix, _ = opt.compute_shear_dual_basis(similarity_metric)
        domain_matrix, codomain_matrix, domain_coords, codomain_coords = penner.compute_similarity_optimization_domain(similarity_metric, shear_basis_matrix)
        optimized_metric_coords = opt.optimize_domain_coordinates(similarity_metric, opt_energy, domain_matrix, codomain_matrix, domain_coords, codomain_coords, proj_params, opt_params)
        optimized_similarity_metric = similarity_metric.set_metric_coordinates(optimized_metric_coords)

    # Save metric coordinate information
    simp_path = os.path.join(output_dir, name + '_metric_coords')
    logger.info("Saving metric coordinates at {}".format(simp_path))
    np.savetxt(simp_path, optimized_similarity_metric.get_reduced_metric_coordinates())

    # Get overlay and write to file
    cut_h = []
    _, V_o, F_o, uv_o, FT_o, is_cut_h, _, fn_to_f, endpoints = penner.generate_VF_mesh_from_similarity_metric(V, F, Th_hat, optimized_similarity_metric, cut_h)

    # Save new meshes
    uv_mesh_path = os.path.join(output_dir, name + '_overlay_with_uv.obj')
    logger.info("Saving uv mesh at {}".format(uv_mesh_path))
    opt.write_obj_with_uv(uv_mesh_path, V_o, F_o, uv_o, FT_o)

    # Save cut information
    simp_path = os.path.join(output_dir, name + '_is_cut_h')
    logger.info("Saving cut information at {}".format(simp_path))
    np.savetxt(simp_path, is_cut_h)

    # Save cut to singularity information
    # TODO Generate this from file data instead of pickle
    cut_to_sin_list = render.add_cut_to_sin(similarity_metric.n, similarity_metric.opp, similarity_metric.to, cones, similarity_metric.type, is_cut_h, vtx_reindex, build_double)
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
    face_map_path = os.path.join(output_dir, name + '_overlay_with_uv_fn_to_f')
    logger.info("Saving new to old face map at {}".format(face_map_path))
    np.savetxt(face_map_path, fn_to_f, fmt='%i')

    # Write vn_to_v to file
    vertex_map_path = os.path.join(output_dir, name + '_overlay_with_uv_vn_to_v')
    logger.info("Saving trivial new to old vertex map at {}".format(vertex_map_path))
    vn_to_v = np.arange(len(uv_o))
    np.savetxt(vertex_map_path, vn_to_v, fmt='%i')

    # Write endpoints to file
    endpoints_path = os.path.join(output_dir, name + '_overlay_with_uv_endpoints')
    logger.info("Saving endpoints at {}".format(endpoints_path))
    np.savetxt(endpoints_path, endpoints, fmt='%i')

def optimize_similarity_many(args):
    script_util.run_many(optimize_similarity_one, args)

def add_optimize_similarity_arguments(parser):
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("--similarity_energy_choice",     help="similarity energy to optimize", default="integrated")
    parser.add_argument("--optimization_method",
                        help="optimization method to use",
                        default="metric")
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")
    script_util.add_parameter_arguments(parser)

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Run optimization method")
    add_optimize_similarity_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel optimization method
    optimize_similarity_many(args)
