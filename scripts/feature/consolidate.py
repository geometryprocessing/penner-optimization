# Script to analyze the final parametrized mesh output of the method

import pandas as pd
import penner
import numpy as np
import os, math, shutil
import sys, multiprocessing
import igl
base_dir = os.path.dirname(__file__)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts')
sys.path.append(script_dir)
module_dir = os.path.join(base_dir, '..', 'py')
sys.path.append(module_dir)
import script_util
import error_table
import optimize_relaxed_angles

def add_arguments(parser):
    optimize_relaxed_angles.add_arguments(parser)

    parser.add_argument("--failure_dir",
                        help="directory for failure case records")


def run_one(output_dir, fname, args):
    # list feature analyses to run
    analyses = [
        'seamless_error',
        'length_error',
        'feature_error',
        'max_error',
        'max_hard_error',
        'found_quad_mesh',
        'found_robust_quad_mesh',
        'min_angle',
        'max_angle',
    ]

    # get output directory for mesh
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    mesh_output_dir = script_util.get_mesh_output_directory(output_dir, m)
    name = m

    try:
        # get per iteration data
        iteration_data_dir = os.path.join(mesh_output_dir, 'feature_iteration_log.csv')
        iteration_data = pd.read_csv(iteration_data_dir)
    except:
        iteration_data = None

    try:
        # get per stability data
        stability_data_dir = os.path.join(mesh_output_dir, 'iteration_stability_log.csv')
        stability_data = pd.read_csv(stability_data_dir)
    except:
        stability_data = None

    try:
        # get uv iteration data
        uv_data_dir = os.path.join(mesh_output_dir, 'uv_analysis.csv')
        uv_data = pd.read_csv(uv_data_dir)
    except:
        uv_data = None

    analysis_dict = {}
    for analysis in analyses:
        try:
            if analysis == 'seamless_error':
                analysis_dict[analysis] = float(uv_data['seamless_error'])

            if analysis == 'length_error':
                analysis_dict[analysis] = float(uv_data['length_error'])

            if analysis == 'feature_error':
                analysis_dict[analysis] = float(uv_data['feature_error'])

            if analysis == 'max_error':
                analysis_dict[analysis] = float(iteration_data['max_error'].tail(1))

            if analysis == 'max_hard_error':
                analysis_dict[analysis] = float(iteration_data['max_hard_error'].tail(1))

            if analysis == 'found_quad_mesh':
                if os.path.isfile(os.path.join(mesh_output_dir, "qm.obj")):
                    analysis_dict[analysis] = True
                elif not os.path.isfile(os.path.join(args['output_dir'], 'status_logs', m  + "_refined_with_uv_end")):
                    #print("{} still running".format(m))
                    #analysis_dict[analysis] = "RUNNING"
                    analysis_dict[analysis] = False
                else:
                    analysis_dict[analysis] = False

            if analysis == 'found_robust_quad_mesh':
                if os.path.isfile(os.path.join(mesh_output_dir, m + "_quad.obj")):
                    analysis_dict[analysis] = True
                elif os.path.isfile(os.path.join(mesh_output_dir, m + "_refined_with_uv_quad.obj")):
                    analysis_dict[analysis] = True
                elif os.path.isfile(os.path.join(mesh_output_dir, m + "_refined_with_uv_connected_quad.obj")):
                    analysis_dict[analysis] = True
                elif os.path.isfile(os.path.join(output_dir, 'connected', 'output', m + "_refined_with_uv_connected_robust_quad.obj")):
                    analysis_dict[analysis] = True
                else:
                    analysis_dict[analysis] = False

                if args['failure_dir'] != "" and not analysis_dict[analysis]:
                    os.makedirs(args['failure_dir'], exist_ok=True)
                    # check for degeneracy
                    try:
                        uv_dir = args['uv_dir']
                        uv_mesh = os.path.join(uv_dir, m + "_output", name + ".obj")
                        v3d, uv, _, f, fuv, _ = igl.read_obj(uv_mesh)
                    except:
                        print("Could not open uv at {}".format(uv_dir))
                        print("Could not open uv at {}".format(uv_mesh))
                        continue 
                    uv_length_error, uv_angle_error, uv_length, uv_angle = penner.compute_seamless_error(f, uv, fuv)
                    if (np.max(uv_length_error) > 1e-8) or (np.max(uv_angle_error) > 1e-8):
                        print("Mesh is not seamless")
                        continue
                    if (np.isnan(uv_length_error).any()) or (np.isnan(uv_angle_error).any()):
                        print("Mesh is not seamless")
                        continue

                    print("Copying {}".format(mesh_output_dir))
                    #shutil.copytree(mesh_output_dir, os.path.join(args['failure_dir']))
                    copy_all = False
                    if copy_all:
                        shutil.copytree(
                            mesh_output_dir,
                            os.path.join(args['failure_dir'], m + '_output'),
                            dirs_exist_ok=True)
                    else:
                        shutil.copy(
                            os.path.join(mesh_output_dir, name + '.obj'),
                            args['failure_dir'])
                        shutil.copy(
                            os.path.join(mesh_output_dir, name + '_igmtest_QGP.log'),
                            args['failure_dir'])
                    print("Done copying {}".format(mesh_output_dir))
                    


            if analysis == 'min_angle':
                analysis_dict[analysis] = float(stability_data['min_corner_angle'].tail(1))

            if analysis == 'max_angle':
                analysis_dict[analysis] = float(stability_data['max_corner_angle'].tail(1))

        except:
            analysis_dict[analysis] = -1

    return analysis_dict

def run_many(args):
    # Create output directory for the mesh
    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, 'feature_data.log')
    logger = script_util.get_logger(log_path)
    logger.info("Building feature data table")

    # get list of models
    models = []
    for fname in args['fname']:
        dot_index = fname.rfind(".")
        m = fname[:dot_index]
        models.append(m)

    # get statistics
    pool_args = [(args['output_dir'], m, args) for m in args['fname']]
    with multiprocessing.Pool(processes=32) as pool:
      all_analyses = pool.starmap(run_one, pool_args, chunksize=1)

    # Save dataframe to file
    analysis_df = pd.DataFrame(all_analyses, index=models)
    csv_path = os.path.join(
        args['output_dir'], 'feature_data.csv')
    logger.info(f"Saving analysis table to {csv_path}")
    analysis_df.to_csv(csv_path)

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Consolidating analysis of feature aligned parametrizations")
    add_arguments(parser)
    args = vars(parser.parse_args())

    run_many(args)
