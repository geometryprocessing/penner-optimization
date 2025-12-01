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
import multiprocessing 
import optimization_scripts.script_util as script_util

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def run_one(fname, args):
    # list feature analyses to run
    analyses = [
        'seamless_error',
        'length_error',
        'angle_error',
        'feature_error',
        'min_area',
        'height',
        'min_angle',
        'max_angle',
        'num_flipped',
        'num_boundary',
    ]

    # get output directory for mesh
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']

    # Load uv information
    try:
        uv_dir = args['uv_dir']
        uv_path = os.path.join(uv_dir, m + "_output", name + ".obj")
        v3d, uv, _, f, fuv, _ = igl.read_obj(uv_path)
        feature_edges = feature.load_mesh_edges(uv_path)
    except:
        return {}

    # run basic analysis
    uv_embed = np.zeros((len(uv), 3))
    uv_embed[:, :2] = uv[:, :2]
    uv_length_error, uv_seamless_error, uv_length, uv_angle = feature.compute_seamless_error(f, uv, fuv)
    uv_angle_error = feature.compute_angle_error(f, uv, fuv)
    feature_error = feature.compute_feature_alignment(f, uv, fuv, feature_edges)
    uv_areas = 0.5 * igl.doublearea(uv_embed, fuv)
    corner_angles = igl.internal_angles(uv_embed, fuv)
    height = feature.compute_height(uv, fuv)

    analysis_dict = {}
    for analysis in analyses:
        try:
            if analysis == 'length_error':
                analysis_dict['length_error'] = np.max(uv_length_error)

            if analysis == 'seamless_error':
                analysis_dict['seamless_error'] = np.max(uv_seamless_error)

            if analysis == 'angle_error':
                analysis_dict['angle_error'] = np.max(uv_angle_error)

            if analysis == 'feature_error':
                analysis_dict['feature_error'] = np.max(feature_error) if feature_error else 0.

            if analysis == 'min_area':
                analysis_dict['min_area'] = np.min(uv_areas)

            if analysis == 'height':
                analysis_dict['height'] = np.min(height)

            if analysis == 'min_angle':
                analysis_dict['min_angle'] = np.min(corner_angles)

            if analysis == 'max_angle':
                analysis_dict['max_angle'] = np.max(corner_angles)

            if analysis == 'num_flipped':
                analysis_dict['num_flipped'] = feature.check_flip(uv, fuv)

            if analysis == 'num_boundary':
                analysis_dict[analysis] = len(igl.boundary_facets(f))

        except:
            analysis_dict[analysis] = -1

    return analysis_dict

def run_many(args):
    # Create output directory for the mesh
    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, 'uv_analysis.log')
    logger = script_util.get_logger(log_path)
    logger.info("Building uv analysis data table")

    # get list of models
    models = []
    for fname in args['fname']:
        dot_index = fname.rfind(".")
        m = fname[:dot_index]
        models.append(m)

    # get statistics
    pool_args = [(m, args) for m in args['fname']]
    with multiprocessing.Pool(processes=32) as pool:
      all_analyses = pool.starmap(run_one, pool_args, chunksize=1)

    # Save dataframe to file
    analysis_df = pd.DataFrame(all_analyses, index=models)
    csv_path = os.path.join(
        args['output_dir'], 'uv_analysis_' + args['suffix'] + '.csv')
    logger.info(f"Saving analysis table to {csv_path}")
    analysis_df.to_csv(csv_path)

def add_arguments(parser):
    alg_params = penner.NewtonParameters()
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Optimize angles with relaxed alignment")
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel method
    run_many(args)
