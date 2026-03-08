# Script to project a marked metric to holonomy constraints with feature alignment

import os, sys
base_dir = os.path.dirname(__file__)
module_dir = os.path.join(base_dir, '..', 'py')
sys.path.append(module_dir)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization')
sys.path.append(script_dir)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts')
sys.path.append(script_dir)
opt_script_dir = os.path.join(script_dir, 'optimization_scripts')
import optimize_angles
import numpy as np
import argparse
import pandas as pd
import penner
import igl
import multiprocessing
import optimization_scripts.script_util as script_util

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def run_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    name = m

    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['field_dir'], fname))
    except:
        return

    # load feature edges
    try:
        feature_edges = np.loadtxt(os.path.join(args['field_dir'], name + "_features"), dtype=int)
        spanning_edges = np.loadtxt(os.path.join(args['field_dir'], name + "_hard_features"), dtype=int)
    except:
        return {}

    # load frame field
    try:
        reference_field, theta, kappa, period_jump = feature.load_frame_field(os.path.join(args['field_dir'], name + ".ffield"))
    except:
        return

    # load optimized metric
    try:
        mesh_metric_dir = script_util.get_mesh_output_directory(args['lambdas_dir'], m)
        metric_coords = np.loadtxt(os.path.join(mesh_metric_dir, name + '_metric_coords'), dtype=float)
    except:
        return {}

    # build optimizer and get the initial metric
    regularization_factor = 1.
    use_minimal_forest = False
    aligned_metric_generator = feature.AlignedMetricGenerator(
        V,
        F,
        feature_edges,
        spanning_edges,
        reference_field,
        theta,
        kappa,
        period_jump,
        regularization_factor,
        use_minimal_forest)
    metric_init = aligned_metric_generator.get_metric()

    # flatten and exponentiate the logarithmic coordinates
    l = np.exp(metric_coords.flatten() / 2.)
    l_init = np.exp(metric_init.flatten() / 2.)
    
    # compute the RMSRE from the coordinates
    analysis_dict = {}
    analysis_dict['rmsre'] = penner.root_mean_square_relative_error(l, l_init)

    return analysis_dict

def run_many(args):
    # Create output directory for the mesh
    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # get list of models
    models = []
    for fname in args['fname']:
        dot_index = fname.rfind(".")
        m = fname[:dot_index]
        models.append(m)

    # get statistics
    pool_args = [(args, m) for m in args['fname']]
    with multiprocessing.Pool(processes=48) as pool:
      all_analyses = pool.starmap(run_one, pool_args, chunksize=1)

    # Save dataframe to file
    analysis_df = pd.DataFrame(all_analyses, index=models)
    csv_path = os.path.join(
        args['output_dir'], 'distortion_data.csv')
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
