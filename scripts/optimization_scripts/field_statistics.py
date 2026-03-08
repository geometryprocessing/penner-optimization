# Script to generate a table summarizing various energies and other metrics from output
# Penner coordinates

import pandas as pd
import penner
import numpy as np
import os, math
import sys, multiprocessing
import igl
base_dir = os.path.dirname(__file__)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts')
sys.path.append(script_dir)
module_dir = os.path.join(base_dir, '..', 'py')
sys.path.append(module_dir)
import script_util


def add_arguments(parser):
    parser.add_argument(
        "-i", "--input_dir",
        help="directory for input meshes"
    )
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for output"
    )

def run_one(args, fname):
    # Build dictionary of statistics
    statistics = [
        'num_faces',
        'num_features',
        'genus',
        'num_hard_features',
    ]
    compute_cones = False
    if compute_cones:
        statistics += [
            'min_cone',
            'max_cone',
        ]

    statistics_dict = {}

    # get output directory for mesh
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    name = m

    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['field_dir'], fname))

        # load feature edges
        feature_edges = np.loadtxt(os.path.join(args['field_dir'], name + "_features"), dtype=int)
        spanning_edges = np.loadtxt(os.path.join(args['field_dir'], name + "_hard_features"), dtype=int)

        # load frame field
        reference_field, theta, kappa, period_jump = feature.load_frame_field(os.path.join(args['field_dir'], name + ".ffield"))
    except:
        return {}

    # generate hard feature constraints
    feature_finder = feature.FeatureFinder(V, F)
    feature_finder.mark_features(feature_edges)

    # generate cut mesh and feature masks
    if compute_cones:
        V_cut, F_cut, V_map, F_is_feature = feature_finder.generate_feature_cut_mesh()

        # build metric with full constraints and marked feature corners
        marked_metric_params = penner.MarkedMetricParameters()
        marked_metric_params.remove_trivial_torus = False
        marked_metric_params.use_log_length = True
        marked_metric_params.remove_loop_constraints = True
        cut_metric_generator = feature.CutMetricGenerator(V_cut, F_cut, marked_metric_params, [])
        cut_metric_generator.set_fields(F_cut, reference_field, theta, kappa, period_jump)
        embedding_metric, vtx_reindex, face_reindex, rotation_form, Th_hat = cut_metric_generator.get_aligned_metric(V_map, marked_metric_params)

    for statistic in statistics:
        try:
            if statistic == 'num_faces':
                statistics_dict[statistic] = len(F)

            if statistic == 'genus':
                statistics_dict[statistic] = int((2 - igl.euler_characteristic(F)) / 2)

            if statistic == 'num_features':
                statistics_dict[statistic] = len(feature_edges)

            if statistic == 'num_hard_features':
                statistics_dict[statistic] = len(spanning_edges)

            if statistic == 'min_cone':
                statistics_dict[statistic] = np.min(Th_hat)

            if statistic == 'max_cone':
                statistics_dict[statistic] = np.max(Th_hat)
        except:
            statistics_dict[statistic] = -1

    return statistics_dict

def run_many(args):
    # Create output directory for the mesh
    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, 'statistics.log')
    logger = script_util.get_logger(log_path)
    logger.info("Building statistics table")

    # get list of models
    models = []
    for fname in args['fname']:
        dot_index = fname.rfind(".")
        m = fname[:dot_index]
        models.append(m)

    logger.info("Statistics for {} models".format(len(models)))
    # get statistics
    pool_args = [(args, m) for m in args['fname']]
    with multiprocessing.Pool(processes=32) as pool:
      all_statistics = pool.starmap(run_one, pool_args, chunksize=1)

    # save as csv file
    statistics_df = pd.DataFrame(all_statistics, index=models)
    csv_path = os.path.join(output_dir, 'field_statistics.csv')
    logger.info("Saving to {}".format(csv_path))
    statistics_df.to_csv(csv_path)

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Generate statistics")
    add_arguments(parser)
    args = vars(parser.parse_args())

    run_many(args)
