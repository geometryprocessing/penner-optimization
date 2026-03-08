# Script to generate a table summarizing various energies and other metrics from output
# Penner coordinates

import pandas as pd
import numpy as np
import os, math
import sys, multiprocessing
import igl
import random
base_dir = os.path.dirname(__file__)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts')
sys.path.append(script_dir)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts', 'optimization_scripts')
sys.path.append(script_dir)
module_dir = os.path.join(base_dir, '..', 'py')
sys.path.append(module_dir)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'py')
sys.path.append(script_dir)
script_dir = os.path.join(base_dir, '..', 'scripts')
sys.path.append(script_dir)
import script_util
import penner


def add_statistics_arguments(parser):
    #parser.add_argument(
    #    "-i", "--input_dir",
    #    help="directory for input meshes"
    #)
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for output"
    )

def get_statistics(input_dir, output_dir, fname):
    # Build dictionary of statistics
    statistics = [
        'genus',
        'faces',
        'is_manifold',
        'boundary',
        #'iter',
        #'solves',
        #'max_error',
        #'max_hard_error',
        #'min_cone',
        #'max_cone',
        #'RMSRE',
    ]
    do_relaxation = False
    if do_relaxation:
        statistics.append('relaxed_max_error')
        statistics.append('relaxed_junctions')

    statistics_dict = {}

    # get output directory for mesh
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    mesh_output_dir = script_util.get_mesh_output_directory(output_dir, m)

    try:
        # Get mesh
        V, F = igl.read_triangle_mesh(os.path.join(input_dir, fname))
    except:
        return {}

    for statistic in statistics:
        try:
            if statistic == 'genus':
                n_v = len(V)
                n_f = len(F)
                n_e = len(igl.edges(F))
                euler_char = n_v - n_e + n_f
                statistics_dict[statistic] = int((2 - euler_char) / 2)

            if statistic == 'is_manifold':
                statistics_dict[statistic] = feature.is_manifold(F)

            if statistic == 'boundary':
                statistics_dict[statistic] = len(igl.boundary_facets(F))

            if statistic == 'faces':
                statistics_dict[statistic] = len(F)

            if statistic == 'min_cone':
                statistics_dict[statistic] = np.min(Th_hat)

            if statistic == 'max_cone':
                statistics_dict[statistic] = np.max(Th_hat)

            if statistic == 'junctions':
                statistics_dict[statistic] = len(junctions)

            if statistic == 'relaxed_junctions':
                junctions_string = ""
                for datum in relaxed_junctions:
                    junctions_string += str(len(datum)) + ";"
                statistics_dict[statistic] = junctions_string

            if statistic == 'relaxed_max_error':
                max_error_string = ""
                for datum in relaxed_iteration_data:
                    max_error_string += str(float(datum['max_error'].tail(1))) + ";"
                statistics_dict[statistic] = max_error_string

            if statistic == 'RMSRE':
                statistics_dict[statistic] = float(iteration_data['rmsre'].tail(1))

            if statistic == 'max_hard_error':
                statistics_dict[statistic] = float(iteration_data['max_hard_error'].tail(1))

            if statistic == 'solves':
                statistics_dict[statistic] = float(iteration_data['solves'].tail(1))

            if statistic == 'iter':
                statistics_dict[statistic] = int(iteration_data['num_iter'].tail(1))

            if statistic == 'max_error':
                statistics_dict[statistic] = float(iteration_data['max_error'].tail(1))
        except:
            statistics_dict[statistic] = -1

    return statistics_dict

def run_statistics(args):
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

    # get statistics
    pool_args = [(args['input_dir'], args['output_dir'], m) for m in args['fname']]
    with multiprocessing.Pool(processes=32) as pool:
      all_statistics = pool.starmap(get_statistics, pool_args, chunksize=1)

    # save as csv file
    statistics_df = pd.DataFrame(all_statistics, index=models)
    csv_path = os.path.join(output_dir, 'statistics.csv')
    statistics_df.to_csv(csv_path)

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Generate statistics")
    add_statistics_arguments(parser)
    args = vars(parser.parse_args())

    if 'fname' not in args or not args['fname']:
        files = os.listdir(args['input_dir'])
        obj_files = [f for f in files if f.endswith(".obj")]
        random.shuffle(obj_files)
        args['fname'] = obj_files

    run_statistics(args)
