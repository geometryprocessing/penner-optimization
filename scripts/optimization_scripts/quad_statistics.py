# Script to generate a table summarizing various metrics from a quad mesh

import pandas as pd
import numpy as np
import os, math
import sys, multiprocessing
import igl, argparse
base_dir = os.path.dirname(__file__)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts')
sys.path.append(script_dir)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts', 'optimization_scripts')
sys.path.append(script_dir)
module_dir = os.path.join(base_dir, '..', 'py')
sys.path.append(module_dir)
import script_util

def add_arguments(parser):
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for output"
    )

def get_statistics(args, mesh_filename):
    # Build dictionary of statistics
    statistics = [
        'faces',
    ]
    statistics_dict = {}

    try:
        # Get mesh
        V, _, _, F, _, _ = igl.read_obj(mesh_filename)
    except:
        print(f"Cannot open mesh at {mesh_filename}")
        return {}

    for statistic in statistics:
        try:
            if statistic == 'faces':
                statistics_dict[statistic] = len(F)
        except:
            print("Cannot compute statistic")
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
    files = []
    for fname in args['fname']:
        dot_index = fname.rfind(".")
        m = fname[:dot_index]
        models.append(m)
        mesh_output_dir = script_util.get_mesh_output_directory(output_dir, m)
        files.append(os.path.join(mesh_output_dir, "qm.obj"))

    # get statistics
    pool_args = [(args, m) for m in files]
    with multiprocessing.Pool(processes=16) as pool:
      all_statistics = pool.starmap(get_statistics, pool_args, chunksize=1)

    # save as csv file
    logger.info("Saving statistics table")
    statistics_df = pd.DataFrame(all_statistics, index=models)
    csv_path = os.path.join(output_dir, 'quad_statistics.csv')
    statistics_df.to_csv(csv_path)

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Generate quad mesh statistics")
    #parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = vars(parser.parse_args())
    files = os.listdir(args['input_dir'])

    # get statistics
    pool_args = [(args, os.path.join(args['input_dir'], m)) for m in files]
    with multiprocessing.Pool(processes=8) as pool:
      all_statistics = pool.starmap(get_statistics, pool_args, chunksize=1)

    # save as csv file
    output_dir = args['output_dir']
    models = [f[:f.find('.')] for f in files]
    statistics_df = pd.DataFrame(all_statistics, index=models)
    csv_path = os.path.join(output_dir, 'quad_statistics.csv')
    statistics_df.to_csv(csv_path)
