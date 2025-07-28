# Script to analyze the final parametrized mesh output of the method

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
import error_table
import optimize_aligned_angles

def add_arguments(parser):
    optimize_aligned_angles.add_arguments(parser)

def run_one(input_dir, uv_dir, fname, args):
    suffix = args['suffix']
    analysis_dict = error_table.run_optimization_analysis(input_dir, uv_dir, fname, suffix)

    # list feature analyses to run
    analyses = [
        'max_cut_alignment_error',
        'avg_cut_alignment_error',
        'avg_relaxed_alignment_error',
        'num_relaxed_halfedges',
    ]
    analyses = [
        'seamless_error',
    ]

    # get output directory for mesh
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    if (suffix == ""):
        name = m
    else:
        name = m + '_' + suffix

    try:
        # Load input mesh info
        input_path = os.path.join(input_dir, m+'.obj')
        v3d_orig, f_orig = igl.read_triangle_mesh(input_path)

        # Get final output mesh
        uv_path = os.path.join(uv_dir, m + "_output", name + ".obj")
        v3d, uv, _, f, fuv, _ = igl.read_obj(uv_path)

        # embed uv
        uv_embed = np.zeros((len(uv), 3))
        uv_embed[:, :2] = uv[:, :2]

        # get endpoints
        endpoints_path = os.path.join(uv_dir, m + "_output", m + '_endpoints')
        endpoints = np.loadtxt(endpoints_path, dtype=int)
    except:
        return {}

    for analysis in analyses:
        try:
            if analysis == 'seamless_error':
                uv_length_error, uv_angle_error, uv_length, uv_angle = penner.compute_seamless_error(f, uv, fuv);
                analysis_dict[analysis] = np.max(uv_angle_error)

            if analysis == 'max_cut_alignment_error':
                F_o_is_cut = penner.generate_overlay_cut_mask(f, endpoints, F_cut, F_is_cut)
                cut_corners = penner.compute_mask_corners(F_o_is_cut)
                uv_alignment = penner.compute_uv_alignment(uv, fuv, cut_corners)

                #for i, v in enumerate(uv_alignment):
                #    if (v > 1e-12):
                #        print(f"{i}th corner {cut_corners[i]} has alignment {v}")

                analysis_dict[analysis] = np.max(uv_alignment)

            if analysis == 'avg_cut_alignment_error':
                F_o_is_cut = penner.generate_overlay_cut_mask(f, endpoints, F_cut, F_is_cut)
                cut_corners = penner.compute_mask_corners(F_o_is_cut)
                uv_alignment = penner.compute_uv_alignment(uv, fuv, cut_corners)
                analysis_dict[analysis] = np.average(uv_alignment)


            if analysis == 'avg_relaxed_alignment_error':
                F_o_is_relaxed = penner.generate_overlay_cut_mask(f, endpoints, F_cut, F_is_relaxed)
                cut_corners = penner.compute_mask_corners(F_o_is_relaxed)
                uv_alignment = penner.compute_uv_alignment(uv, fuv, cut_corners)
                analysis_dict[analysis] = np.average(uv_alignment)

            if analysis == 'num_relaxed_halfedges':
                cut_corners = penner.compute_mask_corners(F_is_relaxed)
                analysis_dict[analysis] = len(cut_corners)
        except:
            analysis_dict[analysis] = -1

    return analysis_dict

def run_many(args):
    # Create output directory for the mesh
    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, 'feature_analysis.log')
    logger = script_util.get_logger(log_path)
    logger.info("Building statistics table")

    # get list of models
    models = []
    for fname in args['fname']:
        dot_index = fname.rfind(".")
        m = fname[:dot_index]
        models.append(m)

    # get statistics
    pool_args = [(args['input_dir'], args['uv_dir'], m, args) for m in args['fname']]
    with multiprocessing.Pool(processes=32) as pool:
      all_analyses = pool.starmap(run_one, pool_args, chunksize=1)

    # Save dataframe to file
    analysis_df = pd.DataFrame(all_analyses, index=models)
    csv_path = os.path.join(
        args['output_dir'], 'feature_analysis_' + args['suffix'] + '.csv')
    logger.info(f"Saving analysis table to {csv_path}")
    analysis_df.to_csv(csv_path)

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Generate analysis of feature aligned parametrizations")
    add_arguments(parser)
    args = vars(parser.parse_args())

    run_many(args)
