# Script to generate a table summarizing various energies and other metrics from output
# parameterizations from uv coordinates

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)

import optimize_impl.energies as energies
import script_util
import igl
import pandas as pd
import penner
import numpy as np
import multiprocessing


def add_error_table_arguments(parser):
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for output table"
    )
    parser.add_argument(
        "--uv_dir",
        help="path to the directory of meshes with uv coordinates"
    )
    parser.add_argument(
        "--suffix",
        help="suffix for output files",
        default=""
    )


def run_optimization_analysis(input_dir, uv_dir, fname, suffix=""):
    analysis_dict = {}
    
    analyses = [
        'uv_length_error',
        'min_overlay_area',
        'max_overlay_area',
        'min_uv_area',
        'max_uv_area',
        'num_faces',
        'num_overlay_faces',
        'is_manifold',
        'num_components',
        'max_sym_dir',
        'max_quadratic_sym_dir',
    ]

    # get output directory for mesh
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    if (suffix == ""):
        name = m
    else:
        name = m + '_' + suffix

    try:
        # Load input mesh information
        input_path = os.path.join(input_dir, m+'.obj')
        v3d_orig, f_orig = igl.read_triangle_mesh(input_path)

        # Get final output mesh
        uv_path = os.path.join(uv_dir, m + "_output", name + ".obj")
        v3d, uv, _, f, fuv, _ = igl.read_obj(uv_path)

        # embed uv
        uv_embed = np.zeros((len(uv), 3))
        uv_embed[:, :2] = uv[:, :2]
    except:
        return {}

    for analysis in analyses:
        try:
            if analysis == 'num_faces':
                analysis_dict[analysis] = len(f_orig)

            if analysis == 'num_overlay_faces':
                analysis_dict[analysis] = len(f)

            if analysis == 'is_manifold':
                analysis_dict[analysis] = igl.is_edge_manifold(f)

            if analysis == 'num_components':
                analysis_dict[analysis] = np.max(igl.face_components(fuv)) + 1

            if analysis == 'uv_length_error':
                analysis_dict[analysis] = penner.compute_uv_length_error(f, uv, fuv)

            if analysis == 'min_overlay_area':
                mesh_areas = 0.5 * igl.doublearea(v3d, f)
                analysis_dict[analysis] = np.min(mesh_areas)

            if analysis == 'max_overlay_area':
                mesh_areas = 0.5 * igl.doublearea(v3d, f)
                analysis_dict[analysis] = np.max(mesh_areas)

            if analysis == 'min_uv_area':
                uv_embed = np.zeros((len(uv), 3))
                uv_embed[:, :2] = uv[:, :2]
                uv_areas = 0.5 * igl.doublearea(uv_embed, fuv)
                analysis_dict[analysis] = np.min(uv_areas)

            if analysis == 'max_uv_area':
                uv_areas = 0.5 * igl.doublearea(uv_embed, fuv)
                analysis_dict[analysis] = np.max(uv_areas)

            if analysis == 'max_sym_dir':
                sym_dirichlet_energy = energies.sym_dirichlet_vf(
                    v3d, f, uv_embed, fuv) - 4
                analysis_dict[analysis] = np.max(sym_dirichlet_energy)

            if analysis == 'max_quadratic_sym_dir':
                quad_sym_dir_energy = energies.quadratic_sym_dirichlet_vf(
                    v3d, f, uv_embed, fuv)
                analysis_dict[analysis] = np.max(quad_sym_dir_energy)

        except:
            analysis_dict[analysis] = -1

    return analysis_dict

def error_table(args):

    # Create output directory for the mesh
    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, 'error_table.log')
    logger = script_util.get_logger(log_path)
    logger.info("Building error table")

    # get list of models
    models = []
    for fname in args['fname']:
        dot_index = fname.rfind(".")
        m = fname[:dot_index]
        models.append(m)

    # run analysis
    pool_args = [(args['input_dir'], args['uv_dir'], m, args['suffix']) for m in args['fname']]
    with multiprocessing.Pool(processes=32) as pool:
      all_analyses = pool.starmap(run_optimization_analysis, pool_args, chunksize=1)

    # Save dataframe to file
    analysis_df = pd.DataFrame(all_analyses, index=models)
    csv_path = os.path.join(
        args['output_dir'], 'analysis_' + args['suffix'] + '.csv')
    logger.info(f"Saving analysis table to {csv_path}")
    analysis_df.to_csv(csv_path)


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Generate error table for analysis")
    add_error_table_arguments(parser)
    args = vars(parser.parse_args())

    # Run method in parallel
    error_table(args)
