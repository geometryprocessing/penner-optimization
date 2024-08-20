# Script to generate a table summarizing various energies and other metrics from output
# parameterizations from uv coordinates

import optimize_impl.energies as energies
import script_util
import igl
import pandas as pd
import penner
import numpy as np
import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)


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


def error_table(args):
    error_dict = {}
    models = []
    error_dict['uv_length_error'] = []
    error_dict['min_overlay_area'] = []
    error_dict['max_overlay_area'] = []
    error_dict['min_uv_area'] = []
    error_dict['max_uv_area'] = []
    error_dict['num_faces'] = []
    error_dict['num_overlay_faces'] = []
    error_dict['is_manifold'] = []
    error_dict['num_components'] = []
    error_dict['max_sym_dir'] = []
    error_dict['max_quadratic_sym_dir'] = []

    # Create output directory for the mesh
    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, 'error_table.log')
    logger = script_util.get_logger(log_path)
    logger.info("Building error table")

    for fname in args['fname']:
        dot_index = fname.rfind(".")
        m = fname[:dot_index]
        models.append(m)

        try:
            logger.info("Processing mesh {}".format(m))
            if (args['suffix'] == ""):
                name = m
            else:
                name = m + '_'+args['suffix']

            # Load input mesh information
            try:
                input_dir = args['input_dir']
                input_path = os.path.join(input_dir, m+'.obj')
                logger.info("Loading initial mesh at {}".format(input_path))
                v3d_orig, f_orig = igl.read_triangle_mesh(input_path)
            except:
                logger.error("Could not load initial mesh")
                return

            # Get final output mesh
            try:
                uv_dir = args['uv_dir']
                uv_path = os.path.join(uv_dir, m + "_output", name + ".obj")
                logger.info("Loading uv coordinates at {}".format(uv_path))
                v3d, uv, _, f, fuv, _ = igl.read_obj(uv_path)
            except:
                logger.error("Could not load uvs for {}".format(name))

            # Get topology information
            logger.info("Getting count information")
            error_dict['num_faces'].append(len(f_orig))
            error_dict['num_overlay_faces'].append(len(f))

            logger.info("Getting topology information")
            error_dict['is_manifold'].append(igl.is_edge_manifold(f))
            error_dict['num_components'].append(-1)
            #    np.max(igl.face_components(fuv)) + 1)

            # Get areas
            logger.info("Computing areas")
            uv_embed = np.zeros((len(uv), 3))
            uv_embed[:, :2] = uv[:, :2]
            mesh_areas = 0.5 * igl.doublearea(v3d, f)
            uv_areas = 0.5 * igl.doublearea(uv_embed, fuv)
            error_dict['uv_length_error'].append(
                penner.compute_uv_length_error(f, uv, fuv))
            error_dict['min_overlay_area'].append(np.min(mesh_areas))
            error_dict['max_overlay_area'].append(np.max(mesh_areas))
            error_dict['min_uv_area'].append(np.min(uv_areas))
            error_dict['max_uv_area'].append(np.max(uv_areas))

            # Get energies
            logger.info("Computing distortion energies")
            sym_dirichlet_energy = energies.sym_dirichlet_vf(
                v3d, f, uv_embed, fuv) - 4
            quad_sym_dir_energy = energies.quadratic_sym_dirichlet_vf(
                v3d, f, uv_embed, fuv)
            error_dict['max_sym_dir'].append(np.max(sym_dirichlet_energy))
            error_dict['max_quadratic_sym_dir'].append(
                np.max(quad_sym_dir_energy))
        except:
            logger.error("Missing output for {}".format(m))
            for key in error_dict:
                error_dict[key].append(-1)

    # Save dataframe to file
    energies_df = pd.DataFrame(error_dict, index=models)
    csv_path = os.path.join(
        args['output_dir'], 'errors_' + args['suffix'] + '.csv')
    energies_df.to_csv(csv_path)


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Generate error table for analysis")
    add_error_table_arguments(parser)
    args = vars(parser.parse_args())

    # Run method in parallel
    error_table(args)
