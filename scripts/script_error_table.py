import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import optimization_py as opt
import pandas as pd
import igl
import script_util
import optimize_impl.energies as energies

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
    error_dict['max_sym_dirichlet'] = []
    error_dict['max_arap'] = []
    error_dict['max_surface_hencky_strain'] = []
    error_dict['is_manifold'] = []

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
            # Get input mesh
            m, C, lambdas_init, lambdas_target, v_orig, f_orig, Th_hat = script_util.generate_mesh(args, fname)
            name = m + '_'+args['suffix']
            he2e, e2he = opt.build_edge_maps(C)
            proj, embed = opt.build_refl_proj(C)

            # Load input mesh information
            try:
                input_dir = args['input_dir']
                logger.info("Loading initial mesh at {}".format(input_dir))
                v3d_orig, f_orig = igl.read_triangle_mesh(os.path.join(input_dir, m+'.obj'))
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

            # Get areas
            uv_embed = np.zeros((len(uv), 3))
            uv_embed[:,:2] = uv[:,:2]
            mesh_areas = 0.5 * igl.doublearea(v3d, f)
            uv_areas = 0.5 * igl.doublearea(uv_embed, fuv)

            # Get energies
            sym_dirichlet_energy = energies.sym_dirichlet_vf(v3d, f, uv_embed, fuv) - 4
            arap_energy = energies.arap_vf(v3d, f, uv_embed, fuv)
            surface_hencky_strain_energy = energies.surface_hencky_strain_vf(v3d, f, uv_embed, fuv)

            # Record error_dict
            error_dict['uv_length_error'].append(opt.compute_uv_length_error(f, uv, fuv))
            error_dict['min_overlay_area'].append(np.min(mesh_areas))
            error_dict['max_overlay_area'].append(np.max(mesh_areas))
            error_dict['min_uv_area'].append(np.min(uv_areas))
            error_dict['max_uv_area'].append(np.max(uv_areas))
            error_dict['num_faces'].append(len(f_orig))
            error_dict['num_overlay_faces'].append(len(f))
            error_dict['max_sym_dirichlet'].append(np.max(sym_dirichlet_energy))
            error_dict['max_arap'].append(np.max(arap_energy))
            error_dict['max_surface_hencky_strain'].append(np.max(surface_hencky_strain_energy))
            error_dict['is_manifold'].append(igl.is_edge_manifold(f))
        except:
            logger.error("Missing output for {}".format(m))
            error_dict['uv_length_error'].append(-1)
            error_dict['min_overlay_area'].append(-1)
            error_dict['max_overlay_area'].append(-1)
            error_dict['min_uv_area'].append(-1)
            error_dict['max_uv_area'].append(-1)
            error_dict['num_faces'].append(-1)
            error_dict['num_overlay_faces'].append(-1)
            error_dict['max_sym_dirichlet'].append(-1)
            error_dict['max_arap'].append(-1)
            error_dict['max_surface_hencky_strain'].append(-1)
            error_dict['is_manifold'].append(-1)

    energies_df = pd.DataFrame(error_dict, index=models)
    csv_path = os.path.join(args['output_dir'], 'energies_' + args['suffix'] + '.csv')
    energies_df.to_csv(csv_path)


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Generate error table for analysis")
    add_error_table_arguments(parser)
    args = vars(parser.parse_args())
 
    # Run method in parallel
    error_table(args)