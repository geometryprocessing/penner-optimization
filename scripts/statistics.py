# Script to generate a table summarizing various energies and other metrics from output
# Penner coordinates

import optimize_impl.energies as energies
import script_util
import pandas as pd
import optimization_py as opt
import holonomy_py as holonomy
import numpy as np
import os, math
import sys
import igl
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)


def add_statistics_arguments(parser):
    parser.add_argument(
        "--use_delaunay",
        action="store_true",
        help="use delaunay connectivity as base mesh"
    )
    parser.add_argument(
        "-i", "--input_dir",
        help="directory for input meshes"
    )
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for output"
    )

def run_statistics(args):
    models = []

    # Build dictionary of statistics
    statistics = [
        'genus',
        'faces',
        'cones',
        'RMSRE',
        'iter',
        'max_stretch',
        'min_cone',
        'max_cone',
        'surface_area',
    ]
    statistics_dict = {statistic : [] for statistic in statistics}

    # Create output directory for the mesh
    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, 'statistics.log')
    logger = script_util.get_logger(log_path)
    logger.info("Building statistics table")

    for fname in args['fname']:
        dot_index = fname.rfind(".")
        m = fname[:dot_index]
        name = m
        models.append(m)

        try:
            # Get mesh
            V, F = igl.read_triangle_mesh(os.path.join(args['input_dir'], fname))
            Th_hat = np.loadtxt(os.path.join(args['input_dir'], m + "_Th_hat"), dtype=float)
            rotation_form = np.loadtxt(os.path.join(args['input_dir'], m + "_kappa_hat"), dtype=float)

            # Generate metric TODO use constructor
            free_cones = []
            marked_metric_params = holonomy.MarkedMetricParameters()
            marked_metric, _ = holonomy.generate_marked_metric(V, F, V, F, Th_hat, rotation_form, free_cones, marked_metric_params)

            # Get target metric
            if (args['use_delaunay']):
                logger.info("Using Delaunay connectivity")
                marked_target = marked_metric.clone_cone_metric()
                marked_target.make_discrete_metric()
                flip_seq = np.array(marked_target.get_flip_sequence())
                penner_target = marked_target.get_reduced_metric_coordinates()
            else:
                penner_target = marked_metric.get_reduced_metric_coordinates()


            # get final metric coordinates
            mesh_output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
            metric_coords_path = os.path.join(mesh_output_dir, name + "_metric_coords")
            logger.info("Loading metric coordinates from {}".format(metric_coords_path))
            reduced_metric_coords = np.loadtxt(metric_coords_path)
            marked_metric = marked_metric.set_metric_coordinates(reduced_metric_coords)

            # ensure coordinates are defined on same connectivity
            if (args['use_delaunay']):
                logger.info("Flipping to Delaunay connectivity")
                for h in flip_seq:
                    marked_metric.flip_ccw(h, True)
            penner_coords = marked_metric.get_reduced_metric_coordinates()

            # get per iteration data
            iteration_data_dir = os.path.join(mesh_output_dir, 'iteration_data_log.csv')
            logger.info("Loading iteration data from {}".format(iteration_data_dir))
            iteration_data = pd.read_csv(iteration_data_dir)
        except:
            logger.info("Could not open mesh data at {}".format(args['input_dir']))
            for statistic in statistics:
                statistics_dict[statistic].append(-1)
            continue

        for statistic in statistics:
            logger.info("Getting {} statistic".format(statistic))
            try:
                if statistic == 'surface_area':
                    statistics_dict[statistic].append(np.sum(igl.doublearea(V, F)) / 2.)

                if statistic == 'genus':
                    statistics_dict[statistic].append(marked_metric.n_homology_basis_loops() / 2)

                if statistic == 'faces':
                    statistics_dict[statistic].append(marked_metric.n_faces())

                if statistic == 'cones':
                    is_bd = igl.is_border_vertex(V, F)
                    _, vtx_reindex = opt.fv_to_double(V, F, V, F, Th_hat, [], False)
                    cones = np.array([id for id in range(len(Th_hat)) if np.abs(Th_hat[id]-2*math.pi) > 1e-15 and not is_bd[id]], dtype=int)
                    cones = [idx for idx in range(len(vtx_reindex)) if vtx_reindex[idx] in cones]
                    statistics_dict[statistic].append(len(cones))

                if statistic == 'min_cone':
                    statistics_dict[statistic].append(np.min(Th_hat))

                if statistic == 'max_cone':
                    statistics_dict[statistic].append(np.max(Th_hat))

                if statistic == 'RMSRE':
                    statistics_dict[statistic].append(float(iteration_data['rmsre'].tail(1)))

                if statistic == 'iter':
                    statistics_dict[statistic].append(int(iteration_data['num_iter'].tail(1)))

                if statistic == 'max_stretch':
                    X = energies.symmetric_stretches(penner_coords, penner_target)
                    statistics_dict[statistic].append(np.max(X))

            except:
                statistics_dict[statistic].append(-1)

    statistics_df = pd.DataFrame(statistics_dict, index=models)
    csv_path = os.path.join(output_dir, 'statistics.csv')
    statistics_df.to_csv(csv_path)


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Generate statistics")
    add_statistics_arguments(parser)
    args = vars(parser.parse_args())

    run_statistics(args)
