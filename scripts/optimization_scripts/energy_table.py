# Script to generate a table summarizing various energies and other metrics from output
# Penner coordinates

import optimize_impl.energies as energies
import script_util
import pandas as pd
import penner
import numpy as np
import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)


def add_energy_table_arguments(parser):
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for output images"
    )
    parser.add_argument(
        "--suffix",
        help="suffix for output files",
        default=""
    )


def energy_table(args):
    energy_dict = {}
    models = []
    energy_dict['two_norm_energy'] = []
    energy_dict['quadratic_sym_dir_energy'] = []
    energy_dict['scale_energy'] = []
    energy_dict['four_norm_energy'] = []
    energy_dict['symmetric_dirichlet_energy'] = []
    energy_dict['constraint_error'] = []
    energy_dict['range_scale'] = []
    energy_dict['max_sym_dirichlet_distortion'] = []
    energy_dict['max_stretch'] = []
    energy_dict['norm_stretch'] = []
    energy_dict['max_first_invariant'] = []
    energy_dict['norm_first_invariant'] = []
    energy_dict['max_second_invariant'] = []
    energy_dict['norm_second_invariant'] = []
    energy_dict['max_metric_distortion'] = []
    energy_dict['norm_metric_distortion'] = []
    energy_dict['max_area_distortion'] = []
    energy_dict['norm_area_distortion'] = []
    energy_dict['norm_sym_dirichlet_distortion'] = []
    energy_dict['min_scale'] = []
    energy_dict['max_scale'] = []
    energy_dict['avg_scale'] = []

    # Create output directory for the mesh
    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, 'energy_table.log')
    logger = script_util.get_logger(log_path)
    logger.info("Building energy table")

    for fname in args['fname']:
        dot_index = fname.rfind(".")
        m = fname[:dot_index]
        models.append(m)

        try:
            # Get mesh and Penner coordinates
            logger.info("Getting mesh {}".format(m))
            m, _, _, _, C, _, C_embed, C_eucl = script_util.generate_mesh(
                args, fname)
            lambdas_target = C.get_reduced_metric_coordinates()
            name = m + '_'+args['suffix']
            reduction_maps = penner.ReductionMaps(C)
            proj = np.array(reduction_maps.proj)
            he2e = np.array(reduction_maps.he2e)
            try:
                lambdas_dir = os.path.join(
                    args['output_dir'], m + '_output', 'lambdas_' + args['suffix'])
                logger.info("Loading lambdas at {}".format(lambdas_dir))
                lambdas = np.loadtxt(lambdas_dir, dtype=float)
            except:
                logger.error("Could not load lambdas for {}".format(name))
            C = C.set_metric_coordinates(lambdas)

            # Compute symmetric stretches
            logger.info("Getting symmetric stretches")
            stretches = energies.symmetric_stretches(lambdas, lambdas_target)

            # Compute metric tensor invariants
            logger.info("Getting metric invariants")
            f2J1, _ = penner.first_invariant(
                C_embed, lambdas[proj], False)
            f2J2sq, _ = penner.second_invariant_squared(
                C_embed, lambdas[proj], False)
            f2J2 = np.sqrt(np.abs(f2J2sq))

            # Compute distortion measures
            logger.info("Getting distortion measures")
            f2metric_dist, _ = penner.metric_distortion_energy(
                C_embed, lambdas[proj], False)
            f2area_dist, _ = penner.area_distortion_energy(
                C_embed, lambdas[proj], False)
            f2sym_dirichlet_dist, _ = penner.symmetric_dirichlet_energy(
                C_embed, lambdas[proj], False)

            logger.info("Getting scale factors")
            u = penner.best_fit_conformal(C_embed, lambdas[proj[he2e]])

            # Compute constraint errors
            logger.info("Getting constraint error")
            max_constraint = penner.compute_max_constraint(C)

            # Make energy functors
            length_energy = penner.LogLengthEnergy(C_embed, 2)
            quad_energy = penner.QuadraticSymmetricDirichletEnergy(
                C_embed, C_eucl)
            scale_energy = penner.LogScaleEnergy(C_embed)
            pnorm_energy = penner.LogLengthEnergy(C_embed, 4)
            sym_dir_energy = penner.SymmetricDirichletEnergy(C_embed, C_eucl)

            # Record energy_dict
            logger.info("Adding energies")
            energy_dict['two_norm_energy'].append(length_energy.energy(C))
            energy_dict['quadratic_sym_dir_energy'].append(quad_energy.energy(C))
            energy_dict['scale_energy'].append(scale_energy.energy(C))
            energy_dict['four_norm_energy'].append(pnorm_energy.energy(C))
            energy_dict['symmetric_dirichlet_energy'].append(sym_dir_energy.energy(C))

            logger.info("Adding stretch")
            energy_dict['max_stretch'].append(np.max(stretches))
            energy_dict['norm_stretch'].append(np.linalg.norm(stretches))

            logger.info("Adding distortion")
            energy_dict['max_first_invariant'].append(np.max(f2J1))
            energy_dict['norm_first_invariant'].append(np.linalg.norm(f2J1))
            energy_dict['max_second_invariant'].append(np.max(f2J2))
            energy_dict['norm_second_invariant'].append(np.linalg.norm(f2J2))
            energy_dict['max_metric_distortion'].append(np.max(f2metric_dist))
            energy_dict['norm_metric_distortion'].append(
                np.linalg.norm(f2metric_dist))
            energy_dict['max_area_distortion'].append(np.max(f2area_dist))
            energy_dict['norm_area_distortion'].append(
                np.linalg.norm(f2area_dist))
            energy_dict['max_sym_dirichlet_distortion'].append(
                np.max(f2sym_dirichlet_dist))
            energy_dict['norm_sym_dirichlet_distortion'].append(
                np.linalg.norm(f2sym_dirichlet_dist))


            logger.info("Adding scale")
            energy_dict['min_scale'].append(np.min(u))
            energy_dict['max_scale'].append(np.max(u))
            energy_dict['avg_scale'].append(np.average(u))
            energy_dict['range_scale'].append(np.max(u) - np.min(u))

            logger.info("Adding error")
            energy_dict['constraint_error'].append(max_constraint)
        except:
            logger.error("Missing output for {}".format(m))
            for key in energy_dict:
                energy_dict[key].append(-1)

    energies_df = pd.DataFrame(energy_dict, index=models)
    csv_path = os.path.join(
        args['output_dir'], 'energies_' + args['suffix'] + '.csv')
    energies_df.to_csv(csv_path)


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Generate energy table for analysis")
    add_energy_table_arguments(parser)
    args = vars(parser.parse_args())

    # Run method in parallel
    energy_table(args)
