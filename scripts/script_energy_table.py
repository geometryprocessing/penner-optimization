import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import optimization_py as opt
import pandas as pd
import script_util
import optimize_impl.energies as energies

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
    energy_dict['surface_hencky_strain_energy'] = []
    energy_dict['symmetric_dirichlet_energy'] = []
    #energy_dict['delaunay_symmetric_dirichlet_energy'] = []
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
            # Get mesh
            logger.info("Getting mesh {}", m)
            m, C, lambdas_init, lambdas_target, v, f, Th_hat = script_util.generate_mesh(args, fname)
            name = m + '_'+args['suffix']
            reduction_maps = opt.ReductionMaps(C)
            proj = np.array(reduction_maps.proj)
            try:
                lambdas_dir = os.path.join(args['output_dir'], m + '_output', 'lambdas_' + args['suffix'])
                logger.info("Loading lambdas at {}".format(lambdas_dir))
                lambdas = np.loadtxt(lambdas_dir, dtype=float)
            except:
                logger.error("Could not load lambdas for {}".format(name))

            # Get delaunay mesh
            #logger.info("Making mesh Delaunay")
            #C_del, lambdas_init_del_full, _, flip_seq = opt.make_delaunay_with_jacobian(C, lambdas_init[proj], False)
            #_, lambdas_del_full = opt.flip_edges(C, lambdas[proj], flip_seq)

            # Get energy_dict
            logger.info("Getting symmetric stretches")
            stretches = energies.symmetric_stretches(lambdas, lambdas_target)

            # Build energy functor
            _, opt_params = script_util.generate_parameters(args)
            opt_energy = opt.EnergyFunctor(C, lambdas_target[proj], opt_params)
            #del_opt_energy = opt.EnergyFunctor(C_del, lambdas_init_del_full, opt_params)

            logger.info("Getting invariants")
            f2J1, _ = opt.first_invariant(C, lambdas_target[proj], lambdas[proj], False)
            f2J2sq, _ = opt.second_invariant_squared(C, lambdas_target[proj], lambdas[proj], False)
            f2J2 = np.sqrt(np.abs(f2J2sq))

            logger.info("Getting distortion measures")
            f2metric_dist, _ = opt.metric_distortion_energy(C, lambdas_target[proj], lambdas[proj], False)
            f2area_dist, _ = opt.area_distortion_energy(C, lambdas_target[proj], lambdas[proj], False)
            f2sym_dirichlet_dist, _ = opt.symmetric_dirichlet_energy(C, lambdas_target[proj], lambdas[proj], False)

            logger.info("Getting best fit conformal energy")
            u = opt.best_fit_conformal(C, lambdas_target[proj], lambdas[proj])

            logger.info("Getting constraint error")
            constraint, _, _, success = opt.constraint_with_jacobian(C, lambdas[proj], False, args['use_edge_lengths'])
            if not success:
                raise Exception("Triangle inequality error")
            
            # Record energy_dict
            logger.info("Adding energies")
            energy_dict['two_norm_energy'].append(opt_energy.compute_two_norm_energy(lambdas[proj]))
            energy_dict['surface_hencky_strain_energy'].append(opt_energy.compute_surface_hencky_strain_energy(lambdas[proj]))
            energy_dict['symmetric_dirichlet_energy'].append(opt_energy.compute_symmetric_dirichlet_energy(lambdas[proj]))
            #energy_dict['delaunay_symmetric_dirichlet_energy'].append(del_opt_energy.compute_symmetric_dirichlet_energy(lambdas_del_full))
            energy_dict['constraint_error'].append(np.max(np.abs(constraint)))
            energy_dict['max_stretch'].append(np.max(stretches))
            energy_dict['norm_stretch'].append(np.linalg.norm(stretches))
            energy_dict['max_first_invariant'].append(np.max(f2J1))
            energy_dict['norm_first_invariant'].append(np.linalg.norm(f2J1))
            energy_dict['max_second_invariant'].append(np.max(f2J2))
            energy_dict['norm_second_invariant'].append(np.linalg.norm(f2J2))
            energy_dict['max_metric_distortion'].append(np.max(f2metric_dist))
            energy_dict['norm_metric_distortion'].append(np.linalg.norm(f2metric_dist))
            energy_dict['max_area_distortion'].append(np.max(f2area_dist))
            energy_dict['norm_area_distortion'].append(np.linalg.norm(f2area_dist))
            energy_dict['max_sym_dirichlet_distortion'].append(np.max(f2sym_dirichlet_dist))
            energy_dict['norm_sym_dirichlet_distortion'].append(np.linalg.norm(f2sym_dirichlet_dist))
            energy_dict['min_scale'].append(np.min(u))
            energy_dict['max_scale'].append(np.max(u))
            energy_dict['avg_scale'].append(np.average(u))
            energy_dict['range_scale'].append(np.max(u) - np.min(u))
        except:
            logger.error("Missing output for {}".format(m))
            energy_dict['two_norm_energy'].append(-1)
            energy_dict['surface_hencky_strain_energy'].append(-1)
            energy_dict['symmetric_dirichlet_energy'].append(-1)
            #energy_dict['delaunay_symmetric_dirichlet_energy'].append(-1)
            energy_dict['constraint_error'].append(-1)
            energy_dict['max_stretch'].append(-1)
            energy_dict['norm_stretch'].append(-1)
            energy_dict['max_first_invariant'].append(-1)
            energy_dict['norm_first_invariant'].append(-1)
            energy_dict['max_second_invariant'].append(-1)
            energy_dict['norm_second_invariant'].append(-1)
            energy_dict['max_metric_distortion'].append(-1)
            energy_dict['norm_metric_distortion'].append(-1)
            energy_dict['max_area_distortion'].append(-1)
            energy_dict['norm_area_distortion'].append(-1)
            energy_dict['max_sym_dirichlet_distortion'].append(-1)
            energy_dict['norm_sym_dirichlet_distortion'].append(-1)
            energy_dict['min_scale'].append(-1)
            energy_dict['max_scale'].append(-1)
            energy_dict['avg_scale'].append(-1)
            energy_dict['range_scale'].append(-1)

    energies_df = pd.DataFrame(energy_dict, index=models)
    csv_path = os.path.join(args['output_dir'], 'energies_' + args['suffix'] + '.csv')
    energies_df.to_csv(csv_path)


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Generate energy table for analysis")
    add_energy_table_arguments(parser)
    args = vars(parser.parse_args())
 
    # Run method in parallel
    energy_table(args)