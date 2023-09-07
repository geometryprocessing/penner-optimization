import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import optimization_py as opt
import script_util
import numpy as np


def interpolate_metric(
    C,
    lambdas_start,
    lambdas_end,
    output_dir,
    num_steps=5,
    proj_params=opt.ProjectionParameters(),
    opt_params=opt.OptimizationParameters()
):
    """
    Interpolate the metric given by lambdas_start to the metric given by lambdas_end, projecting
    to the angle constraints given by C.Th_hat at each step.

    @param[in] Mesh C: mesh
    @param[in] np.array lambdas_start: initial metric coordinates for interpolation
    @param[in] np.array lambdas_end: final metric coordinates for interpolation
    @param[in] int num_steps: number of steps of interpolation
    @param[in] ProjectionParameters proj_params: projection parameters 
    """
    # Linearly interpolate the penner coordinates and project to the constraint
    steps = np.linspace(0, 1, num_steps)
    for i, s in enumerate(steps):
        lambdas = s * lambdas_end + (1 - s) * lambdas_start
        lambdas, _ = opt.project_to_constraint(C,
                                               lambdas,
                                               proj_params,
                                               opt_params)
        np.savetxt(
            os.path.join(output_dir, 'lambdas_' + str(i)),
            lambdas
        )

def add_interpolate_arguments(parser):
    # Input and output directories
    parser.add_argument("--start_lambdas_path",  help="path for start lambda values")
    parser.add_argument("--end_lambdas_path",    help="path for end lambda values")
    parser.add_argument("-o", "--output_dir",          help="directory for output lambdas and logs")
    parser.add_argument("--num_steps",           help="number of interpolation steps", type=int, default=5)

def interpolate_one(args, fname):
    # Get mesh, lambdas, and parameters
    m, C, _, _, _, _, _ = script_util.generate_mesh(args, fname)
    proj_params, opt_params = script_util.generate_parameters(args)
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # If no paths specified, assume conformal to optimized interpolation
    start_lambdas_path = args['start_lambdas_path']
    end_lambdas_path = args['end_lambdas_path']
    if not start_lambdas_path:
        start_lambdas_path = os.path.join(output_dir, 'lambdas_conf')
    if not end_lambdas_path:
        end_lambdas_path = os.path.join(output_dir, 'lambdas_opt')

    # Get initial and final lambdas
    lambdas_start = np.loadtxt(start_lambdas_path, dtype=float)
    lambdas_end = np.loadtxt(end_lambdas_path, dtype=float)
    num_steps = args['num_steps']
    interpolate_metric(C,
                       lambdas_start,
                       lambdas_end,
                       output_dir,
                       num_steps,
                       proj_params,
                       opt_params)

def interpolate_many(args):
    script_util.run_many(interpolate_one, args)

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Interpolate metric coordinates")
    add_interpolate_arguments(parser)
    args = vars(parser.parse_args())
 
    # Run method in parallel
    interpolate_many(args)