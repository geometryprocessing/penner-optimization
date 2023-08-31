# Script to refine a connectivity from an overlay mesh and layout.
#
# By default, runs all meshes specified by the `fname` argument in parallel.
# Functions to run the parallelized script and the method without parallelization
# are also exposed for use in other modules.

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import igl
import script_util
import subprocess
import optimization_py as opt

def add_refine_arguments(parser):
    # Parse arguments for the script
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output refined vf and uv")
    parser.add_argument("--suffix",
                        help="suffix for input/output files",
                        default="")


def refine_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index] 
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']

    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, name+'_refine_vf.log')
    logger = script_util.get_logger(log_path)
    logger.info("Refining {}".format(name))

    # Load uv information
    try:
        uv_dir = args['uv_dir']
        uv_path = os.path.join(uv_dir, m + "_output", name +  "_overlay_with_uv.obj")
        logger.info("Loading uv coordinates at {}".format(uv_path))
        V_o, uv_o, _, F_o, FT_o, _ = igl.read_obj(uv_path)
    except:
        logger.error("Could not load uv coordinates")
        return

    # Load Fn_to_f map
    try:
        fn_to_f_path = os.path.join(uv_dir, m + "_output", name + '_fn_to_f')
        logger.info("Loading face map at {}".format(fn_to_f_path))
        fn_to_f_o = np.loadtxt(fn_to_f_path, dtype=int)
    except:
        logger.error("Could not load face map")
        fn_to_f_o = np.arange(len(F_o))

    # Load endpoints map
    try:
        endpoints_path = os.path.join(uv_dir, m + "_output", name + '_endpoints')
        logger.info("Loading endpoints at {}".format(endpoints_path))
        endpoints_o = np.loadtxt(endpoints_path, dtype=int)
    except:
        logger.error("Could not load endpoints")
        endpoints_o = np.full((len(V_o), 2), -1)

    # Refine original mesh using overlay
    logger.info("Running refinement")
    refinement_mesh = opt.RefinementMesh(V_o, F_o, uv_o, FT_o, fn_to_f_o, endpoints_o)
    V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r = refinement_mesh.get_VF_mesh()
    
    # Check uvs are consistent
    if not opt.check_uv(V_r, F_r, uv_r, FT_r):
        logger.error("UVs are inconsistent")
        return

    # Write combined refined mesh with uv
    uv_mesh_path = os.path.join(output_dir, name + '_refined_with_uv.obj')
    logger.info("Saving refined uv mesh at {}".format(uv_mesh_path))
    opt.write_obj_with_uv(uv_mesh_path, V_r, F_r, uv_r, FT_r)

    # Write face map to file
    face_map_path = os.path.join(output_dir, name + '_refined_with_uv_fn_to_f')
    logger.info("Saving face map at {}".format(face_map_path))
    np.savetxt(face_map_path, fn_to_f_r, fmt='%i')

    # Write endpoints to file
    endpoints_path = os.path.join(output_dir, name + '_refined_with_uv_endpoints')
    logger.info("Saving endpoints at {}".format(endpoints_path))
    np.savetxt(endpoints_path, endpoints_r, fmt='%i')


def refine_many(args):
    script_util.run_many(refine_one, args)


if __name__ == "__main__":
    # Parse arguments for the script 
    parser = script_util.generate_parser("Refine VF output")
    add_refine_arguments(parser)
    args = vars(parser.parse_args())

    # Run method in parallel
    refine_many(args)
