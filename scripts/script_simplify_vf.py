# Script to simplify connectivity for an overlay mesh and layout.
#
# By default, runs all meshes specified by the `fname` argument in parallel.
# Functions to run the parallelized script and the method without parllelization
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

def add_simplify_vf_arguments(parser):
    # Parse arguments for the script
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output simplified vf and uv")
    parser.add_argument("--suffix",
                        help="suffix for input/output files",
                        default="")


def simplify_vf_one(args, fname):
    dot_index = fname.rfind(".")
    m = fname[:dot_index] 
    name = m + '_'+args['suffix']

    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, name+'_simplify_vf.log')
    logger = script_util.get_logger(log_path)
    logger.info("Simplifying {}".format(name))

    # Simplify overlay
    logger.info("Running simplification")
    uv_dir = os.path.join(args['uv_dir'], m + "_output")
    simp_in_path = os.path.join(uv_dir, name+'_o.h5')
    simp_path = os.path.join(uv_dir, name+ '_simplified_with_uv_s.h5')
    logger.info("Getting simplification information at {}".format(simp_in_path))
    subprocess.call(["./ext/mesh_simplification", simp_in_path, simp_path])
    endpoints_s, v_s, f_s, uvt_s, ft_s, is_cut_s, vn_to_v = opt.load_simplify_overlay_output(simp_path)
    logger.info("Saved simplified mesh information to {}".format(simp_path))
    
    # Get cut triangulation for initial mesh
    vt_s = np.zeros((len(uvt_s), 3),dtype=np.float64)
    vt_s[ft_s] = v_s[f_s]
    vn_to_v = vn_to_v[:, 0]
    vnt_to_v = np.zeros(len(uvt_s), dtype=int)
    vnt_to_v[ft_s] = vn_to_v[f_s]

    # Write combined overlay with uv
    uv_mesh_path = os.path.join(output_dir, name + '_simplified_with_uv.obj')
    logger.info("Saving simplified uv mesh at {}".format(uv_mesh_path))
    opt.write_obj_with_uv(uv_mesh_path, v_s, f_s, uvt_s, ft_s)

    # Write vn_to_v to file
    vertex_map_path = os.path.join(output_dir, name + '_simplified_with_uv_vn_to_v')
    logger.info("Saving new to old vertex map at {}".format(vertex_map_path))
    np.savetxt(vertex_map_path, vnt_to_v, fmt='%i')

    # Write endpoints to file
    endpoints_path = os.path.join(output_dir, name + '_simplified_with_uv_endpoints')
    logger.info("Saving endpoints at {}".format(endpoints_path))
    np.savetxt(endpoints_path, endpoints_s, fmt='%i')


def simplify_vf_many(args):
    script_util.run_many(simplify_vf_one, args)


if __name__ == "__main__":
    # Parse arguments for the script 
    parser = script_util.generate_parser("Simplify VF output")
    add_simplify_vf_arguments(parser)
    args = vars(parser.parse_args())

    # Run method in parallel
    simplify_vf_many(args)
