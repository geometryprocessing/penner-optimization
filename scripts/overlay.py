# Script to save overlay mesh with uv layout to standard VF format.

# FIXME make simplificaiton internal
import math
import pickle
import optimize_impl.render as render
import script_util
import optimization_py as opt
import igl
import numpy as np
import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)


def add_overlay_arguments(parser):
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")
    parser.add_argument("--suffix",
                        help="suffix for output files",
                        default="")
    parser.add_argument("--flip_in_original_metric",
                        help="use original metric for edge flips",
                        action="store_true")


def overlay_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    name = m+'_'+args['suffix']

    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, name+'_convert_to_vf.log')
    logger = script_util.get_logger(log_path)
    logger.info("Converting {} to vf".format(name))

    # Get mesh and lambdas
    logger.info("Loading mesh")
    try:
        m, v3d, f, Th_hat, _, _, C_embed, _ = script_util.generate_mesh(
            args, fname)
    except:
        logger.error("Could not load mesh")
        return

    # Get mesh information
    is_bd = igl.is_border_vertex(v3d, f)
    build_double = (np.sum(is_bd) != 0)
    _, vtx_reindex = opt.fv_to_double(v3d, f, v3d, f, Th_hat, [], False)

    # Get cones
    cones = np.array([id for id in range(len(Th_hat)) if np.abs(
        Th_hat[id]-2*math.pi) > 1e-15 and not is_bd[id]], dtype=int)
    cones = [idx for idx in range(len(vtx_reindex))
             if vtx_reindex[idx] in cones]

    # Get final optimized lambdas
    try:
        lambdas_path = os.path.join(
            args['lambdas_dir'], m + "_output", 'lambdas_'+args['suffix'])
        logger.info("Loading lambdas from {}".format(lambdas_path))
        lambdas = np.loadtxt(lambdas_path)
    except:
        logger.error('Could not load lambdas')
        return

    # Get overlay
    # TODO Replace with method akin to generate_VF_mesh_from_metric for log edge lenghts
    if args['use_edge_lengths']:
        logger.info("Using edge lengths")
        u = np.zeros(len(v3d))
        C_o = opt.add_overlay(C_embed, lambdas)
        opt.make_tufted_overlay(C_o, v3d, f, Th_hat)
        v_overlay = v3d[vtx_reindex].T
        endpoints = np.full((len(C.out), 2), -1)

        # Get layout parametrization from overlay
        logger.info("Getting parametrization from overlay")
        parametrize_res = opt.overlay_mesh_to_VL(
            v3d,
            f,
            Th_hat,
            C_o,
            u,
            v_overlay,
            vtx_reindex,
            endpoints,
            -1
        )
        v_o, f_o, u_param_o, v_param_o, ft_o, is_cut_h, _, fn_to_f, endpoints = parametrize_res
        v_o = np.array(v_o)
        f_o = np.array(f_o)
        ft_o = np.array(ft_o)
        uvt_o = np.array([u_param_o, v_param_o]).T
    else:
        logger.info("Using Penner coordinates")
        parametrize_res = opt.generate_VF_mesh_from_metric(
            v3d,
            f,
            Th_hat,
            C_embed,
            lambdas,
            [],
            True
        )
        C_o, v_o, f_o, uvt_o, ft_o, is_cut_h, _, fn_to_f, endpoints = parametrize_res

    # Save new meshes
    uv_mesh_path = os.path.join(output_dir, name + '_overlay.obj')
    logger.info("Saving uv mesh at {}".format(uv_mesh_path))
    opt.write_obj_with_uv(uv_mesh_path, v_o, f_o, uvt_o, ft_o)

    # Save cut information
    simp_path = os.path.join(output_dir, name + '_is_cut_h')
    logger.info("Saving cut information at {}".format(simp_path))
    np.savetxt(simp_path, is_cut_h)

    # Write fn_to_f to file
    face_map_path = os.path.join(output_dir, name + '_fn_to_f')
    logger.info("Saving new to old face map at {}".format(face_map_path))
    np.savetxt(face_map_path, fn_to_f, fmt='%i')

    # Write endpoints to file
    endpoints_path = os.path.join(output_dir, name + '_endpoints')
    logger.info("Saving endpoints at {}".format(endpoints_path))
    np.savetxt(endpoints_path, endpoints, fmt='%i')


def overlay_many(args):
    script_util.run_many(overlay_one, args)


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Build overlay mesh parameterization")
    add_overlay_arguments(parser)
    args = vars(parser.parse_args())

    # Run method in parallel
    overlay_many(args)
