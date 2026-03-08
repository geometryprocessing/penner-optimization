# Script to quadrangulate a mesh

import os, sys
base_dir = os.path.dirname(__file__)
module_dir = os.path.join(base_dir, '..', 'py')
sys.path.append(module_dir)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts')
sys.path.append(script_dir)
import numpy as np
import penner
import igl
import shutil
import optimization_scripts.script_util as script_util
import datetime
import subprocess

def run_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']
    uv_path = os.path.join(args['uv_dir'], m + "_output")
    output_dir = os.path.join(args['output_dir'], m+"_output")
    os.makedirs(output_dir, exist_ok=True)
    status_dir = os.path.join(args['output_dir'], "status_logs")
    os.makedirs(status_dir, exist_ok=True)


    # open logger
    log_path = os.path.join(output_dir, name+'_quadrangulate.log')
    logger = script_util.get_logger(log_path)
    logger.info("Quadrangulating {}".format(name))
    test_name = os.path.basename(args['output_dir'])
    logger.info("Test name: {}".format(test_name))

    # skip processed meshes
    os.makedirs(os.path.join(args['output_dir'], "output"), exist_ok=True)
    quad_filepath = os.path.join(output_dir, "qm.obj")
    robust_filepath = os.path.join(output_dir, name + "_quad.obj")
    if not os.path.isfile(os.path.join(uv_path, name + '.obj')):
        logger.info("Skipping mesh as no uv map available")
        return
    if os.path.isfile(quad_filepath):
        logger.info("mesh is already quantized with bearquad")
        if not args['overwrite']:
            return
    if os.path.isfile(robust_filepath):
        logger.info("mesh already quantized")
        if not args['overwrite']:
            return

    # check for degeneracy
    try:
        uv_dir = args['uv_dir']
        uv_mesh = os.path.join(uv_dir, m + "_output", name + ".obj")
        v3d, uv, _, f, fuv, _ = igl.read_obj(uv_mesh)
    except:
        return
    uv_length_error, uv_angle_error, uv_length, uv_angle = feature.compute_seamless_error(f, uv, fuv)
    if (np.max(uv_length_error) > 1e-8) or (np.max(uv_angle_error) > 1e-8):
        print("Mesh is not seamless")
        return

    # get number of triangles
    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['input_dir'], fname))
    except:
        logger.info("Could not load triangle mesh")
        return
    num_tris = len(F)
    #num_quads = str(int(num_tris / 2))
    num_quads = str(int(2 * 4 * num_tris))
    with open(os.path.join(status_dir, name+"_start"), "a") as f:
        f.write("Start {}".format(datetime.datetime.now()))

    if (uv_path != output_dir):
        logger.info("Moving mesh to new location")
        shutil.copy(os.path.join(uv_path, name+".obj"),os.path.join(output_dir, name+".obj"))
        uv_path = output_dir
    
    param_exec_path = os.path.join(base_dir, "..", "build", "bin", "add_trivial_quad_parameterization")
    if args['use_robust_method']:
        bb_diag = igl.bounding_box_diagonal(V)
        #scale = 0.25 * ((np.sqrt(num_tris)/bb_diag) + 1)
        scale = 1.0
        logger.info("using scale {}".format(scale))
        quad_exec_path = os.path.join(base_dir, "quadrangulate_robust.sh")
        subprocess.run([quad_exec_path, uv_path, name, str(scale), test_name, str(args['final_param'])])

        if not os.path.isfile(robust_filepath):
            logger.info("Could not quantize mesh with robust method")
        else:
            files = os.listdir(output_dir)
            temp_files = [f for f in files if f.startswith("00")]
            for temp_file in temp_files:
                os.remove(os.path.join(output_dir, temp_file))
            #shutil.copy(robust_filepath, target_filepath)
            target_filepath = os.path.join(args['output_dir'], "output", name + "_robust_quad.obj")
            subprocess.run([param_exec_path, "--mesh", robust_filepath, "--output", target_filepath])
    else:
        quad_exec_path = os.path.join(base_dir, "quadrangulate.sh")
        subprocess.run([quad_exec_path, uv_path, name, num_quads, test_name])

        if not os.path.isfile(quad_filepath):
            logger.info("Could not quantize mesh")
        else:
            #shutil.copy(quad_filepath, target_filepath)
            target_filepath = os.path.join(args['output_dir'], "output", name + "_quad.obj")
            subprocess.run([param_exec_path, "--mesh", quad_filepath, "--output", target_filepath])

    with open(os.path.join(status_dir, name+"_end"), "a") as f:
        f.write("End {}".format(datetime.datetime.now()))

def run_many(args):
    script_util.run_many(run_one, args)

def add_arguments(parser):
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("--use_robust_method",      help="use the robust quadrangulation method",
                                                     action="store_true")
    parser.add_argument("--final_param",      help="number of optimization iterations",
                                                    type=int, default=0)
    parser.add_argument("--overwrite",      help="overwrite existing quad mesh",
                                                     action="store_true")


if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Quadrangulate a mesh with a seamless parameterization")
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel method
    run_many(args)
