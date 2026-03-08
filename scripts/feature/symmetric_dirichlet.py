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
import optimization_scripts.script_util as script_util
import subprocess
import shutil

def run_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']
    uv_path = os.path.join(args['uv_dir'], m + "_output")
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(os.path.join(output_dir, name+'_out.obj')):
        print("Skipping mesh as already optimized")
        return

    # open logger
    log_path = os.path.join(uv_path, name+'_symmetric_dirichlet.log')
    logger = script_util.get_logger(log_path)
    logger.info("Optimizing uv for {}".format(name))

    quad_exec = [os.path.join("build", "bin", "symmetric_dirichlet"),]
    if args['json'] == "":
        json_path = os.path.join(base_dir, '..', 'ext', 'example.json')
    else:
        json_path = args['json']
    log_path = os.path.join(output_dir, "symmetric_dirichlet.log")
    quad_exec += [ '-i', uv_path]
    quad_exec += [ '-m', name]
    quad_exec += [ '-j', json_path]
    quad_exec += [ '-f', m + ".ffield"]
    quad_exec += [ '-o', output_dir]
    print(quad_exec)
    with open(log_path, 'w') as log_file:
        subprocess.run(quad_exec, stdout=log_file)
    
    global_output_dir = os.path.join(args['output_dir'], "output")
    os.makedirs(global_output_dir, exist_ok=True)
    shutil.copy(
        os.path.join(output_dir, name+'_out.obj'),
        os.path.join(global_output_dir, name+'_out.obj'))

def run_many(args):
    script_util.run_many(run_one, args)

def add_arguments(parser):
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("-j", "--json",     help="json parameters for optimization", default="")

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Quadrangulate a mesh with a seamless parameterization")
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel method
    run_many(args)
