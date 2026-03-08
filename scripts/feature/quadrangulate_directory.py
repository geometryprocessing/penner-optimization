# Script to quadrangulate a mesh

import os, sys
base_dir = os.path.dirname(__file__)
module_dir = os.path.join(base_dir, '..', 'py')
sys.path.append(module_dir)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts')
sys.path.append(script_dir)
import numpy as np
import igl
import subprocess
import argparse, shutil
import multiprocessing 


def process_file(fname, args):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    uv_path = args['input_dir']
    test_name = os.path.basename(args['input_dir'])
    output_dir = os.path.join(args['input_dir'], m + '_output')
    os.makedirs(output_dir, exist_ok=True)

    robust_filepath = os.path.join(output_dir, m + "_quad.obj")
    target_filepath = os.path.join(args['input_dir'], "output", m + "_robust_quad.obj")
    if os.path.isfile(target_filepath):
        print("Skipping completed mesh")
        return

    # get number of triangles
    try:
        V, F = igl.read_triangle_mesh(os.path.join(args['input_dir'], fname))
    except:
        print("Could not open file")
        return

    num_tris = len(F)
    #num_quads = str(int(num_tris / 2))
    num_quads = 2 * num_tris
    bb_diag = igl.bounding_box_diagonal(V)
    #scale = args['scale'] * (np.sqrt(num_quads / 2)/bb_diag + 1)
    scale = args['scale']

    shutil.copy(os.path.join(uv_path, fname),os.path.join(output_dir, fname))
    uv_path = output_dir

    quad_exec_path = os.path.join(base_dir, "quadrangulate_robust.sh")
    allow_zero_arcs = '1' if args['allow_zero_arcs'] else '0'
    subprocess.run([quad_exec_path, uv_path, m, str(scale), test_name, allow_zero_arcs])

    param_exec_path = os.path.join("build", "bin", "add_trivial_quad_parameterization")
    if os.path.isfile(robust_filepath):
        files = os.listdir(output_dir)
        temp_files = [f for f in files if f.startswith("00")]
        for temp_file in temp_files:
            os.remove(os.path.join(output_dir, temp_file))
        #shutil.copy(robust_filepath, target_filepath)
        os.makedirs(os.path.join(args['input_dir'], "output"), exist_ok=True)
        subprocess.run([param_exec_path, "--mesh", robust_filepath, "--output", target_filepath])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--final_param",      help="number of optimization iterations",
                                                    type=int, default=0)
    parser.add_argument("--allow_zero_arcs",      help="allow zero arcs in quadrangulation",
                                                    type=bool, default=False)
    args = vars(parser.parse_args())

    files = os.listdir(args['input_dir'])
    models = [f for f in files if f.endswith(".obj")]

    pool_args = [(m, args) for m in models]
    with multiprocessing.Pool(processes=8) as pool:
        pool.starmap(process_file, pool_args, chunksize=1)

if __name__ == "__main__":
    main()
