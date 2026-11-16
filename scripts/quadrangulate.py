import os.path
from subprocess import check_call
import multiprocessing
import subprocess, argparse, shutil
import sys

base_dir = os.path.dirname(__file__)

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dir)

def process_file(fname, args):
    dot_index = fname.rfind(".")
    m = fname[:dot_index]

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.output_dir, m + "_quad.obj")):
        print(f"already quadrangulated {fname}")
        return

    scale_exec = [os.path.join("build", "bin", "convert_mesh"),]
    scale_exec += [ '--mesh', os.path.join(args.input_dir, fname)]
    scale_exec += [ '--output', args.output_dir]
    scale_exec += [ '--scale', args.scale]

    quad_exec = [os.path.join("..", "IGM-Source", "build", "Build", "bin", "igm"),]
    quad_exec += ['--in-mesh-seamless', os.path.join(args.output_dir, m + "_scaled.obj")]
    quad_exec += ['--out-quadmesh', os.path.join(args.output_dir, m + "_quad.obj")]
    #quad_exec += ['quantization_allow_zero_arcs=false']
    #quad_exec += ['--Config', os.path.join(args.output_dir, m + "_quad.obj")]
    #quad_exec += ['--in-config', os.path.join("scripts", "json", "quad.json")]

    try:
        log_path = os.path.join(args.output_dir, m + "_scale.log")
        with open(log_path, 'w') as log_file:
            subprocess.run(scale_exec, stdout=log_file)
        log_path = os.path.join(args.output_dir, m + "_quad.log")
        with open(log_path, 'w') as log_file:
            subprocess.run(quad_exec, stdout=log_file, stderr=log_file)
    except:
        print("Could not process mesh")

def main():
    parser = argparse.ArgumentParser("Generate parameterization with coarse voronoi overlay")
    parser.add_argument("-i", "--input_dir",           help="directory with input meshes",
                                                     type=str)
    parser.add_argument("-o", "--output_dir",           help="directory to write mesh with field",
                                                     type=str, default="")
    parser.add_argument("--scale",           help="scale for quad mesh grid",
                                                     type=str, default="1")
    args = parser.parse_args()


    if (args.input_dir.endswith('.obj')) or (args.input_dir.endswith('.h5')):
        dirpath, filename = os.path.split(args.input_dir)
        args.input_dir = dirpath
        files = [filename,]
    else:
        #files = [m for m in os.listdir(args.input_dir) if m.endswith('_opt.h5')]
        files = [m for m in os.listdir(args.input_dir) if m.endswith('_param.obj')]
    print(f"generating quad mesh for {len(files)} meshes")
    print(files)

    if args.output_dir == "":
        args.output_dir = os.path.join(args.input_dir, "quads")
    os.makedirs(args.output_dir, exist_ok=True)
    
    pool_args = [(m, args) for m in files]
    with multiprocessing.Pool(processes=1) as pool:
        pool.starmap(process_file, pool_args, chunksize=1)

if __name__ == "__main__":
    main()
