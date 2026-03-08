import os
import sys
import script_util
import shutil

base_dir = os.path.dirname(__file__)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts')
sys.path.append(script_dir)
module_dir = os.path.join(base_dir, '..', 'py')
sys.path.append(module_dir)


def add_arguments(parser):
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for outputted meshes"
    )
    parser.add_argument(
        "-s", "--symlink_dir",
        help="directory for symlinks"
    )
    parser.add_argument(
        "--make_copy",
        type=bool,
        default=False
    )

def run_one(args, fname):

    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_' + args['suffix']
    symlink_dir = script_util.get_mesh_output_directory(args['symlink_dir'], m)
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    orig_mesh = os.path.join(symlink_dir, name + ".obj")
    orig_ffield = os.path.join(symlink_dir, m + ".ffield")
    orig_edges = os.path.join(symlink_dir, name + "_misaligned_edges")
    symlink_mesh = os.path.join(output_dir, name + ".obj")
    symlink_ffield = os.path.join(output_dir, m + ".ffield")
    symlink_edges = os.path.join(output_dir, name + "_misaligned_edges")

    if os.path.islink(symlink_mesh):
        os.remove(symlink_mesh)
    
    if os.path.islink(symlink_ffield):
        os.remove(symlink_ffield)

    if os.path.islink(symlink_edges):
        os.remove(symlink_edges)

    if args['make_copy']:
        try:
            shutil.copy(orig_mesh, symlink_mesh)
            shutil.copy(orig_ffield, symlink_ffield)
            shutil.copy(orig_edges, symlink_edges)
        except:
            print("Could not copy {}".format(orig_mesh))
    else:
        os.symlink(orig_mesh, symlink_mesh)
        os.symlink(orig_ffield, symlink_ffield)
        os.symlink(orig_edges, symlink_edges)

    
def run_many(args):
    if os.path.abspath(args['symlink_dir']) == os.path.abspath(args['output_dir']):
        print("Symlink directory is the same as the output directory. Skipping symlink creation.")
        return
    script_util.run_many(run_one, args)

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Create symlinks")
    add_arguments(parser)
    args = vars(parser.parse_args())

    run_many(args)
