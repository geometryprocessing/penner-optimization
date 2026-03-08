import os, sys
base_dir = os.path.dirname(__file__)
module_dir = os.path.join(base_dir, '..', '..', 'py')
sys.path.append(module_dir)
import numpy as np
import pandas as pd
import penner
import igl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir")
args = vars(parser.parse_args())

files = os.listdir(args['input_dir'])
suffix='_features'
suffix='_refined_with_uv_out_connected.obj'
suffix='_refined_with_uv_out.obj'
models = [f[:-(len(suffix))] for f in files if f.endswith(suffix)]
for model in models:
    try:
        v3d, uv, _, f, fuv, _ = igl.read_obj(os.path.join(args['input_dir'], model + suffix))
    except:
        print("Could not load mesh")

    seam_path = os.path.join(args['input_dir'], model + '_seams.obj')
    penner.write_boundary(seam_path, v3d, f, fuv)
