import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
opt_script_dir = os.path.join(script_dir, 'optimization_scripts')
sys.path.append(opt_script_dir)
import random
import numpy as np
import optimization_scripts.script_util as script_util
import holonomy_render
import optimize_angles
import optimize_fixed_boundary
import optimize_refined_angles
import optimize_aligned_angles
import render_mesh
import statistics
import holonomy_overlay
import optimize_similarity
import holonomy_histogram
import argparse

def generate_parser(description='Run the optimization method with options.'):
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--num_processes",           help="number of processes for parallelism",
                                                     type=int, default=8)
    return parser

if __name__ == "__main__":
    np.seterr(invalid='raise')

    # Parse arguments for the script
    parser = generate_parser("Run pipeline")
    parser.add_argument("pipeline_path")
    pipeline_args = parser.parse_args()

    pipeline_spec = script_util.load_pipeline(pipeline_args.pipeline_path)
    pipeline_dir = os.path.dirname(pipeline_args.pipeline_path)

    # Load global arguments
    global_args = pipeline_spec['global_args']
    if 'output_dir' not in global_args:
        global_args['output_dir'] = pipeline_dir
    if 'lambdas_dir' not in global_args:
        global_args['lambdas_dir'] = pipeline_dir
    if 'uv_dir' not in global_args:
        global_args['uv_dir'] = pipeline_dir
    if 'input_dir' not in global_args:
        global_args['input_dir'] = pipeline_dir
    if 'fname' not in global_args:
        files = os.listdir(global_args['input_dir'])
        global_args['fname'] = [f for f in files if f.endswith(".obj")]
        random.shuffle(global_args['fname'])

    # Iterate over all scripts to run listed in the pipeline file
    pipeline_list = pipeline_spec['pipeline']
    for pipeline_item in pipeline_list:
        method = pipeline_item['method']
        args_list = pipeline_item['args_list']
        if pipeline_item['skip']:
            continue
        if (method == 'optimize_angles'):
            for args_spec in args_list:
                # Get default arguments for optimization
                parser_method = generate_parser()
                optimize_angles.add_constrain_similarity_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run optimization
                optimize_angles.constrain_similarity_many(args)
        if (method == 'optimize_fixed_boundary'):
            for args_spec in args_list:
                # Get default arguments for optimization
                parser_method = generate_parser()
                optimize_fixed_boundary.add_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run optimization
                optimize_fixed_boundary.run_many(args)
        if (method == 'optimize_similarity'):
            for args_spec in args_list:
                # Get default arguments for optimization
                parser_method = generate_parser()
                optimize_similarity.add_optimize_similarity_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run optimization
                optimize_similarity.optimize_similarity_many(args)
        if (method == 'holonomy_overlay'):
            for args_spec in args_list:
                # Get default arguments for optimization
                parser_method = generate_parser()
                holonomy_overlay.add_similarity_overlay_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run optimization
                holonomy_overlay.similarity_overlay_many(args)
        if (method == 'statistics'):
            for args_spec in args_list:
                # Get default arguments for method
                parser_method = generate_parser()
                statistics.add_statistics_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run method
                statistics.run_statistics(args)
        if (method == 'holonomy_histogram'):
            for args_spec in args_list:
                # Get default arguments for method
                parser_method = script_util.generate_parser()
                holonomy_histogram.add_similarity_histogram_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run method
                holonomy_histogram.similarity_histogram_many(args)
        if (method == 'holonomy_render'):
            for args_spec in args_list:
                # Get default arguments for rendering from uv
                parser_method = generate_parser()
                holonomy_render.add_render_uv_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run method
                holonomy_render.render_uv_many(args)
        if (method == 'render_mesh'):
            for args_spec in args_list:
                # Get default arguments for rendering from uv
                parser_method = generate_parser()
                render_mesh.add_render_mesh_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run method
                render_mesh.render_mesh_many(args)
        if (method == 'optimize_refined_angles'):
            for args_spec in args_list:
                # Get default arguments for optimization
                parser_method = generate_parser()
                optimize_refined_angles.add_optimize_refined_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run optimization
                optimize_refined_angles.optimize_refined_many(args)
        if (method == 'optimize_aligned_angles'):
            for args_spec in args_list:
                # Get default arguments for optimization
                parser_method = generate_parser()
                optimize_aligned_angles.add_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run optimization
                optimize_aligned_angles.run_many(args)
