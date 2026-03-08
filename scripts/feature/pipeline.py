import os
import sys
base_dir = os.path.dirname(__file__)
module_dir = os.path.join(base_dir, '..', '..', 'py')
sys.path.append(module_dir)
script_dir = os.path.join(base_dir, '..')
sys.path.append(script_dir)
opt_script_dir = os.path.join(script_dir, '..', 'optimization_scripts')
sys.path.append(opt_script_dir)

import random
import numpy as np
import optimization_scripts.script_util as script_util
import holonomy_render
import optimize_angles
import optimize_refined_angles
import render_mesh
import statistics as statistics
import holonomy_overlay
import optimize_similarity
import holonomy_histogram
import argparse
import error_table
import render_quads 
import feature.analysis
import feature.overlay
import feature.histogram
import feature.statistics
import feature.consolidate
import feature.layout
import feature.constraint_geometry 
import quadrangulate
import generate_feature_field
import quad_statistics
import feature.symmetric_dirichlet
import field_statistics
import uv_analysis 
import create_symlink 
import compute_distortion
import feature.optimize_closed_angles
import feature.optimize_relaxed_angles

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

    # build dictionary of methods for adding arguments
    argument_funcs = {}
    argument_funcs['optimize_angles'] = optimize_angles.add_constrain_similarity_arguments
    argument_funcs['optimize_similarity'] = optimize_similarity.add_optimize_similarity_arguments
    argument_funcs['holonomy_overlay'] = holonomy_overlay.add_similarity_overlay_arguments
    argument_funcs['statistics'] = statistics.add_statistics_arguments
    argument_funcs['holonomy_histogram'] = holonomy_histogram.add_similarity_histogram_arguments
    argument_funcs['holonomy_render'] = holonomy_render.add_render_uv_arguments
    argument_funcs['render_mesh'] = render_mesh.add_render_mesh_arguments
    argument_funcs['optimize_refined_angles'] = optimize_refined_angles.add_optimize_refined_arguments
    argument_funcs['error_table'] = error_table.add_error_table_arguments
    argument_funcs['quadrangulate'] = quadrangulate.add_arguments
    argument_funcs['quad_statistics'] = quad_statistics.add_arguments
    argument_funcs['symmetric_dirichlet'] = feature.symmetric_dirichlet.add_arguments
    argument_funcs['generate_feature_field'] = generate_feature_field.add_arguments
    argument_funcs['field_statistics'] = field_statistics.add_arguments
    argument_funcs['uv_analysis'] = uv_analysis.add_arguments
    argument_funcs['create_symlink'] = create_symlink.add_arguments
    argument_funcs['compute_distortion'] = compute_distortion.add_arguments
    argument_funcs['feature_statistics'] = feature.statistics.add_statistics_arguments
    argument_funcs['feature_histogram'] = feature.histogram.add_arguments
    argument_funcs['feature_analysis'] = feature.analysis.add_arguments
    argument_funcs['feature_consolidate'] = feature.consolidate.add_arguments
    argument_funcs['feature_overlay'] = feature.overlay.add_arguments
    argument_funcs['feature_layout'] = feature.layout.add_arguments
    argument_funcs['feature_constraint_geometry'] = feature.constraint_geometry.add_arguments
    argument_funcs['optimize_closed_angles'] = feature.optimize_closed_angles.add_arguments
    argument_funcs['optimize_relaxed_angles'] = feature.optimize_relaxed_angles.add_arguments

    # build dictionary of methods
    pipeline_funcs = {}
    pipeline_funcs['optimize_angles'] = optimize_angles.constrain_similarity_many
    pipeline_funcs['optimize_similarity'] = optimize_similarity.optimize_similarity_many
    pipeline_funcs['holonomy_overlay'] = holonomy_overlay.similarity_overlay_many
    pipeline_funcs['statistics'] = statistics.run_statistics
    pipeline_funcs['holonomy_histogram'] = holonomy_histogram.similarity_histogram_many
    pipeline_funcs['holonomy_render'] = holonomy_render.render_uv_many
    pipeline_funcs['render_mesh'] = render_mesh.render_mesh_many
    pipeline_funcs['optimize_refined_angles'] = optimize_refined_angles.optimize_refined_many
    pipeline_funcs['error_table'] = error_table.error_table
    pipeline_funcs['optimize_relaxed_angles'] = feature.optimize_relaxed_angles.run_many
    pipeline_funcs['quadrangulate'] = quadrangulate.run_many
    pipeline_funcs['quad_statistics'] = quad_statistics.run_many
    pipeline_funcs['symmetric_dirichlet'] = feature.symmetric_dirichlet.run_many
    pipeline_funcs['render_quads'] = render_quads.run_many
    pipeline_funcs['generate_feature_field'] = generate_feature_field.run_many
    pipeline_funcs['field_statistics'] = field_statistics.run_many
    pipeline_funcs['uv_analysis'] = uv_analysis.run_many
    pipeline_funcs['create_symlink'] = create_symlink.run_many
    pipeline_funcs['compute_distortion'] = compute_distortion.run_many
    pipeline_funcs['optimize_closed_angles'] = feature.optimize_closed_angles.run_many
    pipeline_funcs['feature_statistics'] = feature.statistics.run_statistics
    pipeline_funcs['feature_histogram'] = feature.histogram.run_many
    pipeline_funcs['feature_analysis'] = feature.analysis.run_many
    pipeline_funcs['feature_consolidate'] = feature.consolidate.run_many
    pipeline_funcs['feature_overlay'] = feature.overlay.run_many
    pipeline_funcs['feature_layout'] = feature.layout.run_many
    pipeline_funcs['feature_constraint_geometry'] = feature.constraint_geometry.run_many

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
        obj_files = [f for f in files if f.endswith(".obj")]
        random.shuffle(obj_files)
        if ('max_meshes' in global_args) and ((len(obj_files) > global_args['max_meshes'])):
            obj_files = obj_files[:global_args['max_meshes']]
        global_args['fname'] = obj_files

    # Iterate over all scripts to run listed in the pipeline file
    pipeline_list = pipeline_spec['pipeline']
    for pipeline_item in pipeline_list:
        method = pipeline_item['method']
        args_list = pipeline_item['args_list']
        if pipeline_item['skip']:
            continue

        for args_spec in args_list:
            # Get default arguments for method
            parser_method = generate_parser()
            argument_funcs[method](parser_method)
            args_default = vars(parser_method.parse_args(""))

            # Overwrite arguments
            args = script_util.overwrite_args(args_default, global_args)
            args = script_util.overwrite_args(args_default, args_spec)

            # Run method
            pipeline_funcs[method](args)
