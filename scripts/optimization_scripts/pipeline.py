import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', '..', 'py')
sys.path.append(module_dir)

import numpy as np
import script_util
import optimize
import optimize_shear
import overlay
import refine
import render_uv
import slim
import energy_table
import error_table
import render_layout
import histogram
import interpolate
import colormap_histogram

if __name__ == "__main__":
    np.seterr(invalid='raise')

    # Parse arguments for the script
    parser = script_util.generate_parser("Run pipeline")
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

    # Iterate over all scripts to run listed in the pipeline file
    pipeline_list = pipeline_spec['pipeline']
    for pipeline_item in pipeline_list:
        # Optionally skip method
        if pipeline_item['skip']:
            continue

        # Get method arguments and function
        method = pipeline_item['method']
        print("Running {}".format(method))
        if (method == 'optimize'):
            add_arguments = optimize.add_optimize_arguments
            method_func = optimize.optimize_many
        elif (method == 'optimize_shear'):
            add_arguments = optimize_shear.add_optimize_shear_arguments
            method_func = optimize_shear.optimize_shear_many
        elif (method == 'slim'):
            add_arguments = slim.add_slim_arguments
            method_func = slim.slim_many
        elif (method == 'interpolate'):
            add_arguments = interpolate.add_interpolate_arguments
            method_func = interpolate.interpolate_many
        elif (method == 'overlay'):
            add_arguments = overlay.add_overlay_arguments
            method_func = overlay.overlay_many
        elif (method == 'refine'):
            add_arguments = refine.add_refine_arguments
            method_func = refine.refine_many
        elif (method == 'energy_table'):
            add_arguments = energy_table.add_energy_table_arguments
            method_func = energy_table.energy_table
        elif (method == 'error_table'):
            add_arguments = error_table.add_error_table_arguments
            method_func = error_table.error_table
        elif (method == 'histogram'):
            add_arguments = histogram.add_histogram_arguments
            method_func = histogram.histogram_many
        elif (method == 'colormap_histogram'):
            add_arguments = colormap_histogram.add_colormap_histogram_arguments
            method_func = colormap_histogram.colormap_histogram_many
        elif (method == 'render_uv'):
            add_arguments = render_uv.add_render_uv_arguments
            method_func = render_uv.render_uv_many
        elif (method == 'render_layout'):
            add_arguments = render_layout.add_render_layout_arguments
            method_func = render_layout.render_layout_many
        else:
            continue

        # Run chosen method
        args_list = pipeline_item['args_list']
        for args_spec in args_list:
            # Get default arguments
            parser_method = script_util.generate_parser()
            add_arguments(parser_method)
            args_default = vars(parser_method.parse_args(""))

            # Overwrite arguments 
            args = script_util.overwrite_args(args_default, global_args)
            args = script_util.overwrite_args(args_default, args_spec)

            # Run method
            method_func(args)
