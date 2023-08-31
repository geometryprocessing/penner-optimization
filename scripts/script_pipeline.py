import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import script_util
import script_optimize
import script_optimize_shear
import script_overlay
import script_simplify_vf
import script_refine
import script_render_uv
import script_slim
import script_energy_table
import script_error_table
import script_render_layout
import script_histogram
import script_interpolate
import script_colormap_histogram

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
        method = pipeline_item['method']
        print("Running {}".format(method))
        args_list = pipeline_item['args_list']
        if pipeline_item['skip']:
            continue
        if (method == 'optimize'):
            for args_spec in args_list:
                # Get default arguments for optimization
                parser_method = script_util.generate_parser()
                script_optimize.add_optimize_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run optimization
                script_optimize.optimize_many(args)
        if (method == 'optimize_shear'):
            for args_spec in args_list:
                # Get default arguments for optimization
                parser_method = script_util.generate_parser()
                script_optimize_shear.add_optimize_shear_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run optimization
                script_optimize_shear.optimize_shear_many(args)
        if (method == 'slim'):
            for args_spec in args_list:
                # Get default arguments for slim
                parser_method = script_util.generate_parser()
                script_slim.add_slim_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run method
                script_slim.slim_many(args)
        if (method == 'interpolate'):
            for args_spec in args_list:
                # Get default arguments for conversion
                parser_method = script_util.generate_parser()
                script_interpolate.add_interpolate_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)
                
                # Run method
                script_interpolate.interpolate_many(args)
        if ((method == 'overlay') or (method == 'vf_pipeline')):
            for args_spec in args_list:
                # Get default arguments for conversion
                parser_method = script_util.generate_parser()
                script_overlay.add_overlay_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)
                
                # Run method
                script_overlay.overlay_many(args)
        if ((method == 'refine') or (method == 'vf_pipeline')):
            for args_spec in args_list:
                # Get default arguments for simplification
                parser_method = script_util.generate_parser()
                script_refine.add_refine_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run method
                script_refine.refine_many(args)
        if (method == 'simplify_vf'):
            for args_spec in args_list:
                # Get default arguments for simplification
                parser_method = script_util.generate_parser()
                script_simplify_vf.add_simplify_vf_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run method
                script_simplify_vf.simplify_vf_many(args)
        if ((method == 'energy_table') or (method == 'analysis_pipeline')):
            for args_spec in args_list:
                # Get default arguments for method
                parser_method = script_util.generate_parser()
                script_energy_table.add_energy_table_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run method
                script_energy_table.energy_table(args)
        if ((method == 'histogram') or (method == 'analysis_pipeline')):
            for args_spec in args_list:
                # Get default arguments for method
                parser_method = script_util.generate_parser()
                script_histogram.add_histogram_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run method
                script_histogram.histogram_many(args)
        if ((method == 'error_table') or (method == 'uv_analysis_pipeline')):
            for args_spec in args_list:
                # Get default arguments for method
                parser_method = script_util.generate_parser()
                script_error_table.add_error_table_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run method
                script_error_table.error_table(args)
        if ((method == 'colormap_histogram') or (method == 'uv_analysis_pipeline')):
            for args_spec in args_list:
                # Get default arguments for method
                parser_method = script_util.generate_parser()
                script_colormap_histogram.add_colormap_histogram_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run method
                script_colormap_histogram.colormap_histogram_many(args)
        if (method == 'render_layout'):
            for args_spec in args_list:
                # Get default arguments for method
                parser_method = script_util.generate_parser()
                script_render_layout.add_render_layout_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run method
                script_render_layout.render_layout_many(args)
        if ((method == 'render_uv') or (method == 'render_pipeline')):
            for args_spec in args_list:
                # Get default arguments for rendering from uv
                parser_method = script_util.generate_parser()
                script_render_uv.add_render_uv_arguments(parser_method)
                args_default = vars(parser_method.parse_args(""))

                # Overwrite arguments 
                args = script_util.overwrite_args(args_default, global_args)
                args = script_util.overwrite_args(args_default, args_spec)

                # Run method
                script_render_uv.render_uv_many(args)