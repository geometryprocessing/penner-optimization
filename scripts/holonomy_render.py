# Script to render mesh with texture from uv coordinates.
#
# By default, runs all meshes specified by the `fname` argument in parallel.
# Functions to run the parallelized script and the method without parllelization
# are also exposed for use in other modules.

import os, sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import numpy as np
import igl
import pickle
import render as render
import penner
import optimize_impl.energies as energies
from conformal_impl.halfedge import *
from matplotlib import cm, colors
import script_util
import matplotlib.pyplot as plt
from tqdm import trange


def render_uv_one(args, fname):
    # Get common rendering parameters
    W = args['width']
    H = args['height']
    bd_thick = args['bd_thick']

    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index] 
    if (args['suffix'] == ""):
        name = m
    else:
        name = m + '_'+args['suffix']

    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, name+'_similarity_render.log')
    logger = script_util.get_logger(log_path)
    logger.info("Rendering {}".format(name))

    # Load mesh information
    try:
        input_dir = args['input_dir']
        logger.info("Loading initial mesh at {}".format(input_dir))
        v3d_orig, f_orig = igl.read_triangle_mesh(os.path.join(input_dir, m+'.obj'))
    except:
        logger.error("Could not load initial mesh")
        return

    # Load uv information
    try:
        uv_dir = args['uv_dir']
        logger.info("Loading uv coordinates at {}".format(uv_dir))
        v3d, uv, _, f, fuv, _ = igl.read_obj(os.path.join(uv_dir, m + "_output", name + ".obj"))
    except:
        logger.error("Could not load uv coordinates")
        return

    # Cut mesh along along uv lines
    #v_cut = script_util.cut_mesh(v3d, f, uv, fuv)
    #v3d = v_cut
    #f = fuv

    # Need to build double mesh when it has boundary
    is_bd = igl.is_border_vertex(v3d_orig, f_orig)
    build_double = (np.sum(is_bd) != 0)
    logger.info("Is double: {}".format(build_double))

	# FIXME Add option to use conversion directly
    # Load vn_to_v from simplification
    try:
        vertex_map_path = os.path.join(uv_dir, m + "_output", name + '_vn_to_v')
        logger.info("Loading vertex map at {}".format(vertex_map_path))
        vn_to_v = np.loadtxt(vertex_map_path, dtype=int)
    except:
        logger.error("Could not load vertex map")
        vn_to_v = np.arange(len(v3d))

    # Load endpoints from simplification
    try:
        endpoints_path = os.path.join(uv_dir, m + "_output", name + '_endpoints')
        logger.info("Loading endpoints at {}".format(endpoints_path))
        endpoints = np.loadtxt(endpoints_path, dtype=int)
    except:
        logger.error("Could not load endpoints")
        endpoints = np.full((len(v3d), 2), -1)

    # Load camera information
    try:
        camera_path = os.path.join(args['camera_dir'], m+'_camera.pickle')
        logger.info("Loading camera at {}".format(camera_path))
        with open(camera_path, 'rb') as fp:
            cam = pickle.load(fp)
            vc = pickle.load(fp)
            fc = pickle.load(fp)
            red_size = pickle.load(fp)
            blue_size = pickle.load(fp)
        (view, proj, vp) = cam
        if not build_double:
            fc = fc[:red_size+blue_size,:]
    except:
        logger.error("Could not load camera")
        return

    # Remove cones if flag set
    if (args["no_cones"]):
        vc = []
        fc = []
        red_size = 0
        blue_size = 0

    # Get cut to singularity edges (or build for double meshes)
    use_boundary_edges = True
    if (args["no_cut"]):
        v_cut_to_sin = []
        f_cut_to_sin = []
        logger.info("Skipping cut to singularity")
    elif (use_boundary_edges):
        v_cut_to_sin, f_cut_to_sin = script_util.get_boundary_edges(v3d, uv, f, fuv, bd_thick)
    else:
        try:
            cut_to_sin_list_path = os.path.join(uv_dir, m + "_output", name+"_cut_to_sin_list.pickle")
            logger.info("Loading cut_to_sin_list at {}".format(cut_to_sin_list_path))
            with open(cut_to_sin_list_path, 'rb') as fp:
                cut_to_sin_list = pickle.load(fp)
        except:
            logger.error("Could not load cut_to_sin_list")
            # FIXME Make independent function
            bd_loops = igl.all_boundary_loop(f)
            cut_to_sin_list = []
            #for loop in bd_loops:
            #    for i in np.arange(len(loop) - 1):
            #        cut_to_sin_list.append([loop[i], loop[i+1]])
        v_cut_to_sin, f_cut_to_sin = penner.get_edges(v3d_orig, f_orig, cut_to_sin_list, bd_thick)

    # Get point matrices
    logger.info("Getting point matrices")
    fid_mat, bc_mat = penner.get_pt_mat(cam, v3d, f, vc, fc, red_size, blue_size, W, H)
    fid_mat_sin, bc_mat_sin = penner.get_pt_mat(
        cam,
        v3d_orig,
        f_orig,
        v_cut_to_sin,
        f_cut_to_sin,
        0,
        0,
        W,
        H
    )
    for i in trange(H):
        for j in range(W):
            if fid_mat_sin[i][j] == -4 and fid_mat[i][j] >= 0:
                fid_mat[i][j] = -5

	# Get connectivity for new vf connectivity
    logger.info('Getting connectivity')
    n, opp, bd, vtx_reind = FV_to_NOB(f)
    C = NOB_to_connectivity(n, opp, bd)
    h = C.f2he
    to = vtx_reind[C.to]
    u = np.zeros(len(n))
    v = np.zeros(len(n))

    # Get per corner uv
    logger.info('Getting per corner uv coordinates')
    u, v = render.get_corner_uv(n, h, to, f, fuv, uv)

    # Get colormap for the mesh
    logger.info('Getting {} colormap'.format(args['colormap']))
    if args['interpolate_from_original']:
        r = energies.get_interpolated_vertex_energy(
            v3d,
            f,
            uv,
            fuv,
            vn_to_v,
            endpoints,
            args['colormap']
        )
    else:
        r = energies.get_vertex_energy(v3d, f, uv, fuv, args['colormap'])
    r = 0.25 + 0.0 * np.abs(r)
    r = 0.5 + 0.0 * np.abs(r)
    #colormap = cm.get_cmap('YlOrBr')
    #colormap = cm.get_cmap('OrRd')
    #colormap = cm.get_cmap('PuRd')
    colormap = cm.get_cmap('cool')
    colormap_scale = args['colormap_scale']
    logger.info('Using colormap scale {}'.format(colormap_scale))
    if (colormap_scale > 0):
        norm = colors.CenteredNorm(colormap_scale*0.5, colormap_scale*0.6)
    else:
        norm = colors.CenteredNorm(0, 1)
        r = r * 0
    norm = colors.NoNorm()

    # Render image
    logger.info('Rendering image')
    uv_scale = args['uv_scale'] * igl.bounding_box_diagonal(v3d)
    color_rgb = render.color_mesh_with_grid(
        fid_mat,
        bc_mat,
        h,
        n,
        to,
        u,
        v,
        r,
        H,
        W,
        colormap,
        norm,
        N_bw = args["N_bw"],
        thick = 0.1,
        uv_scale=uv_scale
    )

    os.makedirs(os.path.join(args['output_dir'], 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    # Save plain image to file
    image_path = os.path.join(output_dir, 'images', name+"_plain.png")
    logger.info('Saving plain image at {}'.format(image_path))
    plt.imsave(image_path, color_rgb)

    # Add shading
    logger.info("Adding shading")
    render.add_shading(color_rgb, v3d, f, fid_mat, bc_mat, cam[0], cam[1])

    # Save shaded image to file
    shaded_image_path = os.path.join(output_dir, 'images', name+".png")
    logger.info('Saving final image at {}'.format(shaded_image_path))
    plt.imsave(shaded_image_path, color_rgb)

    # Save shaded image to file in global directory
    shaded_image_path = os.path.join(args['output_dir'], 'images', name+".png")
    logger.info('Saving final image at {}'.format(shaded_image_path))
    plt.imsave(shaded_image_path, color_rgb)


def render_uv_many(args):
    script_util.run_many(render_uv_one, args)


def add_render_uv_arguments(parser):
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for output images"
    )
    parser.add_argument(
        "--uv_dir", 
        help="path to the directory of meshes with uv coordinates"
    )
    parser.add_argument(
        "--camera_dir", 
        help="path to the directory with camera objects",
        default="data/cameras"
    )
    parser.add_argument(
        "--uv_scale",
        help="ratio to scale uv coordinates by",
        type=float,
        default=1
    )
    parser.add_argument(
        "--colormap",
        help="energy for colormap",
        default="scale_factors"
    )
    parser.add_argument(
        "--interpolate_from_original",
        help="use interpolated colormap from original values",
        action="store_true"
    )
    parser.add_argument(
        "--colormap_scale",
        help="ratio to scale colormap by",
        type=float,
        default=1
    )
    parser.add_argument(
        "--suffix",
        help="suffix for output files",
        default=""
    )
    parser.add_argument(
        "-H", "--height",
        help="image height",
        type=int,
        default=800
    )
    parser.add_argument(
        "-W", "--width",
        help="image width",
        type=int,
        default=1280
    )
    parser.add_argument(
        "--N_bw",
        help="number of grid lines ",
        type=int,
        default=50
    )
    parser.add_argument(
        "--bd_thick",
        help="line thickness",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--no_cones",
        help="remove cones from image",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--no_cut",
        help="remove cut to singularity from image",
        type=bool,
        default=False
    )

if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Render mesh from uv coordinates")
    add_render_uv_arguments(parser)
    args = vars(parser.parse_args())
 
    # Run method in parallel
    render_uv_many(args)
