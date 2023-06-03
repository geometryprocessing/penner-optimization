
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os, pickle

def print_flap(C, arr, h):
    """
    Print value of the halfedge indexed array arr at halfedge h and the other halfedges
    in the flap around h
    
    param[in] Mesh C: mesh containing h
    param[in] np.array arr: array to print local information for
    param[in] int h: halfedge index
    """
    # Get flap halfedges around h
    ho = C.opp[h]
    hb = C.n[h]; ha = C.n[C.n[h]]
    hbo = C.n[C.n[ho]]; hao = C.n[ho]

    # Print flap information in arr
    print("Halfedge {} with val {}".format(h, arr[h]))
    print("Opposite {} with val {}".format(ho, arr[ho]))
    print("Next {} with val {}".format(hb, arr[hb]))
    print("Prev {} with val {}".format(ha, arr[ha]))
    print("Opposite next {} with val {}".format(hao, arr[hao]))
    print("Opposite prev {} with val {}".format(hbo, arr[hbo]))


def generate_histogram(
    X,
    label,
    binrange,
    output_path,
    ylim=50,
    width=7,
    height=5,
    use_percentage=True,
    log=False
):
    # Set width and height of figure
	matplotlib.rcParams['figure.figsize'] = (width, height)

    # Set percentage or absolute scale for y axis
	if use_percentage:
		hist = seaborn.histplot(X, bins = 21, stat='percent', binrange=binrange)
		hist.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
		plt.ylim(0, ylim)
	else:
		hist = seaborn.histplot(X, bins = 21, binrange=binrange)

    # Set axes labels
	hist.set_xlabel(label, fontsize=50)
	hist.set_ylabel("")
	hist.tick_params(labelsize=30)
    
    # Optionally use log scale
	if log:
		hist.set_xscale('log')

    # Save figure to file
	plt.savefig(output_path, bbox_inches='tight')

def generate_comparison_histogram(
    X,
    label,
    binrange,
    output_path,
    ylim=50,
    width=7,
    height=5,
    use_percentage=True,
    log=False
):
    # Set colors for seaborn
    colors = ["#b90f29", "#3c4ac8"]
    seaborn.set_palette(seaborn.color_palette(colors))

    # Set width and height of figure
    matplotlib.rcParams['figure.figsize'] = (width, height)

    # Set percentage or absolute scale for y axis
    if use_percentage:
        hist = seaborn.histplot(X, bins = 21, stat='percent', binrange=binrange)
        hist.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        plt.ylim(0, ylim)
    else:
        hist = seaborn.histplot(X, bins = 21, binrange=binrange)
 
    # Set axes labels
    hist.set_xlabel(label, fontsize=50)
    hist.set_ylabel("")
    hist.tick_params(labelsize=30)
    hist.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    # Format legend
    plt.setp(hist.get_legend().get_texts(), fontsize='22') 

    # Optionally use log scale
    if log:
        hist.set_xscale('log')

    # Save figure to file
    plt.savefig(output_path, bbox_inches='tight')

def pickle_load(
    output_dir,
    file_name
):
    with open(os.path.join(output_dir, file_name), 'rb') as f:
        return pickle.load(f)

def pickle_save(
    output_dir,
    file_name,
    object
):
    with open(os.path.join(output_dir, file_name), 'wb') as f:
        pickle.dump(object, f)