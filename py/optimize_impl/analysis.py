
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os, pickle


def generate_histogram(
    X,
    label,
    binrange,
    output_path,
    ylim=50,
    width=7,
    height=5,
    use_percentage=True,
    log=False,
    bins=21,
    logy=False
):
    # Set width and height of figure
    matplotlib.rcParams['figure.figsize'] = (width, height)

    # Set percentage or absolute scale for y axis
    fig, ax = plt.subplots(1)
    if use_percentage:
        hist = seaborn.histplot(X, bins = bins, stat='percent', binrange=binrange, ax=ax)
        hist.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.set_ylim(0, ylim)
    else:
        hist = seaborn.histplot(X, bins = bins, binrange=binrange)
        ax.set_ylim(0, ylim)

    # Set axes labels
    hist.set_xlabel(label, fontsize=50)
    hist.set_ylabel("")
    hist.tick_params(labelsize=30)
    
    # Optionally use log scale
    if log:
        hist.set_xscale('log')
    if logy:
        hist.set_yscale('log')

    # Save figure to file
    fig.savefig(output_path, bbox_inches='tight')

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