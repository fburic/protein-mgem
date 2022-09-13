import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def scatter_density(x, y, gauss_kern_bandwidth=0.05, **kwargs):
    """
    Scatter plot with density coloring using matplotlib.
    kwargs are passed to the pyplot.scatter() function,
    with defaults s = 4 and cmap = 'plasma'
    """
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy, bw_method=gauss_kern_bandwidth)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    if 's' not in kwargs:
        kwargs['s'] = 4
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'plasma'

    plt.scatter(x, y, c=z, **kwargs)
