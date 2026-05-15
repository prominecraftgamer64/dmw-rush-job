"""Utility functions for SVD and LSA notebook"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


def plot_energy_ratio(energy_ratio, tol=0.90):
    """Return energy ratio plot given energy ratio

    Parameters
    ----------
    energy_ratio : numpy array
        Array containing the percentage of energy preserved by
        each of the singular vectors.

    tol : float, default=0.90
        Default tolerance value for the optimal threshold

    Returns
    -------
    fig, ax : matplotlib figure and axes
        Figure and axes of the plot
    """
    # Get cumsum of energy ratio
    energy_ratio = (energy_ratio).cumsum()

    # Get index where energy ratio exceeds tolerance
    thresh = np.min(np.arange(len(energy_ratio))[energy_ratio >= tol]) + 1

    # Initialize figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # Plot explained varianced
    ax.plot(range(0, len(energy_ratio) + 1), [0] + energy_ratio.tolist(),
            lw=3.0, marker='o')

    # Plot threshold line
    ax.axvline(thresh, linestyle='-', lw=2.5, color='tab:orange')
    
    # Annotate threshold
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.05*thresh, 0.05, f"Number of SVs: {int(thresh)}",
            color='tab:orange', weight='bold', fontsize=12, transform=trans)

    # Set ylim
    ax.set_ylim([min(energy_ratio), 1.05])
    ax.set_xlim([0., len(energy_ratio)])

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set axis labels
    ax.set_xlabel('Number of SVs', fontsize=12)
    ax.set_ylabel('Cumulative Energy Preserved', fontsize=12)
    fig.suptitle('Cumulative Energy Preserved versus Number of SVs',
                 fontsize=14, weight='bold')

    # Set axis limits
    ax.set_ylim(0, 1)
    
    return fig, ax, thresh


def plot_reconstruction_error(reconstruction_error, tol=1e-2):
    """Return energy ratio plot given energy ratio

    Parameters
    ----------
    reconstruction_error : numpy array
        Array containing the reconstruction error.

    tol : float, default=1e-2
        Default tolerance value for the optimal threshold

    Returns
    -------
    fig, ax : matplotlib figure and axes
        Figure and axes of the plot
    """
    # Get index where energy ratio exceeds tolerance
    thresh = np.min(np.arange(len(reconstruction_error))[reconstruction_error <= tol]) + 1

    # Initialize figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # Plot explained varianced
    ax.plot(range(0, len(reconstruction_error) + 1), [None] + reconstruction_error.tolist(),
            lw=3.0, marker='o')

    # Plot threshold line
    ax.axvline(thresh, linestyle='-', lw=2.5, color='tab:orange')
    
    # Annotate threshold
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.05*thresh, 0.05, f"Number of SVs: {int(thresh)}",
            color='tab:orange', weight='bold', fontsize=12, transform=trans)

    # Set ylim
    ax.set_xlim([0., len(reconstruction_error)])

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set axis labels
    ax.set_xlabel('Number of SVs', fontsize=12)
    ax.set_ylabel('Reconstruction Error', fontsize=12)
    fig.suptitle('Reconstruction Error versus Number of SVs',
                 fontsize=14, weight='bold')
    
    return fig, ax, thresh


def plot_principal_components(X_transformed, W, column_1, column_2, margin=0.05,
                 hue=None, vals=None, figsize=(16, 8), wspace=0.03,
                 palette='default'):
    """
    Plot the principal components given X_transformed, W

    Parameters
    ----------
    X_transformed : pandas Data Frame
        Data Frame containing the tranformed feature vectors

    W : pandas Data Frame
        The data frame containing the principle components

    column_1, column 2 : str
        Principal component columns to be inspected

    margin : float, default=0.05
        Text margin on subplot 2 for prettification

    hue : numpy array, default=None
        Discriminator array to use as hue

    vals : list, default=None
        Specify to focus on specific hue values for pair plot.

    figsize : tuple, default=(16, 5)
        Specify the figure size of the lsa analysis plot

    wspace : float, default=0.03
        Specify the horizontal space between subplots

    palette : list of rgb, default='default'
        palette to use for plotting

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
        Figure and axes of the PC analysis
    """
    # Set color palette
    if palette == 'default':
        palette = sns.color_palette('tab10')

    # Initialize figure
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'wspace': wspace})

    # Set vals if not specified
    if vals is None:
        if hue is not None:
            vals = set(hue)

    # Plot word scatter plot for first 2 W's
    if hue is not None:
        for i, val in enumerate(vals):
            axes[0].plot(X_transformed.loc[hue == val, column_1],
                         X_transformed.loc[hue == val, column_2], 'o',
                         color=palette[i],
                         label=val)
        axes[0].legend()
    else:
        axes[0].plot(X_transformed.loc[:, column_1], X_transformed.loc[:, column_2], 'o')

    # Set axis labels
    axes[0].set_xlabel(column_1, fontsize=12)
    axes[0].set_ylabel(column_2, fontsize=12)

    # Remove spines
    for spine in ['top', 'right']:
        axes[0].spines[spine].set_visible(False)

    # Get lsas
    W = W.T
    lsas = np.append(W.loc[:, [column_1]],
                     W.loc[:, [column_2]],
                     axis=1)

    # Compute for weights, rank, then get indices
    weights = np.linalg.norm(lsas, axis=1)
    indices = weights.argsort()[-20:]

    # Get features
    features = W.index

    # Iterate through all top features
    for feature, vec in zip(features[indices], lsas[indices]):
        # Draw vector representation
        axes[1].annotate('', xy=(vec[0], vec[1]),  xycoords='data',
                         xytext=(0, 0), textcoords='data',
                         arrowprops=dict(facecolor=palette[0],
                                         edgecolor='none'))

        # Draw corresponding feature
        axes[1].text(vec[0], vec[1], feature, ha='center', color=palette[1],
                     fontsize=12, weight='bold', zorder=10)

        # Adjust xlim and ylim
        xlim = [np.min(W.loc[:, column_1]),
                np.max(W.loc[:, column_1])]
        xlim_range = xlim[1] - xlim[0]
        ylim = [np.min(W.loc[:, column_2]),
                np.max(W.loc[:, column_2])]
        ylim_range = ylim[1] - ylim[0]
        axes[1].set_xlim(xlim[0] - xlim_range*margin,
                         xlim[1] + xlim_range*margin)
        axes[1].set_ylim(ylim[0] - ylim_range*margin,
                         ylim[1] + ylim_range*margin)

        # Off axis for the vector plot
        axes[1].tick_params(axis='both',which='both',top=False, bottom=False,
                            labelbottom=False, labelleft=False, left=False)

    # Set axis labels
    axes[1].set_xlabel(column_1, fontsize=12)
    axes[1].set_ylabel(column_2, fontsize=12)

    # Remove spines
    for spine in ['top', 'right', 'left', 'bottom']:
        axes[1].spines[spine].set_visible(False)

    return fig, axes


def plot_topic_vector(term_topic_matrix, column, num_terms=20):
    """Return a plot of the weights of the topic vector with largest
    magnitude"""
    fig, ax = plt.subplots(figsize=(8, 6))
    (term_topic_matrix.loc[
        term_topic_matrix[column].abs().nlargest(num_terms).index[::-1], column
        ]
                      .plot(kind='barh'))

    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel("Weight", fontsize=12)
    ax.set_ylabel("Term", fontsize=12)

    fig.suptitle(column, fontsize=14, weight='bold')
    
    return fig, ax
