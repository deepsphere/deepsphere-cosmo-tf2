"""Plotting module."""

from __future__ import division

from builtins import range
import datetime

import numpy as np
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import utils


def plot_filters_gnomonic(filters, order=10, ind=0, title='Filter {}->{}', graticule=False):
    """Plot all filters in a filterbank in Gnomonic projection."""
    nside = hp.npix2nside(filters.G.N)
    reso = hp.pixelfunc.nside2resol(nside=nside, arcmin=True) * order / 100
    rot = hp.pix2ang(nside=nside, ipix=ind, nest=True, lonlat=True)

    maps = filters.localize(ind, order=order)

    nrows, ncols = filters.n_features_in, filters.n_features_out

    if maps.shape[0] == filters.G.N:
        # FIXME: old signal shape when not using Chebyshev filters.
        shape = (nrows, ncols, filters.G.N)
        maps = maps.T.reshape(shape)
    else:
        if nrows == 1:
            maps = np.expand_dims(maps, 0)
        if ncols == 1:
            maps = np.expand_dims(maps, 1)

    # Plot everything.
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8/ncols*nrows),
                             squeeze=False, sharex='col', sharey='row')
    # turn of axes
    [axi.set_axis_off() for axi in axes.ravel()]

    # handle margins
    if title is None:
        margins = [0.003, 0.003, 0.003, 0.003]
        title = ""
    else:
        margins = [0.015, 0.015, 0.015, 0.015]

    cm = plt.cm.seismic
    cm.set_under('w')
    a = max(abs(maps.min()), maps.max())
    ymin, ymax = -a,a
    for row in range(nrows):
        for col in range(ncols):
            map = maps[row, col, :]
            hp.gnomview(map.flatten(), fig=fig, nest=True, rot=rot, reso=reso, sub=(nrows, ncols, col+row*ncols+1),
                    title=title.format(row, col), notext=True,  min=ymin, max=ymax, cbar=False, cmap=cm,
                    margins=margins)

    fig.suptitle('Gnomoinc view of the {} filters in the filterbank'.format(filters.n_filters), fontsize=25, y=1.05)

    if graticule:
        with utils.HiddenPrints():
            hp.graticule(verbose=False)

    return fig


def plot_filters_section(filters,
                         order=10,
                         xlabel='out map {}',
                         ylabel='in map {}',
                         title='Sections of the {} filters in the filterbank',
                         figsize=None,
                         **kwargs):
    """Plot the sections of all filters in a filterbank."""

    nside = hp.npix2nside(filters.G.N)
    npix = hp.nside2npix(nside)

    # Create an inverse mapping from nest to ring.
    index = hp.reorder(range(npix), n2r=True)

    # Get the index of the equator.
    index_equator, ind = get_index_equator(nside, order)
    nrows, ncols = filters.n_features_in, filters.n_features_out

    maps = filters.localize(ind, order=order)
    if maps.shape[0] == filters.G.N:
        # FIXME: old signal shape when not using Chebyshev filters.
        shape = (nrows, ncols, filters.G.N)
        maps = maps.T.reshape(shape)
    else:
        if nrows == 1:
            maps = np.expand_dims(maps, 0)
        if ncols == 1:
            maps = np.expand_dims(maps, 1)

    # Make the x axis: angular position of the nodes in degree.
    angle = hp.pix2ang(nside, index_equator, nest=True)[1]
    angle -= abs(angle[-1] + angle[0]) / 2
    angle = angle / (2 * np.pi) * 360

    if figsize==None:
        figsize = (12, 12/ncols*nrows)

    # Plot everything.
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             squeeze=False, sharex='col', sharey='row')

    ymin, ymax = 1.05*maps.min(), 1.05*maps.max()
    for row in range(nrows):
        for col in range(ncols):
            map = maps[row, col, index_equator]
            axes[row, col].plot(angle, map, **kwargs)
            axes[row, col].set_ylim(ymin, ymax)
            if row == nrows - 1:
                #axes[row, col].xaxis.set_ticks_position('top')
                #axes[row, col].invert_yaxis()
                axes[row, col].set_xlabel(xlabel.format(col))
            if col == 0:
                axes[row, col].set_ylabel(ylabel.format(row))
    fig.suptitle(title.format(filters.n_filters))#, y=0.90)
    return fig

def get_index_equator(nside, radius):
    """Return some indexes on the equator and the center of the index."""
    npix = hp.nside2npix(nside)

    # Create an inverse mapping from nest to ring.
    index = hp.reorder(range(npix), n2r=True)

    # Center index
    center = index[npix // 2]

    # Get the value on the equator back.
    equator_part = range(npix//2-radius, npix//2+radius+1)
    index_equator = index[equator_part]

    return index_equator, center
