#!/usr/bin/env python3
# coding: utf-8

"""
General Plot Parameters Dictionary
----------------------------------
- 'canvas_dimensions'  --> 'figsize' -- size of the plot (xy)
- 'x_title_dimension'  --> 'size' -- size of the x axis title
- 'y_title_dimension'  --> 'size' -- size of the x axis title
- 'z_title_dimension'  --> 'size' -- size of the x axis title
- 'output_quality'     --> 'dpi' -- density of pixels per inch
- 'labels_params'      --> 'rotation' -- label text angle
                       --> 'fontproperties' -- fonttype, dimensions, etc.
- 'ticks_params'       --> 'direction' -- tick facing inward or outward
                       --> 'width' -- tick width
                       --> 'length' -- tick length
                       --> 'color' -- tick color
- 'minor_ticks_params' --> 'which' -- select 1 (x, y) or 2 axis (xy)
                       --> 'width' -- minor tick width
                       --> 'color' -- minor tick color
                       --> 'direction' -- minor tick direction

Contour Plot Parameters Dictionary
----------------------------------
- 'vmin' -- contour level minimum
- 'vmax' -- contour level maximum
- 'nlev' -- number of contour levels
- 'cmap' -- selected color map
- 'ccol' -- contour line color
- 'cwid' -- contour line width
- 'cfon' -- fontsize of the contour line level
- 'scol' -- scatter points color
- 'swid' -- scatter points size
"""

import os
import sys
import csv
import numpy as np
from gridData import Grid
from scipy.interpolate import griddata

from matplotlib import cm, rcParams
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Matplotlib settings
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = ( r'\usepackage{siunitx} ' 
                                    r'\sisetup{detect-all}'
                                    )


general_params = { 'canvas_dimensions'  : { 'figsize' : (6.0,4.0) },
                   'x_title_dimension'  : { 'size' : 12 },
                   'y_title_dimension'  : { 'size' : 12 },
                   'z_title_dimension'  : { 'size' : 12 },
                   'output_quality'     : { 'dpi' : 600 },
                   'labels_params'      : { 'rotation' : 'horizontal','fontproperties' : fm.FontProperties(size=12) },
                   'ticks_params'       : { 'direction' : 'in', 'axis' : 'both', 'width' : 1,'length' : 3,'color' : '#000000' },
                   'minor_ticks_params' : { 'which' : 'minor','width' : 0.6,'color' : '#000000','direction' : 'in' }
                   }


contour_params = { 'vmin': -4.0,
                   'vmax':  4.0,
                   'nlev':  200,
                   'cmap':'RdBu',
                   #'cmap':'Spectral',
                   #'cmap':'PiYG',
                   #'cmap':'RdYlBu',
                   'ccol': '#000000',
                   'cwid': 0.2,
                   'cfon': 8,
                   'scol': '#000000',
                   'swid': 30
                   }


def _plot_slice(yzp, **kwargs):

    # Data processing
    y, z, p = yzp.T

    ny = np.unique(y).shape[0]
    nz = np.unique(z).shape[0]

    Y = y.reshape(ny, nz)
    Z = z.reshape(ny, nz)
    P = p.reshape(ny, nz)

    # Figure options
    fig, ax = plt.subplots(**general_params['canvas_dimensions'])
    vmin, vmax, nlev, cmap, ccol, cwid, cfon, scol, swid = contour_params.values()

    img = plt.imshow(np.array([[vmin,vmax]]), cmap=plt.cm.get_cmap(cmap, nlev), aspect='auto')
    img.set_visible(False)
    cbar = plt.colorbar(orientation="vertical")
    cbar.ax.tick_params(direction='in')
    ctr = ax.contourf(Y, Z, P, np.linspace(vmin, vmax, nlev), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    for c in ctr.collections:
        c.set_edgecolor("face")

    if nlev <= 100:
        contours = ax.contour(Y, Z, P, np.linspace(vmin, vmax, nlev), colors=ccol,
                              linewidths=cwid)

        for line, lvl in zip(contours.collections, contours.levels):
            line.set_linestyle('-')


    # Other fancy stuff
    ax.set_xlabel(r'$x$ / \si{\angstrom}')
    ax.set_ylabel(r'$y$ / \si{\angstrom}')
    # cbar.set_label(r'$V_{Coul}$ / \si{\volt}', labelpad=10)

    off = 0.1
    ax.set_xlim(ax.get_xlim()[0] + off, ax.get_xlim()[1] - off)
    ax.set_ylim(ax.get_ylim()[0] - off, ax.get_ylim()[1] + off)
    ax.minorticks_on()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(**general_params['minor_ticks_params'])
    ax.tick_params(**general_params['ticks_params'])
    ax.yaxis.set_major_formatter(FormatStrFormatter('$%.0f$'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('$%.0f$'))
    ax.invert_yaxis()

    return ax, fig


def _plot_silhouette(pts, ax, fig, **kwargs):

    x = pts[:,0]
    y = pts[:,1]
    z = pts[:,2]

    ax.scatter(x, y, color='k', **kwargs)

    return ax, fig


def read_xyz_esp(fname):
    '''
    Wrapper function to read grid coordinates and esp from an xyz file.

    Parameters
    ----------
    fname: str.
        filename to read.

    Returns
    -------
    xyzv: np.array (N, 4).
        coordinates of the grid and potential.
    '''

    xyzv = np.loadtxt(fname, skiprows=2, usecols=[1,2,3,4])

    return xyzv


def read_cub_esp(fname):
    '''
    Wrapper function to read grid coordinates and esp from a cub file.

    Parameters
    ----------
    fname: str.
        filename to read.

    Returns
    -------
    xyzv: np.array (N, 4).
        coordinates of the grid and potential.
    '''

    c = Cube(fname)
    xyzv = np.c_[ c.grid * au2ang, c.data ]

    return xyzv


def read_dx_esp(fname):
    '''
    Wrapper function to read grid coordinates and esp from a dx file.

    Parameters
    ----------
    fname: str.
        filename to read.

    Returns
    -------
    xyzv: np.array (N, 4).
        coordinates of the grid and potential.
    '''

    g = Grid(fname)
    o = g.origin
    d = g.delta
    nx, ny, nz = g.grid.shape
    grid = np.zeros((nx, ny, nz, 3))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                grid[i,j,k,0] = o[0] + i * d[0]
                grid[i,j,k,1] = o[1] + j * d[1]
                grid[i,j,k,2] = o[2] + k * d[2]

    grid = grid.reshape(nx * ny * nz, 3, order="F")
    V = g.grid.reshape(nx * ny * nz)
    xyzv = np.c_[ grid, V ]

    return xyzv


if __name__ == "__main__":

    fname = sys.argv[1]
    xyzv = read_dx_esp(fname)
    yzv = np.c_[ xyzv[:,:2], xyzv[:,-1] ]

    ax, fig =_plot_slice(yzv)

    # read silhouette
    ats = np.loadtxt(sys.argv[2], skiprows=2, usecols=[0], dtype=str)
    sil = np.loadtxt(sys.argv[2], skiprows=2, usecols=[1, 2, 3])
    ax, fig = _plot_silhouette(sil[ats != "H" ], ax, fig)

    plt.show()
