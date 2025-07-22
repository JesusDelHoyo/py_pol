# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------
# Authors:    Jesus del Hoyo and Luis Miguel Sanchez Brea
# Date:       2019/02/03 (version 1.0)
# License:    GPL
# ------------------------------------
""" functions for drawing """

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import (array, asarray, cos, exp, linspace, matrix, meshgrid,
                   ndarray, ones, outer, real, remainder, sin, size, sqrt,
                   zeros_like)
from scipy.signal import fftconvolve
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .__init__ import np, degrees, eps
from .utils import nearest2, obj_2_xyz, azel_2_xyz

# print(matplotlib.__version__)

Axes3D = Axes3D  # pycharm auto import
colors = matplotlib.colors.TABLEAU_COLORS
name_colors = list(colors)
# linestyles = [('dashdot', 'dashdot'),
#               ('loosely dashdotted', (0, (3, 10, 1, 10))),
#               ('dashdotted', (0, (3, 5, 1, 5))),
#               ('densely dashdotted', (0, (3, 1, 1, 1))),
#               ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
#               ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
#               ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
linestyles = [
    'dotted', 'dashed', 'dashdot', 'loosely dotted', 'loosely dashed',
    'loosely dashdot'
]

# Use only when creating docs
# import plotly.io as pio
# pio.renderers.default = "jupyterlab"
# print(pio.renderers)


def draw_ellipse(E,
                 N_angles=91,
                 filename='',
                 figsize=(6, 6),
                 limit='',
                 draw_arrow=True,
                 draw_labels=True,
                 draw_titles=True,
                 depol_central=False,
                 depol_contour=False,
                 depol_prob=False,
                 subplots=None,
                 N_prob=256,
                 contour_levels=0.9,
                 cmap='hot', *args, **kwargs):
    """Draws polarization ellipse of Jones vector.

    Parameters:
        E (Jones_vector or Stokes): Light object.
        N_angles (int): Number of angles to plot the ellipses. Default: 91.
        filename (str): name of filename to save the figure.
        figsize (tuple): A tuple of length 2 containing the figure size. Default: (8,8).
        limit (float): limit for drawing. If empty, it is obtained from amplitudes.
        draw_arrow (bool): If True, draws an arrow containing the turning sense of the polarization. Does not work with linear polarization vectors. Default: True.
        draw_labels (bool): It True, draws axis labels and tick labels. Default: True.
        draw_titles (bool): If True, draws axis titles. Default: True.
        depol_central (bool): If True, draws a central circle containing the unpolarized field amplitude. Default: False.
        depol_contour (bool): If True, draws a line enveloping the polarization ellipse in order to plot the depolarization. Default: False.
        depol_prob (bool): If True, plots the probability distribution of the electric field. Default: False.
        subplots (string, tuple or None): If AS_SHAPE, divides the figure in several subplots as the shape of the py_pol object. If INDIVIDUAL, each vector is represented in its own subaxis, trying to use a square grid. If tuple, divides the figure in that same number of subplots. If None, all ellipses are plot in the same axes. Default: None.
        N_prob (int): Number of points in each dimension for probability distributions. Default: 256.
        contour_levels (float, np.ndarray, tuple or list): Contains the contour levels (normalized to 1). Default: 0.9.
        cmap (str or color object): Default colormap for probability distributions. Default: hot.

    Returns:
        fig (handle): handle to figure.
        ax (list of handles): handles to axes.
    """
    dif_index_arrow = 4
    # Calculate the electric field amplitudes and the delays
    if E._type == 'Jones_vector':
        E0x, E0y = E.parameters.amplitudes(shape=False)
        E0u = np.zeros(1)
    else:
        E0x, E0y, E0u = E.parameters.amplitudes(shape=False, give_unpol=True)
    delay = E.parameters.delay(shape=False)
    phase = E.parameters.global_phase(shape=False)
    if phase is None:
        phase = np.zeros_like(E0x)
    if np.isnan(phase).any():
        phase[np.isnan(phase)] = 0
    # Create the angle variables
    angles = linspace(0, 360 * degrees, N_angles)
    Angles, E0X = np.meshgrid(angles, E0x)
    _, E0Y = np.meshgrid(angles, E0y)
    _, Delay = np.meshgrid(angles, delay)
    _, Phase = np.meshgrid(angles, phase)
    if E._type == 'Jones_vector':
        is_linear = E.checks.is_linear(shape=False, out_number=False)
    else:
        is_linear = E.checks.is_linear(shape=False,
                                       out_number=False,
                                       use_nan=False)
    # Create the electric field distributions
    # Ex = E0X * np.cos(Angles + Phase)
    # Ey = E0Y * np.cos(Angles + Phase + Delay)    
    Ex = E0X * np.cos(Phase - Angles)
    Ey = E0Y * np.cos(Phase + Delay - Angles)
    if draw_arrow:
        dx = np.diff(Ex)
        dy = np.diff(Ey)
        dif = dx**2 + dy**2
        index_arrow = np.argmax(dif, axis=-1)
        cond = index_arrow >= N_angles -  + dif_index_arrow
        if np.any(cond):
            index_arrow[cond] = index_arrow[cond] - N_angles
    # Calculate the depolarization central distribution
    if E._type == 'Stokes' and depol_central:
        _, E0U = np.meshgrid(angles, E0u)
        Exu = E0U * np.cos(Angles)
        Eyu = E0U * np.sin(Angles)
    # Safety arrays
    if E._type == 'Stokes':
        is_pol = E.checks.is_polarized(shape=False, out_number=False)
        is_depol = E.checks.is_depolarized(shape=False, out_number=False)
    else:
        if E.size < 2:
            is_pol = np.array([True])
        else:
            is_pol = np.ones_like(E0x).flatten()
    # Set automatic limits
    if limit in [0, '', [], None]:
        if depol_contour or depol_prob:
            limit = np.array([E0x.max() + E0u.max(),
                              E0y.max() + E0u.max()]).max() * 1.2
        else:
            limit = np.array([E0x.max(), E0y.max(), E0u.max()]).max() * 1.2

    # Prepare the figure and the subplots
    fig = plt.figure(figsize=figsize)
    if depol_prob:
        if type(subplots) is tuple and E.size == np.prod(np.array(subplots)):
            pass  # Only case subplots is not overwritten
        else:
            subplots = 'individual'
    if subplots is None:
        # Just one subplot
        Nx, Ny, Nsubplots, Ncurves = (1, 1, 1, E.size)
    elif type(subplots) is tuple:
        # Set number of subplots
        Nsubplots = np.prod(np.array(subplots[0:2]))
        if E.size % Nsubplots != 0:
            raise ValueError(
                'Shape {} is not valid for the object {} of {} elements'.
                format(subplots, E.name, E.size))
        Ncurves = E.size / Nsubplots
        Nx, Ny = subplots[0:2]
        indS, indE = (0, 0)
    elif subplots in ('AS_SHAPE', 'as_shape', 'As_shape'):
        # Subplots given by phase
        if E.ndim < 2:
            Nx, Ny = (1, E.size)
            Nsubplots, Ncurves = (E.size, 1)
        else:
            Nx, Ny = E.shape[0:2]
            Nsubplots, Ncurves = (Nx * Ny, E.size / (Nx * Ny))
        indS, indE = (0, 0)
    elif subplots in ('individual', 'Individual', 'INDIVIDUAL'):
        Ny = int(np.floor(np.sqrt(E.size)))
        Nx = int(np.ceil(E.size / Ny))
        Nsubplots, Ncurves = (E.size, 1)
    else:
        raise ValueError('{} is not a valid subplots option.')
    # If contour lines or probability must be plotted, calculate the probability distributions and linestyles
    if depol_contour or depol_prob:
        # Create the basic probability distribution
        x = np.linspace(-limit, limit, N_prob)
        X, E0U, Y = np.meshgrid(x, E0u, x)
        prob = np.exp(-(X**2 + Y**2) / (E0U**2))
        # Create the ellipse distribution
        indX = np.abs(np.subtract.outer(x, Ex)).argmin(0).flatten()
        indY = np.abs(np.subtract.outer(x, Ey)).argmin(0).flatten()
        indE = np.repeat(np.arange(E.size), N_angles)
        # indE = np.flip(indE)
        ellipse_3D = zeros_like(X, dtype=float)
        ellipse_3D[indE, indY, indX] = 1
        # Convolute them adn normalize to 1
        prob = fftconvolve(ellipse_3D, prob, mode='same', axes=(1, 2))
        _, MAX, _ = meshgrid(x, prob.max(axis=(1, 2)), x)
        prob = prob / MAX
        # Remove info for totally polarized vectors
        prob[~is_depol, :, :] = 0
        # Linestyles
        if depol_contour:
            if len(contour_levels) <= len(linestyles):
                line_styles = linestyles[:len(contour_levels)]
            else:
                line_styles = [linestyles[0]]

    # Main loop
    ax = []
    for ind in range(E.size):  # Loop in curves
        # Initial considerations for the subplot
        indS = int(np.floor(ind / Ncurves))
        indC = int(ind % Ncurves)
        if indC == 0:
            axis = fig.add_subplot(Nx, Ny, indS + 1)
            ax.append(axis)
            if draw_titles:
                if Nsubplots > 1:
                    if subplots in ('individual', 'Individual', 'INDIVIDUAL'):
                        string = str(indS)
                    else:
                        string = str(list(np.unravel_index(indS, (Nx, Ny))))
                    plt.title(string, fontsize=18)
                else:
                    plt.title(E.name, fontsize=26)
        # Other considerations
        if depol_prob:
            color = 'w'
        else:
            color = colors[name_colors[ind % 10]]
        if subplots in ('AS_SHAPE', 'as_shape',
                        'As_shape') and Nx * Ny > 1 and Ncurves > 1:
            string = str(list(np.unravel_index(ind, E.shape)[2:]))
        else:
            if Ncurves == 1:
                string = 'Polarized'
            else:
                string = str(list(np.unravel_index(ind, E.shape)))
        # Plot the probability distribution
        if depol_prob and is_depol[ind]:
            IDimage = axis.imshow(prob[ind, :, :],
                                  interpolation='bilinear',
                                  aspect='equal',
                                  origin='lower',
                                  extent=[-limit, limit, -limit, limit])
            # axis = axis[0]
        # Plot the curve
        if is_pol[ind]:
            axis.plot(Ex[ind, :], Ey[ind, :], label=string, color=color,  *args, **kwargs)
            if draw_arrow and ~is_linear[ind]:
                axis.arrow(Ex[ind, index_arrow[ind]],
                           Ey[ind, index_arrow[ind]],
                           Ex[ind, index_arrow[ind] + dif_index_arrow] - Ex[ind, index_arrow[ind]],
                           Ey[ind, index_arrow[ind] + dif_index_arrow] - Ey[ind, index_arrow[ind]],
                           width=0,
                           head_width=0.2 * limit,
                           linewidth=0,
                           color=color,
                           length_includes_head=True)
        elif E._type == 'Stokes' and depol_central and ~is_depol[ind]:
            axis.plot(np.zeros(2),
                      np.zeros(2),
                      label=string,
                      color=color, *args, **kwargs)
        elif depol_central or depol_prob:
            print('Field {} is empty.'.format(string))
        else:
            print('Field {} is empty or totally depolarized.'.format(string))
        # Add the depolarization for Stokes vectors
        if E._type == 'Stokes' and depol_central and is_depol[ind]:
            axis.plot(Exu[ind, :], Eyu[ind, :], color=color, ls='--')
        if E._type == 'Stokes' and depol_contour and is_pol[ind] and is_depol[
                ind]:
            CS = axis.contour(x,
                              x,
                              prob[ind, :, :],
                              contour_levels,
                              colors=(color),
                              linewidths=1.5,
                              linestyles=line_styles, *args, **kwargs)
            # linestyles=('dashdot'))
        # Additions to figure
        if indC == Ncurves - 1:
            plt.axis('equal')
            plt.axis('square')
            plt.grid(True)
            axis.set_xlim(-limit, limit)
            axis.set_ylim(-limit, limit)
            if draw_labels:
                axis.set_xlabel('$E_x$', fontsize=14)
                axis.set_ylabel('$E_y$', fontsize=14)
            else:
                axis.set_xticklabels([])
                axis.set_yticklabels([])

            plt.tight_layout()
            if Ncurves > 1:
                plt.legend()
            elif depol_contour and indC == Ncurves - 1:
                for ind, elem in enumerate(contour_levels):
                    CS.collections[ind].set_label('P = {}'.format(elem))
                plt.legend()
            if depol_prob and is_depol[indS]:
                divider = make_axes_locatable(axis)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(IDimage, cax=cax)
                IDimage.set_cmap(cmap)
    if Nsubplots > 1 and draw_titles:
        fig.suptitle(E.name, fontsize=26)
    # Save the image if required
    if filename not in (None, [], ''):
        plt.savefig(filename)
        print('Image {} saved succesfully!'.format(filename))
    return fig, ax


#
# def draw_ellipse_stokes(stokes_0,
#                         kind='',
#                         limit='',
#                         has_line=True,
#                         filename=''):
#     """ Draws polarization ellipse in stokes vector. If unpolarized light is present, a distribution of probability is given.
#
#     Parameters:
#         stokes_0 (Stokes): Stokes vector
#         kind (str): 'line' 'probabilities'. 'Line': polarized + unpolarized ellipses. 'probabilities' is for unpolarized. Provides probabilities'
#         limit (float): limit for drawing. If empty itis obtained from ampltiudes
#         has_line (bool or float): If True  draws polarized and 0.1 probability lines. If it is a number draws that probability.
#         filename (str): if filled, name for drawing
#
#     Returns:
#         ax (handle): handle to axis.
#         fig (handle): handle to figure.
#     """
#
#     parameters = stokes_0.parameters.get_all()
#
#     E0x, E0y, E0_unpol = parameters['amplitudes']
#     delay = parameters['delay']
#
#     angles = linspace(0, 360 * degrees, 256)
#     Ex = E0x * cos(angles)
#     Ey = E0y * cos(angles + delay)
#     E_unpolarized_x = E0_unpol * cos(angles)
#     E_unpolarized_y = E0_unpol * sin(angles)
#
#     if limit in [0, '', [], None]:
#         radius_max = sqrt(
#             ((Ex + E_unpolarized_x)**2 + (Ey + E_unpolarized_y)**2).max())
#         limit = radius_max * 1.25
#
#     x = linspace(-limit, limit, 256)
#     y = linspace(-limit, limit, 256)
#     X, Y = meshgrid(x, y)
#
#     if abs(E0_unpol) < eps or kind == 'line':
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.plot(Ex, Ey, 'k', lw=2, label='polarized')
#         ax.plot(E_unpolarized_x,
#                 E_unpolarized_y,
#                 'r--',
#                 lw=2,
#                 label='unpolarized')
#         plt.grid(True)
#     else:
#         sigma = E0_unpol
#
#         u_random = exp(-(X**2 + Y**2) / (sigma**2))
#
#         ellipse_2D = zeros_like(X, dtype=float)
#         i_positions, _, _ = nearest2(x, Ex)
#         j_positions, _, _ = nearest2(y, Ey)
#         ellipse_2D[j_positions, i_positions] = 1
#
#         prob = fftconvolve(ellipse_2D, u_random, mode='same')
#         prob = prob / prob.max()
#
#         fig, ax, IDimage = draw2D(prob, x, y)
#         if isinstance(has_line, (int, float)):
#             plt.contour(x,
#                         y,
#                         prob, (has_line, ),
#                         colors=('w'),
#                         linestyles=('dashed'))
#         if has_line is True:
#             plt.contour(x,
#                         y,
#                         prob, (0.1, ),
#                         colors=('w'),
#                         linestyles=('dashed'))
#         if has_line is not False:
#             plt.plot(Ex, Ey, 'k', lw=1)
#
#         plt.grid(False)
#
#     plt.axis('equal')
#     plt.axis('square')
#     ax.set_xlabel('$E_x$', fontsize=22)
#     ax.set_ylabel('$E_y$', fontsize=22)
#     ax.set_xlim(-limit, limit)
#     ax.set_ylim(-limit, limit)
#     plt.legend()
#     plt.tight_layout()
#     if filename not in (None, [], ''):
#         plt.savefig(filename)
#     return ax, fig
#
#
# def set_aspect_equal_3d(ax):
#     """Fix equal aspect bug for 3D plots."""
#     xlim = (-1, 1)
#     ylim = (-1, 1)
#     zlim = (-1, 1)
#
#     xmean = mean(xlim)
#     ymean = mean(ylim)
#     zmean = mean(zlim)
#
#     plot_radius = max([
#         abs(lim - mean_)
#         for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
#         for lim in lims
#     ])
#
#     factor = 1
#     ax.set_xlim3d([xmean - factor * plot_radius, xmean + factor * plot_radius])
#     ax.set_ylim3d([ymean - factor * plot_radius, ymean + factor * plot_radius])
#     ax.set_zlim3d([zmean - 1 * plot_radius, zmean + 1 * plot_radius])


def draw_poincare(S,
                  fig=None,
                  figsize=(6, 6),
                  draw_axes=True,
                  draw_guides=True,
                  kind="scatter",
                  depol=False,
                  param=None,
                  subplots=False,
                  in_degrees=False,
                  log=False,
                  color_limits=(None, None),
                  hover="components",
                  colormap="Blackbody",
                  show_fig=False):
    """Function to draw the Jones or Stokes vectors in the Poincare sphere using plotly.

    Args:
        S (Jones_vector or Stokes): Object to be represented.
        fig (plotly.graph_objects.Figure): Figure to plot the data. Default: None
        figsize (2-element iterable): Figure size. Default: (6,6)
        draw_axes (bool): If True, it draws the three axes of the Poincare space. Default: True
        draw_guides (bool): If True, it draws the circles of S1, S2 and S3 = 0. Default: True
        kind (str): Choose between 'scatter', 'line', 'scatterline' or 'surf' to represent the data. If surf, the object must be 2D, 3D or 4D. Else, it must be 0D, 1D, 2D or 3D. Default: 'scatter'.
        depol (bool): If True, the depolarization is taking into account for scatter and line plots shrinking the radius below 1. Default: False.
        param (str, np.ndarray or None): If str, parameter to use as information for the color. Must be a method of Parameters class which returns a single variable. If np.ndarray, must have the same shape as S. Default: None.
        subplots (bool): If True, it tries to use the first two dimensions as rows and columns. If method == 'surf', this is mandatory. Default: False.
        in_degrees (bool): If True, transforms the parameters to degrees. Default: False.
        log (bool): If True, it calculates it in logarithmic scale. Default: False.
        color_limits (float, float): limits in color
        hover (str): Choose between 'components' (S1, S2, S3) or 'angles' (azimuth, ellipticity). Default: 'components'.
        colormap (str): Colormap of the plots. Default: 'Blackbody'.
        show_fig (bool): If True, the figure is inmediately plotted. Default: False.

    Returns:
        fig (go.Figure): Plotly figure.
    """
    # Unrelated problems
    if hover == "angles":
        if kind == "surf":
            hover = "components"
            print(
                "WARNING: Plotly has an issue in the hover of surfaces. Hover mode changed to components."
            )
        else:
            print(
                "WARNING: Plotly has an issue in the hover of surfaces. Values are not correct."
            )
    # Allow several figures. Indices: [row, column, trace, dots inside trace]
    S = S.copy()
    Ndim = S.ndim
    if kind == "surf":
        # Surf requires pseudo-1D, 2D, and higher dimensions to be separated in subplots
        if Ndim > 4 or Ndim < 1:
            raise ValueError(
                "Object has a wrong number of dimensions ({})".format(Ndim))
        elif Ndim == 1:
            if param is None:
                raise ValueError(
                    "Object has a too few dimensions ({})".format(Ndim))
            else:
                # TODO: Interpolate
                Nrows = 1
                Ncolumns = 1
                S.shape = [1, 1, 1] + [S.size]
        elif Ndim == 2:
            Nrows = 1
            Ncolumns = 1
            S.shape = [1, 1, 1] + S.shape
        elif Ndim == 3:
            Nrows = 1
            Ncolumns = S.shape[0]
            S.shape = [S.shape[0]] + [1, 1] + S.shape[1:]
        elif Ndim == 4:
            Nrows = S.shape[0]
            Ncolumns = S.shape[1]
            S.shape = S.shape[0:2] + [1] + S.shape[2:]
        Ones = np.ones(S.shape[3:])
    elif subplots:
        # When subplots is True, elements of 2D and 3D are separated in differentsubplots. Higher dimensions are taken as part of the 3rd dimension
        if Ndim < 2:
            Nrows = 1
            Ncolumns = 1
            S.shape = [1, 1, 1] + [S.size]
        elif Ndim == 2:
            Nrows = 1
            Ncolumns = S.shape[0]
            S.shape = [1] + [S.shape[0]] + [1] + [S.shape[1]]
        elif Ndim == 3:
            Nrows = S.shape[0]
            Ncolumns = S.shape[1]
            S.shape = S.shape[0:2] + [1] + [S.shape[2]]
        else:
            Nrows = S.shape[0]
            Ncolumns = S.shape[1]
            S.shape = S.shape[0:3] + [np.prod(S.shape[3:])]
        Ones = np.ones(S.shape[3])
    else:
        # When subplots is false, all traces are plotted in the same sphere. Dimensions higher than 2D are included in the 2nd dimension.
        Nrows = 1
        Ncolumns = 1
        if Ndim <= 1:
            S.shape = [1, 1, 1, S.size]
        else:
            S.shape = [1, 1, S.shape[0]] + [np.prod(S.shape[1:])]
        Ones = np.ones(S.shape[3])
    # Reshape param
    if param is not None and not isinstance(param, str):
        param = np.reshape(param, S.shape)

    # Figure and figure options
    add_auxiliar = False
    if fig is None:
        specs = [[{'type': 'surface'}] * Ncolumns] * Nrows
        fig = make_subplots(rows=Nrows, cols=Ncolumns, specs=specs)
        add_auxiliar = True
    lighting = dict(ambient=0.9,
                    diffuse=0.,
                    roughness=0.5,
                    specular=0.05,
                    fresnel=0.2)
    if hover == 'components':
        hovertemplate = "S1: %{x:.3f}<br>S2: %{y:.3f}<br>S3: %{z:.3f}<br>Parameter: %{customdata:.3f}"
    else:
        hovertemplate = "Az: %{customdata[0]:.1f}<br>El: %{customdata[1]:.1f}"

    # Axes
    if draw_axes and add_auxiliar:

        line_p = dict(width=6, color="red")
        line_m = dict(width=6, color="red", dash="dash")
        marker = dict(size=1)
        axis_px = go.Scatter3d(x=[0, 1.2],
                               y=[0, 0],
                               z=[0, 0],
                               line=line_p,
                               marker=marker,
                               hoverinfo="skip",
                               name="S1")
        axis_mx = go.Scatter3d(x=[0, -1.2],
                               y=[0, 0],
                               z=[0, 0],
                               line=line_m,
                               marker=marker,
                               hoverinfo="skip",
                               name="-S1")
        line_p["color"], line_m["color"] = ("green", "green")
        axis_py = go.Scatter3d(x=[0, 0],
                               y=[0, 1.2],
                               z=[0, 0],
                               line=line_p,
                               marker=marker,
                               hoverinfo="skip",
                               name="S2")
        axis_my = go.Scatter3d(x=[0, 0],
                               y=[0, -1.2],
                               z=[0, 0],
                               line=line_m,
                               marker=marker,
                               hoverinfo="skip",
                               name="-S2")
        line_p["color"], line_m["color"] = ("blue", "blue")
        axis_pz = go.Scatter3d(x=[0, 0],
                               y=[0, 0],
                               z=[0, 1.2],
                               line=line_p,
                               marker=marker,
                               hoverinfo="skip",
                               name="S3")
        axis_mz = go.Scatter3d(x=[0, 0],
                               y=[0, 0],
                               z=[0, -1.2],
                               line=line_m,
                               marker=marker,
                               hoverinfo="skip",
                               name="-S3")
        size = 30
        annotations = [
            dict(showarrow=False,
                 x=1.2,
                 y=0,
                 z=0,
                 text="S1",
                 font=dict(color="red", size=size),
                 xshift=15,
                 yshift=15),
            dict(showarrow=False,
                 x=-1.2,
                 y=0,
                 z=0,
                 text="-S1",
                 font=dict(color="red", size=size),
                 xshift=-15,
                 yshift=-15),
            dict(showarrow=False,
                 x=0,
                 y=1.2,
                 z=0,
                 text="S2",
                 font=dict(color="green", size=size),
                 xshift=15,
                 yshift=15),
            dict(showarrow=False,
                 x=0,
                 y=-1.2,
                 z=0,
                 text="-S2",
                 font=dict(color="green", size=size),
                 xshift=-15,
                 yshift=-15),
            dict(showarrow=False,
                 x=0,
                 y=0,
                 z=1.2,
                 text="S3",
                 font=dict(color="blue", size=size),
                 xshift=15,
                 yshift=15),
            dict(showarrow=False,
                 x=0,
                 y=0,
                 z=-1.2,
                 text="-S3",
                 font=dict(color="blue", size=size),
                 xshift=-15,
                 yshift=-15),
        ]
    else:
        annotations = []

    # Curves
    if draw_guides and add_auxiliar:
        angle = np.linspace(0, 360 * degrees, 361)
        line = dict(width=2, color="darkslategrey", dash="dashdot")
        x = np.sin(angle)
        y = np.cos(angle)
        z = np.zeros_like(angle)
        circle_z = go.Scatter3d(x=x,
                                y=y,
                                z=z,
                                line=line,
                                mode="lines",
                                name="S3=0")
        x = np.sin(angle)
        z = np.cos(angle)
        y = np.zeros_like(angle)
        circle_y = go.Scatter3d(x=x,
                                y=y,
                                z=z,
                                line=line,
                                mode="lines",
                                name="S2=0")
        y = np.sin(angle)
        z = np.cos(angle)
        x = np.zeros_like(angle)
        circle_x = go.Scatter3d(x=x,
                                y=y,
                                z=z,
                                line=line,
                                mode="lines",
                                name="S1=0")

    # Poincare sphere
    if kind != "surf" and add_auxiliar:
        # az, el = np.mgrid[0:180*degrees:100j, -45*degrees:45*degrees:100j]
        el, az = np.mgrid[-45 * degrees:45 * degrees:100j,
                          0:180 * degrees:100j]
        x, y, z = azel_2_xyz(az, el)
        customdata = np.dstack(
            (az / degrees, el / degrees)) if hover == "angles" else []
        level = 0.2
        Psphere = go.Surface(x=x,
                             y=y,
                             z=z,
                             surfacecolor=np.ones_like(x) * level,
                             cmin=0,
                             cmax=1,
                             opacity=0.7,
                             colorscale="Greys",
                             showscale=False,
                             lighting=lighting,
                             customdata=customdata,
                             hovertemplate=hovertemplate,
                             name="Sphere",
                             showlegend=False)

    # Loop in rows
    for indR, Srow in enumerate(S):
        for indC, Scol in enumerate(Srow):

            # Axes
            if draw_axes and add_auxiliar:
                fig.add_traces(
                    [axis_px, axis_mx, axis_py, axis_my, axis_pz, axis_mz],
                    rows=indR + 1,
                    cols=indC + 1)

            # Curves
            if draw_guides and add_auxiliar:
                fig.add_traces([circle_z, circle_y, circle_x],
                               rows=indR + 1,
                               cols=indC + 1)

            # Loop in traces
            for indT, Strace in enumerate(Scol):

                # Parameter
                if isinstance(param, str):
                    colorbar = dict(title=param,orientation='h',ticklabelposition='outside bottom',y=-0.1)
                    Scolor = eval("Strace.parameters." + param +
                                  "(out_number=False)")
                elif param is not None:
                    colorbar = dict(title=S.name,orientation='h',ticklabelposition='outside bottom',y=-0.1,tickfont=dict(size=30))
                    Scolor = param[indR, indC, indT, ...]
                else:
                    colorbar = {}
                    Scolor = Ones * (indT + 1) / Scol.shape[0]
                if param is not None:
                    if log:
                        cond = Scolor <= 0
                        if np.any(cond):
                            Scolor[cond] = np.inf
                            Scolor[cond] = np.min(Scolor) / 10
                        Scolor = np.log10(Scolor)
                        colorbar["title"] = "log(" + colorbar["title"] + ")"
                        
                    elif in_degrees:
                        Scolor = Scolor / degrees
                        colorbar["title"] += " (deg)"

                # Plot data
                if kind != "surf":
                    # Poincare sphere
                    if add_auxiliar:
                        fig.add_trace(Psphere, row=indR + 1, col=indC + 1)

                    # Data
                    if kind in ("scatter", "scatterline"):
                        x, y, z, az, el, _ = obj_2_xyz(Strace,
                                                       in_degrees=True,
                                                       depol=depol)
                        customdata = np.squeeze(np.dstack(
                            (az, el))) if hover == "angles" else []
                        if kind == "scatter":
                            marker = dict(size=10,
                                          color=Scolor,
                                          colorbar=colorbar,
                                          colorscale=colormap)
                        else:
                            marker = dict(size=10,
                                          color=Scolor,
                                          colorscale=colormap)
                        Fdata = go.Scatter3d(x=x,
                                             y=y,
                                             z=z,
                                             marker=marker,
                                             name=S.name,
                                             mode="markers",
                                             customdata=customdata,
                                             hovertemplate=hovertemplate)
                        # TODO: Should be possible to link colorbars. Not right now due to a bug in Plotly.
                        # marker = dict(size=10, color=Scolor, colorscale=colormap)
                        # Fdata = go.Scatter3d(x=x, y=y, z=z, marker=marker, name=S.name,
                        #                 mode="markers", customdata=customdata, hovertemplate=hovertemplate, coloraxis="coloraxis")
                        fig.add_trace(Fdata, row=indR + 1, col=indC + 1)
                    if kind in ("line", "scatterline"):
                        x, y, z, az, el, Scolor = obj_2_xyz(Strace,
                                                            DAinterp=1 *
                                                            degrees,
                                                            in_degrees=True,
                                                            param=Scolor,
                                                            depol=depol)
                        customdata = np.squeeze(np.dstack(
                            (az, el))) if hover == "angles" else []
                        line = dict(width=8,
                                    color=Scolor,
                                    colorbar=colorbar,
                                    colorscale=colormap)
                        Fdata = go.Scatter3d(x=x,
                                             y=y,
                                             z=z,
                                             line=line,
                                             name=S.name + " int.",
                                             mode="lines",
                                             customdata=customdata,
                                             hovertemplate=hovertemplate)
                        fig.add_trace(Fdata, row=indR + 1, col=indC + 1)

                else:
                    x, y, z, az, el, Scolor = obj_2_xyz(Strace,
                                                        in_degrees=True,
                                                        param=Scolor,
                                                        interp_to_surf=True)
                    
                    # customdata = [az[0,:], el[:, 0]] if hover == "angles" else None
                    customdata = np.squeeze(np.stack(
                        (az, el), axis=-1)) if hover == "angles" else None
                    customdata = np.stack((Scolor), axis=-1)
                    if color_limits == None:
                        Psphere = go.Surface(x=x,
                                             y=y,
                                             z=z,
                                             surfacecolor=Scolor,
                                             opacity=1,
                                             colorscale=colormap,
                                             lighting=lighting,
                                             customdata=customdata,
                                             hovertemplate=hovertemplate,
                                             name=S.name,
                                             colorbar=colorbar)
                    else:
                        cmin, cmax = color_limits
                        if cmin is None:
                            cmin = Scolor.min()
                        if cmax is None:
                            cmax = Scolor.max()
                        Psphere = go.Surface(x=x,
                                             y=y,
                                             z=z,
                                             surfacecolor=Scolor,
                                             opacity=1,
                                             colorscale=colormap,
                                             showscale=True,
                                             lighting=lighting,
                                             customdata=customdata,
                                             hovertemplate=hovertemplate,
                                             name=S.name,
                                             cmin=cmin,
                                             cmax=cmax,
                                             colorbar=colorbar)
                    fig.add_trace(Psphere, row=indR + 1, col=indC + 1)

    # Plot figure
    axis = dict(showbackground=False,
                showgrid=False,
                zeroline=False,
                visible=False)
    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=0.75, y=0.75, z=0.75))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                      width=int(figsize[0] * 100),
                      height=int(figsize[1] * 100),
                      showlegend=False)
    fig.update_scenes(annotations=annotations,
                      xaxis=axis,
                      yaxis=axis,
                      zaxis=axis,
                      camera=camera)
    # TODO: Annotations do not render. Why???

    if show_fig:
        fig.show()

    return fig
