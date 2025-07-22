#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------
# Authors:    Luis Miguel Sanchez Brea and Jesus del Hoyo
# Date:       2019/01/09 (version 1.0)
# License:    GPL
# ------------------------------------
""" Common functions to classes """

import multiprocessing
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from numpy import array, cos, matrix, pi, sin, sqrt, tan
from scipy.optimize import leastsq

from .__init__ import (degrees, eps, limAlpha, limDelta, limAz, limEl, limRet,
               figsize_default, number_types)

tol_default = eps
NoneType = type(None)


def prepare_variables(vars,
                      expand=[False],
                      length=1,
                      obj=None,
                      give_shape=False):
    """Function that forces all variables of creation methods to be arrays, checks if all variables are a number or arrays of the same size, and expands the required variables if neccessary. If an object is introduced, it also checks that the length of the variables corresponds to the required lengtth for the object.

    Parameters:
        vars (list): List of variables.
        expand (bool or list): List of bools indicating which variables must be expanded.
        length (int): If some variables must be expanded but all of them have length 1, this will be the length of the new variables.
        obj (Py_pol object): Optical element object.

    Returns:
        vars (list): List of prepared variables.
        obj (Py_pol object): Optical element object (only if an obj was given to the function).
        shape (tuple): Shape of the variable with higher dimensionality.
    """
    # Calculate number of dimensions and shapes
    N = len(vars)
    if len(expand) == 1 and N > 1:
        expand = expand[0] * np.ones(N, dtype=bool)
    lengths = np.zeros(N)
    ndims = np.zeros(N)
    shapes = []
    for ind, elem in enumerate(vars):
        elem = np.array(elem)
        shapes.append(elem.shape)
        ndims[ind] = elem.ndim
        lengths[ind] = elem.size
        vars[ind] = elem.flatten()

    # Check if all the elements have size 1 or the same
    lengths_long = lengths[lengths > 1]
    if lengths_long.size > 0:
        if not all(lengths_long == lengths_long[0]):
            raise ValueError(
                'All parameters must be numbers or have the same number of elements'
            )
    # Check if the length is good for the object
    if not (obj is None):
        length2 = obj.size
        if lengths_long.size > 0 and length2 > 1:
            if not (length2 == lengths_long[0]):
                raise ValueError(
                    'The size of the variables is not the same as the number of elements in {}'
                    .format(obj.name))
        else:
            length = length2
    # Expand parameters if required
    if lengths_long.size > 0:
        length_vars = int(lengths_long.max())
        if length_vars > 1:
            length = length_vars
    for ind, elem in enumerate(vars):
        if expand[ind] and lengths[ind] == 1 and length > 1:
            vars[ind] = elem * np.ones(int(length))

    # if only one var was inserted, take it out of the list
    if N == 1:
        vars = vars[0]

    # Expand object.M if required
    if not (obj is None):
        if length2 == 1 and length > 1:
            # if obj._type == 'Jones_vector':
            #     new_M = np.array([
            #         obj.M[0, 0] * np.ones(length),
            #         obj.M[1, 0] * np.ones(length)
            #     ])
            #     obj.from_matrix(new_M)
            # elif obj._type is 'Jones_matrix':
            #     new_M = np.array([
            #         [
            #             obj.M[0, 0, 0] * np.ones(length),
            #             obj.M[0, 1, 0] * np.ones(length)
            #         ],
            #         [
            #             obj.M[1, 0, 0] * np.ones(length),
            #             obj.M[1, 1, 0] * np.ones(length)
            #         ],
            #     ])
            #     obj.from_matrix(new_M)
            # else:
            #     raise ValueError('Not implemented yet')
            new_M = np.multiply.outer(np.squeeze(obj.M), np.ones(length))
            if obj._type in ('Stikes', 'Mueller'):
                obj.from_matrix(new_M, global_phase=obj.global_phase)
            else:
                obj.from_matrix(new_M)
    # Check the shape with highest dimensionality
    if give_shape:
        ind = np.argmax(ndims)
        shapes, _ = select_shape(obj, shape_var=shapes[ind])
    # Output
    if obj is None:
        if give_shape:
            return vars, shapes
        else:
            return vars
    else:
        if give_shape:
            return vars, obj, shapes
        else:
            return vars, obj


def expand_objects(lista, length=1, copy=False):
    """Expand a list of objects.

    Parameters:
        lista (list): List of py_pol objects.
        length (int): If some variables must be expanded but all of them have length 1, this will be the length of the new variables. Default: 1.
        copy (bool): If True, the object are copied. Default: False.

    Returns:
        new_list (list): List of expanded objects.
    """
    # Take the sizes and shapes
    N = 1
    shape = None
    for obj in lista:
        # Update size
        if obj.size > 1:
            if N == 1:
                N = obj.size
            elif N > 1 and N != obj.size:
                raise ValueError('Objects are of different size')
        # Update shape
        if shape is None:
            shape = obj.shape
        else:
            if obj.shape is not None and len(obj.shape) > len(shape):
                shape = obj.shape
    if N == 1 and length > 1:
        N = length
        shape = [N]

    # Expand
    if N > 1:
        new_list = []
        for obj in lista:
            if copy:
                new_obj = obj.copy()
            else:
                new_obj = obj
            if new_obj.size == 1:
                if new_obj._type == 'Jones_vector':
                    new_M = np.array([
                        new_obj.M[0, 0] * np.ones(N),
                        new_obj.M[1, 0] * np.ones(N)
                    ])
                    new_obj.from_matrix(new_M)
                elif new_obj._type == 'Jones_matrix':
                    new_M = np.array([
                        [
                            new_obj.M[0, 0, 0] * np.ones(N),
                            new_obj.M[0, 1, 0] * np.ones(N)
                        ],
                        [
                            new_obj.M[1, 0, 0] * np.ones(N),
                            new_obj.M[1, 1, 0] * np.ones(N)
                        ],
                    ])
                    new_obj.from_matrix(new_M)
            new_list += [new_obj]
    else:
        new_list = lista
    return new_list


def reshape(vars, shape_like=None, shape_fun=None, shape=None, obj=None):
    """Reshapes an array of parameters to have the desired shape.

    Parameters:
        vars (list): List of variables.
        shape_like (numpy.ndarray or py_pol object): Use the shape of this object to reshape the variables.
        shape_fun (tuple, list or string): User-defined shape. If it is a string (like 'array') no manipulation is performed.
        shape (tuple or list): Default shape.
        obj (py_pol object): Py_pol object to take the shape from. Its preference is less than shape_like.

    Returns:
        vars (list): Reshaped list of variables.
    """
    # Check how many variables we have
    N = len(vars)
    # Select the correct shape
    shape, _ = select_shape(
        shape_var=shape, shape_fun=shape_fun, shape_like=shape_like, obj=obj)
    # Act only if there is a shape
    if shape is not None:
        for ind, elem in enumerate(vars):
            # Act only if the element is not a number
            if not isinstance(elem, number_types):
                vars[ind] = elem.reshape(shape)
    # If only one variable was given, take it out of the list
    if N == 1:
        return vars[0]
    else:
        return vars


def select_shape(obj=None, shape_var=None, shape_fun=None, shape_like=None):
    """Selects which shape to save.

    Parameters:
        obj (py_pol object): Object to be applied the shape. It is given here to check if the new shape is correct.
        shape_var (tuple or list): Shape of variables.
        shape_fun (tuple or list): Shape introduced manually by the user.
        shape_like (numpy.ndarray or py_pol object): Object to take the shape from.

    Returns:
        shape (tuple or list): Selected shape.
        ndim (int): Number of dimensions.
    """
    # If the size of the object is 0 or 1, just use None
    if obj is not None and obj.size <= 1:
        shape = None
        ndim = 0
    # If shape_fun is False (different than None), flatten the object
    if obj is not None and shape_fun == False:
        ndim = 1
        shape = [obj.size]
    # Now, take the shape with highest priority that is valid
    else:
        # Extract the shape from shape_like object
        if shape_like is not None:
            shape_like = shape_like.shape
        # If obj is None, we don't need to check if the sahpe is valid, just take the one with higherst priority
        if obj is None:
            shape = None
            shapes = [shape_like, shape_fun, shape_var]
            for s in shapes:
                if s is not None:
                    # If shape is False, force a None shape
                    if s is False:
                        shape = None
                        break
                    elif len(s) > 0:
                        shape = s
                        break

        # When we have an object, we have to check if the shape is correct
        else:
            shape = [obj.size]
            shapes = [shape_like, shape_fun, shape_var, obj.shape]
            # print(obj.shape)
            for s in shapes:
                if s is not None:
                    # If shape is False, force a None shape
                    if s is False:
                        shape = None
                        break
                    # Check that the shape is correct
                    elif isinstance(s, int):
                        shape = [s]
                    elif len(s) > 0 and np.prod(np.array(s)) == obj.size:
                        shape = s
                        break
        # Make sure that the shape is a list
        if shape is not None:
            shape = np.array(shape, dtype=int)
        # Calculate the dimension of the object
        if isinstance(shape, (tuple, list, np.ndarray)):
            ndim = len(shape)
        elif isinstance(shape, int):
            ndim = 1
        else:
            ndim = 0
    return shape, ndim


def multitake(arr, indices, axes):
    """Function that implements the numpy.take function in multiple axes.

    Parameters:
        arr (numpy.ndarray): Original array.
        indices (tuple or list): List with the indices values. In this function, only one value per axis is allowed.
        axes (tuple or list): List of axes.

    Returns:
        arr (numpy.ndarray): Taken array.
    """
    # Make all numpy arrays
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    indices = np.array(indices)
    axes = np.array(axes)
    # Order for increasing Axis
    indices = indices[np.argsort(axes)]
    axes = np.sort(axes)
    # Loop along axes
    for ind, ax in enumerate(axes):
        # Take in this axis
        arr = np.take(arr, indices[ind], ax - ind)
    return arr


def merge_indices(ind1, ind2, axis):
    """Merges two sets of indices with poritions given by axis."""
    # Make all arrays
    ind1, ind2, axis = (np.array(ind1), np.array(ind2), np.array(axis))
    # print(ind1, ind2, axis)
    # Sort the indices
    if axis.size == 1:
        ind1 = np.insert(ind1, axis, ind2)
    else:
        order = np.argsort(axis)
        ind2 = ind2[order]
        axis = axis[order]
        for ind, elem in enumerate(ind2):
            # print('entrada', ind1, axis, ind, elem)
            ind1 = np.insert(ind1, axis[ind], elem)
            # axis = axis + 1
    return ind1


def combine_indices(list_ind):
    """Combines the information given by np.unravel_index."""
    N_elem = len(list_ind)
    N_ind = list_ind[0].size
    final = []
    for i1 in range(N_ind):
        aux = []
        for elem in list_ind:
            aux.append(elem[i1])
        final.append(aux)
    return final


def PrintParam(
        param,
        verbose=True,
        draw=True,
        statistics=True,
        shape=None,
        kind='jones',
        # shape_like=None,
        # shape_obj=None,
        title='',
        heading=''):
    """Function to print the information during the calculation of some parameters.

    Parameters:
        param (np.array, list or tuple): List of variables to be represented.
        verbose (bool): If True, print the numeric values. Default: True.
        draw (bool): If True, plot 1D and 2D parameters. Default: True.
        statistics (bool): If True, a basic statistical analysis will be performed. Default: True.
        shape (list or tuple): Shape of the elements.
        kind (string): Type of jones matrix representation. If 'jones' represents real and imaginary part. If 'amplitude_phase' represents the amplitude and phase of the Jones matrix. Default: 'jones'.
        title (string or list): Title or list of titles (1-D and 2-D elements).
    """
    # Print heading
    print(heading)
    # Calculate the length of the number of parameters
    try:
        if isinstance(param, list) or isinstance(param, tuple):
            l = len(param)
            if l == 1:
                param = param[0]
                if isinstance(title, list) or isinstance(title, tuple):
                    title = title[0]
        else:
            l = 1
    except:
        l = 1
    # Calculate the dimension
    # if shape is not None:
    #     ndim = len(shape)
    # else:
    #     try:
    #         if l == 1:
    #             if param.size == 1:
    #                 ndim = 0
    #             else:
    #                 ndim = param.ndim
    #         else:
    #             if param[0].size == 1:
    #                 ndim = 0
    #             else:
    #                 ndim = param[0].ndim
    #     except:
    #         ndim = 0
    # print(ndim)
    try:
        if l == 1:
            if param.size == 1:
                ndim = 0
            else:
                ndim = param.ndim
        else:
            if param[0].size == 1:
                ndim = 0
            else:
                ndim = param[0].ndim
    except:
        if shape is not None:
            ndim = len(shape)
        else:
            ndim = 0
    # Check if there are complex or bool parameters
    if l == 1:
        is_complex = np.iscomplex(param).any()
        if isinstance(param, np.ndarray):
            is_bool = param.dtype is np.dtype(bool)
        else:
            is_bool = False

    else:
        is_bool = []
        for p in param:
            is_complex = np.iscomplexobj(p)
            if isinstance(p, np.ndarray):
                is_bool.append(p.dtype is np.dtype(bool))
            else:
                is_bool.append(False)

    # Calculate desired size for figs
    if ndim == 1 or ndim == 2:
        figsize = [5, 5]
        if is_complex:
            figsize[1] = l * figsize_default[1]
            figsize[0] = 2 * figsize_default[0]
        else:
            f, c = NumberOfSubplots(l)
            figsize[1] = f * figsize_default[1]
            figsize[0] = c * figsize_default[0]

    # If verbose, print
    if verbose:
        # 0-D: just one element. Use letters only.
        if l == 1:
            print(param)
        else:
            for ind, p in enumerate(param):
                print('  ' + title[ind])
                print(p)


    # Draws
    if draw and ndim == 0:
        print('Low dimensionality, figure not available.')
    elif draw and ndim == 1:
        # 1-D: row element. Use plot.
        fig = plt.figure(figsize=figsize)
        if l == 1:
            if is_complex:
                plt.subplot(1, 2, 1)
                plt.plot(np.real(param))
                plt.title(title + ' (real)')
                # plt.colorbar()
                plt.subplot(1, 2, 2)
                plt.plot(np.imag(param))
                plt.title(title + ' (imaginary)')
                # plt.colorbar()
            else:
                plt.plot(param)
                plt.title(title)
        else:
            # Check if there are complex parameters
            for ind, p in enumerate(param):
                if is_complex:
                    plt.subplot(l, 2, 2 * ind + 1)
                    plt.plot(np.real(p))
                    plt.title(title[ind] + ' (real)')
                    plt.subplot(l, 2, 2 * ind + 2)
                    plt.plot(np.imag(p))
                    plt.title(title[ind] + ' (imaginary)')
                else:
                    plt.subplot(f, c, ind + 1)
                    if is_bool[ind]:
                        plt.plot(p.astype(int))
                    else:
                        plt.plot(p)
                    plt.title(title[ind])
        fig.canvas.draw_idle()
    elif draw and ndim == 2:
        # 2-D: Use imshow
        fig = plt.figure(figsize=figsize)
        if l == 1:
            if is_complex:
                plt.subplot(1, 2, 1)
                plt.imshow(np.real(param),origin='lower')
                plt.title(title + ' (real)')
                plt.colorbar()
                plt.subplot(1, 2, 2)
                plt.imshow(np.imag(param),origin='lower')
                plt.title(title + ' (imaginary)')
                plt.colorbar()
            else:
                if is_bool:
                    param = param.astype(float)
                plt.imshow(param,origin='lower',cmap='hot')
                plt.title(title)
                plt.colorbar()
        else:
            if kind in ('intensity'):
                plt.imshow((np.abs(param[0])+np.abs(param[1]))**2,origin='lower',cmap='hot')
                plt.title('Intensity')
                plt.clim(0,1)
                plt.colorbar()

            else:
                for ind, p in enumerate(param):
                    if is_complex:
                        if kind in ('amplitude_phase'):
                            plt.subplot(l, 2, 2 * ind + 1)
                            plt.imshow(np.abs(p),origin='lower',cmap='gist_heat')
                            plt.title(title[ind] + ' (amplitude)')
                            plt.colorbar()
                            #plt.clim(0,1)
                            plt.subplot(l, 2, 2 * ind + 2)
                            plt.imshow((np.angle(p)/degrees),origin='lower',cmap='twilight')
                            plt.title(title[ind] + ' (phase)')
                            plt.clim(-180,180)
                            plt.colorbar()
                            
                        else:
                            plt.subplot(l, 2, 2 * ind + 1)
                            plt.imshow(np.real(p),origin='lower',cmap='seismic')
                            #plt.axis('off')
                            #plt.xticks([])
                            #plt.yticks([])
                            plt.title(title[ind] + ' (real)',fontsize=15)
                            plt.colorbar()
                            plt.clim(-1,1)
                            plt.subplot(l, 2, 2 * ind + 2)
                            plt.imshow(np.imag(p),origin='lower',cmap='seismic')
                            plt.title(title[ind] + ' (imaginary)',fontsize=15)
                            #plt.xticks([])
                            #plt.yticks([])
                            #plt.axis('off')
                            plt.clim(-1,1)
                            plt.colorbar()
                    else:
                        plt.subplot(f, c, ind + 1)
                        if is_bool[ind]:
                            p = p.astype(float)
                        plt.imshow(p,origin='lower',cmap='hot')
                        plt.title(title[ind])
                        plt.colorbar()
        fig.canvas.draw_idle()
    elif draw:
        print('High dimensionality, figure not available.')

    # Statistics
    if statistics and ndim > 0:
        if l == 1:
            mean = np.mean(param)
            std = np.std(param)
            print('The mean value is {} +- {}'.format(mean, std))
        else:
            for ind, p in enumerate(param):
                mean = np.mean(p)
                std = np.std(p)
                print('The mean value of param {} is {} +- {}'.format(
                    title[ind], mean, std))
    # End
    plt.show()
    print('')
    return None


def take_shape(objs):
    """Calculates the shape with higher dimensionality."""
    if isinstance(objs, (list, tuple)):
        l = 0
        shape = [0]
        for obj in objs:
            # Differenciate between lists of objects and shapes
            if isinstance(obj, (list, tuple, np.ndarray, NoneType)):
                s = obj
            else:
                s = obj.shape
            if s is not None:
                if len(s) > l:
                    l = len(s)
                    shape = s
                elif len(s) == l and np.prod(s) > np.prod(shape):
                    shape = s
    else:
        shape = objs.shape
    return shape


def kron_axis(a, b, axis=None):
    """Function that implements the kronecker product along a given axis.

    Parameters:
        a, b (numpy.ndarray): Arrays to perform the operation.
        axis (int): Axis to perform the operation. a and b sizes in the rest of dimensions must be the same. If None, default np.kron is used.

    Result:
        (numpy.ndarray): Result.
    """
    if axis is None:
        return np.kron(a, b)
    elif isinstance(axis, int):
        # Check dimensions
        shape1 = list(a.shape)
        del shape1[axis]
        shape2 = list(b.shape)
        del shape2[axis]
        if shape1 != shape2:
            raise ValueError(
                'a and b are of incompatible shapes {} and {}'.format(
                    a.shape, b.shape))
        # Expand dimensions and multiply
        a = np.expand_dims(a, axis=axis + 1)
        b = np.expand_dims(b, axis=axis)
        res = a * b
        return res
    else:
        raise ValueError('Axis is of type {} instead of int'.format(
            type(axis)))


def prepare_variables_blocks(M00=None,
                             Dv=None,
                             Pv=None,
                             m=None,
                             extend=[False, False, False, False],
                             multiply_by_M00=False,
                             length=1,
                             obj=None,
                             other_shape=None,
                             give_shape=False):
    """Function that forces Mueller matrix block variables to be arrays of the desired size, checks they are compatible between them, and expands the required variables. If an object is introduced, it also checks that the length of the variables corresponds to the required lengtth for the object.

    Parameters:
        M00 (numpy.ndarray or float): Mean transmission coefficient.
        Dv (numpy.ndarray): Diattenuation vector. At least one of its dimensions must be of size 3.
        Pv (numpy.ndarray): Polarizance vector. At least one of its dimensions must be of size 3.
        m (numpy.ndarray): Small matrix m. At least two of its dimensions must be of size 3.
        extend (tuple or list with 4 bools): For each value, stretch the corresponding variable to match max size. Default: [False] * 4.
        multiply_by_M00 (bool): If True, multiplies Dv, Pv and m by M00.
        length (int): If some variables must be expanded but all of them have length 1, this will be the length of the new variables.
        obj (Py_pol object): Optical element object.
        give_shape (bool): If True, the output includes the shape variable.
        other_shape (int, tuple or list): Other suggested shape given by prepare_variables function.

    Returns:
        M00 (numpy.ndarray or float): 1xN array.
        Dv (numpy.ndarray): 3xN array.
        Pv (numpy.ndarray): 3xN array.
        m (numpy.ndarray): 3x3xN array.
        obj (Py_pol object): Optical element object (only if an obj was given to the function).
        shape (tuple): Shape of the variable with higher dimensionality.
    """
    # Check which variables are None
    is_None = [Dv is None, Pv is None, m is None, M00 is None]
    # Check that the variable sizes are correct
    M00, Dv, Pv, m = (np.array(M00, dtype=float), np.array(Dv, dtype=float),
                      np.array(Pv, dtype=float), np.array(m, dtype=float))
    sizes = np.array([Dv.size / 3, Pv.size / 3, m.size / 9, M00.size])
    sizes[is_None] = 1
    max_size = np.max(sizes)
    cond = (sizes == max_size) + (sizes == 1)
    if any(~cond):
        raise ValueError(
            'M00 ({}), Pv ({}), Dv ({}) and m ({}) have diferent number of elements'
            .format(sizes[3], sizes[0], sizes[2], sizes[1]))
    # Reshape M00
    if not is_None[3]:
        shape1 = M00.shape
        M00 = M00.flatten()
        # Multiply variables if required
        if multiply_by_M00:
            cte = M00
        else:
            cte = 1
    else:
        shape1 = None
        cte = 1
    # Reshape Dv
    if not is_None[0]:
        if Dv.ndim == 1:
            if Dv.size % 3 == 0:
                Dv = np.reshape(Dv, (3, int(Dv.size / 3)))
                shape2 = [Dv.size / 3]
            else:
                raise ValueError(
                    'Dv must have a number of elements multiple of 3.')
        else:
            shape2 = np.array(Dv.shape)
            if (shape2 == 3).any:
                # Store the shape of the desired outputs
                ind = np.argmin(~(shape2 == 3))
                # Store info
                Dv = np.array([
                    cte * np.take(Dv, 0, axis=ind).flatten(),
                    cte * np.take(Dv, 1, axis=ind).flatten(),
                    cte * np.take(Dv, 2, axis=ind).flatten(),
                ])
                shape2 = np.delete(shape2, ind)
            else:
                raise ValueError(
                    'Dv must have one axis with exactly 3 elements')
    else:
        shape2 = None
    # Reshape Pv
    if not is_None[1]:
        if Pv.ndim == 1:
            if Pv.size % 3 == 0:
                Pv = np.reshape(Pv, (3, int(Pv.size / 3)))
                shape3 = None
            else:
                raise ValueError(
                    'Pv must have a number of elements multiple of 3.')
        else:
            shape3 = np.array(Pv.shape)
            if (shape3 == 3).any:
                # Store the shape of the desired outputs
                ind = np.argmin(~(shape3 == 3))
                # Store info
                Pv = np.array([
                    cte * np.take(Pv, 0, axis=ind).flatten(),
                    cte * np.take(Pv, 1, axis=ind).flatten(),
                    cte * np.take(Pv, 2, axis=ind).flatten(),
                ])
                shape3 = np.delete(shape3, ind)
            else:
                raise ValueError(
                    'Pv must have one axis with exactly 3 elements')
    else:
        shape3 = None
    # Reshape m
    if not is_None[2]:
        if m.ndim == 1 or m.ndim == 2:
            if m.size % 9 == 0:
                m = np.reshape(m, (3, 3, int(m.size / 9)))
            else:
                raise ValueError(
                    'm must have a number of elements multiple of 9.')
            shape4 = None
        else:
            shape4 = np.array(m.shape)
            N = np.sum(shape4 == 3)
            if N > 1:
                # Find the matrix indices and the final shape
                ind1 = np.argmin(~(shape4 == 3))
                shape4 = np.delete(shape4, ind1)
                ind2 = np.argmin(~(shape4 == 3))
                shape4 = np.delete(shape4, ind2)
                ind2 = ind2 + 1
                # Calculate the components and construct the matrix from them
                m = np.array([[
                    cte * multitake(m, [0, 0], [ind1, ind2]).flatten(),
                    cte * multitake(m, [0, 1], [ind1, ind2]).flatten(),
                    cte * multitake(m, [0, 2], [ind1, ind2]).flatten()
                ], [
                    cte * multitake(m, [1, 0], [ind1, ind2]).flatten(),
                    cte * multitake(m, [1, 1], [ind1, ind2]).flatten(),
                    cte * multitake(m, [1, 2], [ind1, ind2]).flatten()
                ], [
                    cte * multitake(m, [2, 0], [ind1, ind2]).flatten(),
                    cte * multitake(m, [2, 1], [ind1, ind2]).flatten(),
                    cte * multitake(m, [2, 2], [ind1, ind2]).flatten()
                ]])

            else:
                raise ValueError(
                    'm must have 3 elements in at least 2 dimensions.')
    else:
        shape4 = None

    # Stretch object if required
    if obj is not None and obj.size == 1:
        obj.M = np.multiply.outer(np.squeeze(obj.M), np.ones(max_size))
    # Return
    ret = [M00, Dv, Pv, m]
    if obj is not None:
        ret.append(obj)
    if give_shape:
        ret.append(take_shape([shape1, shape2, shape3, shape4]))
    return ret


def NumberOfSubplots(n):
    """Auxiliar function to calculate the number of desired subplots."""
    if n == 1:
        f, c = (1, 1)
    elif n == 2:
        f, c = (1, 2)
    elif n == 3:
        f, c = (1, 3)
    elif n == 4:
        f, c = (2, 2)
    elif n == 5 or n == 6:
        f, c = (2, 3)
    elif n == 8:
        f, c = (2, 4)
    elif n > 6 and n < 10:
        f, c = (3, 3)
    elif n > 9 and n < 17:
        f, c = (4, 4)
    else:
        f, c = (int(np.ceil(n / 4)), 4)
    return f, c


def PrintMatrices(list, Nspaces=3):
    """Creates the print string of a list of matrices to be printed in a line.

    Parameters:
        list (list): list of matrices.
        Nspaces (int): Number of spaces between matrices.

    Returns:
        str (string): String to be printed.
    """
    # Safety
    if list is None or list == []:
        return "This object is empty"

    # Get size and shape
    N = len(list)
    shape = list[0].shape
    str = ''
    # Iterate in rows
    for row in range(shape[0]):
        for M in list:
            # Check if the printed numbers are complex
            cond_complex = M.dtype in (np.complex64, np.complex128) and np.sum(M.imag**2)/M.size > tol_default
            # Start of matrix
            str = str + '['
            # Print full row of matrix
            for col in range(shape[1]):
                if cond_complex:
                    str = str + '{:+1.3f} + {:+1.3f}j '.format(M[row, col].real, M[row, col].imag)
                else:
                    str = str + '{:+1.3f} '.format(M[row, col].real)
            # End of matrix
            str = str + '\b]'
            str = str + Nspaces * ' '
        # End of line
        str = str + '\n'
    return str


def rotation_matrix_Jones(angle, length=1):
    """Creates an array of Jones 2x2 rotation matrices.

    Parameters:
        angle (np.array): array of angle of rotation, in radians.
        length (int): If some variables must be expanded but all of them have length 1, this will be the length of the new variables.

    Returns:
        numpy.array: 2x2xN matrix
    """
    angle = prepare_variables([angle], length=length)
    M = array([[cos(angle), sin(angle)], [-sin(angle), cos(angle)]])
    return M


def rotation_matrix_Mueller(angle=0):
    """Mueller 4x4 matrix for rotation

    References:
        Gil, Ossikovski (4.30) - p. 131
        Handbook of Optics vol 2. 22.16 (eq.8) is with changed sign in sin

    Parameters:
        angle (float): angle of rotation with respect to 0 deg.

    Returns:
        (numpy.array): 4x4xN rotation matrix.
    """
    # Definicion de la matrix
    c2b, s2b = (cos(2 * angle), sin(2 * angle))
    zero, one = (np.zeros_like(c2b), np.ones_like(c2b))
    M = np.array(
        [[one, zero, zero, zero], [zero, c2b, s2b, zero],
         [zero, -s2b, c2b, zero], [zero, zero, zero, one]],
        dtype=float)
    return M


def azimuth_elipt_2_charac_angles(azimuth, ellipticity, out_number=True):
    """Function that converts azimuth and elipticity to characteristic angles in Jones space.

    .. math:: cos(2 \\alpha) = cos(2 \phi) * cos(2 \chi)

    .. math:: tan(\delta) = \\frac{tan(2 \chi)}{sin(2 \phi)}

    TODO: Improve the 2D way of calculating

    References:
        J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 137 and 1543.


    Parameters:
        azimuth (float or numpy.ndarray) [0,np.pi]: Azimuth (angle of rotation).
        ellipticity (float or numpy.ndarray) [-pi/4,np.pi/4]: Elipticity angle.
        out_number (bool): if True and the result is a 1x1 array, return a number instead. Default: True.


    Returns:
        alpha (float) [0,np.pi]: tan(alpha) is the ratio between the maximum amplitudes of the polarization elipse in X-Y coordinates.
        delta (float) [0, 2*pi]: phase difference between both components of the eigenstates in Jones formalism.
    """
    # Prepare variables
    (azimuth, ellipticity), shape = prepare_variables(
        vars=[azimuth, ellipticity], expand=[True, True], give_shape=True)
    # Check that the angles belong to the correct interval. If not, fix it.
    azimuth = put_in_limits(azimuth, "azimuth", out_number=False)
    ellipticity = put_in_limits(ellipticity, "ellipticity", out_number=False)
    # Initialize variables
    alpha = np.zeros_like(azimuth)  # * np.nan
    delta = np.zeros_like(azimuth)  # * np.nan
    # Start by checking the particular cases of the exteme values
    # Check linear polarization case
    cond1 = np.abs(ellipticity) < tol_default**2
    alpha[cond1] = azimuth[cond1]
    cond2 = cond1 * (azimuth > np.pi / 2)
    delta[cond2] = np.pi
    # Check circular polarization case
    cond2 = (~cond1) * ((np.abs(ellipticity + np.pi / 4) < tol_default**2) +
                        (np.abs(ellipticity - np.pi / 4) < tol_default**2))
    alpha[cond2] = np.pi / 4
    delta[cond2] = np.pi / 2 + ((1 - np.sign(ellipticity[cond2])) * np.pi / 2)
    # Calculate values using trigonometric functions
    cond3 = (~cond1) * (~cond2)
    alpha[cond3] = 0.5 * np.arccos(
        np.cos(2 * azimuth[cond3]) * np.cos(2 * ellipticity[cond3]))
    # Avoid dividing by 0
    cond4 = cond3 * ((azimuth == 0) + (azimuth == np.pi))
    delta[cond4] = np.sign(np.tan(2 * ellipticity[cond4])) * np.pi / 2
    cond4 = (~cond4) * cond3
    delta[cond4] = np.arctan(
        np.tan(2 * ellipticity[cond4]) / np.sin(2 * azimuth[cond4]))
    # Use the other possible value from the arcs functions if necessary
    Qel = which_quad(ellipticity)
    Qaz = which_quad(azimuth)
    # Adjust
    cond1 = Qaz >= 3
    delta[cond1] = delta[cond1] + np.pi
    cond1 = Qel == 1.5
    delta[cond1] = np.pi / 2
    cond1 = Qel == -1.5
    delta[cond1] = 3 * np.pi / 2
    cond1 = (Qaz >= 3) * (Qel == 0)
    delta[cond1] = np.pi

    # Make sure the output values are in the allowed limits
    alpha = put_in_limits(alpha, "alpha", out_number=out_number)
    delta = put_in_limits(delta, "delta", out_number=out_number)
    # Recover the shape
    alpha = np.reshape(alpha, shape)
    delta = np.reshape(delta, shape)
    # End
    return alpha, delta


def charac_angles_2_azimuth_elipt(alpha, delta):
    """Function that converts azimuth and elipticity to characteristic angles in Jones space.

    .. math:: cos(2 \\alpha) = cos(2 \phi) * cos(2 \chi)

    .. math:: tan(\delta) = \\frac{tan(2 \chi)}{sin(2 \phi)}


    References:
        J. J. Gil, "Polarized light and the Mueller Matrix approach", pp 137 and 154.

    Parameters:
        alpha (float) [0,np.pi]: tan(alpha) is the ratio between the maximum
        amplitudes of the polarization elipse in X-Y coordinates.
        delta (float) [0, 2*pi]: phase difference between both components of
            the eigenstates in Jones formalism.

    Returns:
        azimuth (float) [0,np.pi]: Azimuth (angle of rotation).
        ellipticity (float) [-pi/4,np.pi/4]: Elipticity angle.
    """
    # Prepare variables
    (alpha, delta), shape = prepare_variables(
        vars=[alpha, delta], expand=[True, True], give_shape=True)
    # Check that the angles belong to the correct interval. If not, fix it.
    alpha = put_in_limits(alpha, "alpha", out_number=False)
    delta = put_in_limits(delta, "delta", out_number=False)
    azimuth = np.zeros_like(alpha)
    ellipticity = np.zeros_like(alpha)
    # Check circular polarization case
    cond1 = (np.abs(alpha - np.pi / 4) < tol_default**2) * (
        (np.abs(delta - np.pi / 2) < tol_default**2) +
        (np.abs(delta - 3 * np.pi / 2) < tol_default**2))
    ellipticity[cond1] = np.sign(delta[cond1] - np.pi) * np.pi / 4
    # Check the linear polarization at 0 and 90º case
    cond2 = (alpha < tol_default) + (np.abs(alpha - np.pi / 2) < tol_default)
    azimuth[cond2] = alpha[cond2]
    # Calculate values using trigonometric functions
    cond1 = np.logical_not(cond1 + cond2)
    azimuth[cond1] = 0.5 * np.arctan(tan(2 * alpha[cond1]) * cos(delta[cond1]))
    ellipticity[cond1] = 0.5 * np.arcsin(
        sin(2 * alpha[cond1]) * sin(delta[cond1]))
    # Use the other possible value from the arcs functions if necessary
    Qalpha = which_quad(alpha)
    Mdelta = which_quad(delta, octant=False)
    cond2 = Qalpha == 2
    azimuth[cond1 * cond2] = azimuth[cond1 * cond2] + np.pi / 2
    cond2 = np.logical_not(cond2)
    cond3 = (Mdelta == 2) + (Mdelta == 3)
    azimuth[cond1 * cond2 * cond3] = azimuth[cond1 * cond2 * cond3] + np.pi
    cond2 = (Mdelta == 0) + (Qalpha == 2.5)
    azimuth[cond1 * cond2] = azimuth[cond1 * cond2] + np.pi
    cond2 = np.logical_not(cond2)
    cond3 = ((Mdelta == 1.5) + (Mdelta == 2.5) + (Mdelta == 3.5)) * (
        (Qalpha == 1.5) + (Qalpha == 2.5))
    azimuth[cond1 * cond2 * cond3] = azimuth[cond1 * cond2 * cond3] + np.pi
    cond = (Qalpha == 1.5) * (Mdelta == 1.5)
    ellipticity[cond] = np.pi / 4
    azimuth[cond] = np.pi / 4
    cond = (Qalpha == 1.5) * (Mdelta == 3.5)
    ellipticity[cond] = -np.pi / 4
    azimuth[cond] = np.pi / 4
    cond = (Qalpha == 1.5) * (Mdelta >= 1.5) * (Mdelta <= 3.5)
    azimuth[cond] = 3 * np.pi / 4

    # if np.abs(alpha - np.pi / 4) < tol_default and (
    #         np.abs(delta - np.pi / 2) < tol_default
    #         or np.abs(delta - 3 * np.pi / 2) < tol_default):
    #     azimuth = np.nan
    #     ellipticity = np.sign(delta - np.pi) * np.pi / 4
    # else:
    #     # Calculate values using trigonometric functions
    #     azimuth = 0.5 * np.arctan(tan(2 * alpha) * cos(delta))
    #     ellipticity = 0.5 * np.arcsin(sin(2 * alpha) * sin(delta))
    #     # Use the other possible value from the arcs functions if necessary
    #     Qalpha = which_quad(alpha)
    #     Mdelta = which_quad(delta, octant=False)
    #     if Qalpha == 2:
    #         azimuth += np.pi / 2
    #     else:
    #         if Mdelta == 2 or Mdelta == 3:
    #             azimuth += np.pi
    #     if Mdelta == 0 and Qalpha == 2.5:
    #         azimuth += np.pi
    #     elif Mdelta == 1.5 or Mdelta == 2.5 or Mdelta == 3.5:
    #         if Qalpha == 1.5 or Qalpha == 2.5:
    #             azimuth += np.pi
    # Check that the outpit values are in the correct interval
    azimuth = put_in_limits(azimuth, "azimuth")
    ellipticity = put_in_limits(ellipticity, "ellipticity")
    # Reshape and return
    azimuth = np.reshape(azimuth, shape)
    ellipticity = np.reshape(ellipticity, shape)
    return azimuth, ellipticity


def extract_azimuth_elipt(vector, use_nan=True):
    """Function that extracts azimuth and ellipticity from a diattenuation, polarizance or retardance vector. All of them are of the form of: TODO.

    .. math:: cos(2 \\alpha) = cos(2 \phi) * cos(2 \chi)

    .. math:: tan(\delay) = \\tan(2 \chi) / sin(2 \phi)


    References:
        J. J. Gil, "Polarized light and the Mueller Matrix approach", pp 128 and 142.

    Parameters:
        vector (np.array 1x3 or 3x1): vector to be measured

    Returns:
        azimuth (float) [0,np.pi]: Azimuth (angle of rotation).
        ellipticity (float) [-pi/4,np.pi/4]: Elipticity angle.
    """
    # Normalize
    norm = np.linalg.norm(vector, axis=0)
    cond = norm > tol_default
    if np.any(cond):
        vector[0, :][cond] /= norm[cond]
        vector[1, :][cond] /= norm[cond]
        vector[2, :][cond] /= norm[cond]

    # Start by calculating ellipticity, which is isolated
    ellipticity = np.arcsin(vector[2, :]) / 2
    # Now, calculate azimuth using the first therm, as it has the cos
    azimuth = 0.5 * np.arccos(vector[0, :] / np.cos(2 * ellipticity))
    # # Adjust when we have azimuths over 90º
    cond = vector[1, :] < 0
    if np.any(cond):
        azimuth[cond] = np.pi - azimuth[cond]

    # Correct nans
    if not use_nan:
        cond = np.isnan(azimuth)
        if np.any(cond):
            azimuth[cond] = 0
        cond = np.isnan(ellipticity)
        if np.any(ellipticity):
            ellipticity[cond] = 0
    return azimuth, ellipticity


def extract_charac_angles(vector, use_nan=True, type='diattenuator'):
    """Function that extracts azimuth and ellipticity from a diattenuation, polarizance or retardance vector. All of them are of the form of: TODO.

    .. math:: cos(2 \\alpha) = cos(2 \phi) * cos(2 \chi)

    .. math:: tan(\delay) = \\frac{tan(2 \chi)}{sin(2 \phi)}


    References:
        J. J. Gil, "Polarized light and the Mueller Matrix approach", pp 128 and 142.

    Parameters:
        vector (np.array 1x3 or 3x1): vector to be measured

    Returns:
        alpha (float) [0,np.pi]: tan(alpha) is the ratio between the maximum amplitudes of the polarization elipse in X-Y coordinates.
        delay (float) [0, 2*pi]: phase difference between both components of the eigenstates in Jones formalism.
    """
    # Do it with azimuth and ellipticity and convert
    azimuth, ellipticity = extract_azimuth_elipt(
        vector=vector, use_nan=use_nan, type=type)
    alpha, delay = azimuth_elipt_2_charac_angles(azimuth, ellipticity)
    return alpha, delay


def which_quad(angle, octant=True):
    """Auxiliary function to calculate which quadrant or octant angle belongs to.
    Half angles means that it is exactly between two quarants.

    Parameters:
        (float): Angle to determine the quadrant.

    Returns:
        (float): Quadrant
    """
    q = np.zeros_like(angle)
    if octant:
        q = q - 4.5 * (angle <= -np.pi + tol_default)
        q = q - 4 * (angle > -np.pi + tol_default) * (
            angle < -np.pi * 3 / 4 - tol_default)
        q = q - 3.5 * (np.abs(angle + np.pi * 3 / 4) <= tol_default)
        q = q - 3 * (angle > -np.pi * 3 / 4 + tol_default) * (
            angle < -np.pi / 2 - tol_default)
        q = q - 2.5 * (np.abs(angle + np.pi / 2) <= tol_default)
        q = q - 2 * (angle > -np.pi / 2 + tol_default) * (
            angle < -np.pi / 4 - tol_default)
        q = q - 1.5 * (np.abs(angle + np.pi / 4) <= tol_default)
        q = q - 1 * (angle > -np.pi / 4 + tol_default) * (angle < -tol_default)
        q = q + 1 * (angle > tol_default) * (angle < np.pi / 4 - tol_default)
        q = q + 1.5 * (np.abs(angle - np.pi / 4) <= tol_default)
        q = q + 2 * (angle > np.pi / 4 + tol_default) * (
            angle < np.pi / 2 - tol_default)
        q = q + 2.5 * (np.abs(angle - np.pi / 2) <= tol_default)
        q = q + 3 * (angle > np.pi / 2 + tol_default) * (
            angle < np.pi * 3 / 4 - tol_default)
        q = q + 3.5 * (np.abs(angle - np.pi * 3 / 4) <= tol_default)
        q = q + 4 * (angle > np.pi * 3 / 4 + tol_default) * (
            angle < np.pi - tol_default)
        q = q + 4.5 * (angle >= np.pi - tol_default)
    else:
        angle = angle % (2 * np.pi)
        q = q + 1 * (angle > tol_default) * (angle < np.pi / 2 - tol_default)
        q = q + 1.5 * (np.abs(angle - np.pi / 2) < tol_default)
        q = q + 2 * (angle > np.pi / 2 + tol_default) * (angle <
                                                         np.pi - tol_default)
        q = q + 2.5 * (np.abs(angle - np.pi) < tol_default)
        q = q + 3 * (angle > np.pi + tol_default) * (
            angle < np.pi * 3 / 2 - tol_default)
        q = q + 3.5 * (np.abs(angle - np.pi * 3 / 2) < tol_default)
        q = q + 4 * (angle > np.pi * 3 / 2 + tol_default) * (
            angle < np.pi * 2 - tol_default)
    return q


def put_in_limits(x, typev, out_number=True):
    """When dealing with polarization elipse coordinates, make sure that they
    are in the valid limits, which are set in the declaration of this class.

    Parameters:
        x (1xN array): Value
        typev (string): Which type of variable is: alpha, delta, azimuth or ellipticity.
        out_number (bool): if True and the result is a 1x1 array, return a number instead. Default: True.

    Returns:
        y (1xN array): Corresponding angle inside the valid limits.
    """
    # In case it is a number, transform it into an array
    x = np.array(x)
    # Avoid unnecessary nan warnings
    condNan = np.isnan(x)
    x[condNan] = 0
    # Change x only if necessary
    if typev in ("alpha", "Alpha"):
        cond = np.logical_or(x < limAlpha[0], x >= limAlpha[1])
        aux = sin(x[cond])
        x[cond] = np.arcsin(abs(aux))

    elif typev in ("delta", "Delta", "delay", "Delay"):
        cond = np.logical_or(x < limDelta[0], x >= limDelta[1])
        x[cond] = x[cond] % (2 * np.pi)

    elif typev in ("azimuth", "Az", 'azimuth', 'Azimuth'):
        cond = np.logical_or(x < limAz[0], x >= limAz[1])
        # aux = cos(x[cond])
        # x[cond] = np.arccos(aux)
        x[cond] = x[cond] % np.pi

    elif typev in ("ellipticity", "El", 'ellipticity', 'Ellipticity'):
        cond = np.logical_or(x < limEl[0], x >= limEl[1])
        aux = tan(x[cond])
        cond2 = np.abs(aux) > 1
        # print(x / degrees, '\n', cond, '\n', cond2)
        aux[cond2] = 1 / aux[cond2]
        x[cond] = np.arctan(aux)
    elif typev in ('ret', 'Ret', 'RET', 'Retardance', 'retardance'):
        cond = np.logical_or(x < limRet[0], x > limRet[1])
        aux = cos(x[cond])
        x[cond] = np.arccos(aux)
    # Recover the nan values
    if np.any(condNan):
        x[condNan] = np.nan
    # If the result is a number and the user asks for it, return a float
    if out_number and x.size == 1:
        if np.isnan(x):
            x = np.nan
        else:
            x = float(x)

    return x


def execute_multiprocessing(__function_process__,
                            dict_parameters,
                            num_processors,
                            verbose=False):
    """Executes multiprocessing reading a dictionary.

    Parameters:
        __function_process__ (func): function to process, it only accepts a dictionary
        dict_parameters (dict): dictionary / array with parameters
        num_processors (int): Number of processors. if 1 no multiprocessing is used
        verbose (bool): Prints processing time.

    Returns:
        data: results of multiprocessing
        (float): processing time

    Example:

    .. code-block:: python

        def __function_process__(xd):
            x = xd['x']
            y = xd['y']
            # grt = copy.deepcopy(grating)
            suma = x + y
            return dict(sumas=suma, ij=xd['ij'])

        def creation_dictionary_multiprocessing():
            # create parameters for multiprocessing
            t1 = time.time()
            X = sp.linspace(1, 2, 10)
            Y = sp.linspace(1, 2, 1000)
            dict_parameters = []
            ij = 0
            for i, x in enumerate(X):
                for j, y in enumerate(Y):
                    dict_parameters.append(dict(x=x, y=y, ij=[ij]))
                    ij += 1
            t2 = time.time()
            print "time creation dictionary = {}".format(t2 - t1)
            return dict_parameters
    """
    t1 = time.time()
    if num_processors == 1 or len(dict_parameters) < 2:
        data_pool = [__function_process__(xd) for xd in dict_parameters]
    else:
        pool = multiprocessing.Pool(processes=num_processors)
        data_pool = pool.map(__function_process__, dict_parameters)
        pool.close()
        pool.join()
    t2 = time.time()
    if verbose is True:
        print("num_proc: {}, time={}".format(num_processors, t2 - t1))
    return data_pool, t2 - t1


def divide_in_blocks(M):
    """Function that creates a mueller matrix from their block components.

    References: J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

    Parameters:
        M (4x4 matrix): Mueller matrix of the diattenuator.

    Returns:
        Dv (1x3 or 3x1 float): Diattenuation vector.
        Pv (1x3 or 3x1 float): Diattenuation vector.
        m (3x3 matrix): Small m matrix.
        m00 (float, default 1): [0, 1] Parameter of average intensity.
    """
    m00 = M[0, 0]
    if m00 > 0:
        M = M / m00
    Dv = matrix(M[0, 1:4])
    Pv = matrix(M[1:4, 0])
    m = matrix(M[1:4, 1:4])
    return Dv, Pv, m, m00


def list_of_objects_depercated(size, type_object):
    """Creates a list of objects."""
    try:
        N = len(size)
    except:
        N = 1
    dim0 = []
    if N == 1:
        for ind in range(size):
            dim0.append(type_object(' '))
    elif N == 2:
        for ind in range(size[0]):
            dim1 = []
            for ind in range(size[1]):
                dim1.append(type_object(' '))
            dim0.append(dim1)
    elif N == 3:
        for ind in range(size[0]):
            dim1 = []
            for ind in range(size[1]):
                dim2 = []
                for ind in range(size[2]):
                    dim2.append(type_object(' '))
                dim1.append(dim2)
            dim0.append(dim1)
    elif N == 4:
        for ind in range(size[0]):
            dim1 = []
            for ind in range(size[1]):
                dim2 = []
                for ind in range(size[2]):
                    dim3 = []
                    for ind in range(size[3]):
                        dim3.append(type_object(' '))
                    dim2.append(dim3)
                dim1.append(dim2)
            dim0.append(dim1)
    return dim0


def inv_pypol(M):
    """Calculates the inverse matrix of the matrix of a py_pol object.

    Parameters:
        M (numpy.ndarray): Array to calculate the inverse. Its shape must be (NxNxM). The result will have the same shape.

    Returns:
        (numpy.ndarray): Result.
    """
    M = np.moveaxis(M, -1, 0)
    M = np.linalg.inv(M)
    M = np.moveaxis(M, 0, -1)
    return M


def obj_error(sol, test, shape=None, shape_like=None, out_number=None):
    """Calculates the difference between two abjects and calculates the error as the norm."""
    # Check that both of them are of the same type
    if sol._type != test._type:
        raise ValueError(
            'sol ({}) and test ({}) are of different types.'.format(
                sol._type, test._type))
    # Make them to have the same size
    if sol.size < test.size:
        obj1 = sol.stretch(test.size, keep=True)
        obj2 = test.copy()
    elif sol.size > test.size:
        obj1 = sol.copy()
        obj2 = test.stretch(sol.size, keep=True)
    else:
        obj1 = sol.copy()
        obj2 = test.copy()
    # Calculate the matrix difference
    M = obj1.M - obj2.M
    obj1.shape = take_shape((obj1, obj2))
    # Calculate the norm
    if sol._type in ('Jones_vector', 'Stokes'):
        norm = np.linalg.norm(M, axis=0)
    else:
        norm = np.linalg.norm(M, axis=(0, 1))
    # Reshape the norm
    if out_number and norm.size == 1:
        norm = norm[0]
    norm = reshape([norm], shape_like=shape_like, shape_fun=shape, obj=obj1)
    return norm


def matmul_pypol(M1, M2):
    """Calculates the multiplication of two matrices from pypol objects.

    Parameters:
        M1, (numpy.ndarray): Left array to multiply. Its shape must be (NxNxM).
        M1, (numpy.ndarray): Right array to multiply. Its shape must be (NxNxM) or (NxM). The result will have the same shape.

    Returns:
        (numpy.ndarray): Result.
    """
    M1 = np.moveaxis(M1, -1, 0)
    M2 = np.moveaxis(M2, -1, 0)
    if M2.ndim == 2:
        M2 = np.expand_dims(M2, 2)
    M = M1 @ M2
    M = np.moveaxis(M, 0, -1)
    return M


def list_of_objects(size, type_object):
    """Creates a list of objects."""
    if isinstance(size, (int, float)):
        size = [size]
    Ndims = len(size)
    list = []
    for ind in range(size[0]):
        if Ndims > 1:
            list.append(list_of_objects(size[1:Ndims + 1], type_object))
        else:
            list.append(type_object(' '))

    return list


def _pickle_method(method):
    """
    function for multiprocessing in class
    """
    func_name = method.__func__.__name__
    obj = method.__self__
    cls = method.__self__.__class__
    # deal with mangled names
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    """
    function for multiprocessing in class
    """
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


def iscolumn(v):
    """Checks if the array v is a column array or not.

    Parameters:
        v (array): Array to be tested.

    Returns:
        cond (bool): True if v is a column array."""

    cond = False
    s = v.shape
    if len(s) == 2:
        if s[1] == 1 and s[1] > 1:
            cond = True
    return cond


def isrow(v):
    """Checks if the array v is a row array or not.

    Parameters:
        v (array): Array to be tested.

    Returns:
        cond (bool): True if v is a row array."""

    cond = False
    s = v.shape
    if len(s) == 1:
        cond = True
    elif len(s) == 2:
        if s[0] == 1:
            cond = True
    return cond


def delta_kron(a, b):
    """Computes the Kronecker delta.

    Parameters:
        a, b (int): Numbers.

    Returns:
        d (int): Result."""

    if a == b:
        d = 1
    else:
        d = 0
    return d


def order_eig(val, vect, kind='normal'):
    """Function that orders the eigenvalues from max to min, and then orders
    the eigenvectors following the same order.

    Parameters:
        val (numpy.ndarray): Array of eigenvalues.
        vect (numpy.ndarray): Matrix with the eigenvectors as columns.
        kind (string): Choses if the sort order is normal (min to max) or reverse (max to min).

    Returns:
        q (numpy.ndarray): Array of ordered eigenvalues.
        m (numpy.ndarray): Matrix with the eigenvectors ordered as columns.
        """
    # Values must be real, so make them real
    val = np.array(val, dtype=float)
    vect = np.array(vect, dtype=float)

    # Reverse order if desired
    if kind.upper() != 'NORMAL':
        val = -val

    # Make sure val is a 4xN array and vect a 4x4xN array
    if val.ndim > 2:
        # Save the old shape
        shape = list(val.shape)
        # Reshape
        size = val.size / 4
        val = np.reshape(val, [4, size])
        vect = np.reshape(vect, [4, 4, size])
    else:
        shape = None

    # Find the correct order
    order = np.argsort(val, axis=0)
    # Reorder
    val = np.sort(val, axis=0)
    for ind in range(4):
        vect[ind, :, :] = np.take_along_axis(vect[ind, :, :], order, axis=0)
    # # Find correct order
    # order = np.flip(np.argsort(q), 0)
    # # Order eigenvalues
    # q = np.flip(np.sort(q), 0)
    # # Order eigenvectors
    # s = m.shape
    # m2 = np.zeros(s, dtype=complex)
    # for ind in range(s[1]):
    #     ind2 = order[ind]
    #     m2[:, ind] = np.squeeze(m[:, ind2])
    # # Differentiate between real and complex cases
    # Im = np.linalg.norm(np.imag(q))
    # if Im < tol_default:
    #     q = np.real(q)
    # Im = np.linalg.norm(np.imag(m2))
    # if Im < tol_default:
    #     m2 = np.real(m2)
    # # Return
    # return q, np.matrix(m2, dtype=complex)

    # Reshape again if necessary
    if shape is not None:
        val = np.reshape(val, shape)
        vect = np.reshape(vect, [4] + shape)

    # Reverse order if desired
    if kind.upper() != 'NORMAL':
        val = -val

    return val, vect


def check_eig(q, m, M):
    """Function that checks the eigenvalues and eigenvectors."""
    dif = np.zeros(len(q))
    for ind, qi in enumerate(q):
        v = m[:, ind]
        v2 = M * v
        d = v2 - qi * v
        dif[ind] = np.linalg.norm(d)
        print(("The eigenvalue {} has an eigenvector {}.".format(qi, v.T)))
    M2 = m * M * m.T
    d = M2 - M
    dif2 = sqrt(np.sum(np.square(d)))
    dif3 = (abs(d)).max()
    d = m.T - m.I
    dif4 = sqrt(np.sum(np.square(d)))
    print('The eigenvalues are:')
    print(q)
    print('The deviation respect to the eigenvectors is:')
    print(dif)
    print(
        ('The mean square difference in the decomposition is: {}.'.format(dif2)
         ))
    print(('The maximum difference in the decomposition is: {}.'.format(dif3)))
    print(('The matrix of eigenvalues is orthogonal with deviation {}'.format(
        dif4)))
    print(M)
    print(M2)


# def seq(start, stop, step=1):
#     n = int(round((stop - start) / float(step)))
#     if n > 1:
#         return ([start + step * i for i in range(n + 1)])
#     else:
#         return ([])


def distance(x1, x2):
    """
    Compute distance between two vectors.

    Arguments:
        x1 (numpy.ndarray): vector 1
        x2 (numpy.ndarray): vector 2

    Returns:
        (float): distance between vectors.
    """
    x1 = array(x1)
    x2 = array(x2)
    # print(x1.ndim)

    dist2 = 0
    for i in range(x1.ndim):
        dist2 = dist2 + (x1[i] - x2[i])**2

    return sp.sqrt(dist2)


def nearest(vector, number):
    """Computes the nearest element in vector to number.

        Parameters:
            vector (numpy.ndarray): array with numbers
            number (float):  number to determine position

        Returns:
            (int): index - index of vector which is closest to number.
            (float): value  - value of vector[index].
            (float): distance - difference between number and chosen element.
    """
    indexes = np.abs(vector - number).argmin()
    values = vector.flat[indexes]
    distances = values - number
    return indexes, values, distances


def nearest2(vector, numbers):
    """Computes the nearest element in vector to numbers.

        Parameters:
            vector (numpy.ndarray): array with numbers
            number (numpy.ndarray):  numbers to determine position

        Returns:
            (numpy.ndarray): index - indexes of vector which is closest to number.
            (numpy.ndarray): value  - values of vector[indexes].
            (numpy.ndarray): distance - difference between numbers and chosen elements.

    """
    indexes = np.abs(np.subtract.outer(vector, numbers)).argmin(0)
    values = vector[indexes]
    distances = values - numbers
    return indexes, values, distances


def repair_name(name_initial):
    """
    Repairs name when several angles are included.

    Example:
        M1 @45.00deg @45.00deg @45.00deg @45.00deg @45.00deg @45.00deg @45.00deg

        passes to:

        M1 @135.00deg

    Parameters:
        name_initial (str): String with the name.

    Returns:
        (str): Repaired name

    """

    num_angles = name_initial.count('@')
    if num_angles == 0:
        return name_initial
    try:
        text = "{} ".format(name_initial)

        text = text.split("@")
        name_original = text[0][0:-1]
        angle_number = 0
        for i, ti in enumerate(text[1:]):
            ri = ti.rfind('d')
            angle = float(ti[:ri])
            angle_number = angle_number + angle
        angle_number = np.remainder(angle_number, 360)
        name_final = "{} @ {:2.2f} deg".format(name_original, angle_number)
    except:
        name_final = name_initial
    return name_final


def comparison(proposal, solution, maximum_diff=tol_default):
    """This functions is mainly for testing. It compares compares proposal to solution.

    Parameters:
        proposal (numpy.matrix): proposal of result.
        solution (numpy.matrix): results of the test.
        maximum_diff (float): maximum difference allowed.

    Returns:
        (bool): True if comparison is possitive, else False.
    """
    # Just in case
    proposal = np.array(proposal)
    solution = np.array(solution)

    comparison1 = np.linalg.norm(proposal - solution) < maximum_diff
    comparison2 = np.linalg.norm(proposal + solution) < maximum_diff

    return comparison1 or comparison2


def params_to_list(J, verbose=False):
    """Makes a list from data provided at parameters.dict_params

        Parameters:
            J (object): Object Jones_vector, Jones_matrix, Stokes or Mueller.
            verbose (bool): If True prints the parameters

        Returns:
            (list): List with parameters from dict_params.
    """

    if J._type == 'Jones_vector':
        params = J.parameters.dict_params

        intensity = params['intensity']
        alpha = params['alpha']
        delay = params['delay']
        azimuth = params['azimuth']
        ellipticity_angle = params['ellipticity_angle']
        a, b = params['ellipse_axes'][0], params['ellipse_axes'][1]

        if verbose is True:
            print("({}, {}, {}, {}, {}, {}, {})".format(
                intensity, alpha, delay, azimuth, ellipticity_angle, a, b))

        return intensity, alpha, delay, azimuth, ellipticity_angle, a, b

    elif J._type == 'Stokes':
        params = J.parameters.dict_params

        intensity = params['intensity']
        amplitudes = params['amplitudes']
        degree_pol = params['degree_pol']
        degree_linear_pol = params['degree_linear_pol']
        degree_circular_pol = params['degree_circular_pol']
        alpha = params['alpha']
        delay = params['delay']
        azimuth = params['azimuth']
        ellipticity_angle = params['ellipticity_angle']
        ellipticity_param = params['ellipticity_param']
        polarized = params['polarized']
        unpolarized = params['unpolarized']

        if verbose is True:
            print("({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
                intensity, amplitudes, degree_pol, degree_linear_pol,
                degree_circular_pol, alpha, delay, azimuth, ellipticity_angle,
                ellipticity_param, polarized, unpolarized))

        return intensity, amplitudes, degree_pol, degree_linear_pol, degree_circular_pol, alpha, delay, azimuth, ellipticity_angle, ellipticity_param, polarized, unpolarized

    elif J._type == 'Jones_matrix':
        params = J.parameters.dict_params

        delay = params['delay']
        diattenuation = params['diattenuation']
        is_homogeneous = params['is_homogeneous']

        if verbose is True:
            print("({}, {}, {})".format(delay, diattenuation, is_homogeneous))

        return delay, diattenuation, is_homogeneous

    elif J._type == 'Mueller':
        pass


# def fit_phase(param, phase, irrelevant=0):
#     s = phase.shape
#     Th0, X = np.meshgrid(param[:s[1]], np.arange(s[0]))
#     W, _ = np.meshgrid(param[s[1]:2 * s[1]], np.arange(s[0]))
#     # print('phase shape = {}\nX shape = {}\n'.format(s, X.shape))
#     test = (Th0 + X * W) % (2 * np.pi)
#     # print('phase', phase % (2 * np.pi))
#     # print('test', test)
#     error = (test - phase % (2 * np.pi)).flatten()
#     print('error', error, np.linalg.norm(error))
#     return error


def fit_distribution(param, dist, irrelevant=0):
    s = dist.shape
    Th0, X = np.meshgrid(param[:s[1]], np.arange(s[0]))
    W, _ = np.meshgrid(param[s[1]:2 * s[1]], np.arange(s[0]))
    A, _ = np.meshgrid(param[2 * s[1]:], np.arange(s[0]))
    test = np.cos(Th0 + X * W) * A
    error = (test - dist).flatten()
    # print(np.linalg.norm(error))
    return error


def fit_cos(par, x, y):
    """Function to fit a cos function using the least_squares function from scipy.optimize."""
    c = par[0] * np.cos(par[1] + x * par[2])
    return y - c


def fit_sine(t, data, has_draw=True):
    """fit a sine function
    """

    def optimize_func(x):
        return x[0] * np.sin(x[1] * t + x[2]) + x[3] - data

    guess_mean = np.mean(data)
    guess_std = 3 * np.std(data) / (2**0.5) / (2**0.5)
    guess_phase = 0
    guess_freq = 1
    guess_amp = 1

    # This might already be good enough for you
    data_first_guess = guess_std * np.sin(t + guess_phase) + guess_mean

    # Define the function to optimize, in this case, we want to minimize
    # the difference between the actual data and our "guessed" parameters

    est_amp, est_freq, est_phase, est_mean = leastsq(
        optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters

    if has_draw is True:
        fine_t = np.arange(0, max(t), 0.1)
        data_fit = est_amp * np.sin(est_freq * fine_t + est_phase) + est_mean

        plt.figure()
        plt.plot(t, data, '.')
        plt.plot(t, data_first_guess, label='first guess')
        plt.plot(fine_t, data_fit, label='after fitting')
        plt.legend()
        plt.show()

    return est_amp, est_freq, est_phase, est_mean


def obj_2_xyz(S, Ninterp=1, DAinterp=None, in_degrees=False, param=None, interp_to_surf=False, depol=False):
    """Function to calculate the x, y and z coordinates of the Poincare sphere.

    Args:
        S (Jones_vector or Stokes): 0D or 1D object to be represented.
        Ninterp (number): Multiplication factor to interpolate angles if DAinterp is not None. Default: 1.
        DAinterp (number): Angle step for interpolation. Default: None.
        return_azel (bool): If True, returns interpolated azimuth and ellipticity. Default: False.
        in_degrees (bool): If True, azimuth and ellipticity are returned in degrees. Default: False.
        interp_to_surf (bool): If True and S is 1D, transform it to 2D. Default: True.
        depol (bool): If True, the depolarization is taking into account for scatter and line plots shrinking the radius below 1. Default: False.

    Returns:
        x, y, z (np.ndarray): Calculated coordinates.
        az, el (np.ndarray): Interpolated azimuth and ellipticity.
        param (np.ndarray): Calculated and interpolated parameter.
    """
    # Calculate actual spherical coordinates
   
    az, el = S.parameters.azimuth_ellipticity(out_number=False, use_nan=False)
    if depol:
        pol = S.parameters.degree_polarization(out_number=False, use_nan=False)
        pol[np.isnan(pol)] = eps
        pol[pol < eps] = eps
        pol[pol > 1] = 1
    else:
        pol = 1

    # Interpolate 1D data to plot in the 2D sphere
    if interp_to_surf and S.ndim == 1:
        # Calculate spherical coordinates
        az_int = np.linspace(0, 180*degrees, int(np.ceil(179.999*degrees/(DAinterp or 1*degrees))+1))
        el_int = np.linspace(-45*degrees, 45*degrees, int(np.ceil(90*degrees/(DAinterp or 1*degrees))+1))
        Az_int, El_int = np.meshgrid(az_int, el_int)
        #print(Az_int.shape,El_int.shape)
        # Calculate interpolation
       
        param = griddata((az, el), param, (Az_int, El_int), method='cubic')

        #param = griddata((az, el), param, (Az_int, El_int), method='cubic')
        az, el = Az_int, El_int

    # Interpolate scatter points to plot the line along the sphere
    else:
        # Interpolate in angles step
        if DAinterp is not None:
            Daz, Del = (np.diff(az), np.diff(el))
            Dstep = np.maximum(Daz, Del)
            Nstep = np.array(np.ceil(Dstep/DAinterp), dtype=int)
            Nincr = np.repeat(1/Nstep, Nstep)
            x = np.cumsum(Nincr)
        # Interpolate multiplying number of points
        elif Ninterp > 1:
            x = np.linspace(0, az.size-1, (az.size-1)*Ninterp)
        # Interpolate
        if (DAinterp is not None) or (Ninterp > 1):
            xp = np.arange(az.size)
            az = np.interp(x, xp, az)
            el = np.interp(x, xp, el)
            if param is not None:
                param = np.interp(x, xp, param)
            if depol:
                pol = np.interp(x, xp, pol)

    # Calculate cartesian points.
    x, y, z = azel_2_xyz(az, el, pol=pol)

    # Return
    if in_degrees:
        ret = [x, y, z, az/degrees, el/degrees, param]
    else:
        ret = [x, y, z, az, el, param]
    return ret


def azel_2_xyz(az, el, pol=1):
    """Transforms azimuth and ellipticity angle to x, y and z coordinates of the Poincare sphere.

    Args:
        az (np.ndarray): Azimuth.
        el (np.ndarray): Ellipticity angle.
        pol (np.ndarray): Polarization degree.

    Returns:
        x, y, z (np.ndarray): Calculated coordinates.
    """
    phi = 2*az
    chi = -2*el + 90*degrees
    x = pol * np.sin(chi) * np.cos(phi)
    y = pol * np.sin(chi) * np.sin(phi)
    z = pol * np.cos(chi)
    return x, y, z


# recreate the fitted curve using the optimized parameters
