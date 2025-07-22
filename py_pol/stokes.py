# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------
# Authors:    Luis Miguel Sanchez Brea and Jesus del Hoyo
# Date:       2019/01/09 (version 1.0)
# License:    GPL
# -------------------------------------
"""
Stokes objects describe light polarization states in the Mueller-Stokes formalism.

**Class fields**
    * **M**: 2xN array containing all the Stokes vectors.
    * **global_phase**: Global phase of the light state.
    * **name**: Name of the object for print purposes.
    * **shape**: Shape desired for the outputs.
    * **size**: Number of stores Stokes vectors.
    * **type**: Type of the object ('Jones_vector'). This is used for determining the object class as using isinstance may throw unexpected results in .ipynb files.
    * **parameters**: Object of class *Parameters_Jones_vector*.
    * **checks**: Object of class *Checks_Jones_vector*.


**Parent methods**
    * **clear**:  Removes data and name form Jones vector.
    * **copy**:  Creates a copy of the Jones_vector object.
    * **stretch**:  Stretches a Jones vector of size 1.
    * **shape_like**:  Takes the shape of another object to use as its own.
    * **reshape**: Changes the shape of the object.
    * **flatten**:  Transforms N-D objects into 1-D objects (0-D if only 1 element).
    * **flip**: Flips the object along some dimensions.
    * **get_list**: Creates a list with single elements.
    * **from_list**: Creates the object from a list of single elements.
    * **concatenate**: Canocatenates several objects into a single one.
    * **draw**: Draws the components of the object.
    * **clear**: Clears the information of the object.


**Generation methods**
    * **from_components**: Creates Stokes vectors directly from the 4 elements $S_0$, $S_1$, $S_2$, $S_3$.
    * **from_matrix**: Creates Stokes vectors from an external 4 x shape numpy array.
    * **from_list**: Creates a Jones_vector object directly from a list of 4 or 4x1 numpy arrays.
    * **from_Jones**: Creates Stokes vectors from a Jones_vector object.
    * **linear_light**: Creates Stokes vectors for pure linear polarizer light.
    * **circular_light**: Creates Stokes vectors for pure circular polarizer light.
    * **elliptical_light** Creates Stokes vectors for polarizer elliptical light.
    * **general**: Creates the most general Stokes vectors.
    * **general_charac_angles** Creates Stokes vectors given by their characteristic angles.
    * **general_azimuth_ellipticity** Creates Stokes vectors given by their azimuth and ellipticity.


**Manipulation methods**
    * **simplify**:  Simplifies the Stokes vectors in several ways.
    * **rotate**: Rotates the Stokes vectors.
    * **sum**: Calculates the summatory of the Stokes vectors in the object.
    * **reciprocal**: Calculates the Stokes vectors that propagates backwards.
    * **orthogonal**: Calculates the orthogonal Stokes vectors.
    * **normalize**: Normalize the electric field to be normalized in electric field amplitude or intensity.
    * **rotate_to_azimuth**: Rotates the Stokes vectors to have a certain azimuth.
    * **remove_global_phase**: Calculates the global phase of the electric field (respect to the X component) and removes it.
    * **add_global_phase**: Adds a global phase to the Stokes vectors.
    * **set_global_phase**: Sets the global phase of the Stokes vectors.
    * **set_depolarization**: Sets the degree of depolarization.
    * **add_depolarization**: Increases the degree of depolarization.


**Drawing methods**
    * **draw_ellipse**:  Draws the polarization ellipse of the Stokes vectors.
    * **draw_poincare**: Draws the Stokes vectors in the Poincare sphere.


**Parameters subclass methods**
    * **matrix**:  Gets a numpy array with the Stokes vectors.
    * **components**: Calculates the electric field components of the Stokes vectors.
    * **amplitudes**: Calculates the electric field amplitudes of the Stokes vectors.
    * **intensity**: Calculates the intensity of the Stokes vectors.
    * **irradiance**: Calculates the irradiance of the Stokes vectors.
    * **alpha**: Calculates the ratio between electric field amplitudes ($E_x$/$E_y$).
    * **delay / delta**: Calculates the delay (phase shift) between Ex and Ey components of the electric field.
    * **charac_angles**: Calculates both alpha and delay, the characteristic angles of the Stokes vectors.
    * **azimuth**: Calculates azimuth, that is, the orientation angle of the major axis.
    * **ellipticity_angle**: Calculates the ellipticity angle.
    * **azimuth_ellipticity**: Calculates both azimuth and ellipticity angles.
    * **ellipse_axes**: Calculates the length of major and minor axis (a,b).
    * **ellipticity_param**: Calculates the ellipticity parameter, b/a.
    * **eccentricity**: Calculates the eccentricity, the complementary of the ellipticity parameter.
    * **global_phase**: Calculates the global phase of the Stokes vectors (respect to the X component of the electric field).
    * **degree_polarization**: Calculates the degree of polarization of the Stokes vectors.
    * **degree_depolarization**: Calculates the degree of depolarization of the Stokes vectors.
    * **degree_linear_polarization**: Calculates the degree of linear polarization of the Stokes vectors.
    * **degree_circular_polarization**: Calculates the degree of circular polarization of the Stokes vectors.
    * **norm**: Calculates the norm of the Stokes vectors.
    * **polarized_unpolarized**: Divides the Stokes vector in Sp+Su, where Sp is fully-polarized and Su fully-unpolarized.

    * **get_all**: Returns a dictionary with all the parameters of Stokes vectors.



**Checks subclass methods**
    * **is_physical**: Checks if the Stokes vectors are physically realizable.
    * **is_linear**: Checks if the Stokes vectors are lienarly polarized.
    * **is_circular**: Checks if the Stokes vectors are circularly polarized.
    * **is_right_handed**: Checks if the Stokes vectors rotation direction are right handed.
    * **is_left_handed**: Checks if the Stokes vectors rotation direction are left handed.
    * **is_polarized**: Checks if the Stokes vectors are at least partially polarized.
    * **is_totally_polarized**: Checks if the Stokes vectors are totally polarized.
    * **is_depolarized**: Checks if the Stokes vectors are at least partially depolarized.
    * **is_totally_depolarized**: Checks if the Stokes vectors are totally depolarized.

    * **get_all**: Returns a dictionary with all the checks of Stokes vectors.


**Analysis subclass methods**
    * **filter_physical_conditions**: Forces the Stokes vectors to be physically realizable.
    * **density**: Calculates the density of states around the Poincaré sphere.
    * **existence**: Checks the existence of states around the Poincaré sphere.
"""

import warnings
from functools import wraps

import numpy as np
from numpy import arctan2, array, cos, matrix, pi, sin, sqrt
from scipy import optimize
from copy import deepcopy

from .__init__ import degrees, eps, num_decimals, number_types, eta
from .py_pol import Py_pol
from .drawings import *
from .jones_vector import Jones_vector
from .utils import (azimuth_elipt_2_charac_angles, put_in_limits, repair_name,
                    rotation_matrix_Mueller, prepare_variables, reshape,
                    PrintParam, take_shape, select_shape, PrintMatrices,
                    combine_indices, merge_indices, multitake,
                    fit_distribution, azel_2_xyz)

warnings.filterwarnings('ignore')

stokes_0 = np.zeros((4, 1), dtype=float)
tol_default = eps
N_print_list = 10
print_list_spaces = 3
change_names = True
unknown_phase = False
default_phase = 0

# TODO: función para luz eliptica (a, b, angulo, polarización)? No sabía hacer

# # define Python user-defined exceptions
# class Error(Exception):
#     """Base class for other exceptions"""
#     pass
#
#
# class CustomError(Error):
#     """Raised when a custom error is produced"""
#     pass

################################################################################
# Functions
################################################################################


def create_Stokes(name='S', N=1, out_object=True):
    """Function that creates several Stokes objects at the same time from a list of names or a number.

    Parameters:
        names (str, list or tuple): name of vector for string representation. If list or tuple, it also represents the number of objects to be created. Default: 'S'.
        N (int): Number of elements to be created. This parameter is overrided if name is a list or tuple. Defeult: 1.
        out_object (bool): If True and the result is a list of length 1, return a Stokes object instead. Default: True.

    Returns:
        S (Stokes or list): List of Stokes vectors
    """
    S = []
    if isinstance(name, list) or isinstance(name, tuple):
        for n in name:
            S.append(Stokes(n))
    else:
        for _ in range(N):
            S.append(Stokes(name))
    if len(S) == 1 and out_object:
        S = S[0]
    return S


def set_printoptions(N_list=None, list_spaces=None):
    """Function that modifies the global print options parameters.

    TODO: Single global function for all modules.

    Parameters:
        N_list (int): Number of matrices that will be printed as a list if the shape of the object is 1D. Default: None.
        list_spaces (int): Number of spaces between matrices if they are printed as a list. Default: None.
    """
    global N_print_list, print_list_spaces
    if list_spaces is not None:
        print_list_spaces = list_spaces
    if N_list is not None:
        N_print_list = N_list


################################################################################
# Main class
################################################################################


class Stokes(Py_pol):
    """Class for Stokes vectors

    Parameters:
        M (float or numpy.ndarray): 4xN array containing all the Stokes vectors.
        name (string): Name of the object for print purposes.
        shape (tuple or list): Shape desired for the outputs.
        size (int): Number of stored Stokes vectors.
        ndim (int): Number of dimensions for representation purposes.
        global_phase (float or numpy.ndarray): Global phase of the Stokes vector. If it is a numpy.ndarray, it must have the same number of elements than self.size. It is used for addition and substraction.
        type (string): Type of the object ('Stokes'). This is used for determining the object class as using isinstance may throw unexpected results in .ipynb files.

    Attributes:
        self.parameters (class): Class containing the measurable parameters of Stokes vectors.
        self.checks (class): Class containing the methods that check something about the Stokes vectors.
        self.analysis (class): Class containing the methods to analyze the Stokes vectors.
    """
    __array_priority__ = 15000

    def __init__(self, name='S'):
        """Triggers during the Stokes inicialization.

        Parameters:
            name (string): Name of the object for representation purposes. Default: 'S'.

        Returns:
            (Stokes):
        """
        super().__init__(name=name, _class="Stokes")
        self._global_phase = None

        self.parameters = Parameters_Stokes_vector(self)
        self.analysis = Analysis_Stokes(self)
        self.checks = Check_Stokes(self)

    def __add__(self, other):
        """Adds two fields represented by their Stokes or Stokes vectors.

        Parameters:
            other (Stokes or Jones_vector): 2nd field to add.

        Returns:
            (Stokes): Result.
        """
        M3 = Stokes()
        if other.type == 'Stokes':
            # Easy case, incoherent sum
            if self.global_phase is None or other.global_phase is None or np.all(
                    self.global_phase == other.global_phase):
                M3.from_matrix(self.M + other.M)
            elif np.all(self.global_phase == other.global_phase):
                M3.from_matrix(self.M + other.M,
                               global_phase=self.global_phase)
            # More complicated case, coherent sum
            else:
                # Divide Stokes vectors in polarized and unpolarized counterparts
                s_pol, s_unpol = self.parameters.polarized_unpolarized()
                o_pol, o_unpol = other.parameters.polarized_unpolarized()
                # Create the Stokes vectors corresponding to the polarized parts and add them
                s = Jones_vector(self.name)
                s.from_Stokes(s_pol)
                o = Jones_vector(other.name)
                o.from_Stokes(o_pol)
                m3 = s + o
                # Reconvert to Stokes and add the unpolarized parts
                M3.from_Jones(m3)
                M3.M = M3.M + s_unpol.M + o_unpol.M
        elif other.type == 'Jones_vector':
            # If self has unknown global phase, make the sum incoherent
            if self.global_phase is None:
                S = Stokes(other.name)
                S.from_Jones(other)
                M3.from_matrix(self.M + S.M)
            # Coherent sum
            else:
                # Divide Stokes vectors in polarized and unpolarized counterparts
                s_pol, s_unpol = self.parameters.polarized_unpolarized()
                # Create the Stokes vectors corresponding to the polarized parts and add them
                s = Jones_vector(self.name)
                s.from_Stokes(s_pol)
                m3 = s + other
                # Reconvert to Stokes and add the unpolarized parts
                M3.from_Jones(m3)
                M3.M = M3.M + s_unpol.M
        else:
            raise ValueError(
                'other is {} instead of Stokes or Jones_vector.'.format(
                    type(other)))
        # Fix name and update common variables
        M3.shape = take_shape((self, other))
        if change_names:
            M3.name = self.name + " + " + other.name
        return M3

    def __sub__(self, other):
        """Substracts two fields represented by their Stokes or Jones vectors.

        Parameters:
            other (Stokes or Jones_vector): 2nd field to substract.

        Returns:
            (Stokes): Result.
        """
        M3 = self + ((-1) * other)
        if change_names:
            M3.name = self.name + " - " + other.name
        return M3

    def __mul__(self, other):
        """Multiplies a Stokes vector by a number. If the number is complex or real negative, the absolute value is used and the global phase is updated acordingly.

        Parameters:
            other (float, complex or numpy.ndarray): number to multiply.

        Returns:
            (Stokes): Result.
        """
        M3 = self.copy()
        # If we have a number, name can be updated
        if isinstance(other, number_types) and change_names:
            M3.name = str(other) + " * " + self.name
        # Calculate components
        S0, S1, S2, S3 = self.parameters.components(shape=False)
        # Save the Number of elements, and then flatten
        if isinstance(other, np.ndarray):
            N = other.size
            other2 = other.flatten()
        else:
            N = 1
            other2 = other
        # Check that the multiplication can be performed
        if N == self.size or self.size == 1 or N == 1:
            # Calculate the absolute value and complex phase of the number
            mod, phase = (np.abs(other2), np.angle(other2))
            # Create the object
            M3.from_components((S0 * mod, S1 * mod, S2 * mod, S3 * mod),
                               global_phase=self.global_phase)
            M3.add_global_phase(phase)
            if isinstance(other, np.ndarray):
                M3.shape = take_shape((self, other.shape))
            elif isinstance(other, number_types):
                M3.shape = self.shape
        else:
            raise ValueError(
                'The number of elements in other ({}) and self {} ({}) is not the same'
                .format(N, self.name, self.size))
        return M3

    def __rmul__(self, other):
        """Multiplies a Stokes vector by a number. If the number is complex or real negative, the absolute value is used and the global phase is updated acordingly.

        Parameters:
            other (float, complex or numpy.ndarray): number to multiply.

        Returns:
            (Stokes): Result.
        """
        M3 = self * other
        if isinstance(other, number_types) and change_names:
            M3.name = str(other) + " * " + self.name
        return M3

    def __truediv__(self, other):
        """Divides a Stokes vector by a number. If the number is complex or real negative, the absolute value is used and the global phase is updated acordingly.

        Parameters:
            other (float or numpy.ndarray): Divisor.

        Returns:
            (Stokes): Result.
        """
        M3 = self * (other**(-1))
        if isinstance(other, number_types) and change_names:
            M3.name = self.name + " / " + str(other)
        return M3

    def __repr__(self):
        """
        Represents the Stokes vector with print().
        """
        # Extract the components
        S0, S1, S2, S3 = self.parameters.components()
        # If the object is empty, say it
        if self.size == 0 or np.all(self.M == 0):
            return '{} is empty\n'.format(self.name)
        # If the object is 0D or 1D, print it like a list or inline
        elif self.size == 1 or self.shape is None or len(self.shape) < 2:
            if self.size <= N_print_list:
                list = self.get_list(out_number=False)
                l0_name = "{} = \n".format(self.name)
                l1_name = PrintMatrices(list, print_list_spaces)
                return l0_name + l1_name
            else:
                l0_name = "{} S0 = {}".format(self.name, S0)
                l1_name = " " * len(self.name) + " S1 = {}".format(S1)
                l2_name = " " * len(self.name) + " S2 = {}".format(S2)
                l3_name = " " * len(self.name) + " S3 = {}".format(S3)
        # Print higher dimensionality as pure arrays
        else:
            l0_name = "{} S0 = \n{}".format(self.name, S0)
            l1_name = "{} S1 = \n{}".format(self.name, S1)
            l2_name = "{} S2 = \n{}".format(self.name, S2)
            l3_name = "{} S3 = \n{}".format(self.name, S3)
        return l0_name + '\n' + l1_name + '\n' + l2_name + '\n' + l3_name + '\n'

    def __getitem__(self, index):
        """
        Implements object extraction from indices.
        """
        if change_names:
            S = Stokes(self.name + '_picked')
        else:
            S = Stokes(self.name)
        # If the indices are 1D, act upon the matrix directly
        cond = (isinstance(index, (int, slice))
                and self.ndim > 1) or (isinstance(index, np.ndarray)
                                       and index.ndim == 1 and self.ndim > 1)
        if cond:
            S.from_matrix(self.M[:, index])
            # Add global phase
            if self.global_phase is not None:
                if self.global_phase.size == 1:
                    S.global_phase = self.global_phase
                else:
                    S.global_phase = self.global_phase[index]

        # If not, act upon the components
        else:
            S0, S1, S2, S3 = self.parameters.components(out_number=False)
            M = np.array([S0[index], S1[index], S2[index], S3[index]])
            S.from_matrix(M)
            # Add global phase
            if self.global_phase is not None:
                if self.global_phase.size == 1:
                    S.global_phase = self.global_phase
                else:
                    phase = self.parameters.global_phase(out_number=False)
                    S.set_global_phase(phase[index])

        return S

    def __setitem__(self, index, data):
        """
        Implements object inclusion from indices.
        """
        # Check that data is a correct pypol object
        if data.type == 'Jones_vector':
            data2 = Stokes(data.name)
            data2.from_Jones(data)
        elif data.type == 'Stokes':
            data2 = data
        else:
            raise ValueError(
                'data is type {} instead of Jones_vector or Stokes.'.format(
                    data.type))
        # Expand phase if required
        if self.global_phase is None:
            self.global_phase = np.zeros(self.size) * np.nan
        if data2.global_phase is None:
            data2.global_phase = np.nan
        # If the indices are 1D, act upon the matrix directly
        if isinstance(index, int) and self.ndim > 1:
            self.M[:, index] = np.squeeze(data2.M)
            # Add global phase
            self.global_phase[index] = data2.global_phase
        elif isinstance(index, slice) and self.ndim > 1:
            if data2.size == 1:
                if index.step is None:
                    step = 1
                else:
                    step = index.step
                N = int((index.stop - index.start) / step)
                data3 = data2.stretch(length=N, keep=True)
            else:
                data3 = data2
            self.M[:, index] = np.squeeze(data3.M)
            # Add global phase
            self.global_phase[index] = data3.global_phase
        elif isinstance(index,
                        np.ndarray) and index.ndim == 1 and self.ndim > 1:
            self.M[:, index] = data2.M
            # Add global phase
            self.global_phase[index] = data2.global_phase
        # If not, act upon the components
        else:
            # Extract phase and components
            S0, S1, S2, S3 = self.parameters.components(out_number=False)
            phase = self.parameters.global_phase(out_number=False)
            S0_new, S1_new, S2_new, S3_new = data2.parameters.components(
                out_number=False)
            phase_new = data2.parameters.global_phase(out_number=False)
            # Set the new values
            S0[index] = np.squeeze(S0_new)
            S1[index] = np.squeeze(S1_new)
            S2[index] = np.squeeze(S2_new)
            S3[index] = np.squeeze(S3_new)
            phase[index] = np.squeeze(phase_new)
            # Update the object
            self.from_components((S0, S1, S2, S3), global_phase=phase)

    def __eq__(self, other):
        """
        Implements equality operation.
        """
        try:
            # Calculate the difference object
            if other.type == 'Jones_vector':
                S = Stokes()
                S.from_Jones(other)
                j3 = self - other
            elif other.type == 'Stokes':
                j3 = self - other
            else:
                return False
            # Compare matrices
            norm = j3.parameters.norm()
            cond1 = norm < tol_default
            # Compare phases
            if j3.global_phase is None:
                # Check if one of the original phases was different than None
                if other.type == 'Jones_vector':
                    cond2 = (self.global_phase is None) * \
                        (S.global_phase is None)
                elif other.type == 'Stokes':
                    cond2 = (self.global_phase is None) * \
                        (other.global_phase is None)
            else:
                print(j3, j3.global_phase)
                cond2 = j3.global_phase == 0
            # Merge conditions
            cond = cond1 * cond2
            new_shape = take_shape((self, other))
            if new_shape is not None:
                cond = np.reshape(cond, new_shape)
            return cond
        except:
            return False

    ################
    ## PROPERTIES
    ################

    @property
    def global_phase(self):
        return self._global_phase

    @global_phase.setter
    def global_phase(self, value):
        if value is None or isinstance(value, number_types):
            self._global_phase = value
        elif value.size == 1:
            self._global_phase = np.ones(self.size) * value
        elif value.size != self.size:
            raise ValueError(
                "Global phase has diffrent size ({}) than object ({})".format(
                    value.size, self.size))
        else:
            self._global_phase = value

    #####################
    ## MANIPULATION
    #####################

    def sum(self, axis=None, keep=False, change_name=change_names):
        """Calculates the sum of Stokes vectors stored in the object.

        Parameters:
            axis (int, list or tuple): Axes along which the sum is performed. If None, all vectors are summed. Default: None.
            keep (bool): If True, the original element is not updated. Default: False.
            change_name (bool): If True, changes the object name adding Sum of at the beggining of the name. Default: True.

        Returns:
            Modified object.
        """
        new_obj, S = create_Stokes(N=2)
        # Simple case
        if axis is not None:
            N_axis = np.array(axis).size
        if axis is None or self.ndim <= 1 or self.ndim == N_axis:
            gp = self.parameters.global_phase(out_number=False, shape=False)
            # Sum all elements
            for ind in range(0, self.size):
                S.from_matrix(self.M[:, ind])
                S.set_global_phase(gp[ind])
                new_obj = new_obj + S
        # Complicated case
        else:
            # Calculate maximum axis
            if isinstance(axis, int):
                m = axis + 1
            else:
                axis = np.array(axis)
                m = np.max(axis) + 1
            # Check that the axes are correct
            if m >= self.ndim + 1:
                raise ValueError(
                    'Axis {} greater than the number of dimensions of {}, which is {}'
                    .format(m, self.name, self.ndim))
            # Calculate shapes, sizes and indices
            if isinstance(axis, int):
                shape_removed = self.shape[axis]
            else:
                shape_removed = np.array(self.shape)[axis]
            N_removed = np.prod(shape_removed)
            ind_removed = combine_indices(
                np.unravel_index(np.array(range(N_removed)), shape_removed))
            shape_matrix = np.delete(self.shape, axis)
            N_matrix = np.prod(shape_matrix)
            ind_matrix = combine_indices(
                np.unravel_index(np.array(range(N_matrix)), shape_matrix))
            shape_final = [4] + list(shape_matrix)
            axes_aux = np.array(range(1, self.ndim + 1))
            shape_orig = [4] + list(self.shape)
            # Prealocate memory
            M_orig = np.reshape(self.M, shape_orig)
            phase_orig = self.parameters.global_phase(out_number=False)
            M = np.zeros(shape_final)
            phase = np.zeros(shape_final)
            # Make the for loop of the matrix to be calculated
            for indM in range(N_matrix):
                # Prepare the Stokes vector to sum
                indices = merge_indices(ind_matrix[indM], ind_removed[0], axis)
                new_obj.from_matrix(multitake(M_orig, indices, axes_aux))
                new_obj.set_global_phase(phase_orig[tuple(indices)])
                # Make the summation loop
                for indR in range(1, N_removed):
                    indices = merge_indices(ind_matrix[indM],
                                            ind_removed[indR], axis)
                    S.from_matrix(multitake(M_orig, indices, axes_aux))
                    S.set_global_phase(phase_orig[tuple(indices)])
                    new_obj = new_obj + S
                # Store the result
                ind_aux = tuple([0] + list(ind_matrix[indM]))
                M[ind_aux] = new_obj.M[0, 0]
                ind_aux = tuple([1] + list(ind_matrix[indM]))
                M[ind_aux] = new_obj.M[1, 0]
                ind_aux = tuple([2] + list(ind_matrix[indM]))
                M[ind_aux] = new_obj.M[2, 0]
                ind_aux = tuple([3] + list(ind_matrix[indM]))
                M[ind_aux] = new_obj.M[3, 0]
                ind_aux = tuple(ind_matrix[indM])
                phase = new_obj.parameters.global_phase()
            # Create the object and return it
            new_obj.from_matrix(M)
            new_obj.set_global_phase(phase)
        if change_names:
            new_obj.name = 'Sum of ' + self.name
        else:
            new_obj.name = self.name
        # Act differently if we want to keep self intact
        if ~keep:
            self = new_obj
        return new_obj

    # @_actualize_
    def rotate(self, angle=0, keep=False, change_name=change_names):
        """Rotates a jones vector a certain angle.

        M_rotated = rotation_matrix_Jones(-angle) * self.M

        Parameters:
            angle (float): Rotation angle in radians. Default: 0.
            keep (bool): If True, the original element is not updated. Default: False.
            change_name (bool): If True and angle is of size 1, changes the object name adding @ XX deg, being XX the total rotation angle. Default: True.

        Returns:
            (Stokes): Rotated object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Prepare variables
        angle, new_obj, new_shape = prepare_variables([angle],
                                                      expand=[True],
                                                      obj=new_obj,
                                                      give_shape=True)
        # Calculate the array of rotation matrices
        Mrot = rotation_matrix_Mueller(-angle)
        # The 1-D case is much simpler. Differenciate it.
        if new_obj.size <= 1 and angle.size == 1:
            S = np.squeeze(Mrot) @ new_obj.M
        else:
            # Move axes of the variables to allow multiplication
            Mrot = np.moveaxis(Mrot, 2, 0)
            S = np.moveaxis(new_obj.M, 1, 0)
            S = np.expand_dims(S, 2)
            # Multiply
            S = Mrot @ S
            # Reshape again to accomodate to our way of representing elements
            S = np.moveaxis(np.squeeze(S), 0, 1)
        # Update
        new_obj.from_matrix(S, global_phase=self.global_phase)
        # Update name if required
        if change_name and angle.size == 1:
            if angle[0] != 0:
                new_obj.name = new_obj.name + \
                    " @ {:1.2f} deg".format(angle[0] / degrees)
                new_obj.name = repair_name(new_obj.name)
        new_obj.shape, _ = select_shape(new_obj, new_shape)
        # Return
        return new_obj

    # @_actualize_
    def set_depolarization(self,
                           degree_pol,
                           degree_depol=None,
                           ratio=np.ones(3),
                           keep=True,
                           change_name=change_names):
        """Function that reduces de polarization degree of a Stokes vector among the three last components.

        Parameters:
            degree_pol (float) [0, 1]: Polarization degree.
            degree_depol (float) [0, 1]: Depolarization degree. Overrides degree_pol if different than None. Default: None.
            ratio (np.array): Ratio between the three components of the depolarization degree. When different from [1, 1, 1]. Default: [1, 1, 1].
            keep (bool): If True, the original element is not updated. Default: True.
            change_name (bool): If True, changes the object name adding Normalized of at the end of the name. Default: True.

        Returns:
            (Stokes): Result.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Transform to polarization degree if required
        if degree_depol is not None:
            degree_pol = np.sqrt(1 - degree_depol**2)
        # Prepare variables
        (degree_pol), new_shape = prepare_variables(vars=[degree_pol],
                                                    expand=[True],
                                                    length=1,
                                                    give_shape=True)
        # Extract the components and the current polarization degree
        S0, S1, S2, S3 = new_obj.parameters.components(shape=False)
        degree_pol_prov = np.sqrt(
            np.abs(ratio[0])**2 * S1**2 + np.abs(ratio[1])**2 * S2**2 +
            np.abs(ratio[2])**2 * S3**2) / S0
        # Calculate the proportionallity coefficients
        c1 = degree_pol * np.abs(ratio[0]) / degree_pol_prov
        c2 = degree_pol * np.abs(ratio[1]) / degree_pol_prov
        c3 = degree_pol * np.abs(ratio[2]) / degree_pol_prov
        # Calculate the object
        new_obj.from_components((S0, S1 * c1, S2 * c2, S3 * c3))
        # End operations
        self.shape, _ = select_shape(self, shape_var=new_shape)
        if change_names:
            new_obj.name = new_obj.name + ' depolarized'
        return new_obj

    def add_depolarization(self,
                           degree_pol,
                           degree_depol=None,
                           ratio=np.ones(3),
                           keep=True,
                           change_name=change_names):
        """Function that reduces de polarization degree of a Stokes vector among the three last components.

        Parameters:
            degree_pol (float) [0, 1]: Polarization degree.
            degree_depol (float) [0, 1]: Depolarization degree. Overrides degree_pol if different than None. Default: None.
            ratio (np.array): Ratio between the three components of the depolarization degree. When different from [1, 1, 1]. Default: [1, 1, 1].
            keep (bool): If True, the original element is not updated. Default: True.
            change_name (bool): If True, changes the object name adding Normalized of at the end of the name. Default: True.

        Returns:
            (Stokes): Result.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Transform to polarization degree if required
        if degree_depol is not None:
            degree_pol = np.sqrt(1 - degree_depol**2)
        # Prepare variables
        (degree_pol), new_shape = prepare_variables(vars=[degree_pol],
                                                    expand=[True],
                                                    length=1,
                                                    give_shape=True)
        # Calculate the new polarization degree
        degree_pol_old = new_obj.parameters.degree_polarization(shape=False)
        degree_pol = degree_pol + degree_pol_old
        degree_pol[degree_pol < 0] = 0
        degree_pol[degree_pol > 1] = 1
        # Set the new polarization degree
        new_obj.set_depolarization(degree_pol=degree_pol)
        # End operations
        self.shape, _ = select_shape(self, shape_var=new_shape)
        if change_names:
            new_obj.name = new_obj.name + ' depolarized'
        return new_obj

    def normalize(self, keep=False, change_name=change_names):
        """Function that normalizes the Stokes vectors to have Intensity = 1.

        Parameters:
            keep (bool): If True, the original element is not updated. Default: False.
            change_name (bool): If True, changes the object name adding Normalized of at the end of the name. Default: True.

        Returns:
            (Stokes): Result.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Calculate the normalized components
        S0, S1, S2, S3 = new_obj.parameters.components(out_number=False)
        S1, S2, S3 = (S1 / S0, S2 / S0, S3 / S0)
        # Avoid dividing by 0
        cond = S0 == 0
        S1[cond] = 0
        S2[cond] = 0
        S3[cond] = 0
        new_obj.from_components((1, S1, S2, S3))
        # End operations
        if change_names:
            new_obj.name = new_obj.name + ' normalized'
        return new_obj

    def add_global_phase(self,
                         phase=0,
                         unknown_as_zero=unknown_phase,
                         keep=False):
        """Function that adds a phase to the Stokes object.

        Parameters:
            phase (float or np.ndarray): Phase to be added to the Stokes vectors. Default: 0.
            unknown_as_zero (bool): If True, takes unknown phase as zero. Default: False.
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Stokes): Recalculated Stokes object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Prepare variables
        phase, new_obj, new_shape = prepare_variables([phase],
                                                      expand=[True],
                                                      obj=new_obj,
                                                      give_shape=True)
        phase = phase.flatten()
        # Add the phase
        if self.global_phase is None:
            if unknown_as_zero:
                self.global_phase = phase
        else:
            self.global_phase = self.global_phase + phase
        # End
        new_obj.shape, _ = select_shape(new_obj, new_shape)
        return new_obj

    def set_global_phase(self,
                         phase=0,
                         keep=False,
                         length=1,
                         shape_like=None,
                         shape=None):
        """Function that sets the phase to the Stokes object.

        Parameters:
            phase (float or np.ndarray): Phase to be added to the Stokes vectors. Default: 0.
            keep (bool): If True, self is not updated. Default: False.
            length (int): If amplitude and azimuth are not specified, it is created a 4 x length array of jones vectors. Default: 1.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Stokes): Recalculated Stokes object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # If None or Nan, skip some steps
        if phase is None or np.all(np.isnan(phase)):
            self.phase = None
        else:
            # Prepare variables
            phase, new_obj, new_shape = prepare_variables([phase],
                                                          expand=[True],
                                                          obj=new_obj,
                                                          give_shape=True,
                                                          length=length)
            phase = phase.flatten()
            new_obj.shape, _ = select_shape(new_obj,
                                            shape_var=new_shape,
                                            shape_fun=shape,
                                            shape_like=shape_like)
            self.global_phase = phase
        # End
        return new_obj

    def remove_global_phase(self, keep=False):
        """Function that removes the phase to the Stokes object.

        Parameters:
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Stokes): Recalculated Stokes object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Set the phase
        self.global_phase = default_phase
        # End
        return new_obj

    def rotate_to_azimuth(self, azimuth=0, keep=False):
        """Function that rotates the Stokes vectors to have a certain azimuth.

        Parameters:
            azimuth (string or np.ndarray): Azimuth of the Stokes vectors. 'X', 'Y', '-X' and '-Y' are the same as 0, 90, 180 and 270 degrees respectively. Default: 0.
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Stokes): Normalized Stokes vectors.
        """
        # Translate the valid strings
        if isinstance(azimuth, str):
            if azimuth in ('X', 'x'):
                azimuth = 0
            elif azimuth in ('Y', 'y'):
                azimuth = 90 * degrees
            elif azimuth in ('-X', '-x'):
                azimuth = 180 * degrees
            elif azimuth in ('-Y', '-y'):
                azimuth = 270 * degrees
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Prepare variables
        azimuth, new_obj = prepare_variables([azimuth],
                                             expand=[True],
                                             obj=new_obj)
        # Calculate the new vector
        I = new_obj.parameters.intensity()
        el = new_obj.parameters.ellipticity_angle()
        new_obj.general_azimuth_ellipticity(azimuth=azimuth,
                                            ellipticity=el,
                                            intensity=I)
        # Return
        return new_obj

    def rotate_Poincare(self, x_angle=0, y_angle=0, z_angle=0, rot_vector=None, angle=None, keep=False):
        """Function to rotate the states in the Poincare sphere around X, Y and Z axes (in that order) if no rotation vector is specified. Rotation angle sign is defined using the right hand rule.

        Args:
            x_angle (int): Angle around axis X. Default: 0.
            y_angle (int): Angle around axis Y. Default: 0.
            z_angle (int): Angle around axis Z. Default: 0.
            rot_vector (iterable): 3xN iterable with the rotation vector. Default: None.
            angle (iterable): rotation angles around the rotation vector. Default: None.
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Stokes): Rotated Stokes vectors.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self

        # First option: rotation around X,Y,Z axes
        if rot_vector is None or angle is None:
            new_obj.rotate_Poincare(angle=x_angle, rot_vector=[1, 0, 0])
            new_obj.rotate_Poincare(angle=y_angle, rot_vector=[0, 1, 0])
            new_obj.rotate_Poincare(angle=z_angle, rot_vector=[0, 0, 1])

        # Defined rotation vector
        else:
            # Prepare variables
            try:
                ux, uy, uz = rot_vector
                sum = np.sqrt(ux**2 + uy**2 + uz**2)
                ux = ux / sum
                uy = uy / sum
                uz = uz / sum
            except:
                raise ValueError("rot_vector must be an iterable with 3 elements in the first dimension")
            S0, S1, S2, S3 = self.parameters.components(out_number=False, shape=False)
            (angle, S1, S2, S3, ux, uy, uz), new_shape = prepare_variables(
                vars=[angle, S1, S2, S3, ux, uy, uz], expand=[True] * 7, length=1, give_shape=True)
            angle = angle.flatten()
            one = 1 if S0.size == 1 else np.ones_like(S0)
        
            # Rotate 
            S_array =  np.array([S1, S2, S3]).T[:, :, np.newaxis]
            sinA = np.sin(angle)
            cosA = np.cos(angle)
            rot = np.array([
                [cosA + (one - cosA)*ux**2, ux*uy * (one - cosA) - uz*sinA, ux*uz * (one - cosA) + uy*sinA], 
                [ux*uy * (one - cosA) + uz*sinA, cosA + (one - cosA)*uy**2, uy*uz * (one - cosA) - ux*sinA],
                [ux*uz * (one - cosA) - uy*sinA, uz*uy * (one - cosA) + ux*sinA, cosA + (one - cosA)*uz**2],
            ])
            rot = np.moveaxis(rot, -1, 0)
            S_array = rot @ S_array
            # Recreate S
            new_obj.from_components([S0, S_array[:, 0, 0], S_array[:, 1, 0], S_array[:, 2, 0]])
            # End
            new_obj.shape, _ = select_shape(new_obj, new_shape)

        return new_obj

    def rotate_Poincare_orig(self, x_angle=0, y_angle=0, z_angle=0, angles=None, keep=False):
        """Function to rotate the states in the Poincare sphere around X, Y and Z axes (in that order). Rotation angle sign is defined using the right hand rule.

        Args:
            x_angle (int): Angle around axis X. Default: 0.
            y_angle (int): Angle around axis Y. Default: 0.
            z_angle (int): Angle around axis Z. Default: 0.
            angles (iterable): 3xN iterable which overrides x_angle, y_angle and z_angle. Default: None.
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Stokes): Rotated Stokes vectors.
        """
        # Use angles in the same variable if allowed
        if angles is not None:
            try:
                x_angle, y_angle, z_angle = angles
            except:
                raise ValueError("Angles must be an iterable with 3 elements in the first dimension")
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Prepare variables
        S0, S1, S2, S3 = self.parameters.components(out_number=False, shape=False)
        (x_angle, y_angle, z_angle, S1, S2, S3), new_shape = prepare_variables(
            vars=[x_angle, y_angle, z_angle, S1, S2, S3], expand=[True] * 6, length=1, give_shape=True)
        x_angle = x_angle.flatten()
        y_angle = y_angle.flatten()
        z_angle = z_angle.flatten()
        zero = 0 if S0.size == 1 else np.zeros_like(S0)
        one = 1 if S0.size == 1 else np.ones_like(S0)
        # Rotate X
        S_array =  np.array([S1, S2, S3]).T[:, :, np.newaxis]
        # S_array =  np.array([S1, S2, S3])
        # if S0.size > 1:
        #     S_array = S_array.T[:, :, np.newaxis]
        sinA = np.sin(x_angle)
        cosA = np.cos(x_angle)
        rot = np.array([[one, zero, zero], [zero, cosA, -sinA], [zero, sinA, cosA]])
        rot = np.moveaxis(rot, -1, 0)
        # print(type(S0), S0)
        # print("S_array", S_array.shape)
        # print("rot", rot.shape)
        S_array = rot @ S_array
        # Rotate Y
        sinA = np.sin(y_angle)
        cosA = np.cos(y_angle)
        rot = np.array([[cosA, zero, sinA], [zero, one, zero], [-sinA, zero, cosA]])
        rot = np.moveaxis(rot, -1, 0)
        S_array = rot @ S_array
        # Rotate Z
        sinA = np.sin(z_angle)
        cosA = np.cos(z_angle)
        rot = np.array([[cosA, -sinA, zero], [sinA, cosA, zero], [zero, zero, one]])
        rot = np.moveaxis(rot, -1, 0)
        S_array = rot @ S_array
        # Recreate S
        # print(S_array.shape)
        # print(S_array[0,:])
        new_obj.from_components([S0, S_array[:, 0, 0], S_array[:, 1, 0], S_array[:, 2, 0]])
        # End
        new_obj.shape, _ = select_shape(new_obj, new_shape)
        return new_obj


    ###########################################################################
    # Creation
    ###########################################################################

    def from_components(self,
                        components,
                        degree_pol=1,
                        degree_depol=None,
                        global_phase=default_phase,
                        length=1,
                        shape_like=None,
                        shape=None):
        """Creates Stokes vectors directly from the 4 elements [s0, s1, s2, s3]

        Parameters:
            components (tuple): A 4 element tuple containing the 4 components of the Stokes vectors (S0, S1, S2, S3).
            degree_pol (float) [0, 1]: Polarization degree. Default: 1
            degree_depol (float) [0, 1]: Depolarization degree. Overrides degree_pol if different than None. Default: None.
            global_phase (numpy.ndarray): Adds a global phase to the Stokes object. Default: 0.
            length (int): If amplitude and azimuth are not specified, it is created a 4 x length array of stokes vectors. Default: 1.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Stokes): Created object.
        """
        # Transform to polarization degree if required
        if degree_depol is not None:
            degree_pol = np.sqrt(1 - degree_depol**2)
        # Prepare variables
        S0, S1, S2, S3 = components
        (S0, S1, S2, S3, degree_pol), new_shape = prepare_variables(
            vars=[S0, S1, S2, S3, degree_pol],
            expand=[True, True, True, True, False],
            length=length,
            give_shape=True)
        # Store
        # print("From_matrix", S0.shape, (degree_pol * S1).shape, (degree_pol * S2).shape, (degree_pol * S3).shape)
        if S0.size == 1:
            self.M = np.array(
                [S0, degree_pol * S1, degree_pol * S2, degree_pol * S3], dtype=float)
        else:
            self.M = np.array(
                [S0, degree_pol * S1, degree_pol * S2, degree_pol * S3])
        # print(self.M.shape)
        # self.size = S0.size
        self.shape, _ = select_shape(self,
                                     shape_var=new_shape,
                                     shape_fun=shape,
                                     shape_like=shape_like)
        self.set_global_phase(global_phase)
        return self

    def from_matrix(self,
                    M,
                    degree_pol=1,
                    degree_depol=None,
                    global_phase=default_phase,
                    shape_like=None,
                    shape=None):
        """Creates a Stokes object from an external array.

        Parameters:
            M (numpy.ndarray): New matrix. At least one dimension must be of size 4.
            degree_pol (float) [0, 1]: Polarization degree. Default: 1
            degree_depol (float) [0, 1]: Depolarization degree. Overrides degree_pol if different than None. Default: None.
            global_phase (numpy.ndarray): Adds a global phase to the Stokes object. Default: 0.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Stokes): Created object.
        """
        # Check if the matrix is of the correct Size
        M = np.array(M)
        s = M.size
        # Case 1D
        if M.ndim == 1:
            if M.size % 4 == 0:
                M = np.reshape(M, (4, int(M.size / 4)))
                new_shape = None
            else:
                raise ValueError(
                    'M must have a number of elements multiple off 4.')
        # Case 2D
        elif M.ndim == 2:
            s = M.shape
            if s[0] != 4:
                if s[1] == 4:
                    M = M.transpose()
                    new_shape = [s[0]]
                else:
                    raise ValueError(
                        'The Stokes vectors must be a 4xN or Nx4 array. Current shape is {}'
                        .format(s))
            else:
                new_shape = [s[1]]
        # Case 3+D
        else:
            sh = np.array(M.shape)
            if (sh == 4).any:
                # Store the shape of the desired outputs
                ind = np.argmin(~(sh == 4))
                # Store info
                M = np.array([
                    np.take(M, 0, axis=ind).flatten(),
                    np.take(M, 1, axis=ind).flatten(),
                    np.take(M, 2, axis=ind).flatten(),
                    np.take(M, 3, axis=ind).flatten()
                ])
                new_shape = np.delete(sh, ind)
            else:
                raise ValueError(
                    'The matrix must have one axis with exactly 4 elements')
        self.M = M
        # self.size = M.size / 4
        # End operations
        self.shape, _ = select_shape(self,
                                     shape_var=new_shape,
                                     shape_fun=shape,
                                     shape_like=shape_like)
        # Add global phase and depolarization
        self.set_global_phase(global_phase)
        self.set_depolarization(degree_pol=degree_pol,
                                degree_depol=degree_depol)
        return self

    def from_distribution(self,
                          Ex,
                          Ey,
                          ind_d=-1,
                          method='direct',
                          N_periods=1,
                          shape_like=None,
                          shape=None):
        """Determine the Stokes vectors from a temporal or spatial electric field distribution [(Ex(t), Ey(t)].

        Parameters:
            Ex (numpy.ndarray or float): X component of the electric field.
            Ey (numpy.ndarray or float): Y component of the electric field.
            ind_d (int): Index of the spatial or temporal dimension. Default: -1.
            method (string): Method for calculating the field amplitude and delay: DIRECT or FIT. Default: direct.
            N_periods (float): Number of periods in the representation data. It is used by the fit algorithm (real case only) for calculating the frequency. If the value is not exact, convergency may decrease. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_vector): Created object.
        """
        # Measure sizes, dims, etc
        Ex = np.array(Ex)
        Ey = np.array(Ey)
        dimsX, dimsY = (Ex.ndim, Ey.ndim)
        dims = max(dimsX, dimsY)
        shapeX, shapeY = (np.array(Ex.shape), np.array(Ey.shape))
        Nx, Ny = (Ex.size, Ey.size)
        new_shape = None
        # Check that the data is compatible
        if Nx != Ny and dimsX != 1 and dimsY != 1:
            raise ValueError(
                'Ex and Ey components must have the same number of elements')
        elif Nx != Ny and dimsX == 1 and dimsY == 1:
            raise ValueError(
                'Ex and Ey components must have the same number of elements')
        elif dimsX == 1 and not (shapeY[ind_d] == Nx):
            raise ValueError(
                'Number of elements of X and Y components does not match')
        elif dimsY == 1 and not (shapeX[ind_d] == Ny):
            raise ValueError(
                'Number of elements of X and Y components does not match')
        elif shapeX[ind_d] != shapeY[ind_d]:
            raise ValueError(
                'Temporal dimension of X and Y components does not match')
        # Check if the variables are real or complex
        complex_X = Ex.dtype == np.dtype('complex128')
        complex_Y = Ey.dtype == np.dtype('complex128')
        if not complex_X == complex_Y:
            raise ValueError(
                'Both Ex and Ey must be in the same representation type: real or complex'
            )

        # Prepare the data as a 2D arrays
        if dimsX != 1:
            Nt = shapeX[ind_d]
            NelemX = int(Nx / Nt)
            if complex_X:
                ex = np.zeros((Nt, NelemX), dtype=complex)
            else:
                ex = np.zeros((Nt, NelemX))
            for ind in range(Nt):
                ex[ind, :] = np.take(Ex, ind, axis=ind_d).flatten()
            new_shape = np.delete(shapeX, ind_d)
        else:
            Nt = Nx
            NelemX = 1
            ex = np.reshape(Ex, (Nt, 1))

        if dimsY != 1:
            NelemY = int(Ny / Nt)
            if complex_X:
                ey = np.zeros((Nt, NelemY), dtype=complex)
            else:
                ey = np.zeros((Nt, NelemY))
            for ind in range(Nt):
                ey[ind, :] = np.take(Ey, ind, axis=ind_d).flatten()
            if dimsY > dimsX:
                new_shape = np.delete(shapeY, ind_d)
        else:
            ey = np.reshape(Ey, (Nt, 1))
            NelemY = 1

        # Calculate
        if complex_X:
            # Amplitude here is easy to extract
            if method in ('fit', 'FIT', 'Fit'):
                # Use the mean for polarized amplitude
                Ex = np.abs(ex).mean(axis=0)
                Ey = np.abs(ey).mean(axis=0)
                # Use variance for unpolarized intensity
                dIx = np.abs(ex).var(axis=0)
                dIy = np.abs(ey).var(axis=0)
                # Fit the phase
                phaseX = np.unwrap(np.angle(ex), axis=0)
                phaseY = np.unwrap(np.angle(ey), axis=0)
                x = np.arange(ex.shape[0])
                _, phaseX = np.polyfit(x=x, y=phaseX, deg=1)
                _, phaseY = np.polyfit(x=x, y=phaseY, deg=1)
                # Calculate the components
                S0 = Ex**2 + Ey**2 + (dIx + dIy) / 2
                S1 = Ex**2 - Ey**2
                S2 = 2 * Ex * Ey * np.cos(phaseY - phaseX)
                S3 = -2 * Ex * Ey * np.sin(phaseY - phaseX)
            else:
                # Use the average values
                S0 = np.mean(np.abs(ex)**2, axis=0).real + \
                    np.mean(np.abs(ey)**2, axis=0).real
                S1 = np.mean(np.abs(ex)**2, axis=0).real - \
                    np.mean(np.abs(ey)**2, axis=0).real
                S2 = 2 * np.mean(ex * ey.conjugate(), axis=0).real
                S3 = 2 * np.mean(ex * ey.conjugate(), axis=0).imag
                phaseX = np.angle(ex[0, :])
        else:
            # Real evolution according to cos(k*x) or cos(w*t). Start by calculating the direct values
            Ex = np.abs(ex).max(axis=0)
            Ey = np.abs(ey).max(axis=0)
            phaseX = np.arccos(ex[0, :] / Ex)
            cond = ex[1, :] > ex[0, :]
            phaseX[cond] = np.pi * 2 - phaseX[cond]
            phaseY = np.arccos(ey[0, :] / Ey)
            cond = ey[1, :] > ey[0, :]
            phaseY[cond] = np.pi * 2 - phaseY[cond]
            # If using fit, use those values as first guess
            if method in ('fit', 'FIT', 'Fit'):
                # Prepare variables for X fit
                s = ex.shape[1]
                zero, one = (np.zeros(s), np.ones(s))
                cte = 2 * np.pi * N_periods / ex.shape[0]
                par0 = np.concatenate((phaseX, one * cte, Ex))
                min_limit = np.concatenate((zero, zero, zero))
                max_limit = np.concatenate((2 * np.pi * one, np.pi * one, Ex))
                # X fit
                result = optimize.least_squares(fit_distribution,
                                                par0,
                                                args=(ex, 0),
                                                bounds=(min_limit, max_limit))
                phaseX = result.x[:s]
                Ex = result.x[2 * s:]
                dIx = np.reshape(result.fun, ex.shape).var(axis=0)
                # prepare variables for Y fit
                par0 = np.concatenate((phaseY, one * cte, Ey))
                max_limit = np.concatenate((2 * np.pi * one, np.pi * one, Ey))
                # Y fit
                result = optimize.least_squares(fit_distribution,
                                                par0,
                                                args=(ey, 0),
                                                bounds=(min_limit, max_limit))
                phaseY = result.x[:s]
                Ey = result.x[2 * s:]
                dIy = np.reshape(result.fun, ey.shape).var(axis=0)
            else:
                # Without fit, it is impossible to asses the unpolarized part
                dIx, dIy = (0, 0)
            # Calculate the components
            S0 = Ex**2 + Ey**2 + (dIx + dIy) / 2
            S1 = Ex**2 - Ey**2
            S2 = 2 * Ex * Ey * np.cos(phaseY - phaseX)
            S3 = -2 * Ex * Ey * np.sin(phaseY - phaseX)
        # Create the object
        self.from_components((S0, S1, S2, S3), global_phase=phaseX)
        self.shape, _ = select_shape(self,
                                     shape_var=new_shape,
                                     shape_fun=shape,
                                     shape_like=shape_like)
        return self

    def from_Jones(self,
                   E,
                   degree_pol=1,
                   degree_depol=None,
                   length=1,
                   shape_like=None,
                   shape=None):
        """Creates a Stokes object from a Jones vectors object or a matrix corresponding to a Jones vectors.

        .. math:: s_0 = abs(E_x)^2 + abs(E_y)^2

        .. math:: s_1 = (abs(E_x)^2 - abs(E_y)^2)   p_1

        .. math:: s_2 = 2  real(E_x  E_y^*)   p_1

        .. math:: s_3 = -2  imag(E_x  E_y^*)   p_2


        Parameters:
            E (Jones_vector object): Jones vectors.
            degree_pol (float or numpy.ndarray): Degree of polarization of the new Stokes vector. Default: 1.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Stokes): Created object.
        """
        # Transform to polarization degree if required
        if degree_depol is not None:
            degree_pol = np.sqrt(1 - degree_depol**2)
        # Prepare variables
        (degree_pol), new_shape = prepare_variables(vars=[degree_pol],
                                                    expand=[False],
                                                    length=max(length, E.size),
                                                    give_shape=True)
        # Calculate electric field components
        Ex, Ey = E.parameters.components(shape=False)
        global_phase = E.parameters.global_phase(shape=False)
        # Calculate the Stokes vector
        S0 = E.parameters.intensity(shape=False)
        S1 = degree_pol * (np.abs(Ex)**2 - np.abs(Ey)**2)
        S2 = 2 * degree_pol * np.real(Ex * np.conj(Ey))
        S3 = -2 * degree_pol * np.imag(Ex * np.conj(Ey))
        self.from_components((S0, S1, S2, S3), global_phase=global_phase)
        # Select final shape
        self.shape, _ = select_shape(self, shape_var=new_shape, shape_like=E)
        self.shape, _ = select_shape(self,
                                     shape_var=E.shape,
                                     shape_fun=shape,
                                     shape_like=shape_like)
        # Return
        return self

    # @_actualize_
    # def from_distribution(self, E, is_normalized=False):
    #     """Creates Stokes vectors from a [Ex(t), Ey(t)] electric field.
    #
    #     Parameters:
    #         E (numpy.array): [Ex(t), Ey(t)]
    #         is_normalized (bool): If True intensity is normalized
    #
    #     Returns:
    #         S (4x1 numpy.matrix): Stokes vector (I, Q, U, V).
    #     """
    #
    #     Ex, Ey = E[:, 0], E[:, 1]
    #
    #     S = np.matrix(np.array([[0.0], [0.0], [0.0], [0.0]]))
    #     S[0] = (np.conjugate(Ex) * Ex + np.conjugate(Ey) * Ey).mean().real
    #     S[1] = (np.conjugate(Ex) * Ex - np.conjugate(Ey) * Ey).mean().real
    #     S[2] = 2 * (Ex * np.conjugate(Ey)).mean().real
    #     # S[2] = (Ex * np.conjugate(Ey)+Ey*np.conjugate(Ex)).mean().real
    #     S[3] = -2 * (Ex * np.conjugate(Ey)).mean().imag
    #     # S[3] = (1j*(Ex * np.conjugate(Ey)-Ey*np.conjugate(Ex)).mean()).real
    #
    #     if is_normalized:
    #         S = S / S[0]
    #
    #     self.from_matrix(S)
    #
    # # @_actualize_
    # def from_distribution_deprecated(self, E, is_normalized=False):
    #     """Creates Stokes vectors from a [Ex(t), Ey(t)] electric field.
    #
    #     Parameters:
    #         E (numpy.array): [Ex(t), Ey(t)]
    #         is_normalized (bool): If True intensity is normalized
    #
    #     Returns:
    #         S (4x1 numpy.matrix): Stokes vector (I, Q, U, V).
    #     """
    #
    #     Ex, Ey = E[0], E[1]
    #
    #     if is_normalized is True:
    #         intensity = np.sqrt(np.conjugate(Ex) * Ex + np.conjugate(Ey) * Ey)
    #         Ex = Ex / intensity
    #         Ey = Ey / intensity
    #
    #     S = np.matrix(np.array([[0.0], [0.0], [0.0], [0.0]]))
    #     S[0] = (np.conjugate(Ex) * Ex + np.conjugate(Ey) * Ey).mean.real
    #     S[1] = (np.conjugate(Ex) * Ex - np.conjugate(Ey) * Ey).real
    #     S[2] = 2 * (Ex * np.conjugate(Ey)).real
    #     S[3] = 2 * (Ex * np.conjugate(Ey)).imag
    #     self.from_matrix(S)

    # # @_actualize_
    # def to_Jones(self):
    #     """Function that converts Stokes light states to Jones states.
    #
    #     Returns:
    #         j (Jones_vector object): Stokes state."""
    #     j = Jones_vector(self.name)
    #     j.from_Stokes(self)
    #     return j

    def linear_light(self,
                     intensity=1,
                     azimuth=0,
                     amplitude=None,
                     degree_pol=1,
                     degree_depol=None,
                     global_phase=default_phase,
                     length=1,
                     shape_like=None,
                     shape=None):
        """Creates a Stokes vector of linear polarizer light.

        Parameters:
            intensity (numpy.array or float): Array of intensity. Default: 1.
            azimuth (numpy.array or float): Array of azimuths. Default: 0.
            amplitude (numpy.array or float): Array of electric field amplitude. Overrides intensity if it is different than None. Default: None.
            degree_pol (numpy.array or float): Array of polarization degree. Default: 1.
            degree_depol (float) [0, 1]: Depolarization degree. Overrides degree_pol if different than None. Default: None.
            global_phase (float or numpy.ndarray): Adds a global phase to the Stokes object. Default: default_phase.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Stokes): Created object.
        """
        # Transform to polarization degree if required
        if degree_depol is not None:
            degree_pol = np.sqrt(1 - degree_depol**2)
        # Use amplitude if specified
        if amplitude is not None:
            intensity = amplitude**2
        # Prepare variables
        (intensity, azimuth, degree_pol), new_shape = prepare_variables(
            vars=[intensity, azimuth, degree_pol],
            expand=[True, False, False],
            length=length,
            give_shape=True)
        # Calculate the components
        S0 = intensity
        S1 = intensity * degree_pol * cos(2 * azimuth)
        S2 = intensity * degree_pol * sin(2 * azimuth)
        S3 = np.zeros_like(intensity)
        # Calculate the vector
        self.from_components((S0, S1, S2, S3), global_phase=global_phase)
        # Store
        self.shape, _ = select_shape(self,
                                     shape_var=new_shape,
                                     shape_fun=shape,
                                     shape_like=shape_like)
        return self

    # @_actualize_
    def circular_light(self,
                       kind='d',
                       intensity=1,
                       amplitude=None,
                       degree_pol=1,
                       degree_depol=None,
                       global_phase=default_phase,
                       length=1,
                       shape_like=None,
                       shape=None):
        """Creates Stokes vectors for pure circular polarizer light.

        Parameters:
            kind (str): Can be 'd','r', 'right', 'dextro' or 'l', 'i', 'left', 'levo'. Default: 'r'.
            intensity (numpy.array or float): Array of intensity. Default: 1.
            amplitude (numpy.array or float): Array of electric field amplitude. Overrides inetnsity if it is different than None. Default: None.
            degree_pol (numpy.array or float): Array of polarization degree. Default: 1.
            degree_depol (float) [0, 1]: Depolarization degree. Overrides degree_pol if different than None. Default: None.
            global_phase (float or numpy.ndarray): Adds a global phase to the Stokes object. Default: default_phase.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Stokes): Created object.
        """
        # Use the correct sign
        if kind.lower() in ["r", "d", "right", "dextro"]:
            sign = 1
        else:
            sign = -1
        # Transform to polarization degree if required
        if degree_depol is not None:
            degree_pol = np.sqrt(1 - degree_depol**2)
        # Use amplitude if specified
        if amplitude is not None:
            intensity = amplitude**2
        # Prepare variables
        (intensity, degree_pol), new_shape = prepare_variables(
            vars=[intensity, degree_pol],
            expand=[True, False],
            length=length,
            give_shape=True)
        # Calculate the vector
        self.from_components((intensity, 0, 0, sign * degree_pol * intensity),
                             global_phase=global_phase)
        # Store
        self.shape, _ = select_shape(self,
                                     shape_var=new_shape,
                                     shape_fun=shape,
                                     shape_like=shape_like)
        return self

    # @_actualize_
    def elliptical_light(self,
                         a=1,
                         b=1,
                         kind='r',
                         azimuth=0,
                         degree_pol=1,
                         degree_depol=None,
                         global_phase=default_phase,
                         length=1,
                         shape_like=None,
                         shape=None):
        """Stokes object of the most general light calculated from the polarization ellipse parameters.

        Parameters:
            a (numpy.array or float): Array of electric amplitude of x axis. Default: 1.
            b (numpy.array or float): Array of electric amplitude of y axis. Default: 1.
            kind (str): Can be 'd','r', 'right', 'dextro' or 'l', 'i', 'left', 'levo'. Default: 'r'.
            azimuth (numpy.array or float): Angle of the a axis respect to the x axis. Default: 0.
            degree_pol (float) [0, 1]: Polarization degree. Default: 1
            degree_depol (float) [0, 1]: Depolarization degree. Overrides degree_pol if different than None. Default: None.
            global_phase (float or numpy.ndarray): Adds a global phase to the Stokes object. Default: default_phase.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Stokes): Created object.
        """
        # Transform to polarization degree if required
        if degree_depol is not None:
            degree_pol = np.sqrt(1 - degree_depol**2)
        # Calculate it as Stokes vectors (much easier)
        J = Jones_vector()
        J.elliptical_light(a=a,
                           b=b,
                           kind=kind,
                           azimuth=azimuth,
                           global_phase=global_phase,
                           length=length,
                           shape_like=shape_like,
                           shape=shape)
        # Transform it to Stokes
        self.from_Jones(J)
        # Depolarize
        self.set_depolarization(degree_pol=degree_pol)
        return self

    # @_actualize_
    def general(self,
                azimuth=0,
                ellipticity=0,
                alpha=None,
                delay=None,
                *args,
                **kwargs):
        """Create the most general Stokes vector.

        Parameters:
            azimuth (numpy.array or float): [0, pi]: Azimuth. Only used if alpha and delay are None. Default: 0.
            ellipticity (numpy.array or float): [-pi/4, pi/4]: Ellipticity angle. Only used if alpha and delay are None. Default: 0.
            alpha (numpy.ndarray, float or none): Ratio between amplitude of components Ex/Ey of electric field. Overrides azimuth and ellipticity if not None. Default: None.
            delay (numpy.ndarray, float or None): Phase delay between Ex and Ey components of the electric field. Overrides azimuth and ellipticity if not None. Default: None.
            amplitude (numpy.array or float): Array of electric field amplitude. Overrides inetnsity if it is different than None. Default: None.
            intensity (numpy.array or float): Array of intensity. Default: 1.
            amplitude (numpy.array or float): Array of electric field amplitude. Overrides intensity if it is different than None. Default: None.
            degree_pol (numpy.array or float): Array of polarization degree. Default: 1.
            degree_depol (float) [0, 1]: Depolarization degree. Overrides degree_pol if different than None. Default: None.
            global_phase (float or numpy.ndarray): Adds a global phase to the Stokes object. Default: default_phase.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of azimuth and ellipticity or alpha and delay are forced to be in their correct range. Default: True.

        Returns:
            S (4x1 numpy.matrix): Stokes vector.
        """
        if alpha is None or delay is None:
            return self.general_azimuth_ellipticity(
                azimuth=azimuth,
                ellipticity=ellipticity,
                *args,
                **kwargs)
        else:
            return self.general_charac_angles(
                alpha=alpha,
                delay=delay,
                *args,
                **kwargs)

    # @_actualize_
    def general_azimuth_ellipticity(self,
                                    azimuth=0,
                                    ellipticity=0,
                                    intensity=1,
                                    amplitude=None,
                                    degree_pol=1,
                                    degree_depol=None,
                                    global_phase=default_phase,
                                    length=1,
                                    shape_like=None,
                                    shape=None,
                                    force_limits=False):
        """Creates Stokes vectors given by their azimuth and ellipticity.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 137.

        Parameters:
            azimuth (float): [0, pi]: azimuth. Default: 0
            ellipticity (float): [-pi/4, pi/4]: ellipticity. Default: 0
            intensity (numpy.array or float): Array of intensity. Default: 1.
            amplitude (numpy.array or float): Array of electric field amplitude. Overrides intensity if it is different than None. Default: None.
            degree_pol (numpy.array or float): Array of polarization degree. Default: 1.
            degree_depol (float) [0, 1]: Depolarization degree. Overrides degree_pol if different than None. Default: None.
            global_phase (float or numpy.ndarray): Adds a global phase to the Stokes object. Default: default_phase.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of azimuth and ellipticity are forced to be in the range [0, pi) and [-pi/4, pi/4] respectively. Default: False.

        Returns:
            S (4x1 numpy.matrix): Stokes vector.
        """
        # Transform to polarization degree if required
        if degree_depol is not None:
            degree_pol = np.sqrt(1 - degree_depol**2)
        # Use amplitude if specified
        if amplitude is not None:
            intensity = amplitude**2
        # Prepare variables
        (intensity, azimuth, ellipticity, degree_pol,
         global_phase), new_shape = prepare_variables(
             vars=[intensity, azimuth, ellipticity, degree_pol, global_phase],
             expand=[True, False, False, False, False],
             length=length,
             give_shape=True)
        # Restrict possible values
        if force_limits:
            azimuth = put_in_limits(azimuth, 'azimuth')
            ellipticity = put_in_limits(ellipticity, 'ellipticity')
        # Calculate the components
        S0 = intensity
        S1 = intensity * degree_pol * cos(2 * azimuth) * cos(2 * ellipticity)
        S2 = intensity * degree_pol * sin(2 * azimuth) * cos(2 * ellipticity)
        S3 = intensity * degree_pol * sin(2 * ellipticity)
        # Separate totally depolarization from circular polarization
        cond = np.isnan(azimuth) + np.isnan(ellipticity)
        if np.any(cond):
            S1[cond] = 0
            S2[cond] = 0
        cond = np.isnan(ellipticity) + (
            np.isnan(azimuth) *
            (np.abs(ellipticity) < np.pi / 4 - tol_default))
        if np.any(cond):
            S3[cond] = 0
        # From components
        self.from_components((S0, S1, S2, S3),
                             global_phase=global_phase,
                             length=length,
                             shape=False)
        # Store
        self.shape, _ = select_shape(self,
                                     shape_var=new_shape,
                                     shape_fun=shape,
                                     shape_like=shape_like)
        return self

    # @_actualize_
    def general_charac_angles(self,
                              alpha=0,
                              delay=0,
                              intensity=1,
                              amplitude=None,
                              degree_pol=1,
                              degree_depol=None,
                              global_phase=default_phase,
                              length=1,
                              shape_like=None,
                              shape=None,
                              force_limits=False):
        """Creates Stokes vectors given by their characteristic angles.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016),pp 137.

        Parameters:
            alpha (float): [0, pi]: tan(alpha) is the ratio between field amplitudes of X and Y components. Default: 0
            delay (float): [0, 2*pi]: phase difference between X and Y field components. Default: 0
            intensity (numpy.array or float): Array of intensity. Default: 1.
            amplitude (numpy.array or float): Array of electric field amplitude. Overrides intensity if it is different than None. Default: None.
            degree_pol (numpy.array or float): Array of polarization degree. Default: 1.
            degree_depol (float) [0, 1]: Depolarization degree. Overrides degree_pol if different than None. Default: None.
            global_phase (float or numpy.ndarray): Adds a global phase to the Stokes object. Default: default_phase.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of alpha and delay are forced to be in the range [0, pi/2] and [0, 2*pi) respectively. Default: False.

        Returns:
            S (4x1 numpy.matrix): Stokes vector.
        """
        # Transform to polarization degree if required
        if degree_depol is not None:
            degree_pol = np.sqrt(1 - degree_depol**2)
        # Use amplitude if specified
        if amplitude is not None:
            intensity = amplitude**2
        # Prepare variables
        (intensity, alpha, delay, degree_pol,
         global_phase), new_shape = prepare_variables(
             vars=[intensity, alpha, delay, degree_pol, global_phase],
             expand=[False, False, False, False, False],
             length=length,
             give_shape=True)
        # Restrict possible values
        if force_limits:
            alpha = put_in_limits(alpha, 'alpha')
            delay = put_in_limits(delay, 'delay')
        # Calculate the components
        S0 = intensity
        S1 = intensity * degree_pol * cos(2 * alpha)
        S2 = intensity * degree_pol * sin(2 * alpha) * cos(delay)
        S3 = intensity * degree_pol * sin(2 * alpha) * sin(delay)
        # Nan cases are totally depolarized
        cond = np.isnan(S3)
        if np.any(cond):
            S1[cond] = 0
            S2[cond] = 0
            S3[cond] = 0
        # From components
        self.from_components((S0, S1, S2, S3),
                             global_phase=global_phase,
                             length=length,
                             shape=False)
        # Store
        self.shape, _ = select_shape(self,
                                     shape_var=new_shape,
                                     shape_fun=shape,
                                     shape_like=shape_like)
        return self

    #######################################################################
    # Draw
    #######################################################################

    def draw_poincare(self, *args, **kwargs):
        """"Draws stokes vector.

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
            hover (str): Choose between 'components' (S1, S2, S3) or 'angles' (azimuth, ellipticity). Default: 'components'.
            colormap (str): Colormap of the plots. Default: 'Blackbody'.
            show_fig (bool): If True, the figure is inmediately plotted. Default: False.

        Returns:
            fig (go.Figure): Plotly figure.
        """
        fig = draw_poincare(self, *args, **kwargs)
        return fig

    def draw_ellipse(self, *args, **kwargs):
        """Draws polarization ellipse of Stokes vector.

        Args:
            E (Jones_vector or Stokes): Light object.
            N_angles (int): Number of angles to plot the ellipses. Default: 91.
            filename (str): name of filename to save the figure.
            figsize (tuple): A tuple of length 2 containing the figure size. Default: (8,8).
            limit (float): limit for drawing. If empty, it is obtained from amplitudes.
            draw_arrow (bool): If True, draws an arrow containing the turning sense of the polarization. Does not work with linear polarization vectors. Default: True.
            depol_central (bool): If True, draws a central circle containing the unpolarized field amplitude. Default: False.
            depol_contour (bool): If True, draws a line enveloping the polarization ellipse in order to plot the depolarization. Default: False.
            depol_prob (bool): If True, plots the probability distribution of the electric field. Default: False.
            subplots (string, tuple or None): If AS_SHAPE, divides the figure in several subplots as the shape of the py_pol object. If INDIVIDUAL, each vector is represented in its own subaxis, trying to use a square grid. If tuple, divides the figure in that same number of subplots. If None, all ellipses are plot in the same axes. Default: None.
            N_prob (int): Number of points in each dimension for probability distributions. Default: 256.
            contour_levels (float, np.ndarray, tuple or list): Contains the contour levels (normalized to 1). Default: 0.9.
            cmap (str or color object): Default colormap for probability distributions. Default: hot.

        Returns:
            ax (handle): handle to axis.
            fig (handle): handle to figure.
        """
        ax, fig = draw_ellipse(self, *args, **kwargs)
        return ax, fig

        # ax, fig = draw_ellipse_stokes(self, kind, limit, has_line, filename)
        # return ax, fig

    # def pseudoinverse(self, returns_matrix=True, keep=True):
    #     """Calculates the pseudoinverse of the Stokes vector.
    #     TODO
    #
    #     Parameters:
    #         keep (bool): if True, the original element is not updated. Default: False.
    #         returns_matrix (bool): if True returns a matrix, else returns an instance to object. Default: True.
    #
    #     Returns:
    #         (numpy.matrix or Mueller object): 4x4 matrix.
    #     """
    # # Calculate pseudoinverse
    # S = np.matrix(self.M)
    # Sinv = (S.T * S).I * S.T
    #
    # # Caluclate inverse
    # if keep:
    #     S2 = Stokes(self.name + '_inv')
    #     S2.from_matrix(Sinv)
    # else:
    #     self.from_matrix(Sinv)
    # if returns_matrix:
    #     return Sinv
    # else:
    #     if keep:
    #         return S2
    #     else:
    #         return self


class Parameters_Stokes_vector(object):
    """Class for Stokes vector Parameters

    Parameters:
        Stokes_vector (Stokes_vector): Stokes Vector

    Attributes:
        self.parent (Stokes_vector): parent object.
    """

    def __init__(self, Stokes):
        self.parent = Stokes

    def __repr__(self):
        """Prints all the parameters."""
        self.get_all(verbose=True, draw=True)
        return ''

    def get_all(self, verbose=False, draw=False):
        """Creates a dictionary with all the parameters of Stokes vectors.

        Parameters:
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.
        """
        dict_params = {}
        dict_params['intensity'] = self.intensity(verbose=verbose, draw=draw)
        dict_params['E0x'], dict_params['E0y'], dict_params[
            'E0u'] = self.amplitudes(verbose=verbose,
                                     draw=draw,
                                     give_unpol=True)
        dict_params['global_phase'] = self.global_phase(verbose=verbose,
                                                        draw=draw)
        dict_params['degree_depol'] = self.degree_depolarization(
            verbose=verbose, draw=draw)
        dict_params['degree_pol'] = self.degree_polarization(verbose=verbose,
                                                             draw=draw)
        dict_params['degree_linear_pol'] = self.degree_linear_polarization(
            verbose=verbose, draw=draw)
        dict_params['degree_circular_pol'] = self.degree_circular_polarization(
            verbose=verbose, draw=draw)
        dict_params['alpha'] = self.alpha(verbose=verbose, draw=draw)
        dict_params['delay'] = self.delay(verbose=verbose, draw=draw)
        dict_params['ellipticity_param'] = self.ellipticity_param(
            verbose=verbose, draw=draw)
        dict_params['ellipticity_angle'] = self.ellipticity_angle(
            verbose=verbose, draw=draw)
        dict_params['azimuth'] = self.azimuth(verbose=verbose, draw=draw)
        dict_params['eccentricity'] = self.eccentricity(verbose=verbose,
                                                        draw=draw)
        dict_params['S_p'], dict_params['S_u'] = self.polarized_unpolarized(
            verbose=verbose)
        dict_params['norm'] = self.norm(verbose=verbose, draw=draw)
        return dict_params

    def matrix(self, shape=None, shape_like=None):
        """Returns the numpy array of the Stokes object.

        Parameters:
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (float or numpy.ndarray) 2xN numpy array.
        """
        shape, _ = select_shape(obj=self.parent,
                                shape_fun=shape,
                                shape_like=shape_like)
        if shape is not None and len(shape) > 1:
            shape = tuple([4] + list(shape))
            M = np.reshape(self.parent.M, shape)
        else:
            M = self.parent.M
        return M

    def global_phase(self,
                     give_nan=True,
                     out_number=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """Extracts the global phase of the Stokes vector.

        Parameters:
            give_nan (bool): If False, NaN values are transformed into 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result.
        """
        gp = self.parent.global_phase
        # If the phase is None (unknown), give nans or zeros
        if gp is None:
            if self.parent.shape is None:
                gp = np.array([0])
            else:
                gp = np.zeros(self.parent.size)
            if give_nan:
                gp = gp * np.nan
        # If the result is a number and the user asks for it, return a float
        if out_number and gp.size == 1:
            gp = gp[0]
        # Calculate Ez and reshape if required
        gp = reshape([gp],
                     shape_like=shape_like,
                     shape_fun=shape,
                     obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The global phase of {} is (deg):'.format(
                self.parent.name)
            PrintParam(param=(gp / degrees),
                       shape=self.parent.shape,
                       title=("Global phase (deg)"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return gp

    def components(self,
                   out_number=True,
                   shape_like=None,
                   shape=None,
                   verbose=False,
                   draw=False):
        """Calculates the $S_0$, $S_1$, $S_2$ and $S_3$ components of the Stokes vector.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            S0 (float or numpy.ndarray): Array of total intensity.
            S1 (float or numpy.ndarray): Array of linearly horizontal or vertical polarized intensity.
            S2 (float or numpy.ndarray): Array of linearly 45º or 135º polarized intensity.
            S3 (float or numpy.ndarray): Array of circularly polarized intensity.
        """
        # Calculate the components
        S0 = self.parent.M[0, :]
        S1 = self.parent.M[1, :]
        S2 = self.parent.M[2, :]
        S3 = self.parent.M[3, :]
        # If the result is a number and the user asks for it, return a float
        if out_number and S0.size == 1:
            (S0, S1, S2, S3) = (S0[0], S1[0], S2[0], S3[0])
        # Calculate Ez and reshape if required
        S0, S1, S2, S3 = reshape([S0, S1, S2, S3],
                                 shape_like=shape_like,
                                 shape_fun=shape,
                                 obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The intensity components of {} are (a.u.):'.format(
                self.parent.name)
            PrintParam(param=(S0, S1, S2, S3),
                       shape=self.parent.shape,
                       title=("S0: (a.u.)", "S1: (a.u.)", "S2: (a.u.)",
                              "S3: (a.u.)"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return S0, S1, S2, S3

    def intensity(self,
                  out_number=True,
                  shape_like=None,
                  shape=None,
                  verbose=False,
                  draw=False):
        """
        Calculates the intensity of the Stokes vector ($S_0$).

        References:
            Handbook of Optics vol 2. 22.16 (eq.2)

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result.
        """
        # Calculate intensity
        S0 = self.parent.M[0, :]
        # If the result is a number and the user asks for it, return a float
        if out_number and S0.size == 1:
            S0 = S0[0]
        # Calculate Ez and reshape if required
        S0 = reshape([S0],
                     shape_like=shape_like,
                     shape_fun=shape,
                     obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The intensity of {} is (a.u.):'.format(self.parent.name)
            PrintParam(param=(S0),
                       shape=self.parent.shape,
                       title=("Intensity: (a.u.)"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return S0

    def irradiance(self,
                   n=1,
                   out_number=True,
                   shape_like=None,
                   shape=None,
                   verbose=False,
                   draw=False):
        """
        Calculates the irradiance of the Stokes vector ($S_0$).

        References:
            Handbook of Optics vol 2. 22.16 (eq.2)

        Parameters:
            n (float): Refractive index. Default: 1.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result.
        """
        # Calculate the irradiance
        Irrad = (n / (2 * eta)) * self.norm(
            out_number=out_number, shape_like=shape_like, shape=shape)
        # Print the result if required
        if verbose or draw:
            heading = 'The irradiance of {} is (W/m^2):'.format(
                self.parent.name)
            PrintParam(param=(Irrad),
                       shape=self.parent.shape,
                       title=("Irradiance: (W/m^2)"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return Irrad

    def degree_polarization(self,
                            use_nan=True,
                            out_number=True,
                            shape_like=None,
                            shape=None,
                            verbose=False,
                            draw=False):
        """Calculates the degree of polarization (DP) of the Stokes vectors.

        $DP=\frac{\sqrt{S_{1}^{2}+S_{2}^{2}+S_{3}^{2}}}{S_{0}}$

        References:
            Handbook of Optics vol 2. 22.16 (eq.3)

        Parameters:
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result
        """
        # Calculate the components
        S0, S1, S2, S3 = self.components(shape_like=shape_like,
                                         shape=shape,
                                         out_number=out_number)
        # Calculate the polarization degree
        DOP = np.zeros_like(S0)
        cond = S0 != 0
        DOP[cond] = np.sqrt(S1[cond]**2 + S2[cond]**2 + S3[cond]**2) / S0[cond]
        if use_nan and np.any(~cond):
            DOP[~cond] = np.nan
        # Print the result if required
        if verbose or draw:
            heading = 'The degree of polarization of {} is:'.format(
                self.parent.name)
            PrintParam(param=(DOP),
                       shape=self.parent.shape,
                       title=("Degree of polarization"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return DOP

    def degree_depolarization(self,
                              use_nan=True,
                              out_number=True,
                              shape_like=None,
                              shape=None,
                              verbose=False,
                              draw=False):
        """Calculates the degree of depolarization (DD) of the Stokes vectors.

        Parameters:
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result
        """
        # Calculate the components
        DOP = self.degree_polarization(use_nan=use_nan,
                                       shape_like=shape_like,
                                       shape=shape,
                                       out_number=out_number)
        # Calculate the depolarization degree
        DD = np.sqrt(1 - DOP**2)
        # Print the result if required
        if verbose or draw:
            heading = 'The degree of depolarization of {} is:'.format(
                self.parent.name)
            PrintParam(param=(DD),
                       shape=self.parent.shape,
                       title=("Degree of depolarization"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return DD

    def degree_linear_polarization(self,
                                   use_nan=True,
                                   out_number=True,
                                   shape_like=None,
                                   shape=None,
                                   verbose=False,
                                   draw=False):
        """Calculates the degree of linear polarization (DLP) of the Stokes vectors.

        $DLP=\frac{\sqrt{S_{1}^{2}+S_{2}^{2}{S_{0}}$

        References:
            Handbook of Optics vol 2. 22.16 (eq.4)

        Parameters:
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result
        """
        # Calculate the components
        S0, S1, S2, _ = self.components(shape_like=shape_like,
                                        shape=shape,
                                        out_number=out_number)
        # Calculate the polarization degree
        DOP = np.zeros_like(S0)
        cond = S0 != 0
        DOP[cond] = np.sqrt(S1[cond]**2 + S2[cond]**2) / S0[cond]
        if use_nan and np.any(~cond):
            DOP[~cond] = np.nan
        # Print the result if required
        if verbose or draw:
            heading = 'The degree of linear polarization of {} is:'.format(
                self.parent.name)
            PrintParam(param=(DOP),
                       shape=self.parent.shape,
                       title=("Degree of linear polarization"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return DOP

    def degree_circular_polarization(self,
                                     use_nan=True,
                                     out_number=True,
                                     shape_like=None,
                                     shape=None,
                                     verbose=False,
                                     draw=False):
        """Calculates the degree of circular polarization (DCP) of the Stokes vectors.

        $DCP=\frac{S_{3}}{S_{0}}$

        References:
            Handbook of Optics vol 2. 22.16 (eq.5)

        Parameters:
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result
        """
        # Calculate the components
        S0, _, _, S3 = self.components(shape_like=shape_like,
                                       shape=shape,
                                       out_number=out_number)
        # Calculate the polarization degree
        DOP = np.zeros_like(S0)
        cond = S0 != 0
        DOP[cond] = np.abs(S3[cond]) / S0[cond]
        if use_nan and np.any(~cond):
            DOP[~cond] = np.nan
        # Print the result if required
        if verbose or draw:
            heading = 'The degree of circular polarization of {} is:'.format(
                self.parent.name)
            PrintParam(param=(DOP),
                       shape=self.parent.shape,
                       title=("Degree of circular polarization"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return DOP

    def alpha(self,
              use_nan=True,
              out_number=True,
              shape_like=None,
              shape=None,
              verbose=False,
              draw=False):
        """Calculates the ratio angle between electric field amplitudes.

        .. math:: arcsin(E_y/Ex).

        Parameters:
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result.
        """
        # Calculate the components of the fully polarized part of S
        Sp, _ = self.polarized_unpolarized(shape_like=shape_like, shape=shape)
        S0, S1, _, _ = Sp.parameters.components(shape_like=shape_like,
                                                shape=shape,
                                                out_number=False)
        # Preallocate memory
        alpha = np.zeros_like(S0)
        if use_nan:
            alpha = alpha * np.nan
        # Calculate if intensity is enough
        cond = np.abs(S0) > tol_default**2
        alpha[cond] = 0.5 * np.arccos(S1[cond] / S0[cond])
        # If the result is a number and the user asks for it, return a float
        if out_number and alpha.size == 1:
            alpha = alpha[0]
        # Print the result if required
        if verbose or draw:
            heading = 'The alpha of {} is (deg):'.format(self.parent.name)
            PrintParam(param=(alpha / degrees),
                       shape=self.parent.shape,
                       title=("Alpha: (deg)"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return alpha

    def delay(self,
              use_nan=True,
              out_number=True,
              shape_like=None,
              shape=None,
              verbose=False,
              draw=False):
        """Phase shift between $E_x$ and $E_y$ electric field components.

        .. math:: \delta_2 - \delta_1.

        Parameters:
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result.
        """
        # Calculate the components
        S0, _, S2, S3 = self.components(shape_like=shape_like,
                                        shape=shape,
                                        out_number=False)
        # Preallocate memory
        delta = np.zeros_like(S0)
        if use_nan:
            delta = delta * np.nan
        # Calculate if intensity is enough
        cond = np.abs(S0) > tol_default**2
        delta[cond] = np.arctan2(S3[cond], S2[cond])
        # Make sure the values are in the correct range
        delta = put_in_limits(delta, 'delta', out_number=out_number)
        # Print the result if required
        if verbose or draw:
            heading = 'The delay of {} is (deg):'.format(self.parent.name)
            PrintParam(param=(delta / degrees),
                       shape=self.parent.shape,
                       title=("Delay: (deg)"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return delta

    def delta(self,
              use_nan=True,
              out_number=True,
              shape_like=None,
              shape=None,
              verbose=False,
              draw=False):
        """Phase shift between $E_x$ and $E_y$ electric field components. This is the same as *delay*.

        .. math:: \delta_2 - \delta_1.

        Parameters:
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result.
        """
        return self.delay(use_nan=use_nan,
                          out_number=out_number,
                          shape_like=shape_like,
                          shape=shape,
                          verbose=verbose,
                          draw=draw)

    def charac_angles(self,
                      use_nan=True,
                      out_number=True,
                      shape_like=None,
                      shape=None,
                      verbose=False,
                      draw=False):
        """Calculates the characteristic angles of the Stokes object.

        Parameters:
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            alpha (float): [0, pi/2]: tan(alpha) is the ratio angle between amplitudes of the electric field.
            delay (float): [0, 2*pi]: phase difference between both components of the electric field.
        """
        alpha = self.alpha(use_nan=use_nan,
                           out_number=out_number,
                           shape_like=shape_like,
                           shape=shape,
                           verbose=verbose,
                           draw=draw)
        delta = self.delta(use_nan=use_nan,
                           out_number=out_number,
                           shape_like=shape_like,
                           shape=shape,
                           verbose=verbose,
                           draw=draw)
        return alpha, delta

    def ellipticity_param(self,
                          use_nan=True,
                          out_number=True,
                          shape_like=None,
                          shape=None,
                          verbose=False,
                          draw=False):
        """Calculates the ellipticity parameter: the ratio between the minor and major polarization ellipse axes. It takes only into account the fully polarized part of the Stokes vector. It goes from 0 for linearly polarized light to 1 for circulary polarized light. Positive sign means left-handed and negative values right-handed rotation direction.

        References:
            Handbook of Optics vol 2. 22.16 (eq.7)

        Parameters:
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result.
        """
        # Calculate the components of the fully polarized part of S
        Sp, _ = self.polarized_unpolarized(shape_like=shape_like, shape=shape)
        S0, S1, S2, S3 = Sp.parameters.components(shape_like=shape_like,
                                                  shape=shape,
                                                  out_number=False)
        # Preallocate memory
        el = np.zeros_like(S0)
        if use_nan:
            el = el * np.nan
        # Act if we have some polarization
        cond = (S1 != 0) + (S2 != 0) + (S3 != 0)
        el[cond] = S3[cond] / (S0[cond] + np.sqrt(S1[cond]**2 + S2[cond]**2))
        # Just in case remove nans
        if not use_nan:
            el[np.isnan(el)] = 0
        # If the result is a number and the user asks for it, return a float
        if out_number and el.size == 1:
            el = el[0]
        # Print the result if required
        if verbose or draw:
            heading = 'The ellipticity parameter of {} is:'.format(
                self.parent.name)
            PrintParam(param=(el),
                       shape=self.parent.shape,
                       title=("Ellipticity parameter"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return el

    def ellipticity_angle(self,
                          use_nan=True,
                          out_number=True,
                          shape_like=None,
                          shape=None,
                          verbose=False,
                          draw=False):
        """Calculates the ratio angle between the major and minor axis length. It goes from 0 degrees for linearly polarized light to 45 degrees for circular polarized light. Positive sign means left-handed and negative values right-handed rotation direction.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016) pp 13.

        Parameters:
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result.

        """
        # Calculate the ellipticity parameter
        el = self.ellipticity_param(use_nan=use_nan,
                                    shape_like=shape_like,
                                    shape=shape,
                                    out_number=out_number)
        # Transform to angle
        el = np.arctan(el)
        # Print the result if required
        if verbose or draw:
            heading = 'The ellipticity angle of {} is (deg):'.format(
                self.parent.name)
            PrintParam(param=(el / degrees),
                       shape=self.parent.shape,
                       title=("Ellipticity angle: (deg)"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return el

    def azimuth(self,
                use_nan=True,
                out_number=True,
                shape_like=None,
                shape=None,
                verbose=False,
                draw=False):
        """Calculates the rotation angle of the polarization elipse major axis. If S is not fully polarized, azimuth is computed on the fully polarized part of S. Azimuth ranges from 0 to 180 degres (not included this last value).

        References:
            Handbook of Optics vol 2. 22.16 (eq.8)

        Parameters:
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result.
        """
        # Calculate the components of the fully polarized part of S
        Sp, _ = self.polarized_unpolarized(shape_like=shape_like, shape=shape)
        S0, S1, S2, S3 = Sp.parameters.components(shape_like=shape_like,
                                                  shape=shape,
                                                  out_number=False)
        # Preallocate memory
        azimuth = np.zeros_like(S0)
        if use_nan:
            azimuth = azimuth * np.nan
        # Take some cases apart
        cond1 = np.abs(S1) < tol_default**2
        cond2 = S2 > tol_default**2
        azimuth[cond1 * cond2] = np.pi / 4
        cond2 = S2 < -tol_default**2
        azimuth[cond1 * cond2] = -np.pi / 4
        # General case
        azimuth[~cond1] = 0.5 * np.arctan2(S2[~cond1], S1[~cond1])
        # Just in case remove nans
        if not use_nan:
            azimuth[np.isnan(azimuth)] = 0
        # If the result is a number and the user asks for it, return a float
        if out_number and azimuth.size == 1:
            azimuth = azimuth[0]
        # Make sure the values are in the correct range
        azimuth = put_in_limits(azimuth, 'azimuth')
        # Print the result if required
        if verbose or draw:
            heading = 'The azimuth of {} is (deg):'.format(self.parent.name)
            PrintParam(param=(azimuth / degrees),
                       shape=self.parent.shape,
                       title=("Azimuth: (deg)"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return azimuth

    def azimuth_ellipticity(self,
                            use_nan=True,
                            out_number=True,
                            shape_like=None,
                            shape=None,
                            verbose=False,
                            draw=False):
        """Calculates both the azimuth and ellipticity of the Stokes object.

        Parameters:
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            azimuth (float or numpy.ndarray): [0, pi): Azimuth.
            ellipticity (float or numpy.ndarray): [-pi/4, pi/4]: Ellipticity angle.
        """
        azimuth = self.azimuth(use_nan=use_nan,
                               out_number=out_number,
                               shape_like=shape_like,
                               shape=shape,
                               verbose=verbose,
                               draw=draw)
        ellipticity = self.ellipticity_angle(use_nan=use_nan,
                                             out_number=out_number,
                                             shape_like=shape_like,
                                             shape=shape,
                                             verbose=verbose,
                                             draw=draw)
        return azimuth, ellipticity

    def eccentricity(self,
                     use_nan=True,
                     out_number=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """The eccentricity of the polarization ellipse (0 for circular polarization, 1 for linear).

        References:
            Handbook of Optics vol 2. 22.16 (eq.9)

        Parameters:
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (float or numpy.ndarray): Result.
        """
        # Calculate the ellipticity parameter
        e = self.ellipticity_param(use_nan=use_nan,
                                   out_number=out_number,
                                   shape_like=shape_like,
                                   shape=shape)
        # Now we can calculate the eccentricity
        e2 = np.sqrt(1 - e**2)
        # Print the result if required
        if verbose or draw:
            heading = 'The eccentricity of {} is:'.format(self.parent.name)
            PrintParam(param=(e2),
                       shape=self.parent.shape,
                       title=("Eccentricity"),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return e2

    def ellipse_axes(self,
                     out_number=True,
                     sort=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """Calculates the length of major and minor axis (a,b) of the polarization ellipse. This is a wrapper around the Parameters_Jones_Vector.ellipse_axes function.

        References:
            D. Golstein "Polarized light" 2nd ed Marcel Dekker (2003), 3.4 eq.3-30a and 3-30b

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            sort (bool): If True, it sorts a and b to be the major and minor axis respectively. Default: True. TODO: Check why this is neccessary.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            a (numpy.ndarray or float): Major axis
            b (numpy.ndarray or float): Minor axis
        """
        # Go to Jones, it is much easier
        J = Jones_vector()
        J.from_Stokes(self.parent)
        a, b = J.parameters.ellipse_axes(out_number=out_number,
                                         sort=sort,
                                         shape_like=shape_like,
                                         shape=shape,
                                         verbose=verbose,
                                         draw=draw)
        return a, b

    def polarized_unpolarized(self,
                              shape_like=None,
                              shape=None,
                              verbose=False):
        """Divides the Stokes vector in two, one totally polarized and the other totally unpolarized.

        References:
            Handbook of Optics vol 2. 22.16 (eq.6)

        Parameters:
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.

        Returns:
            Sp (Stokes): Totally polarized Stokes object.
            Su (Stokes): Totally unpolarized Stokes object.
        """
        # Calculate the degree of polarization and the components
        DOP = self.degree_polarization(shape=shape, shape_like=shape_like)
        S0, S1, S2, S3 = self.components(shape=shape, shape_like=shape_like)
        # Create the Stokes objects
        Sp, Su = create_Stokes(('Polarized ' + self.parent.name,
                                'Unpolarized ' + self.parent.name))
        Sp.from_components(
            (DOP * S0, S1, S2, S3),
            global_phase=self.global_phase(shape=shape, shape_like=shape_like))
        Su.from_components(((1 - DOP) * S0, 0, 0, 0))
        # Print the result if required
        if verbose:
            print(Sp, Su)
        return Sp, Su

    def amplitudes(self,
                   give_Ez=False,
                   give_unpol=False,
                   out_number=True,
                   shape_like=None,
                   shape=None,
                   verbose=False,
                   draw=False):
        """Calculates the $E_x$ and $E_y$ field amplitudes of the polarized part of the Stokes vectos. It may also calculate $E_z$ and the field amplitude of the unpolarized part.

        Parameters:
            give_Ez (bool): If True, it returns the z component of the electric field (all values will be 0). Default: False.
            give_unpol (bool): If True, it returns the unpolarized component of the electric field. Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            E0x (numpy.ndarray or float): Array of electric field amplitude along x axis.
            E0y (numpy.ndarray or float): Array of electric field amplitude along y axis.
            E0z (numpy.array, optional): Array of electric field amplitude along z axis.
        """
        # Separate the polarized and the unpolarized parts
        Sp, Su = self.polarized_unpolarized()
        # Extract the important parameters
        S0, S1, _, _ = Sp.parameters.components(shape_like=shape_like,
                                                shape=shape,
                                                out_number=out_number)
        # Calculate the amplitudes
        E0x = np.sqrt((S0 + S1) / 2)
        E0y = np.sqrt((S0 - S1) / 2)
        if give_Ez:
            E0z = np.zeros_like(E0x)
        if give_unpol:
            Iu = Su.parameters.intensity(shape_like=shape_like,
                                         shape=shape,
                                         out_number=out_number)
            E0u = np.sqrt(Iu)
        # Print the result if required
        if verbose or draw:
            heading = 'The elctric field amplitudes of {} are (V/m):'.format(
                self.parent.name)
            if give_Ez and give_unpol:
                PrintParam(param=(E0x, E0y, E0z, E0u),
                           shape=self.parent.shape,
                           title=('Ex (V/m)', 'Ey (V/m)', 'Ez (V/m)',
                                  'Eu (V/m)'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
            elif give_Ez:
                PrintParam(param=(E0x, E0y, E0z),
                           shape=self.parent.shape,
                           title=('Ex (V/m)', 'Ey (V/m)', 'Ez (V/m)'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
            elif give_unpol:
                PrintParam(param=(E0x, E0y, E0u),
                           shape=self.parent.shape,
                           title=('Ex (V/m)', 'Ey (V/m)', 'Eu (V/m)'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
            else:
                PrintParam(param=(E0x, E0y),
                           shape=self.parent.shape,
                           title=('Ex (V/m)', 'Ey (V/m)'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
        # Return
        if give_Ez and give_unpol:
            return E0x, E0y, E0z, E0u
        elif give_Ez:
            return E0x, E0y, E0z
        elif give_unpol:
            return E0x, E0y, E0u
        else:
            return E0x, E0y

    def norm(self,
             out_number=True,
             shape_like=None,
             shape=None,
             verbose=False,
             draw=False):
        """Calculates the algebraic norm of the Stokes vectors.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            S0 (float or numpy.ndarray): Array of total intensity.
        """
        # Calculate the norm
        norm = np.linalg.norm(self.parent.M, axis=0)
        # Reshape if neccessary
        norm = reshape([norm],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The norm of {} is (a.u.):'.format(self.parent.name)
            PrintParam(param=norm,
                       shape=self.parent.shape,
                       title='Vector norm (a.u.)',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return norm

    def density(self, *args, **kwargs):
        """Maintained here just for compatibility. Look for Stokes.analysis.density instead.
        """
        print(
            "DEPERCATION WARNING: Please use Stokes.analysis.density instead.")
        return self.parent.analysis.density(*args, **kwargs)


class Analysis_Stokes(object):
    """Class for Analysis of Stokes vetors.

    Parameters:
        stokes_vector (Stokes): Stokes vector

    Attributes:
        self.parent (Stokes)
    """

    def __init__(self, parent):
        self.parent = parent

    def filter_physical_conditions(self, tol=tol_default, keep=False):
        """Function that filters experimental errors by forcing the Stokes vector to fulfill the conditions necessary for a vector to be real light.

        Parameters:
            tol (float): Tolerance in equalities. Default: tol_default = 1e-6.
            keep (bool): If True, the object is updated to the filtered result. If false, a new fresh copy is created. Default: False.

        Returns:
            S (Stokes): Filtered Stokes vector.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.parent.copy()
        else:
            new_obj = self.parent
        # Check if the Stokes vector is physically realizable
        cond, dict_param = new_obj.checks.is_physical(out_number=False,
                                                      tol=tol,
                                                      give_all=True)
        # Act only if it isn't
        if np.any(cond):
            S0, S1, S2, S3 = new_obj.parameters.components(out_number=False)
            # Fix the first and second conditions
            S0 = np.abs(S0)
            S1, S2, S3 = (np.real(S1), np.real(S2), np.real(S3))
            # Fix the third condition
            PD = np.sqrt(S1**2 + S2**2 + S3**2) / S0
            cond2 = PD > 1 + tol
            if np.any(cond2):
                S1[cond2] = S1[cond2] / PD[cond2]
                S2[cond2] = S2[cond2] / PD[cond2]
                S3[cond2] = S3[cond2] / PD[cond2]
            # Reconstruct the object
            new_obj.from_components((S0, S1, S2, S3))
        return new_obj

    def density(self,
                radius=0.1,
                sampling=1 * degrees,
                draw=False,
                return_coor=False,
                return_fig=False,
                **kwargs):
        """Calculates the density of elements in the Poincare sphere. Unlike many other parameters methods, the return object is always a 2D object covering the whole Poincare Sphere.

        Parameters:
            radius (number): Unit radius to compute density. Must be positive and lower than 1. Default: 0.1.
            sampling (number): Resolution in azimuth and ellipticity. Default: 1 degree.
            draw (bool): If True, plot it. Default: False.
            return_coor (bool): If True, it returns the x, y and z coordinates of the Poincare sphere. Default: False.
            return_fig (bool): If True, the figure object will be returned. Default: False.
            **kwargs: If draw is True, these arguments will be passed to draw_poincare method.

        Returns:
            density (np.ndarray): Result. Units: points / srad.
        """
        # Calculate spherical coordinates
        az = np.linspace(0, 179.99 * degrees,
                         int(np.ceil(179.99 * degrees / sampling) + 1))
        el = np.linspace(-45 * degrees, 45 * degrees,
                         int(np.ceil(90 * degrees / sampling) + 1))
        az, el = np.meshgrid(az, el)
        x, y, z = azel_2_xyz(az, el)

        # Prepare the vector
        obj = self.parent.flatten(keep=True)
        obj.set_depolarization(0)
        obj.normalize()

        # Strecth arrays
        shape = [obj.size] + list((x.shape))
        xb = np.broadcast_to(x, shape)
        yb = np.broadcast_to(y, shape)
        zb = np.broadcast_to(z, shape)
        coor = np.stack((xb, yb, zb))
        M = np.broadcast_to(obj.M[1:, :, np.newaxis, np.newaxis], [3] + shape)

        # Calculate distances
        dist = np.sqrt(np.sum((M - coor)**2, axis=0))

        # Calculate density
        dens = dist < radius
        dens = np.sum(dens, axis=0) / (dens * np.pi * np.sqrt(1 - radius**2))

        # Draw
        if draw:
            S = Stokes("Density").from_components((1, x, y, z))
            fig = S.draw_poincare(kind="surf", param=dens, **kwargs)
            fig.show()

        # Returns
        ret = [dens]
        if return_coor:
            ret = ret + [x, y, z]
        if return_fig:
            ret.append(fig)

        return ret

    def existence(self,
                  radius=0.1,
                  sampling=1 * degrees,
                  draw=False,
                  return_coor=False,
                  return_fig=False,
                  **kwargs):
        """Calculates the density of elements in the Poincare sphere. Unlike many other parameters methods, the return object is always a 2D object covering the whole Poincare Sphere.

        Parameters:
            radius (number): Unit radius to compute density. Must be positive and lower than 1. Default: 0.1.
            sampling (number): Resolution in azimuth and ellipticity. Default: 1 degree.
            draw (bool): If True, plot it. Default: False.
            return_coor (bool): If True, it returns the x, y and z coordinates of the Poincare sphere. Default: False.
            return_fig (bool): If True, the figure object will be returned. Default: False.
            **kwargs: If draw is True, these arguments will be passed to draw_poincare method.

        Returns:
            existence (np.ndarray of bools): Result.
            fig (plotly.graph_objects.Figure): Figure. Returned only if return_fig is True.
        """
        # Calculate density
        density, x, y, z = self.density(radius=radius,
                                        sampling=sampling,
                                        draw=False,
                                        return_coor=True,
                                        return_fig=False)
        existence = np.array(density >= 1, dtype=float)

        # Draw
        if draw:
            S = Stokes("Existence").from_components((1, x, y, z))
            fig = S.draw_poincare(kind="surf", param=existence, **kwargs)
            fig.show()

        # Returns
        ret = [existence]
        if return_coor:
            ret = ret + [x, y, z]
        if return_fig:
            ret.append(fig)

        return ret


class Check_Stokes(object):
    """Class for Check of Stokes vectors.

    Parameters:
        stokes_vector (Stokes): Stokes vector

    Attributes:
        self.parent (Stokes)
    """

    def __init__(self, parent):
        self.parent = parent

    def __repr__(self):
        """Prints all the checks"""
        self.get_all(verbose=True, draw=True)
        return ''

    def get_all(self, verbose=False, draw=False):
        """Creates a dictionary with all the checks of Stokes vectors.

        Parameters:
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.
        """
        dict_params = {}
        dict_params['is_physical'], dict_params[
            'dict_is_physical'] = self.is_physical(verbose=verbose,
                                                   draw=draw,
                                                   give_all=True)
        dict_params['is_linear'] = self.is_linear(verbose=verbose, draw=draw)
        dict_params['is_circular'] = self.is_circular(verbose=verbose,
                                                      draw=draw)
        dict_params['is_right_handed'] = self.is_right_handed(verbose=verbose,
                                                              draw=draw)
        dict_params['is_left_handed'] = self.is_left_handed(verbose=verbose,
                                                            draw=draw)
        dict_params['is_polarized'] = self.is_polarized(verbose=verbose,
                                                        draw=draw)
        dict_params['is_totally_polarized'] = self.is_totally_polarized(
            verbose=verbose, draw=draw)
        dict_params['is_depolarized'] = self.is_depolarized(verbose=verbose,
                                                            draw=draw)
        dict_params['is_totally_depolarized'] = self.is_totally_depolarized(
            verbose=verbose, draw=draw)
        return dict_params

    def is_physical(self,
                    tol=tol_default,
                    give_all=False,
                    out_number=True,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """Tests the conditions of physical realizability:

        * Condition 1: All components must be real.
        * Condition 2: S0 must be positive.
        * Condition 3: The square sum of S1, S2 and S3 must be equal or lower than S0.

        References:
            Handbook of Optics vol 2. 22.34
            "Polarized light and the Mueller Matrix approach", J. J. Gil, pp 187.

        Parameters:
            tol (float): Tolerance in equality conditions. Default: tol_default = 1e-6.
            give_all (bool): If True, the function will return the individual conditions. Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            cond (bool): Is real or not.
            ind (dictionary, optional): dictionary with condition, True/False, distance
        """
        # Calculate the three Conditions
        S0, S1, S2, S3 = self.parent.parameters.components(out_number=False)
        cond1_0, cond1_1, cond1_2, cond1_3 = (np.isreal(S0), np.isreal(S1),
                                              np.isreal(S2), np.isreal(S3))
        cond2 = S0 >= -tol
        DOP = self.parent.parameters.degree_polarization(out_number=False)
        cond3 = DOP <= 1 + tol
        # Calculate Ez and reshape if required
        cond1_0, cond1_1, cond1_2, cond1_3, cond2, cond3 = reshape(
            [cond1_0, cond1_1, cond1_2, cond1_3, cond2, cond3],
            shape_like=shape_like,
            shape_fun=shape,
            obj=self.parent)
        # Calculate final conditions
        cond1 = cond1_0 * cond1_1 * cond1_2 * cond1_3
        cond = cond1 * cond2 * cond3
        # Create the dictionary with all the conditions if desired
        if give_all:
            dict = {}
            dict['cond1'] = cond1
            dict['cond1_0'] = cond1_0
            dict['cond1_1'] = cond1_1
            dict['cond1_2'] = cond1_2
            dict['cond1_3'] = cond1_3
            dict['cond2'] = cond2
            dict['S0'] = S0
            dict['cond3'] = cond3
        # Print the result if required
        if verbose or draw:
            if give_all:
                heading = '{} is physically realizable:'.format(
                    self.parent.name)
                PrintParam(param=(cond1_0, cond1_1, cond1_2, cond1_3, cond1,
                                  cond2, cond3, cond),
                           shape=self.parent.shape,
                           title=("Physicall (cond. 1, S0)",
                                  "Physicall (cond. 1, S1)",
                                  "Physicall (cond. 1, S2)",
                                  "Physicall (cond. 1, S3)",
                                  "Physicall (cond. 1)", "Physicall (cond. 2)",
                                  "Physicall (cond. 3)", "Physicall"),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
            else:
                heading = '{} is physically realizable:'.format(
                    self.parent.name)
                PrintParam(param=(cond),
                           shape=self.parent.shape,
                           title=("Physicall"),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
        # Return
        if give_all:
            return cond, dict
        else:
            return cond

    def is_linear(self,
                  tol=tol_default,
                  use_nan=True,
                  out_number=True,
                  shape_like=None,
                  shape=None,
                  verbose=False,
                  draw=False):
        """Calculates if the Stokes vectors are linearly polarized.

        Parameters:
            tol (float): Tolerance. Default: tol_default = 1e-6.
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to False. Default: True.
            out_number (bool): if True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Array of the condition.
        """
        # Preallocate memory
        if use_nan:
            cond = np.zeros(max(self.parent.size, 1))  # No bool to allow nans
        else:
            cond = np.zeros(max(self.parent.size, 1), dtype=bool)
        # If the Stokes vector is totally depolarized, polarization is not defined
        cond1 = self.is_totally_depolarized(shape=False, out_number=False)
        if use_nan:
            cond[cond1] = np.nan
        # Calculate the relative degree of linear polarization
        DP = self.parent.parameters.degree_polarization(shape=False,
                                                        out_number=False)
        DLP = self.parent.parameters.degree_linear_polarization(
            shape=False, out_number=False)
        cond[~cond1] = DLP[~cond1] / DP[~cond1] > 1 - tol
        # Give a number if required
        if out_number and cond.size == 1:
            cond = cond[0]
        # Reshape if required
        cond = reshape([cond],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is linearly polarized:'.format(self.parent.name)
            PrintParam(param=(cond),
                       shape=self.parent.shape,
                       title=('Linearly polarized'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return cond

    def is_circular(self,
                    tol=tol_default,
                    use_nan=True,
                    out_number=True,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """Calculates if the Stokes vectors are circularly polarized.

        Parameters:
            tol (float): Tolerance. Default: tol_default: 1e-6.
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to False. Default: True.
            out_number (bool): if True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Array of the condition.
        """
        # Preallocate memory
        if use_nan:
            cond = np.zeros(max(self.parent.size, 1))  # No bool to allow nans
        else:
            cond = np.zeros(max(self.parent.size, 1), dtype=bool)
        # If the Stokes vector is totally depolarized, polarization is not defined
        cond1 = self.is_totally_depolarized(shape=False, out_number=False)
        if use_nan:
            cond[cond1] = np.nan
        # Calculate the relative degree of linear polarization
        DP = self.parent.parameters.degree_polarization(shape=False,
                                                        out_number=False)
        DCP = self.parent.parameters.degree_circular_polarization(
            shape=False, out_number=False)
        cond[~cond1] = DCP[~cond1] / DP[~cond1] > 1 - tol
        # Give a number if required
        if out_number and cond.size == 1:
            cond = cond[0]
        # Reshape if required
        cond = reshape([cond],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is circcularly polarized:'.format(self.parent.name)
            PrintParam(param=(cond),
                       shape=self.parent.shape,
                       title=('Circularly polarized'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return cond

    def is_right_handed(self,
                        tol=tol_default,
                        use_nan=True,
                        out_number=True,
                        shape_like=None,
                        shape=None,
                        verbose=False,
                        draw=False):
        """Calculates if the polarization rotation direction is right-handed.

        Parameters:
            tol (float): Tolerance. Default: tol_default: 1e-6.
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to False. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Array of the condition.
        """
        # Preallocate memory
        if use_nan:
            cond = np.zeros(max(self.parent.size, 1))  # No bool to allow nans
        else:
            cond = np.zeros(max(self.parent.size, 1), dtype=bool)
        # If the Stokes vector is linearly polarized or totally depolarized, right and left handed is not defined
        cond1 = self.is_linear(shape=False, out_number=False, use_nan=False) * \
            self.is_totally_depolarized(
                shape=False, out_number=False, tol=tol)
        if use_nan:
            cond[cond1] = np.nan
        # If delay is between 0 and pi (not included), it is right handed
        delay = self.parent.parameters.delay(shape=False,
                                             out_number=False,
                                             use_nan=False)
        cond2 = ~cond1 * (delay > 0) * (delay < np.pi)
        cond[cond2] = True
        # Give a number if required
        if out_number and cond.size == 1:
            cond = cond[0]
        # Reshape if required
        cond = reshape([cond],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is right handed:'.format(self.parent.name)
            PrintParam(param=(cond),
                       shape=self.parent.shape,
                       title=('Right handed'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return cond

    def is_left_handed(self,
                       tol=tol_default,
                       use_nan=True,
                       out_number=True,
                       shape_like=None,
                       shape=None,
                       verbose=False,
                       draw=False):
        """Calculates if the polarization rotation direction is left-handed.

        Parameters:
            tol (float): Tolerance. Default: tol_default: 1e-6.
            use_nan (bool): If True, unknown values are set to np.nan, otherwise they are set to False. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Array of the condition.
        """
        # Preallocate memory
        if use_nan:
            cond = np.zeros(max(self.parent.size, 1))  # No bool to allow nans
        else:
            cond = np.zeros(max(self.parent.size, 1), dtype=bool)
        # If the Stokes vector is linearly polarized or totally depolarized, right and left handed is not defined
        cond1 = self.is_linear(shape=False, out_number=False, use_nan=False) * \
            self.is_totally_depolarized(
                shape=False, out_number=False, tol=tol)
        if use_nan:
            cond[cond1] = np.nan
        # If delay is between 0 and pi (not included), it is right handed
        delay = self.parent.parameters.delay(shape=False, out_number=False)
        cond2 = ~cond1 * (delay > np.pi) * (delay < 2 * np.pi)
        cond[cond2] = True
        # Give a number if required
        if out_number and cond.size == 1:
            cond = cond[0]
        # Reshape if required
        cond = reshape([cond],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is left handed:'.format(self.parent.name)
            PrintParam(param=(cond),
                       shape=self.parent.shape,
                       title=('Left handed'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return cond

    def is_polarized(self,
                     tol=tol_default,
                     out_number=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """Calculates if the Stokes vectors are at least partially polarized.

        Parameters:
            tol (float): Tolerance. Default: tol_default = 1e-6.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Array of the condition.
        """
        # Calculate the polarization degree
        PD = self.parent.parameters.degree_polarization(shape=shape,
                                                        shape_like=shape_like,
                                                        out_number=out_number)
        cond = PD > tol
        # Print the result if required
        if verbose or draw:
            heading = '{} is polarized:'.format(self.parent.name)
            PrintParam(param=(cond),
                       shape=self.parent.shape,
                       title=('Polarized'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return cond

    def is_totally_polarized(self,
                             tol=tol_default,
                             out_number=True,
                             shape_like=None,
                             shape=None,
                             verbose=False,
                             draw=False):
        """Calculates if the Stokes vectors are totally polarized.

        Parameters:
            tol (float): Tolerance. Default: tol_default: 1e-6.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Array of the condition.
        """
        # Calculate the polarization degree
        PD = self.parent.parameters.degree_polarization(shape=shape,
                                                        shape_like=shape_like,
                                                        out_number=out_number)
        cond = PD > 1 - tol
        # Print the result if required
        if verbose or draw:
            heading = '{} is totally polarized:'.format(self.parent.name)
            PrintParam(param=(cond),
                       shape=self.parent.shape,
                       title=('Totally polarized'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return cond

    def is_depolarized(self,
                       tol=tol_default,
                       out_number=True,
                       shape_like=None,
                       shape=None,
                       verbose=False,
                       draw=False):
        """Calculates if the Stokes vectors are at least partially depolarized.

        Parameters:
            tol (float): Tolerance. Default: tol_default: 1e-6.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Array of the condition.
        """
        # Calculate the polarization degree
        DD = self.parent.parameters.degree_depolarization(
            shape=shape, shape_like=shape_like, out_number=out_number)
        cond = DD > tol
        # Print the result if required
        if verbose or draw:
            heading = '{} is depolarized:'.format(self.parent.name)
            PrintParam(param=(cond),
                       shape=self.parent.shape,
                       title=('Depolarized'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return cond

    def is_totally_depolarized(self,
                               tol=tol_default,
                               out_number=True,
                               shape_like=None,
                               shape=None,
                               verbose=False,
                               draw=False):
        """Calculates if the Stokes vectors are totally depolarized.

        Parameters:
            tol (float): Tolerance. Default: tol_default: 1e-6.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Array of the condition.
        """
        # Calculate the polarization degree
        DD = self.parent.parameters.degree_depolarization(
            shape=shape, shape_like=shape_like, out_number=out_number)
        cond = DD > 1 - tol
        # Print the result if required
        if verbose or draw:
            heading = '{} is totally depolarized:'.format(self.parent.name)
            PrintParam(param=(cond),
                       shape=self.parent.shape,
                       title=('Totally depolarized'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return cond
