# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------
# Authors:    Luis Miguel Sanchez Brea and Jesus del Hoyo
# Date:       2019/11/15 (version 1.0)
# License:    GPL
# -------------------------------------
"""
Jones_vector objects describe light polarization states in the Jones formalism.

**Class fields**
    * **M**: 2xN array containing all the Jones vectors.
    * **name**: Name of the object for print purposes.
    * **shape**: Shape desired for the outputs.
    * **size**: Number of stores Jones vectors.
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
    * **from_components**: Creates a Jones_vector object directly from the electric field components $E_x$ and $E_y$.
    * **from_matrix**: Creates a Jones_vector object directly from a 2 x shape numpy array.
    * **from_list**: Creates a Jones_vector object directly from a list of 2 or 2x1 numpy arrays.
    * **from_distribution**: Creates a Jones vector from the temporal evolution of the electric field components.
    * **from_Stokes**: Creates a Jones vector from a Stokes object. Take into account that only pure (totally polarized) Stokes vectors must be transformed to Jones vectors, and thet even for them, the global phase might be unknown.
    * **linear_light**: Creates a state of linear polarization with the desired angle.
    * **circular_light**: Creates a state of circular polarization.
    * **eliptical_light**: Creates a state of eliptical polarization.
    * **general**: Creates the most general Jones vector.
    * **general_azimuth_ellipticity**: Creates a Jones vector from the azimuth, ellipticity parameters.
    * **general_charac_angles**: Creates a Jones vector from the characteristic angles parameters.


**Manipulation methods**
    * **simplify**:  Simplifies the Jones vector in several ways.
    * **rotate**: Rotates the Jones vector.
    * **sum**: Calculates the summatory of the Jones vectors in the object.
    * **reciprocal**: Calculates the Jones vector that propagates backwards.
    * **orthogonal**: Calculates the orthogonal Jones vector.
    * **normalize**: Normalize the electric field to be normalized in electric field amplitude or intensity.
    * **rotate_to_azimuth**: Rotates the Jones vector to have a certain azimuth.
    * **remove_global_phase**: Calculates the global phase of the electric field (respect to the X component) and removes it.
    * **add_global_phase**: Adds a global phase to the Jones vector.
    * **set_global_phase**: Sets the global phase of the Jones vector.
    * **add_delay**: Adds a phase to the Y component of the electric field of the Jones vector.


**Drawing methods**
    * **draw_ellipse**:  Draws the polarization ellipse of the Jones vector.
    * **draw_poincare**: Draws the light state in the Poincare sphere.


**Parameters subclass methods**
    * **matrix**:  Gets a numpy array with all the vectors (similat to object.M).
    * **components**: Calculates the electric field components of the Jones vector.
    * **amplitudes**: Calculates the electric field amplitudes of the Jones vector.
    * **intensity**: Calculates the intensity of the Jones vector.
    * **irradiance**: Calculates the irradiance of the Jones vector.
    * **alpha**: Calculates the ratio between electric field amplitude angle ($|E_x|$/$|E_y|$).
    * **delay / delta**: Calculates the delay (phase shift) between Ex and Ey components of the electric field.
    * **charac_angles**: Calculates both alpha and delay, the characteristic angles of the Jones vector.
    * **azimuth**: Calculates azimuth, that is, the orientation angle of the major axis.
    * **ellipticity_angle**: Calculates the ellipticity angle.
    * **azimuth_ellipticity**: Calculates both azimuth and ellipticity angles.
    * **ellipse_axes**: Calculates the length of major and minor axis (a,b).
    * **ellipticity_param**: Calculates the ellipticity parameter, b/a.
    * **eccentricity**: Calculates the eccentricity, the complementary of the ellipticity parameter.
    * **global_phase**: Calculates the global phase of the Jones vector (respect to the X component of the electric field).
    * **degree_linear_polarization**: Calculates the degree of linear polarization of the Jones vector.
    * **degree_circular_polarization**: Calculates the degree of circular polarization of the Jones vector.
    * **norm**: Calculates the norm of the Jones vector (same as intensity for Jones vectors).

    * **get_all**: Returns a dictionary with all the parameters of Jones vector.


**Checks subclass methods**
    * **is_linear**: Checks if the Jones vector is lienarly polarized.
    * **is_circular**: Checks if the Jones vector is circularly polarized.
    * **is_right_handed**: Checks if the Jones vector rotation direction is right handed.
    * **is_left_handed**: Checks if the Jones vector rotation direction is left handed.

    * **get_all**: Returns a dictionary with all the checks of Jones vector.


**Auxiliar functions**
    * **create_Jones_vectors**: Function to create several Jones_vector objects at once.
"""

import warnings
# from functools import wraps

from scipy import optimize

from .__init__ import degrees, np, eps, eta, number_types
from .py_pol import Py_pol
from .drawings import draw_ellipse, draw_poincare
from .utils import (charac_angles_2_azimuth_elipt, put_in_limits, repair_name,
                    rotation_matrix_Jones, prepare_variables, reshape, 
                    PrintParam, take_shape, select_shape, PrintMatrices,
                    fit_distribution)

warnings.filterwarnings('ignore')

tol_default = eps
N_print_list = 10
print_list_spaces = 3
j_empty = np.array([[0], [0]])
change_names = True

################################################################################
# Functions
################################################################################


def create_Jones_vectors(name='E', N=1, out_object=True):
    """Function that creates several Jones_vector objects at the same time from a list of names or a number.

    Parameters:
        name (str, list or tuple): name of vector for string representation. If list or tuple, it also represents the number of objects to be created.
        N (int): Number of elements to be created. This parameter is overrided if name is a list or tuple. Defeult: 1.
        out_object (bool): If True and the result is a list of length 1, return a Jones_vector object instead. Default: True.

    Returns:
        E (Jones_vector or list): List of Jones vectors
    """
    E = []
    if isinstance(name, list) or isinstance(name, tuple):
        for n in name:
            E.append(Jones_vector(n))
    else:
        for _ in range(N):
            E.append(Jones_vector(name))
    if len(E) == 1 and out_object:
        E = E[0]
    return E


def set_printoptions(N_list=None, list_spaces=None):
    """Function that modifies the global print options parameters.

    Parameters:
        N_print_list (int): Number of matrices that will be printed as a list if the shape of the object is 1D. Default: None
        print_list_spaces (int): Number ofspaces between matrices if they are printed as a list. Default: None
    """
    global N_print_list, print_list_spaces
    if list_spaces is not None:
        print_list_spaces = list_spaces
    if N_list is not None:
        N_print_list = N_list


################################################################################
# Main class
################################################################################


class Jones_vector(Py_pol):
    """Class for Jones vectors.

    Parameters:
        M (np.ndarray): 2xN array containing all the Jones vectors.
        name (string): Name of the object for print purposes.
        shape (tuple or list): Shape desired for the outputs.
        size (int): Number of stores Jones vectors.
        ndim (int): Number of dimensions for representation purposes.
        type (string): Type of the object ('Jones_vector'). This is used for determining the object class as using isinstance may throw unexpected results in .ipynb files.

    Attributes:
        self.parameters (class): Class containing the measurable parameters of the Jones vectors.
        self.checks (class): Class containing the methods that check something about the Jones vectors.
    """
    __array_priority__ = 10000

    ############################################################################
    # Operations
    ############################################################################

    def __init__(self, name='E'):
        """Triggers during the Jones_vector inicialization.

        Parameters:
            name (string): Name of the object for representation purposes. Default: 'E'

        Returns:
            (Jones_vector): Result.
        """
        super().__init__(name=name, _class="Jones_vector")
        self.parameters = Parameters_Jones_Vector(self)
        self.checks = Checks_Jones_Vector(self)

    def __add__(self, other):
        """Adds two Jones vectors.

        Parameters:
            other (Jones_vector): 2nd Jones vector to add.

        Returns:
            (Jones_vector): Result.
        """
        if other.type == 'Jones_vector':
            j3 = Jones_vector()
            j3.M = self.M + other.M
            j3.shape = take_shape((self, other))
            if change_names:
                j3.name = self.name + " + " + other.name
            return j3
        else:
            raise ValueError('other is {} instead of Jones_vector.'.format(
                type(other)))

    def __sub__(self, other):
        """Substracts two Jones vectors.

        Parameters:
            other (Jones_vector): 2nd Jones vector to substract.

        Returns:
            (Jones_vector): Result.
        """
        if other.type == 'Jones_vector':
            j3 = Jones_vector()
            j3.M = self.M - other.M
            j3.shape = take_shape((self, other))
            if change_names:
                j3.name = self.name + " - " + other.name
            return j3
        else:
            raise ValueError('other is {} instead of Jones_vector.'.format(
                type(other)))

    def __mul__(self, other):
        """Multiplies a Jones vectors by a number.

        Parameters:
            other (float or numpy.ndarray): number to multiply.

        Returns:
            (Jones_vector): Result.
        """
        j3 = Jones_vector()

        if isinstance(other, number_types):
            j3.M = self.M * other
            j3.shape = self.shape
            if change_names:
                j3.name = self.name + " * " + str(other)
        elif isinstance(other, np.ndarray):
            if other.size == self.size or self.size == 1:
                Ex, Ey = self.parameters.components(shape=False)
                j3.from_components(Ex * other.flatten(), Ey * other.flatten())
                j3.shape = take_shape((self, other.shape))
            else:
                raise ValueError(
                    'The number of elements in other and {} is not the same'.
                    format(self.name))
        else:
            raise ValueError(
                'other is not a number or a numpy array, is {}'.format(
                    type(other)))
        # print("in __mul__ jones vector")
        return j3

    def __rmul__(self, other):
        """Multiplies a Jones vectors by a number.

        Parameters:
            other (float or numpy.ndarray): number to multiply.

        Returns:
            (Jones_vector): Result.
        """
        j3 = Jones_vector()

        if isinstance(other, number_types):
            j3.M = other * self.M
            j3.shape = self.shape
            if change_names:
                j3.name = self.name + " * " + str(other)
        elif isinstance(other, np.ndarray):
            if other.size == self.size or self.size == 1:
                Ex, Ey = self.parameters.components(shape=False)
                j3.from_components(Ex * other.flatten(), Ey * other.flatten())
                j3.shape = take_shape((self, other.shape))
            else:
                raise ValueError(
                    'The number of elements in other and {} is not the same: {} vs {}'
                    .format(self.name, self.size, other.size))
        else:
            raise ValueError(
                'other is not a number or a numpy array, is {}'.format(
                    type(other)))
        # print("in __rmul__ jones vector")
        return j3

    def __truediv__(self, other):
        """Divides a Jones vector.

        Parameters:
            other (number or numpy.ndarray): 2nd element to divide.

        Returns:
            (Jones_vector): Result.
        """
        j3 = Jones_vector()
        # Find the class of the other object
        if isinstance(other, number_types):
            j3.M = self.M / other
            j3.shape = self.shape
            if change_names:
                j3.name = self.name + r" / " + str(other)
        elif isinstance(other, np.ndarray):
            if other.size == self.size or self.size == 1:
                Ex, Ey = self.parameters.components(shape=False)
                j3.from_components(Ex / other.flatten(), Ey / other.flatten())
                j3.shape = take_shape((self, other.shape))
            else:
                raise ValueError(
                    'The number of elements in other and {} is not the same'.
                    format(self.name))
        else:
            raise ValueError('other is Not number')
        return j3

    def __repr__(self):
        """
        Represents the Jones vector with print().
        """
        # Extract the components
        Ex, Ey = self.parameters.components()
        if np.sum(self.M.imag**2)/self.size < tol_default:
            Ex, Ey = Ex.real, Ey.real
        # If the object is empty, say it
        if self.size == 0:
            return '{} is empty\n'.format(self.name)
        # If the object is 0D or 1D, print it like a list or inline
        elif self.size == 1 or self.shape is None or len(self.shape) < 2:
            if self.size <= N_print_list:
                list = self.get_list(out_number=False)
                l0_name = "{} = \n".format(self.name)
                l1_name = PrintMatrices(list, print_list_spaces)
                return l0_name + l1_name
            else:
                l0_name = "{} Ex = {}".format(self.name, Ex)
                l1_name = " " * len(self.name) + " Ey = {}".format(Ey)
        # Print higher dimensionality as pure arrays
        else:
            l0_name = "{} Ex = \n{}".format(self.name, Ex)
            l1_name = "{} Ey = \n{}".format(self.name, Ey)
        return l0_name + '\n' + l1_name + '\n'

    def __getitem__(self, index):
        """
        Implements object extraction from indices.
        """
        if change_names:
            E = Jones_vector(self.name + '_picked')
        else:
            E = Jones_vector(self.name)
        # If the indices are 1D, act upon the matrix directly
        if isinstance(index, (int, slice)) and self.ndim > 1:
            E.from_matrix(self.M[:, index])
        elif isinstance(index,
                        np.ndarray) and index.ndim == 1 and self.ndim > 1:
            E.from_matrix(self.M[:, index])
        # If not, act upon the components
        else:
            Ex, Ey = self.parameters.components(out_number=False)
            M = np.array([Ex[index], Ey[index]])
            E.from_matrix(M)

        return E

    def __setitem__(self, index, data):
        """
        Implements object inclusion from indices.
        """
        # Check that data is a correct pypol object
        if data.type == 'Jones_vector':
            data2 = data
        elif data.type == 'Stokes':
            data2 = Jones_vector(data.name)
            data2.from_Stokes(data)
        else:
            raise ValueError(
                'data is type {} instead of Jones_vector or Stokes.'.format(
                    data.type))
        # Change to complex if necessary
        if np.iscomplexobj(data2.M):
            self.M = np.array(self.M, dtype=complex)
        # If the indices are 1D, act upon the matrix directly
        if isinstance(index, int) and self.ndim > 1:
            self.M[:, index] = np.squeeze(data2.M)
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
        elif isinstance(index,
                        np.ndarray) and index.ndim == 1 and self.ndim > 1:
            self.M[:, index] = data2.M
        # If not, act upon the components
        else:
            Ex, Ey = self.parameters.components(out_number=False)
            Ex_new, Ey_new = data2.parameters.components(out_number=False)
            Ex[index] = np.squeeze(Ex_new)
            Ey[index] = np.squeeze(Ey_new)
            self.from_components(Ex, Ey)

    def __eq__(self, other):
        """
        Implements equality operation.
        """
        try:
            if other.type == 'Jones_vector':
                j3 = self - other
                norm = j3.parameters.norm()
                cond = norm < tol_default
                return cond
            else:
                return False
        except:
            return False

    ############################################################################
    # Manipulation
    ############################################################################

    def rotate(self, angle=0, keep=False, change_name=change_names):
        """Rotates a jones vector a certain angle.

        M_rotated = rotation_matrix_Jones(-angle) * self.M

        Parameters:
            angle (float): Rotation angle in radians. Default: 0
            keep (bool): If True, the original element is not updated. Default: False.
            change_name (bool): If True and angle is of size 1, changes the object name adding @ XX deg, being XX the total rotation angle. Default: True.

        Returns:
            (Jones_vector): Rotated object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Prepare variables
        angle, new_obj, new_shape = prepare_variables(
            [angle], expand=[True], obj=new_obj, give_shape=True)
        # Calculate the array of rotation matrices
        Mrot = rotation_matrix_Jones(-angle)
        # The 1-D case is much simpler. Differenciate it.
        if new_obj.size <= 1 and angle.size == 1:
            E = np.squeeze(Mrot) @ new_obj.M
        else:
            # Move axes of the variables to allow multiplication
            Mrot = np.moveaxis(Mrot, 2, 0)
            E = np.moveaxis(new_obj.M, 1, 0)
            E = np.expand_dims(E, 2)
            # Multiply
            E = Mrot @ E
            # Reshape again to accomodate to our way of representing elements
            E = np.moveaxis(np.squeeze(E), 0, 1)
        # Update
        new_obj.from_matrix(E)
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
    def reciprocal(self, keep=False, change_name=change_names):
        """Calculates the recirpocal of the Jones vector, so the light is propagated in the opposite direction. It is calculated as Ey = -Ey.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp TODO.

        Parameters:
            keep (bool): If True, the original element is not updated. Default: False.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Jones_vector): Reciprocal Jones matrix.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Extract the components
        old_shape = new_obj.shape
        Ex, Ey = self.parameters.components(shape=False)
        new_obj.from_components(Ex, -Ey)
        new_obj.shape = old_shape
        # Fix the name
        if change_name:
            new_obj.name = 'Reciprocal of ' + new_obj.name
        return new_obj

    def orthogonal(self, keep=False, change_name=change_names):
        """Calculates the orthogonal of the Jones vector.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 125.

        Parameters:
            keep (bool): If True, the original element is not updated. Default: False.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Jones_vector): Reciprocal Jones matrix.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Extract the characteristic angles
        alpha, delay = self.parameters.charac_angles(shape=False)
        phase = self.parameters.global_phase(shape=False)
        # Calculate the new ones
        alpha = np.pi / 2 - alpha
        delay = delay + np.pi
        # Create the object
        old_shape = new_obj.shape
        new_obj.general_charac_angles(
            alpha=alpha, delay=delay, global_phase=phase)
        new_obj.shape = old_shape
        # Fix the name
        if change_name:
            new_obj.name = 'Orthogonal of ' + new_obj.name
        return new_obj



    def sum(self, axis=None, keep=False, change_name=change_names):
        """Calculates the sum of Jones vectors stored in the object.

        Parameters:
            axis (int, list or tuple): Axes along which the summatory is performed. If None, all matrices are summed.
            keep (bool): If True, the original element is not updated. Default: False.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Jones_vector): Modified object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Simple case
        if axis is None or new_obj.ndim <= 1:
            M = np.sum(new_obj.M, axis=1)
        # Complicated case
        else:
            # Calculate maximum axis
            if isinstance(axis, int):
                axis = axis + 1
                m = axis
            else:
                axis = np.array(axis) + 1
                m = np.max(axis)
            # Check that the axes are correct
            if m >= new_obj.ndim + 1:
                raise ValueError(
                    'Axis {} greater than the number of dimensions of {}, which is {}'
                    .format(m, new_obj.name, new_obj.ndim))
            # Reshape M to fit the current shape
            shape = [2] + new_obj.shape
            M = np.reshape(new_obj.M, shape)
            # check if the axis is int or not
            if isinstance(axis, int):
                M = np.sum(M, axis=axis)
            else:
                M = np.sum(M, axis=tuple(axis))
        # Create the object and return it
        new_obj.from_matrix(M)
        if change_names:
            new_obj.name = 'Sum of ' + new_obj.name
        return new_obj



    def normalize(self, kind='amplitude', keep=False):
        """Function that normalizes the Jones vectors in amplitude or intensity.

        Parameters:
            kind (string): Field amplitude or intensity. Default: AMPLITUDE.
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Jones_vector): Normalized Jones vector.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Calculate the normalization factor
        if kind in ('amplitude', 'amplitude', 'AMPLITUDE', 'Amp', 'amp', 'AMP',
                    'a', 'A'):
            norm, _ = self.parameters.amplitudes(shape=False)
        else:
            norm, _ = self.parameters.intensity(shape=False)
        # Calculate the normalized vectors
        Ex, Ey = self.parameters.components(shape=False)
        new_obj.from_components(Ex / norm, Ey / norm)
        # Return
        return new_obj

    def rotate_to_azimuth(self, azimuth=0, keep=False):
        """Function that rotates the Jones vector to have a certain azimuth.

        Parameters:
            azimuth (string or np.ndarray): Azimuth of the Jones vector. 'X', 'Y', '-X' and '-Y' are the same as 0, 90, 180 and 270 degrees respectively. Default: 0.
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Jones_vector): Jones vector rotated.
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
        azimuth, new_obj = prepare_variables(
            [azimuth], expand=[True], obj=new_obj)
        # Calculate the new vector
        I = new_obj.parameters.intensity()
        el = new_obj.parameters.ellipticity_angle()
        new_obj.general_azimuth_ellipticity(
            azimuth=azimuth, ellipticity=el, intensity=I)
        # Return
        return new_obj

    # @_actualize_
    def remove_global_phase(self, keep=False):
        """Function that transforms the Jones vector removing the global phase, so the first component of the electric field is real and positive.

        Parameters:
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Jones_vector): Recalculated Jones vector.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Calculate and remove the global phase
        old_shape = new_obj.shape
        phase = new_obj.parameters.global_phase(shape=False)
        phase = np.exp(1j * phase)
        Ex, Ey = self.parameters.components(shape=False)
        new_obj.from_components(Ex / phase, Ey / phase)
        new_obj.shape = old_shape
        # Return
        return new_obj

    def add_global_phase(self, phase=0, keep=False):
        """Function that adds a phase to the Jones vector.

        Parameters:
            phase (float or np.ndarray): Phase to be added to the Jones vector. Default: 0.
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Jones_vector): Recalculated Jones vector.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Prepare variables
        phase, new_obj, new_shape = prepare_variables(
            [phase], expand=[True], obj=new_obj, give_shape=True)
        # Add the phase
        Ex, Ey = new_obj.parameters.components(shape=False)
        phase = np.exp(1j * phase)
        new_obj.from_components(Ex * phase, Ey * phase)
        new_obj.shape, _ = select_shape(new_obj, new_shape)
        # Return
        return new_obj

    def set_global_phase(self, phase=0, keep=False):
        """Function that sets the phase of the Jones vector.

        Parameters:
            phase (float or np.ndarray): Phase to be added to the Jones vector. Default: 0.
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Jones_vector): Recalculated Jones vector.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Prepare variables
        phase, new_obj, new_shape = prepare_variables(
            [phase], expand=[True], obj=new_obj, give_shape=True)
        # Remove the current phase
        new_obj.remove_global_phase()
        # Add the phase
        new_obj.add_global_phase(phase)
        new_obj.shape, _ = select_shape(new_obj, new_shape)
        # Return
        return new_obj

    def add_delay(self, delay=0, keep=False):
        """Function that adds a phase to the Y component of the electric field of the Jones vector.

        Parameters:
            delay (float or np.ndarray): Phase to be added to the Y component of the Jones vector. Default: 0.
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Jones_vector): Recalculated Jones vector.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Prepare variables
        delay, new_obj, new_shape = prepare_variables(
            [delay], expand=[True], obj=new_obj, give_shape=True)
        # Add the phase
        Ex, Ey = new_obj.parameters.components(shape=False)
        delay = np.exp(1j * delay)
        new_obj.from_components(Ex, Ey * delay)
        new_obj.shape, _ = select_shape(new_obj, new_shape)
        # Return
        return new_obj


    #########################################################################
    # Draw
    #########################################################################

    def draw_ellipse(self, *args, **kwargs):
        """Draws polarization ellipse of Jones vector.

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

    def draw_poincare(self, *args, **kwargs):
        """"Draws jones vector in the Poincare sphere.

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

    ############################################################################
    # Creation
    ############################################################################

    def from_components(self,
                        Ex,
                        Ey=None,
                        global_phase=0,
                        length=1,
                        shape_like=None,
                        shape=None):
        """Creates the Jones vector object form the arrays of electric field components.

        Parameters:
            Ex (numpy.array, float or 2x1 tuple/list): Electric field amplitude along x axis. This variable can also be a 2x1 tuple or list containing both Ex and Ey.
            Ey (numpy.array or float): Electric field amplitude along y axis. Default: None.
            global_phase (float or numpy.ndarray): Adds a global phase to the Jones vector. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_vector): Created object.
        """
        # Separate components if required
        if isinstance(Ex, tuple) or isinstance(Ex, list):
            Ey = Ex[1]
            Ex = Ex[0]
        # Prepare variables
        (Ex, Ey, global_phase), new_shape = prepare_variables(
            vars=[Ex, Ey, global_phase],
            expand=[True, True, False],
            length=length,
            give_shape=True)
        # Add global Phase
        if global_phase != 0:
            Ex = Ex * np.exp(1j * global_phase)
            Ey = Ey * np.exp(1j * global_phase)
        # Store
        self.M = np.array([Ex, Ey])
        self.shape, _ = select_shape(
            self, shape_var=new_shape, shape_fun=shape, shape_like=shape_like)
        return self

    def from_matrix(self, M, shape_like=None, shape=None):
        """Create a Jones vector from an external matrix.

        Parameters:
            M (numpy.ndarray or float): New matrix.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_vector): Created object.
        """
        # Check if the matrix is of the correct Size
        M = np.array(M)
        s = M.size
        # Case 1D
        if M.ndim == 1:
            if M.size % 2 == 0:
                M = np.reshape(M, (2, int(M.size / 2)))
                new_shape = None
            else:
                raise ValueError('M must have an even number of elements.')
        # Case 2D
        elif M.ndim == 2:
            s = M.shape
            if s[0] != 2:
                if s[1] == 2:
                    M = M.transpose()
                    new_shape = [s[0]]
                else:
                    raise ValueError('The Jones vector must be a 2xN array')
            else:
                new_shape = [s[1]]
        # Case 3+D
        else:
            sh = np.array(M.shape)
            if (sh == 2).any:
                # Store the shape of the desired outputs
                ind = np.argmin(~(sh == 2))
                # Store info
                M = np.array([
                    np.take(M, 0, axis=ind).flatten(),
                    np.take(M, 1, axis=ind).flatten()
                ])
                new_shape = np.delete(sh, ind)
            else:
                raise ValueError(
                    'The Jones vector array must have one axis with 2 elements'
                )
        self.M = M
        # self.size = M.size / 2
        self.shape, _ = select_shape(
            self, shape_var=new_shape, shape_fun=shape, shape_like=shape_like)
        return self


    def from_distribution(self,
                          Ex,
                          Ey,
                          ind_d=-1,
                          method='direct',
                          N_periods=1,
                          shape_like=None,
                          shape=None):
        """Determine Jones vector from a temporal or spatial electric field distribution [(Ex(t), Ey(t)].

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
            Ex = np.abs(ex).mean(axis=0)
            Ey = np.abs(ey).mean(axis=0)
            if method in ('fit', 'FIT', 'Fit'):
                phaseX = np.unwrap(np.angle(ex), axis=0)
                phaseY = np.unwrap(np.angle(ey), axis=0)
                x = np.arange(ex.shape[0])
                _, phaseX = np.polyfit(x=x, y=phaseX, deg=1)
                _, phaseY = np.polyfit(x=x, y=phaseY, deg=1)
            else:
                # Use just the first value
                phaseX = np.angle(ex[0, :])
                phaseY = np.angle(ey[0, :])
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
                result = optimize.least_squares(
                    fit_distribution,
                    par0,
                    args=(ex, 0),
                    bounds=(min_limit, max_limit))
                phaseX = result.x[:s]
                Ex = result.x[2 * s:]
                # prepare variables for Y fit
                par0 = np.concatenate((phaseY, one * cte, Ey))
                max_limit = np.concatenate((2 * np.pi * one, np.pi * one, Ey))
                # Y fit
                result = optimize.least_squares(
                    fit_distribution,
                    par0,
                    args=(ey, 0),
                    bounds=(min_limit, max_limit))
                phaseY = result.x[:s]
                Ey = result.x[2 * s:]
        # Finish calculations
        Ex = Ex * np.exp(1j * phaseX)
        Ey = Ey * np.exp(1j * phaseY)
        self.from_components(Ex, Ey)
        self.shape, _ = select_shape(
            self, shape_var=new_shape, shape_fun=shape, shape_like=shape_like)
        return self

    def from_Stokes(self, S, shape_like=None, shape=None, verbose=False):
        """Create a Jones vector from a Stokes vector. This operation is only meaningful for pure (totally polarized) Stokes vectors. For the rest of them, only the polarized part is transformed, and a warning is printed.

        Parameters:
            S (Stokes): Stokes vector object.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.

        Returns:
            (Jones_vector): Created object.
        """
        # Extract the totally polarized part
        Sp, Su = S.parameters.polarized_unpolarized(
            shape=shape, shape_like=shape_like)
        # Print a warning if the unpolarized part is meaningful
        DOP = S.parameters.degree_depolarization(shape=False, out_number=False)
        if np.any(DOP > tol_default):
            warnings.warn(
                'Non-totally polarized Stokes vector transformed into a Jones vector'
            )
        # Calculate required parameters of the polarized Stokes vector
        intensity = Sp.parameters.intensity()
        azimuth = Sp.parameters.azimuth()
        ellipticity = Sp.parameters.ellipticity_angle()
        global_phase = Sp.parameters.global_phase(give_nan=False)
        # Generate a Jones vector from those parameters
        self.general_azimuth_ellipticity(
            azimuth=azimuth,
            ellipticity=ellipticity,
            intensity=intensity)
        self.set_global_phase(global_phase)
        # Print the result if required
        if verbose:
            print(self)
        return self

    # @_actualize_
    # def to_Stokes(self, p=1):
    #     """Function that converts Jones light states to Stokes states.
    #     Parameters:
    #         p (float or 1x2 float): Degree of polarization, or
    #             [linear, circular] degrees of polarization.
    #     Returns:
    #         S (Stokes object): Stokes state."""
    #     # Check if we are using linear/circular or global polarization degree
    #
    #     if np.size(p) T 1:
    #         (p1, p2) = (p, p)
    #     else:
    #         (p1, p2) = (p[0], p[1])
    #
    #     E = self.M
    #     # Calculate the vector
    #     (Ex, Ey) = (E[0,:], E[1,:])
    #     S = np.zeros([1, 4])
    #     s0 = abs(Ex)**2 + abs(Ey)**2
    #     s1 = (abs(Ex)**2 - abs(Ey)**2) * p1
    #     s2 = 2 * np.real(Ex * np.conj(Ey)) * p1
    #     s3 = -2 * np.imag(Ex * np.conj(Ey)) * p2
    #
    #     S1 = Stokes(self.name)
    #     S1.from_elements(s0, s1, s2, s3)
    #     return S1

    def linear_light(self,
                     amplitude=None,
                     azimuth=0,
                     intensity=1,
                     global_phase=0,
                     length=1,
                     shape_like=None,
                     shape=None):
        """Jones vector for polarizer linear light.

        Parameters:
            amplitude (numpy.array or float): Array of electric field amplitude. Overrides inetnsity if it is different than None. Default: None.
            azimuth (numpy.array or float): Array of azimuths. Default: 0.
            intensity (numpy.array or float): Array of intensity. Default: 1.
            global_phase (float or numpy.ndarray): Adds a global phase to the Jones vector. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_vector): Created object.
        """
        # Use amplitude if specified
        if amplitude is None:
            amplitude = np.sqrt(intensity)
        # Prepare variables
        (amplitude, azimuth, global_phase), new_shape = prepare_variables(
            vars=[amplitude, azimuth, global_phase],
            expand=[True, False, False],
            length=length,
            give_shape=True)
        # Add global Phase
        if global_phase != 0:
            amplitude = amplitude * np.exp(1j * global_phase)
        # Calculate
        self.M = np.array(
            [amplitude * np.cos(azimuth), amplitude * np.sin(azimuth)])
        # self.size = amplitude.size
        self.shape, _ = select_shape(
            self, shape_var=new_shape, shape_fun=shape, shape_like=shape_like)
        return self

    def circular_light(self,
                       kind='d',
                       amplitude=None,
                       intensity=1,
                       global_phase=0,
                       length=1,
                       shape_like=None,
                       shape=None):
        """Jones vector for polarizer circular light

        Parameters:
            kind (str): Can be 'd','r', 'right', 'dextro' or 'l', 'i', 'left', 'levo'.
            amplitude (numpy.array or float): Array of electric field amplitude. Overrides intensity if it is different than None. Default: None.
            intensity (numpy.array or float): Array of intensity. Default: 1.
            global_phase (float or numpy.ndarray): Adds a global phase to the Jones vector. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_vector): Created object.
        """
        # Use amplitude if specified
        if amplitude is None:
            amplitude = np.sqrt(intensity)
        (amplitude, global_phase), new_shape = prepare_variables(
            vars=[amplitude, global_phase],
            expand=[True, False],
            length=length,
            give_shape=True)
        # Add global Phase
        if (global_phase != 0).all:
            amplitude = amplitude * np.exp(1j * global_phase)
        # Calculate
        amplitude = amplitude / np.sqrt(2)
        if kind.lower() in ["r", "d", "right", "dextro"]:
            self.M = np.array([amplitude, amplitude * 1j])
        else:
            self.M = np.array([amplitude, -amplitude * 1j])
        self.shape, _ = select_shape(
            self, shape_var=new_shape, shape_fun=shape, shape_like=shape_like)
        return self

    def elliptical_light(self,
                         a=1,
                         b=1,
                         kind='r',
                         azimuth=0,
                         global_phase=0,
                         length=1,
                         shape_like=None,
                         shape=None):
        """Jones vector of the most general light calculated from the polarization ellipse parameters.

        Parameters:
            a (numpy.array or float): Array of electric amplitude of x axis. Default: 1.
            b (numpy.array or float): Array of electric amplitude of y axis. Default: 1.
            kind (str): Can be 'd','r', 'right', 'dextro' or 'l', 'i', 'left', 'levo'. Default: 'r'.
            azimuth (numpy.array or float): Angle of the a axis respect to the x axis. Default: 0.
            global_phase (float or numpy.ndarray): Adds a global phase to the Jones vector. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_vector): Created object.
        """
        (a, b, azimuth, global_phase), new_shape = prepare_variables(
            vars=[a, b, azimuth, global_phase],
            expand=[True, True, False, False, False],
            length=length,
            give_shape=True)
        # Add global Phase
        if (global_phase != 0).all:
            a = a * np.exp(1j * global_phase)
            b = b * np.exp(1j * global_phase)
        # Create the Jones vectors
        if kind.lower() in ["r", "d", "right", "dextro"]:
            M = np.array([a, b * 1j])
        else:
            M = np.array([a, -b * 1j])
        self.from_matrix(M)
        self.rotate(azimuth)
        self.shape, _ = select_shape(
            self, shape_var=new_shape, shape_fun=shape, shape_like=shape_like)
        return self
    
    def general(self,
                azimuth=0,
                ellipticity=0,
                alpha=None,
                delay=None,
                *args,
                **kwargs):
        """General Jones vector.

        Parameters:
            azimuth (numpy.array or float): [0, pi]: Azimuth. Only used if alpha and delay are None. Default: 0.
            ellipticity (numpy.array or float): [-pi/4, pi/4]: Ellipticity angle. Only used if alpha and delay are None. Default: 0.
            alpha (numpy.ndarray, float or none): Ratio between amplitude of components Ex/Ey of electric field. Overrides azimuth and ellipticity if not None. Default: None.
            delay (numpy.ndarray, float or None): Phase delay between Ex and Ey components of the electric field. Overrides azimuth and ellipticity if not None. Default: None.
            amplitude (numpy.array or float): Array of electric field amplitude. Overrides inetnsity if it is different than None. Default: None.
            intensity (numpy.array or float): Array of intensity. Default: 1.
            global_phase (float or numpy.ndarray): Adds a global phase to the Jones vector. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of azimuth and ellipticity or alpha and delay are forced to be in their correct range. Default: True.

        Returns:
            (Jones_vector): Created object.
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
        

    def general_azimuth_ellipticity(self,
                                    azimuth=0,
                                    ellipticity=0,
                                    amplitude=None,
                                    intensity=1,
                                    global_phase=0,
                                    length=1,
                                    shape_like=None,
                                    shape=None,
                                    force_limits=False):
        """Jones vector from azimuth and ellipticity angle. Maintiained for retrocompatibility.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 137.

        Parameters:
            azimuth (numpy.array or float): [0, pi]: Azimuth. Default: 0.
            ellipticity (numpy.array or float): [-pi/4, pi/4]: Ellipticity angle. Default: 0.
            amplitude (numpy.array or float): Array of electric field amplitude. Overrides inetnsity if it is different than None. Default: None.
            intensity (numpy.array or float): Array of intensity. Default: 1.
            global_phase (float or numpy.ndarray): Adds a global phase to the Jones vector. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of azimuth and ellipticity are forced to be in the range [0, pi) and [-pi/4, pi/4] respectively. Default: False.

        Returns:
            (Jones_vector): Created object.
        """
        # Use amplitude if specified
        if amplitude is None:
            amplitude = np.sqrt(intensity)
        # Prepare variables
        (amplitude, azimuth, ellipticity,
         global_phase), new_shape = prepare_variables(
             vars=[amplitude, azimuth, ellipticity, global_phase],
             expand=[True, True, False, False],
             length=length,
             give_shape=True)
        if force_limits:
            azimuth = put_in_limits(azimuth, "azimuth")
            ellipticity = put_in_limits(ellipticity, "ellipticity")
        # Add global Phase
        if (global_phase != 0).all:
            amplitude = amplitude * np.exp(1j * global_phase)
        # NaN check
        if azimuth is np.nan and ellipticity is np.nan:
            raise ValueError(
                "general_azimuth_ellipticity: need total polarized light ")
        # Azimuth can be NaN if ellipticity is +-45º
        cond1 = np.isnan(azimuth)
        cond2 = (ellipticity - 45 * degrees) < eps
        cond3 = (ellipticity + 45 * degrees) < eps
        cond = cond1 * (cond2 + cond3)
        azimuth[cond] = 0
        # Calculate the field
        e1 = np.cos(ellipticity) * np.cos(azimuth) - 1j * \
            np.sin(ellipticity) * np.sin(azimuth)
        e2 = np.cos(ellipticity) * np.sin(azimuth) + 1j * \
            np.sin(ellipticity) * np.cos(azimuth)
        self.M = np.array([amplitude * e1, amplitude * e2])
        # self.size = amplitude.size
        self.shape, _ = select_shape(
            self, shape_var=new_shape, shape_fun=shape, shape_like=shape_like)
        return self

    def general_charac_angles(self,
                              alpha=0,
                              delay=0,
                              amplitude=None,
                              intensity=1,
                              global_phase=0,
                              length=1,
                              shape_like=None,
                              shape=None,
                              force_limits=False):
        """Jones vector from characteristic angles. Maintiained for retrocompatibility.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 137.

        Parameters:
            alpha (numpy.ndarray or float): Ratio between amplitude of components Ex/Ey of electric field. Default:0
            delay (numpy.ndarray or float): Phase delay between Ex and Ey components of the electric field. Default: 0.
            amplitude (numpy.array or float): Array of electric field amplitude. Overrides inetnsity if it is different than None. Default: None.
            intensity (numpy.array or float): Array of intensity. Default: 1.
            global_phase (float or numpy.ndarray): Adds a global phase to the Jones vector. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of alpha and delay are forced to be in the range [0, pi/2] and [0, 2*pi) respectively. Default: False.

        Returns:
            (Jones_vector): Created object.
        """
        # Use amplitude if specified
        if amplitude is None:
            amplitude = np.sqrt(intensity)
        # Prepare variables
        (amplitude, alpha, delay, global_phase), new_shape = prepare_variables(
            vars=[amplitude, alpha, delay, global_phase],
            expand=[True, False, False, False],
            length=length,
            give_shape=True)
        if force_limits:
            alpha = put_in_limits(alpha, "alpha")
            delay = put_in_limits(delay, "delay")
        # Add global Phase
        if (global_phase != 0).all:
            amplitude = amplitude * np.exp(1j * global_phase)
        # NaN check
        if delay is np.nan and alpha is np.nan:
            raise ValueError(
                "general_azimuth_ellipticity: need total polarized light ")
        # Calculate the field
        e1 = np.cos(alpha) * np.exp(-1j * delay / 2)
        e2 = np.sin(alpha) * np.exp(1j * delay / 2)
        self.from_components(amplitude * e1, amplitude * e2)
        # self.size = amplitude.size
        self.shape, _ = select_shape(
            self, shape_var=new_shape, shape_fun=shape, shape_like=shape_like)
        return self


################################################################################
# Parameters
################################################################################


class Parameters_Jones_Vector(object):
    """Class for Jones vector parameters.

    Parameters:
        jones_vector (Jones_vector): Jones Vector

    Attributes:
        self.parent (Jones_vector)
    """

    def __init__(self, Jones_vector):
        self.parent = Jones_vector

    def __repr__(self):
        """Prints all the parameters"""
        self.get_all(verbose=True, draw=True)
        return ''

    def get_all(self, verbose=False, draw=False):
        """Creates a dictionary with all the parameters of Jones vector.

        Parameters:
            verbose (bool): If True, print all parameters. Default: False.
            draw (bool): If True, draw all plots/images of the parameters. Default: False.

        Returns:
            (dict): Dictionary with parameters of Jones vector.
        """
        dict_params = {}
        dict_params['Ex'], dict_params['Ey'] = self.components(
            verbose=verbose, draw=draw)
        dict_params['E0x'], dict_params['E0y'] = self.amplitudes(
            verbose=verbose, draw=draw)
        dict_params['intensity'] = self.intensity(
            verbose=verbose, draw=draw)
        dict_params['norm'] = self.norm(verbose=verbose, draw=draw)
        dict_params['irradiance'] = self.irradiance(
            verbose=verbose, draw=draw)
        dict_params['alpha'] = self.alpha(verbose=verbose, draw=draw)
        dict_params['delay'] = self.delay(verbose=verbose, draw=draw)
        dict_params['azimuth'] = self.azimuth(verbose=verbose, draw=draw)
        dict_params['ellipticity_angle'] = self.ellipticity_angle(
            verbose=verbose, draw=draw)
        dict_params['global_phase'] = self.global_phase(
            verbose=verbose, draw=draw)
        dict_params['a'], dict_params['b'] = self.ellipse_axes(
            verbose=verbose, draw=draw)
        dict_params['ellipticity_param'] = self.ellipticity_param(
            verbose=verbose, draw=draw)
        dict_params['eccentricity'] = self.eccentricity(
            verbose=verbose, draw=draw)
        dict_params[
            'degree_circular_polarization'] = self.degree_circular_polarization(
                verbose=verbose, draw=draw)
        dict_params[
            'degree_linear_polarization'] = self.degree_linear_polarization(
                verbose=verbose, draw=draw)

        return dict_params

    def matrix(self, shape=None, shape_like=None):
        """Returns the numpy array of Jones vectors.

        Parameters:
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.


        Returns:
            (numpy.ndarray): 2xN numpy array.
        """
        shape, _ = select_shape(
            obj=self.parent, shape_fun=shape, shape_like=shape_like)
        if shape is not None and len(shape) > 1:
            shape = tuple([2] + list(shape))
            M = np.reshape(self.parent.M, shape)
        else:
            M = self.parent.M
        return M

    def components(self,
                   give_Ez=False,
                   out_number=True,
                   shape_like=None,
                   shape=None,
                   verbose=False,
                   draw=False,
                   kind='jones'):
        """Calculates the x and y field components of the Jones vector.

        Parameters:
            give_Ez (bool): If True, it returns the z component of the electric field (all values will be 0). Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            Ex (numpy.ndarray or float): Array of electric field along x axis.
            Ey (numpy.ndarray or float): Array of electric field along y axis.
            Ez (numpy.ndarray, optional): Array of electric field along z axis.
        """
        # Calculate the components
        Ex = self.parent.M[0, :]
        Ey = self.parent.M[1, :]
        # If the result is a number and the user asks for it, return a float
        if out_number and Ex.size == 1:
            Ex = Ex[0]
            Ey = Ey[0]
        # Calculate Ez and reshape if required
        if give_Ez:
            Ez = np.zeros_like(Ex)
            Ex, Ey, Ez = reshape(
                [Ex, Ey, Ez],
                shape_like=shape_like,
                shape_fun=shape,
                obj=self.parent)
        else:
            Ex, Ey = reshape(
                [Ex, Ey],
                shape_like=shape_like,
                shape_fun=shape,
                obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The elctric field components of {} are (V/m):'.format(
                self.parent.name)
            if give_Ez:
                PrintParam(
                    param=(Ex, Ey, Ez),
                    shape=self.parent.shape,
                    title=('Ex (V/m)', 'Ey (V/m)', 'Ez (V/m)'),
                    heading=heading,
                    verbose=verbose,
                    draw=draw,
                    kind=kind)
            else:
                PrintParam(
                    param=(Ex, Ey),
                    shape=self.parent.shape,
                    title=('Ex (V/m)', 'Ey (V/m)'),
                    heading=heading,
                    verbose=verbose,
                    draw=draw,
                    kind=kind)
        # Return
        if give_Ez:
            return [Ex, Ey, Ez]
        else:
            return [Ex, Ey]

    def amplitudes(self,
                   give_Ez=False,
                   out_number=True,
                   shape_like=None,
                   shape=None,
                   verbose=False,
                   draw=False):
        """Calculates the x and y field amplitudes of the Jones vector.

        Parameters:
            give_Ez (bool): If True, it returns the z component of the electric field (all values will be 0). Default: False.
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
        # Calculate intensity
        Ex, Ey = self.components(out_number=out_number)
        E0x = np.abs(Ex)
        E0y = np.abs(Ey)
        # Calculate Ez and reshape if required
        if give_Ez:
            E0z = np.zeros_like(E0x)
            E0x, E0y, E0z = reshape(
                [E0x, E0y, E0z],
                shape_like=shape_like,
                shape_fun=shape,
                obj=self.parent)
        else:
            E0x, E0y = reshape(
                [E0x, E0y],
                shape_like=shape_like,
                shape_fun=shape,
                obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The elctric field amplitudes of {} are (V/m):'.format(
                self.parent.name)
            if give_Ez:
                PrintParam(
                    param=(E0x, E0y, E0z),
                    shape=self.parent.shape,
                    title=('Ex (V/m)', 'Ey (V/m)', 'Ez (V/m)'),
                    heading=heading,
                    verbose=verbose,
                    draw=draw)
            else:
                PrintParam(
                    param=(E0x, E0y),
                    shape=self.parent.shape,
                    title=('Ex (V/m)', 'Ey (V/m)'),
                    heading=heading,
                    verbose=verbose,
                    draw=draw)
        # Return
        if give_Ez:
            return E0x, E0y, E0z
        else:
            return E0x, E0y

    def norm(self,
             out_number=True,
             shape_like=None,
             shape=None,
             verbose=False,
             draw=False):
        """Calculates the norm of the Jones vectors.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            norma (numpy.ndarray or float): Norm of Jones vectors.
        """
        # Calculate the norm
        norma = np.linalg.norm(self.parent.M, axis=0)

        # Reshape if neccessary
        norma = reshape(
            [norma], shape_like=shape_like, shape_fun=shape, obj=self.parent)
        if out_number and norma.size == 1:
            norma = norma[0]
        # Print the result if required
        if verbose or draw:
            heading = 'The norm of {} is (a.u.):'.format(self.parent.name)
            PrintParam(
                param=norma,
                shape=self.parent.shape,
                title='Vector norm (a.u.)',
                heading=heading,
                verbose=verbose,
                draw=draw)
        return norma

    def intensity(self,
                  out_number=True,
                  shape_like=None,
                  shape=None,
                  verbose=False,
                  draw=False):
        """Calculates the intensity of the Jones vector.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            intensity (numpy.ndarray or float): Array of intensities.
        """
        # Calculate intensity
        I = self.norm(
            out_number=out_number, shape_like=shape_like, shape=shape)**2
        # Print the result if required
        if verbose or draw:
            heading = 'The intensity of {} is (a.u.):'.format(self.parent.name)
            PrintParam(
                param=I,
                shape=self.parent.shape,
                title='Intensity (a.u.)',
                heading=heading,
                verbose=verbose,
                draw=draw)
        return I

    def irradiance(self,
                   n=1,
                   out_number=True,
                   shape_like=None,
                   shape=None,
                   verbose=False,
                   draw=False):
        """Calculates the irradiance of the Jones vector.

        Parameters:
            n (float): Refractive index of the medium. Default: 1
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            irradiance (numpy.ndarray or float): Array of irradiance.
        """
        # Calculate irradiance
        Irrad = (n / (2 * eta)) * self.norm(
            out_number=out_number, shape_like=shape_like, shape=shape)
        # Print the result if required
        if verbose or draw:
            heading = 'The irradiance of {} is (W/m^2):'.format(
                self.parent.name)
            PrintParam(
                param=Irrad,
                shape=self.parent.shape,
                title='Irradiance (W/m^2)',
                heading=heading,
                verbose=verbose,
                draw=draw)
        return Irrad

    def alpha(self,
              out_number=True,
              shape_like=None,
              shape=None,
              verbose=False,
              draw=False):
        """Calculates the angle ratio between amplitude of components Ex/Ey of electric field.

        References:
            D. Golstein "Polarized light" 2nd ed Marcel Dekker (2003), 3.4 eq.3-35

        Parameters:
            out_number (bool): if True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            E0x (numpy.ndarray or float): Array of the electric field amplitude in x axis.
            E0y (numpy.ndarray or float): Array of the electric field amplitude in y axis.

        """
        # Calculate alpha
        E0x, E0y = self.amplitudes(shape=False, out_number=False)
        alpha = np.arctan(E0y / E0x)
        alpha = put_in_limits(alpha, 'alpha', out_number=out_number)
        # Reshape if neccessary
        alpha = reshape(
            [alpha], shape_like=shape_like, shape_fun=shape, obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The ratio angle between electric field amplitudes of {} is (deg.):'.format(
                self.parent.name)
            PrintParam(
                param=alpha / degrees,
                shape=self.parent.shape,
                title='Alpha (deg.)',
                heading=heading,
                verbose=verbose,
                draw=draw)
        return alpha

    def delay(self,
              out_number=True,
              shape_like=None,
              shape=None,
              verbose=False,
              draw=False):
        """Calculates the delay (phase shift) between Ex and Ey components of the electric field.

        References:
            D. Golstein "Polarized light" 2nd ed Marcel Dekker (2003), 3.4 eq.3-33b.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float) [0, 2*pi]: Array of delay angles.
        """
        # Calculate delay
        Ex, Ey = self.components(shape=False, out_number=False)
        delay = np.angle(Ey) - np.angle(Ex)
        delay = put_in_limits(delay, 'delay', out_number=out_number)
        # Reshape if neccessary
        delay = reshape(
            [delay], shape_like=shape_like, shape_fun=shape, obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'Delay between electric field components of {} is (deg.):'.format(
                self.parent.name)
            PrintParam(
                param=delay / degrees,
                shape=self.parent.shape,
                title='Delay (deg.)',
                heading=heading,
                verbose=verbose,
                draw=draw)
        return delay

    def delta(self,
              out_number=True,
              shape_like=None,
              shape=None,
              verbose=False,
              draw=False):
        """Calculates the delay (phase shift) between Ex and Ey components of the electric field. It is the same method as delay.

    Parameters:
        out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
        shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
        shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
        verbose (bool): If True prints the parameter. Default: False.
        draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

    Returns:
        delta (numpy.ndarray or float) [0, 2*pi]: Array of delay angles.
        """
        return self.delay(
            verbose=verbose,
            draw=draw,
            out_number=out_number,
            shape_like=shape_like,
            shape=shape)

    def charac_angles(self,
                      out_number=True,
                      shape_like=None,
                      shape=None,
                      verbose=False,
                      draw=False):
        """Calculates the characteristic angles (alpha and delay) of a Jones vector.

        References:
            D. Golstein "Polarized light" 2nd ed Marcel Dekker (2003), 3.4 eq.3-33b.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            alpha (numpy.ndarray or float): [0, pi/2]: Array of alpha angles.
            delay (numpy.ndarray or float): [0, 2*pi]: Array of delay angles.
        """
        alpha = self.alpha(
            out_number=out_number, shape_like=shape_like, shape=shape)
        delay = self.delay(
            out_number=out_number, shape_like=shape_like, shape=shape)
        if verbose or draw:
            heading = 'The characteristic angles of {} are:'.format(
                self.parent.name)
            PrintParam(
                param=(alpha / degrees, delay / degrees),
                shape=self.parent.shape,
                title=('Alpha (deg.)', 'Delay (deg.)'),
                heading=heading,
                verbose=verbose,
                draw=draw)

        return alpha, delay

    def azimuth(self,
                out_number=True,
                shape_like=None,
                shape=None,
                verbose=False,
                draw=False):
        """Calculates azimuth, that is, the orientation of the major axis.

        References:
            J. J. Gil, "Polarized light and the Mueller Matrix approach", pp 137.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            azimuth (numpy.ndarray or float): [0, pi]: Array of azimuth angles.
        """
        # Calculate azimuth
        azimuth, _ = self.azimuth_ellipticity(
            out_number=out_number, shape_like=shape_like, shape=shape)
        # Print the result if required
        if verbose or draw:
            heading = 'The azimuth of {} is (deg.):'.format(self.parent.name)
            PrintParam(
                param=azimuth / degrees,
                shape=self.parent.shape,
                title='Azimuth (deg.)',
                heading=heading,
                verbose=verbose,
                draw=draw)
        return azimuth

    def ellipticity_angle(self,
                          out_number=True,
                          shape_like=None,
                          shape=None,
                          verbose=False,
                          draw=False):
        """Calculates the ellipticity angle.

        References:
            J. J. Gil, "Polarized light and the Mueller Matrix approach", pp 137.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            ellipticity (numpy.ndarray or float) [-pi/4, pi/4]: Array of ellipticity angles.
        """
        # Calculate the ellipticity
        _, ellipticity = self.azimuth_ellipticity(
            out_number=out_number, shape_like=shape_like, shape=shape)
        # Print the result if required
        if verbose or draw:
            heading = 'The ellipticity angle of {} is (deg.):'.format(
                self.parent.name)
            PrintParam(
                param=ellipticity / degrees,
                shape=self.parent.shape,
                title='Ellipticity angle (deg.)',
                heading=heading,
                verbose=verbose,
                draw=draw)
        return ellipticity

    def azimuth_ellipticity(self,
                            out_number=True,
                            shape_like=None,
                            shape=None,
                            verbose=False,
                            draw=False, **kwargs):
        """Calculates the azimuth and ellipticity angle of a Jones vector.

        References:
            J. J. Gil, "Polarized light and the Mueller Matrix approach", pp 137.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            azimuth (numpy.ndarray or float): [0, pi]: Array of azimuth angles.
            ellipticity (numpy.ndarray or float) [-pi/4, pi/4]: Array of ellipticity angles.
        """
        # Calculate alpha and delta, then transform
        alpha, delay = self.charac_angles(
            out_number=out_number, shape_like=shape_like, shape=shape)
        azimuth, ellipticity = charac_angles_2_azimuth_elipt(alpha, delay)
        if verbose or draw:
            heading = 'The azimuth and ellipticity angles of {} are (deg.):'.format(
                self.parent.name)
            PrintParam(
                param=(azimuth / degrees, ellipticity / degrees),
                shape=self.parent.shape,
                title=('Azimuth (deg.)', 'Ellipticity angle (deg.)'),
                heading=heading,
                verbose=verbose,
                draw=draw)

        return azimuth, ellipticity

    def global_phase(self,
                     out_number=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """Calculates the phase of the first component of the electric field (which is the reference for global phase in py_pol model).

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float) [0, 2*pi]: Array of global phase.
        """
        # Calculate phase
        Ex, Ey = self.components(shape=False, out_number=False)
        phase = np.angle(Ex) % (2 * np.pi)
        cond = (np.abs(Ex) < tol_default) * (np.abs(Ex) < np.abs(Ey))
        if np.any(cond):
            phase[cond] = np.angle(Ey[cond]) % (2 * np.pi)
        # Reshape if neccessary
        phase = reshape(
            [phase], shape_like=shape_like, shape_fun=shape, obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The global phase of {} is (deg.):'.format(
                self.parent.name)
            PrintParam(
                param=phase / degrees,
                shape=self.parent.shape,
                title='Global phase (deg.)',
                heading=heading,
                verbose=verbose,
                draw=draw)
        if out_number and phase.size == 1:
            phase = phase[0]
        return phase

    def ellipse_axes(self,
                     out_number=True,
                     sort=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """Calculates the length of major and minor axis (a,b) of the polarization elipse.

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
        # Precalculate necessary parameters
        E0x, E0y = self.amplitudes(out_number=False, shape=False)
        delay = self.delay(out_number=False, shape=False)
        azimuth = self.azimuth(out_number=False, shape=False)
        # Calculate the intensity of the axes
        a2 = E0x**2 * np.cos(azimuth)**2 + E0y**2 * np.sin(
            azimuth)**2 + 2 * E0x * E0y * np.cos(azimuth) * np.sin(
                azimuth) * np.cos(delay)
        b2 = E0x**2 * np.sin(azimuth)**2 + E0y**2 * np.cos(
            azimuth)**2 - 2 * E0x * E0y * np.cos(azimuth) * np.sin(
                azimuth) * np.cos(delay)
        # Remove points lower than 0 and calculate field
        a2[a2 < 0] = 0
        b2[b2 < 0] = 0
        a_aux = np.sqrt(a2)
        b_aux = np.sqrt(b2)
        # Order the axes
        if sort:
            a = np.maximum(a_aux, b_aux)
            b = np.minimum(a_aux, b_aux)
        else:
            a = a_aux
            b = b_aux
        # If the result is a number and the user asks for it, return a float
        if out_number and a.size == 1:
            a = a[0]
            b = b[0]
        # Reshape if neccessary
        a, b = reshape(
            [a, b], shape_like=shape_like, shape_fun=shape, obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The ellipse axes of {} are (V/m):'.format(
                self.parent.name)
            PrintParam(
                param=(a, b),
                shape=self.parent.shape,
                title=('a (V/m)', 'b (V/m)'),
                heading=heading,
                verbose=verbose,
                draw=draw)
        return a, b

    def ellipticity_param(self,
                          out_number=True,
                          shape_like=None,
                          shape=None,
                          verbose=False,
                          draw=False):
        """Calculates the ellipticity parameter: the ratio between the minor and major polarization ellipse axes. It's 0 for linearly polarized light and 1 for circulary polarized light.

        References:
            Handbook of Optics vol 2. 22.16 (eq.7)

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            e (numpy.ndarray or float): Array of length of ellipticity parameter.
        """
        # Calculate the ellipse axes
        a, b = self.ellipse_axes(shape=False, out_number=out_number)
        e = b / a
        # Reshape if neccessary
        e = reshape(
            [e], shape_like=shape_like, shape_fun=shape, obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The ellipticity parameter of {} is:'.format(
                self.parent.name)
            PrintParam(
                param=e,
                shape=self.parent.shape,
                title='Ellipticity param.',
                heading=heading,
                verbose=verbose,
                draw=draw)
        return e

    def eccentricity(self,
                     out_number=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """Calculates the eccentricity, the opposite to the ellipticity parameter. It's 1 for linearly polarized light and 0 for circulary polarized light.

        References:
            Handbook of Optics vol 2. 22.16 (eq.9)

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            e (numpy.ndarray or float): Array of length of eccentricity parameter.
        """
        # Calculate the ellipticity parameter
        e = self.ellipticity_param(shape=False, out_number=out_number)
        ecc = np.sqrt(1 - e**2)
        # Reshape if neccessary
        ecc = reshape(
            [ecc], shape_like=shape_like, shape_fun=shape, obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The eccentricity of {} is:'.format(self.parent.name)
            PrintParam(
                param=ecc,
                shape=self.parent.shape,
                title='Eccentricity',
                heading=heading,
                verbose=verbose,
                draw=draw)
        return ecc

    def degree_circular_polarization(self,
                                     out_number=True,
                                     shape_like=None,
                                     shape=None,
                                     verbose=False,
                                     draw=False):
        """Calculates the degree of circular polarization: a coefficient that measures the amount of circular polarization in the beam.

        References:
            J. J. Gil, "Polarized light and the Mueller Matrix approach", pp 39.

        Parameters:
            out_number (bool): if True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            dcp (numpy.ndarray or float): Array of degree of circular polarization.
        """
        # Calculate the degree (take into account possible nans)
        Ex, Ey = self.components(
            shape=shape, shape_like=shape_like, out_number=False)
        I = self.intensity(
            shape=shape, shape_like=shape_like, out_number=False)
        cond = I > 0
        dcp = np.nan * np.zeros_like(I)
        dcp[cond] = -2 * np.imag(Ex[cond] * np.conj(Ey[cond])) / I[cond]
        # If the result is a number and the user asks for it, return a float
        if out_number and dcp.size == 1:
            dcp = dcp[0]
        # Print the result if required
        if verbose or draw:
            heading = 'The degree of circular polarization of {} is:'.format(
                self.parent.name)
            PrintParam(
                param=dcp,
                shape=self.parent.shape,
                title='Degree circ. pol.',
                heading=heading,
                verbose=verbose,
                draw=draw)
        return dcp

    def degree_linear_polarization(self,
                                   out_number=True,
                                   shape_like=None,
                                   shape=None,
                                   verbose=False,
                                   draw=False):
        """Calculates the degree of linear polarization: a coefficient that measures the amount of linear polarization in the beam.

        References:
            J. J. Gil, "Polarized light and the Mueller Matrix approach", pp 39.

        Parameters:
            out_number (bool): if True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            dcp (numpy.ndarray or float): Array of degree of linear polarization.
        """
        # Calculate the degree
        dcp = self.degree_circular_polarization(
            shape=shape, shape_like=shape_like, out_number=out_number)
        dlp = np.sqrt(1 - dcp**2)
        # Print the result if required
        if verbose or draw:
            heading = 'The degree of linear polarization of {} is:'.format(
                self.parent.name)
            PrintParam(
                param=dlp,
                shape=self.parent.shape,
                title='Degree lin. pol.',
                heading=heading,
                verbose=verbose,
                draw=draw)
        return dlp


################################################################################
# Checks
################################################################################


class Checks_Jones_Vector(object):
    """Class for Jones vector checks.

    Parameters:
        jones_vector (Jones_vector): Parent object.

    Attributes:
        self.parent (Jones_vector): Parent object.
    """

    def __init__(self, Jones_vector):
        self.parent = Jones_vector

    def __repr__(self):
        """Prints all the checks."""
        self.get_all(verbose=True, draw=True)
        return ''

    def get_all(self, verbose=False, draw=False):
        """Creates a dictionary with all the checks of Jones vector.

        Parameters:
            verbose (bool): If True, print all parameters. Default: False.
            draw (bool): If True, draw all plots/images of the parameters. Default: False.

        Returns:
            (dict): Dictionary with parameters of Jones vector.
        """
        dict_params = {}
        dict_params['is_linear'] = self.is_linear(
            verbose=verbose, draw=draw)
        dict_params['is_circular'] = self.is_circular(
            verbose=verbose, draw=draw)
        dict_params['is_right_handed'] = self.is_right_handed(
            verbose=verbose, draw=draw)
        dict_params['is_left_handed'] = self.is_left_handed(
            verbose=verbose, draw=draw)

        return dict_params

    def is_linear(self,
                  out_number=True,
                  shape_like=None,
                  shape=None,
                  verbose=False,
                  draw=False):
        """Calculates if the Jones vector is linearly polarized.

        Parameters:
            out_number (bool): if True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Array of the condition.
        """
        # Calculate the degree of linear polarization
        cond = np.abs(
            self.parent.parameters.degree_linear_polarization(
                shape=shape, shape_like=shape_like, out_number=out_number) -
            1) < tol_default
        # Print the result if required
        if verbose or draw:
            heading = '{} is linearly polarized:'.format(self.parent.name)
            PrintParam(
                param=(cond),
                shape=self.parent.shape,
                title=('Linearly polarized'),
                heading=heading,
                verbose=verbose,
                draw=draw)
        # Return
        return cond

    def is_circular(self,
                    out_number=True,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """Calculates if the Jones vector is circularly polarized.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Array of the condition.
        """
        # Calculate the degree of linear polarization
        cond = np.abs(
            self.parent.parameters.degree_circular_polarization(
                shape=False, out_number=out_number) - 1) < tol_default
        # Calculate Ez and reshape if required
        cond = reshape(
            [cond], shape_like=shape_like, shape_fun=shape, obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is circcularly polarized:'.format(self.parent.name)
            PrintParam(
                param=(cond),
                shape=self.parent.shape,
                title=('Circularly polarized'),
                heading=heading,
                verbose=verbose,
                draw=draw)
        # Return
        return cond

    def is_right_handed(self,
                        out_number=True,
                        shape_like=None,
                        shape=None,
                        verbose=False,
                        draw=False):
        """Calculates if the Jones polarization rotation direction is right-handed.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Array of the condition.
        """
        cond = np.zeros(self.parent.size)  # No bool to allow nans
        # If the Jones vector is linearly polarized, right and left handed is not defined
        cond1 = self.is_linear(shape=False, out_number=False)
        cond[cond1] = np.nan
        # If delay is between 0 and pi (not included), it is right handed
        delay = self.parent.parameters.delay(shape=False, out_number=False)
        print(cond1.shape, delay.shape)
        cond2 = ~cond1 * (delay > 0) * (delay < np.pi)
        cond[cond2] = True
        # Give a number if required
        if out_number and cond.size == 1:
            cond = cond[0]
        # Calculate Ez and reshape if required
        cond = reshape(
            [cond], shape_like=shape_like, shape_fun=shape, obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is right handed:'.format(self.parent.name)
            PrintParam(
                param=(cond),
                shape=self.parent.shape,
                title=('Right handed'),
                heading=heading,
                verbose=verbose,
                draw=draw)
        # Return
        return cond

    def is_left_handed(self,
                       out_number=True,
                       shape_like=None,
                       shape=None,
                       verbose=False,
                       draw=False):
        """Calculates if the Jones polarization rotation direction is left-handed.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Array of the condition.
        """
        cond = np.zeros(self.parent.shape)  # No bool to allow nans
        # If the Jones vector is linearly polarized, right and left handed is not defined
        cond1 = self.is_linear(shape=False, out_number=False)
        cond[cond1] = np.nan
        # If delay is between 0 and pi (not included), it is right handed
        delay = self.parent.parameters.delay(shape=False, out_number=False)
        cond2 = ~cond1 * (delay > np.pi) * (delay < 2 * np.pi)
        cond[cond2] = True
        # Give a number if required
        if out_number and cond.size == 1:
            cond = cond[0]
        # Calculate Ez and reshape if required
        cond = reshape(
            [cond], shape_like=shape_like, shape_fun=shape, obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is left handed:'.format(self.parent.name)
            PrintParam(
                param=(cond),
                shape=self.parent.shape,
                title=('Left handed'),
                heading=heading,
                verbose=verbose,
                draw=draw)
        # Return
        return cond
