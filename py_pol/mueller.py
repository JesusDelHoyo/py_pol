# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------
# Authors:    Luis Miguel Sanchez Brea and Jesus del Hoyo
# Date:       2019/01/09 (version 1.0)
# License:    GPL
# -------------------------------------

"""
Mueller objects describe optical polarization elements in the Mueller-Stokes formalism.

**Class fields:**
    * **M**: 4x4xN array containing all the Mueller matrices.
    * **name**: Name of the object for print purposes.
    * **shape**: Shape desired for the outputs.
    * **size**: Number of stored Mueller matrices.
    * **ndim**: Number of dimensions for representation purposes.
    * **no_rotation**: If True, rotation method do not act upon the object. Useful for objects that shouldn't be rotated as mirrors.
    * **type**: Type of the object ('Mueller'). This is used for determining the object class as using isinstance may throw unexpected results in .ipynb files.
    * **parameters**: parameters of the Mueller matrices.
    * **checks**: checks of the Mueller matrices.
    * **analysis**: analysis of the Mueller matrices.


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
    * **from_components**: Creates a Mueller matrix directly from the 16 $M_{ij}$ elements.
    * **from_matrix**: Creates a Mueller object directly from a 4x4xN matrix.
    * **from_normalized**: Creates a Mueller matrix directly from a normalized 4x4 matrix ($M_{norm} = M/M_{00}$).
    * **from_Jones**: Creates a Mueller Matrix equivalent to a Jones matrix.
    * **from_blocks**: Creates a Mueller matrix from the blocks of its decomposition.
    * **from_covariance**: Creates a Mueller matrix from the equivalent covariant matrix.
    * **from_inverse**: Creates a Mueller matrix from the inverse matrix.
    * **from_list**: Creates a Jones_matrix object directly from a list of 4x4 numpy arrays.
    * **vacuum**: Creates the matrix for vacuum.
    * **mirror**: Creates the matrix for a mirror. NOTE: This matrix mus not be rotated.
    * **filter_amplifier**: Creates the matrix for a neutral filter or amplifier element.
    * **depolarizer_perfect**: Creates a perfect depolarizer.
    * **depolarizer_diagonal**: Creates a depolarizer with elements just in the diagonal.
    * **depolarizer_states**: Creates a general depolarizer from the diattenuation, polarizance and eigenstate vectors.
    * **diattenuator_perfect**: Creates a perfect linear polarizer.
    * **diattenuator_linear**: Creates a real diattenuator with perpendicular axes.
    * **diattenuator_general**: Creates the most general diattenuator.
    * **diattenuator_charac_angles**: Creates the most general homogeneous diattenuator with orthogonal eigenstates from the characteristic angles of the main eigenstate.
    * **diattenuator_azimuth_ellipticity**: Creates the most general homogenous diattenuator from the characteristic angles of the main eigenstate.
    * **diattenuator_vector**: Creates the most general homogenous diattenuator from the diattenuation vector.
    * **quarter_waveplate**: Creates a perfect retarder with 90º retardance.
    * **half_waveplate**: Creates a perfect retarder with 180º retardance.
    * **retarder_linear**: Creates a linear retarder.
    * **retarder_general**: Creates the most general homogeneous retarder.
    * **retarder_charac_angles**: Creates the most general homogeneous retarder from the characteristic angles of the fast eigenstate.
    * **retarder_azimuth_ellipticity**: Creates the most general homogeneous retarder from the characteristic angles of the fast eigenstate.
    * **retarder_from_vector**: Creates the most general homogeneous retarder from the retardance vector.
    * **diattenuator_retarder_linear**: Creates an homogeneous linear diattenuator retarder with the same axes for diattenuation and retardance.
    * **diattenuator_retarder_general**: Creates the most general homogeneous diattenuator retarder with the same eigenstates for diattenuation and retardance.
    * **diattenuator_retarder_azimuth_ellipticity**: Creates the most general homogeneous diattenuator retarder with the same eigenstates for diattenuation and retardance from the azimuth and ellipticity angle.
    * **diattenuator_retarder_charac_angles**: Creates the most general homogeneous diattenuator retarder with the same eigenstates for diattenuation and retardance from the characteristic angles.
    * **biaxial_material**: Creates a retarder using physical properties of a anisotropic material.
    * **general_eigenstates**: Generates the most general pure optical element from its eigenstates.


**Manipulation methods**
    * **rotate**: Rotates the Mueller matrix.
    * **sum**: Calculates the summatory of the Jones matrices in the object.
    * **prod**: Calculates the product of the Jones matrices in the object.
    * **remove_global_phase**: Removes the phase introduced by the optical element.
    * **add_global_phase**: Increases the phase introduced by the optical element.
    * **set_global_phase**: Sets the phase introduced by the optical element.
    * **reciprocal**: Flips the optical element so the light transverses it in the opposite direction.
    * **transpose**: Transposes the Mueller matrix of the element.
    * **inverse**: Calculates the inverse matrix of the Mueller matrix.
    * **covariant_matrix**: This method calculates the covariant matrix of the Mueller matrix of the object.


**Parameters subclass methods**
    * **matrix**:  Gets a numpy array with all the matrices.
    * **components**: Extracts the four components of the Mueller matrix.
    * **global_phase**: Extracts the global phase introduced by the object.
    * **blocks**: Method that divides a mueller matrix in their blocks: mean transmission ($M_{00}$), diattenuation and polarizance vectors and small matrix m.
    * **diattenuation_vector**: Extracts the 3xN array of diattenuation vectors.
    * **polarizance_vector**: Extracts the 3xN array of polarizance vectors.
    * **small_matrix**: Extracts the 3x3xN array of small matrix m.
    * **retardance_vector**: Extracts the 3xN array of retardance vectors (if exists).
    * **mean_transmission**: Calculates the mean transmission coefficient.
    * **transmissions**: Calculates the maximum and minimum transmissions.
    * **inhomogeneity**: Calculates the inhomogeneity parameter.
    * **diattenuation**: Calculates the diattenuation of a Mueller matrix.
    * **diattenuation_linear**: Calculates the linear diattenuation of a Mueller matrix.
    * **diattenuation_circular**: Calculates the circular diattenuation of a Mueller matrix.
    * **polarizance**: Calculates the polarizance of a Mueller matrix.
    * **polarizance_linear**: Calculates the linear polarizance of a Mueller matrix.
    * **polarizance_circular**: Calculates the delay of the matrix.
    * **degree_polarizance**: Calculates the degree of polarizance.
    * **spheric_purity**: Calculates the spheric purity grade.
    * **retardance**: Calculates the retardance (also refered as delay) of the Mueller matrix of a pure retarder.
    * **polarimetric_purity**: Calculates the degree of polarimetric purity of a Mueller matrix.
    * **depolarization_index**: Calculates the depolarization_index of a Mueller matrix.
    * **polarimetric_purity_indices**: Calculates the polarimetric purity indices of a Mueller matrix.
    * **eig**: Calculates the eigenvalues and eigenstates (eigenvectors) of the Mueller matrices.
    * **eigenvalues**: Calculates the eigenvalues and of the Mueller matrices.
    * **eigenvectors**: Calculates the eigenvectors of the Mueller matrices.
    * **eigenstates**: Calculates the eigenstates (Stokes vectors of the eigenvectors) of the Mueller matrices.
    * **det**: Calculates the determinant and of the Mueller matrices.
    * **trace**: Calculates the trace of the Mueller matrices.
    * **norm**: Calculates the norm of the Mueller matrices.
    * **get_all**: Returns a dictionary with all the parameters of the object.


**Checks subclass methods**
    * **is_physical**:  Conditions of physical realizability.
    * **is_non_depolarizing / is_pure**: Checks if matrix is non-depolarizing.
    * **is_homogeneous**: Checks if the matrix is homogeneous (eigenstates are orthogonal). It is implemented in two different ways.
    * **is_retarder**: Checks if the matrix M corresponds to a pure retarder.
    * **is_diattenuator / is_polarizer**: Checks if the matrix M corresponds to a pure homogeneous diattenuator.
    * **is_depolarizer**: Checks if the object corresponds to a depolarizer.
    * **is_singular**: Checks if the matrix is singular (at least one of its eigenvalues is 0).
    * **is_symmetric**: Checks if the Mueller matrices are symmetric.
    * **is_eigenstate**: Checks if a given light state is an eigenstate of the objct.
    * **get_all**: Returns a dictionary with all the checks of the object.


**Analysis subclass methods**
    * **diattenuator**: Calculates all the parameters from the Mueller Matrix of a diattenuator.
    * **polarizer**: Calculates all the parameters from the Mueller Matrix of a diattenuator using the polarizance vector. If the polarizer is homogeneous, this is equivalent to the previous method.
    * **retarder**: Calculates all the parameters from the Mueller Matrix of a retarder.
    * **depolarizer**: Calculates some of the parameters from the Mueller matrix of a diattenuator.
    * **filter_physical_conditions**: Method that filters experimental errors by forcing the Mueller matrix M to fulfill the conditions necessary for a matrix to be physicall.
    * **filter_purify_number**: Purifies a Mueller matrix by choosing the number of eigenvalues of the covariant matrix that will be made 0.
    * **filter_purify_threshold**: Purifies a Mueller matrix by making 0 the eigenvalues of the covariant matrix lower than a certain threshold.
    * **decompose_pure**: Polar decomposition of a pure Mueller matrix in a retarder and a diattenuator.
    * **decompose_polar**: Polar decomposition of a general Mueller matrix in a depolarizer, retarder and a diattenuator.
    """
from .utils import *
from .py_pol import Py_pol
from .stokes import Stokes, create_Stokes
from .jones_matrix import Jones_matrix
from .jones_vector import Jones_vector
from .__init__ import degrees, eps, num_decimals, number_types
from sympy.functions.special.tensor_functions import Eijk
from numpy.linalg import inv
from numpy import arctan2, array, cos, exp, matrix, pi, sin, sqrt
import numpy as np
from functools import wraps
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')

tol_default = eps
counter_max = 20
N_print_list = 5
print_list_spaces = 3
empty_matrix = np.zeros((4, 4, 1), dtype=float)
change_names = True
tol_default = eps
tol_default_filter = 1e-10
unknown_phase = False
default_phase = 0
zero_D = np.zeros(3)
zero_m = np.zeros((3, 3))

# Create a list with the base of matrices and its kronecker product
S = [
    np.eye(2),
    array([[1, 0], [0, -1]]),
    array([[0, 1], [1, 0]]),
    array([[0, -1j], [1j, 0]])
]
S_kron = []
for i in range(4):
    for j in range(4):
        S_kron.append(np.kron(S[i], np.conj(S[j])))

#############################################################################
# Methods
#############################################################################


def create_Mueller(name='M', N=1, out_object=True):
    """Method that creates several Mueller_matrix objects at the same time from a list of names or a number.

    Parameters:
        name (string or list): Name of the object for print purposes. Default: 'M'.
        N (int): Number of created elements. Default: 1.
        out_object (bool): If N=1 and out_object is True the output is a Jones_matrix instead of a list. Default: True.

    Attributes:
        self.parameters (class): Class containing the measurable parameters of the Mueller matrices.
        self.checks (class): Class containing the methods that check something about the Mueller matrices.

    Returns:
        J (list or Mueller_matrix): Result.

    """
    J = []
    if isinstance(name, list) or isinstance(name, tuple):
        for n in name:
            J.append(Mueller(n))
    else:
        for _ in range(N):
            J.append(Mueller(name))
    if len(J) == 1 and out_object:
        J = J[0]
    return J


def set_printoptions(N_list=None, list_spaces=None):
    """Method that modifies the global print options parameters.

    Parameters:
        N_print_list (int): Number of matrices that will be printed as a list if the shape of the object is 1D. Default: None
        print_list_spaces (int): Number of spaces between matrices if they are printed as a list. Default: None
    """
    global N_print_list, print_list_spaces
    if list_spaces is not None:
        print_list_spaces = list_spaces
    if N_list is not None:
        N_print_list = N_list


################################################################################
# Main class
################################################################################


class Mueller(Py_pol):
    """Class for Mueller matrices

    Parameters:
        name (str): name of Mueller matrix, for string representation

    Attributes:
        M (np.ndarray): 4x4xN array of floats containing all the Mueller matrices.
        m00 (np.ndarray): 1xN array containing the $M_{00}$ element of all Mueller matrices.
        D (np.ndarray): 3xN array containing the diattenuation vectors of the Mueller matrices.
        P (np.ndarray): 3xN array containing the polarizance vectors of the Mueller matrices.
        m (np.ndarray): 3xN array containing the rest of the Mueller matrices.
        global_phase (np.ndarray): 1xN array storing the global phase introduced by the optical objects.
        name (string): Name of the object for print purposes.
        shape (tuple or list): Shape desired for the outputs.
        size (int): Number of stored Mueller matrices.
        ndim (int): Number of dimensions for representation purposes.
        no_rotation (bool): If True, rotation method do not act upon the object. Useful for objects that shouldn't be rotated as mirrors.
        type (string): Type of the object ('Jones_matrix'). This is used for determining the object class as using isinstance may throw unexpected results in .ipynb files.

    Attributes:
        self.parameters (class): parameters of the Mueller matrices.
        self.checks (class): checks of the Mueller matrices.
        self.analysis (class): analysis of the Mueller matrices.
    """
    __array_priority__ = 30000

    ############################################################################
    # Operations
    ############################################################################

    def __init__(self, name='M'):
        super().__init__(name=name, _class="Mueller")

        self.no_rotation = False
        self._global_phase = default_phase

        self.parameters = Parameters_Mueller(self)
        self.analysis = Analysis_Mueller(self)
        self.checks = Check_Mueller(self)

    def __add__(self, other):
        """Adds two Mueller matrices.

        Parameters:
            other (Mueller or Jones_vector): 2nd matrix to add.

        Returns:
            M2 (Mueller): Result.
        """
        try:
            if other.type in ('Jones_matrix', 'Mueller'):
                # Transform other to Mueller if necessary
                M2 = Mueller()
                if other.type == 'Jones_matrix':
                    M1 = Mueller(other.name)
                    M1.from_Jones(other)
                else:
                    M1 = other
                # Calculate the new global phase
                if self.global_phase is None or M1.global_phase is None:
                    phase = None
                elif np.all(self.global_phase == M1.global_phase):
                    phase = self.global_phase
                else:
                    phase = None
                # Calculate the new matrix
                M2.from_matrix(self.M + M1.M, global_phase=phase)
                M2.shape = take_shape((self, M1))
                if change_names:
                    M2.name = self.name + " + " + M1.name
                return M2
            else:
                raise ValueError('other is {} instead of Jones_matrix.'.format(
                    other.type))
        except:
            raise ValueError('other is not a py_pol object but {}'.format(
                type(other)))

    def __sub__(self, other):
        """Substracts two Mueller matrices.

        Parameters:
            other (Mueller or Jones_vector): 2nd matrix to substract.

        Returns:
            M3 (Mueller): Result.
        """
        M3 = self + ((-1) * other)
        if change_names:
            M3.name = self.name + " - " + other.name
        return M3

    def __rmul__(self, other):
        """Multiplies a Mueller matrix by a number. If the number is complex or real negative, the absolute value is used and the global phase is updated acordingly.

        Parameters:
            other (float, complex or numpy.ndarray): number to multiply.

        Returns:
            (Mueller): Result.
        """
        M3 = Mueller()
        # Save the Number of elements, and then flatten
        if isinstance(other, number_types):
            N = 1
            other2 = other
        elif isinstance(other, np.ndarray):
            N = other.size
            other2 = other.flatten()
        else:
            raise ValueError('Other is not a number or a numpy array')
        # Calculate components
        components = self.parameters.components(shape=False)
        # Check that the multiplication can be performed
        if N == self.size or self.size == 1 or N == 1:
            # Calculate the absolute value and complex phase of the number
            mod, phase = (np.abs(other2), np.angle(other2))
            for ind in range(16):
                components[ind] = components[ind] * mod
            # Create the object
            M3.from_components(components, global_phase=self.global_phase)
            M3.add_global_phase(phase)
            if isinstance(other, np.ndarray):
                M3.shape = take_shape((self, other.shape))
            elif isinstance(other, number_types):
                M3.shape = self.shape
        else:
            raise ValueError(
                'The number of elements in other ({}) and {} ({}) is not the same'
                .format(N, self.name, self.size))

        if isinstance(other, number_types) and change_names:
            M3.name = str(other) + " * " + self.name
        else:
            M3.name = str(other)

        return M3

    def __mul__(self, other):
        """Multiplies the Mueller matrix by a number, an array of numbers, a Stokes or Jones vector, or another Mueller or Jones matrix.

        Parameters:
            other (float, numpy.ndarray, Stokes, Jones_vector, Mueller or Jones_matrix): 2nd object to multiply.

        Returns:
            (Mueller): Result.
        """
        # Multiplication by numbers or arrays is already implemented in __mul__
        if isinstance(other, number_types) or isinstance(other, np.ndarray):
            return other * self
        # Multiply by py_pol objects
        else:
            other_Jones = None
            # try:
            # Transform Jones vectors if required
            if other.type == 'Jones_vector':
                new_other = Stokes()
                new_other.from_Jones(other)
                other_Jones = other
            elif other.type == 'Jones_matrix':
                new_other = Mueller()
                new_other.from_Jones(other)
                other_Jones = other
            else:
                new_other = other
                try:
                    if  other.type == 'Mueller':
                        other_Jones = Jones_matrix().from_Mueller(other)
                    else:
                        other_Jones = Jones_vector().from_Stokes(other)
                except:
                    other_Jones = Jones_matrix().vacuum()
            try:
                self_Jones = Jones_matrix().from_Mueller(self)
            except:
                self_Jones = Jones_matrix().vacuum()

            # Prepare variables
            new_self, new_other = expand_objects([self, new_other], copy=True)
            if new_other.type == 'Stokes':
                M3 = Stokes()
            elif new_other.type == 'Mueller':
                M3 = Mueller()
            else:
                raise ValueError(
                    'other is not a correct py_pol object, but {}.'.format(
                        other.type))
            # Multiply
            Mf = matmul_pypol(new_self.M, new_other.M)
            Jf = self_Jones * other_Jones
            phase = Jf.parameters.global_phase()
            M3.from_matrix(Mf, global_phase=phase)
            #M3.from_matrix(Mf)
            if change_names:
                M3.name = self.name + " * " + other.name
            # except:
            #     raise ValueError(
            #         'other is not number, numpy.ndarray or py_pol object, but {}.'.format(type(other)))
        M3.shape = take_shape((self, other))
        if isinstance(other, number_types) and change_names:
            M3.name = str(other) + " * " + self.name
        return M3

    def __truediv__(self, other):
        """Divides a Mueller matrix by a number. If the number is complex or real negative, the absolute value is used and the global phase is updated acordingly.

        Parameters:
            other (float or numpy.ndarray): Divisor.

        Returns:
            (Mueller): Result.
        """
        M3 = self * (other**(-1))
        if isinstance(other, number_types) and change_names:
            M3.name = self.name + " / " + str(other)
        return M3

    def __repr__(self):
        """prints information about class"""
        # Extract the components
        components = self.parameters.components()
        # If the object is empty, say it
        if self.size == 0 or np.all(self.M == 0):
            return '{} is empty\n'.format(self.name)
        # If the object is 0D or 1D, print it like a list or inline
        elif self.size == 1 or self.shape is None or self.ndim < 2:
            # Short enough objects can be printed as a list of matrices
            if self.size <= N_print_list:
                list = self.get_list(out_number=False)
                str = "{} = \n".format(self.name)
                str = str + PrintMatrices(list, print_list_spaces)
            # Print the rest using a single line for each component
            else:
                str = "{} M00 = {}\n".format(self.name, components[0])
                for ind1 in range(4):
                    for ind2 in range(4):
                        if ind1 + ind2 > 0:
                            str = str + " " * \
                                len(self.name) + " M{}{} = {}\n".format(ind1,
                                                                        ind2, components[ind1 * 4 + ind2])
        # Print higher dimensionality as pure arrays
        else:
            str = ""
            for ind1 in range(4):
                for ind2 in range(4):
                    str = str + \
                        "{} M{}{} =\n {}\n".format(
                            self.name, ind1, ind2, components[ind1 * 4 + ind2])
        return str

    def __getitem__(self, index):
        """
        Implements object extraction from indices.
        """
        if change_names:
            E = Mueller(self.name + '_picked')
        else:
            E = Mueller(self.name)
        # If the indices are 1D, act upon the matrix directly
        if isinstance(index, (int, slice)) and self.ndim > 1:
            E.from_matrix(self.M[:, :, index])
        elif isinstance(index,
                        np.ndarray) and index.ndim == 1 and self.ndim > 1:
            E.from_matrix(self.M[:, :, index])
        # If not, act upon the components
        else:
            components = self.parameters.components(out_number=False)
            for ind in range(16):
                components[ind] = components[ind][index]
            E.from_components(components)

        return E

    def __setitem__(self, index, data):
        """
        Implements object inclusion from indices.
        """
        # Check that data is a correct pypol object
        if data.type == 'Jones_matrix':
            data2 = Mueller(data.name)
            data2.from_Jones(data)
        elif data.type == 'Mueller':
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
            self.M[:, :, index] = np.squeeze(data2.M)
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
            self.M[:, :, index] = np.squeeze(data3.M)
            # Add global phase
            self.global_phase[index] = data3.global_phase
        elif isinstance(index,
                        np.ndarray) and index.ndim == 1 and self.ndim > 1:
            self.M[:, :, index] = data2.M
        # If not, act upon the components
        else:
            # Extract phase and components
            components = self.parameters.components(out_number=False)
            phase = self.parameters.global_phase(out_number=False)
            components_new = data2.parameters.components(out_number=False)
            for ind in range(16):
                components[ind][index] = components_new[ind]
            phase_new = data2.parameters.global_phase(out_number=False)
            phase[index] = phase_new
            # Set the new values
            self.from_components(components, global_phase=phase)

    def __eq__(self, other):
        """
        Implements equality operation.
        """
        try:
            # Calculate the difference object
            if other.type == 'Jones_matrix':
                M2 = Mueller()
                M2.from_Jones(other)
            elif other.type == 'Mueller':
                M2 = other
            else:
                return False
        except:
            return False
        # Stretch if required
        if self.size == 1 and M2.size > 1:
            M1 = self.stretch(length=M2.size, keep=True)
        else:
            M1 = self
        if M1.size > 1 and M2.size == 1:
            M2 = M2.stretch(length=M1.size, keep=True)
        shape = take_shape((self, other))
        # Compare matrices
        norm = np.linalg.norm(M1.M - M2.M, axis=(0, 1))
        cond1 = norm < tol_default
        if shape is not None:
            cond1 = np.reshape(cond1, shape)
        # Compare phases
        if (M1.global_phase is None) or (M1.global_phase is None):
            cond2 = np.nan
        else:
            phase1 = M1.parameters.global_phase(shape=shape)
            phase2 = M2.parameters.global_phase(shape=shape)
            cond2 = phase1 == phase2
        # Merge conditions
        cond = cond1 * cond2
        return cond

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
            raise ValueError("Global phase has diffrent size ({}) than object ({})".format(value.size, self.size))
        else:
            self._global_phase = value

    #####################
    ## MANIPULATION
    #####################


    def add_global_phase(self,
                         phase=0,
                         unknown_as_zero=unknown_phase,
                         keep=False):
        """Method that adds a phase to the Mueller object.

        Parameters:
            phase (float or np.ndarray): Phase to be added to the Mueller object. Default: 0.
            unknown_as_zero (bool): If True, takes unknown phase as zero. Default: False.
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Mueller): Recalculated Mueller object.
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
                         shape_like=None,
                         shape=None):
        """Method that sets the phase to the Mueller object.

        Parameters:
            phase (float or np.ndarray): Phase to be added to the Mueller object. Default: 0.
            keep (bool): If True, self is not updated. Default: False.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Recalculated Mueller object.
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
                                                          give_shape=True)
            phase = phase.flatten()
            new_obj.shape, _ = select_shape(new_obj,
                                                       shape_var=new_shape,
                                                       shape_fun=shape,
                                                       shape_like=shape_like)
            self.global_phase = phase
        # End
        return new_obj

    def remove_global_phase(self, keep=False):
        """Method that removes the phase to the Mueller object.

        Parameters:
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Mueller): Recalculated Mueller object.
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

    # @_actualize_
    def rotate(self, angle=0, keep=False, change_name=change_names):
        """Rotates a Mueller matrix a certain angle

        M_rotated = R(-angle) * self.M * R(angle)

        Parameters:
            angle (float): angle of rotation in radians. Default: 0
            keep (bool): If True, the original element is not updated. Default: False.
            change_name (bool): If True and angle is of size 1, changes the object name adding @ XX deg, being XX the total rotation angle. Default: True.

        Returns:
            (Mueller): Mueller object rotated
        """
        if self.no_rotation:
            print('Warning: Tried to rotate {}, which must not be rotated.'.
                  format(self.name))
            return self
        else:
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
            # Calculate the rotation objects
            Jneg, Jpos = create_Mueller(('-', '+'))
            Jneg.from_matrix(rotation_matrix_Mueller(-angle))
            Jpos.from_matrix(rotation_matrix_Mueller(angle))
            # Rotate
            other = Jneg * (new_obj * Jpos)
            new_obj.from_matrix(other.M)
            # Update name
            if change_name and angle.size == 1:
                if np.abs(angle) > tol_default:
                    new_obj.name = new_obj.name + \
                        " @ {:1.2f} deg".format(angle[0] / degrees)
            # Return
            new_obj.shape, _ = select_shape(obj=new_obj,
                                                       shape_var=new_shape)
            return new_obj

    def covariance_matrix(self,
                          keep=False,
                          shape_like=None,
                          shape=None,
                          change_name=change_names):
        """Calculates the covariance matrix of a Mueller matrix.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016) pp 171.

        Notes:
            The base of matrices S is used in an uncommon order.
            In order to obtain the same result as in the book, the formula must be:

            .. math:: H=0.25\sum(m[i,j]\,kron([S(i),S^{*}(j))].

        Parameters:
            keep (bool): If True, the original element is not updated. Default: False.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Mueller): Modified object.

        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Calculate components
        components = self.parameters.components(shape=False, out_number=False)
        # Calculate the covariance matrices
        H = np.zeros((4, 4, new_obj.size), dtype=complex)
        for ind in range(16):
            H = H + np.multiply.outer(S_kron[ind], components[ind])
        new_obj.from_matrix(H / 4)
        new_obj.shape, _ = select_shape(shape_var=self.shape,
                                                   shape_like=shape_like,
                                                   shape_fun=shape)
        # Fix the name if required
        if change_name:
            new_obj.name = 'Covariant of ' + new_obj.name
        return new_obj

    def inverse(self,
                keep=False,
                shape_like=None,
                shape=None,
                change_name=change_names):
        """Calculates the inverse matrix of the Mueller matrix.

        Parameters:
            keep (bool): If True, the original element is not updated. Default: False.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Mueller): Modified object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Caluclate inverse
        new_obj.from_matrix(inv_pypol(self.M),
                            shape_like=shape_like,
                            shape=shape)
        new_obj.shape, _ = select_shape(shape_var=self.shape,
                                                   shape_like=shape_like,
                                                   shape_fun=shape)
        # Fix the name if required
        if change_name:
            new_obj.name = 'Inverse of ' + new_obj.name
        return new_obj

    # @_actualize_
    def reciprocal(self,
                   keep=False,
                   shape_like=None,
                   shape=None,
                   change_name=change_names):
        """Calculates the reciprocal of the optical element, so the light tranverses it in the opposite direction. In Mueller formalism, it is calculated as:

        .. math:: M^{r}=\left[\begin{array}{cccc}
                    1 & 0 & 0 & 0\\
                    0 & 1 & 0 & 0\\
                    0 & 0 & -1 & 0\\
                    0 & 0 & 0 & 1
                    \end{array}\right]M^{T}\left[\begin{array}{cccc}
                    1 & 0 & 0 & 0\\
                    0 & 1 & 0 & 0\\
                    0 & 0 & -1 & 0\\
                    0 & 0 & 0 & 1
                    \end{array}\right]

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 111.

        Parameters:
            keep (bool): If True, the original element is not updated. Default: False.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Mueller): Result.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Calculate components
        components = new_obj.transpose(keep=True).parameters.components(
            shape=False)
        components[2] = -components[2]
        components[6] = -components[6]
        components[8] = -components[8]
        components[9] = -components[9]
        components[11] = -components[11]
        components[14] = -components[14]
        # Create the object
        new_obj.from_components(components)
        new_obj.shape, _ = select_shape(shape_var=self.shape,
                                                   shape_like=shape_like,
                                                   shape_fun=shape)
        # Fix the name if required
        if change_name:
            new_obj.name = 'Reciprocal of ' + new_obj.name
        return new_obj

    def transpose(self,
                  keep=False,
                  shape_like=None,
                  shape=None,
                  change_name=change_names):
        """Calculates the transposed matrices of the Mueller matrices.

        Parameters:
            keep (bool): If True, the original element is not updated. Default: False.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Jones_matrix): Modified object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Caluclate inverse
        new_obj.from_matrix(np.transpose(new_obj.M, axes=(1, 0, 2)))
        new_obj.shape, _ = select_shape(shape_var=self.shape,
                                                   shape_like=shape_like,
                                                   shape_fun=shape)
        # Fix the name if required
        if change_name:
            new_obj.name = 'Transpose of ' + new_obj.name
        return new_obj

    def sum(self, axis=None, keep=False, change_name=change_names):
        """Calculates the sum of Mueller matrices stored in the object.

        Parameters:
            axis (int, list or tuple): Axes along which the summatory is performed. If None, all matrices are summed. Default: None
            keep (bool): If True, the original element is not updated. Default: False.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Mueller): Modified object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Simple case
        if axis is None or new_obj.ndim <= 1:
            M = np.sum(new_obj.M, axis=2)
        # Complicated case
        else:
            # Calculate maximum axis
            if isinstance(axis, int):
                axis = axis + 2
                m = axis
            else:
                axis = np.array(axis) + 2
                m = np.max(axis)
            # Check that the axes are correct
            if m >= new_obj.ndim + 2:
                raise ValueError(
                    'Axis {} greater than the number of dimensions of {}, which is {}'
                    .format(m, new_obj.name, new_obj.ndim))
            # Reshape M to fit the current shape
            shape = [4, 4] + new_obj.shape
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

    def prod(self, axis=None, keep=False, change_name=change_names):
        """Calculates the product of Mueller matrices stored in the object.

        Parameters:
            axis (int, list or tuple): Axes along which the product is performed. If None, all matrices are multiplied. Default: None
            keep (bool): If True, the original element is not updated. Default: False.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Mueller): Modified object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Simple case
        if axis is not None:
            N_axis = np.array(axis).size
        if axis is None or new_obj.ndim <= 1 or new_obj.ndim == N_axis:
            M = new_obj.M[:, :, 0]
            for ind in range(1, new_obj.size):
                M = M @ new_obj.M[:, :, ind]
        # Complicated case
        else:
            # Calculate maximum axis
            if isinstance(axis, int):
                m = axis + 2
            else:
                axis = np.array(axis)
                m = np.max(axis) + 2
            # Check that the axes are correct
            if m >= new_obj.ndim + 2:
                raise ValueError(
                    'Axis {} greater than the number of dimensions of {}, which is {}'
                    .format(m, new_obj.name, new_obj.ndim))
            # Calculate shapes, sizes and indices
            if isinstance(axis, int):
                shape_removed = new_obj.shape[axis]
            else:
                shape_removed = np.array(new_obj.shape)[axis]
            N_removed = np.prod(shape_removed)
            ind_removed = combine_indices(
                np.unravel_index(np.array(range(N_removed)), shape_removed))
            shape_matrix = np.delete(new_obj.shape, axis)
            N_matrix = np.prod(shape_matrix)
            ind_matrix = combine_indices(
                np.unravel_index(np.array(range(N_matrix)), shape_matrix))
            shape_final = [4, 4] + list(shape_matrix)
            axes_aux = np.array(range(2, new_obj.ndim + 2))
            shape_orig = [4, 4] + list(new_obj.shape)
            # Make the for loop of the matrix to be calculated
            M_orig = np.reshape(new_obj.M, shape_orig)
            M = np.zeros(shape_final)
            for indM in range(N_matrix):
                # Make the multiplication loop
                indices = merge_indices(ind_matrix[indM], ind_removed[0], axis)
                aux = multitake(M_orig, indices, axes_aux)
                for indR in range(1, N_removed):
                    indices = merge_indices(ind_matrix[indM],
                                            ind_removed[indR], axis)
                    aux = aux @ multitake(M_orig, indices, axes_aux)
                # Store the result
                for i1 in range(4):
                    for i2 in range(4):
                        ind_aux = tuple([i1, i2] + list(ind_matrix[indM]))
                        M[ind_aux] = aux[i1, i2]
        # Create the object and return it
        new_obj.from_matrix(M)
        if change_names:
            new_obj.name = 'Prod of ' + new_obj.name
        return new_obj



    ####################################################################
    # Creation
    ####################################################################

    def from_components(self,
                        components,
                        global_phase=0,
                        length=1,
                        shape_like=None,
                        shape=None):
        """Creates the Mueller matrix object form the arrays of its 16 components components.

        Parameters:
            components (tuple or list): A 4 element tuple containing the 6 components of the Mueller matrices (M00, M01, ..., M32, M33).
            global_phase (float or numpy.ndarray): Adds a global phase to the object. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller_matrix): Created object.
        """
        # Prepare variables
        components = list(components)
        components.append(global_phase)
        expand = 16 * [True] + [False]
        (components), new_shape = prepare_variables(vars=components,
                                                    expand=expand,
                                                    length=length,
                                                    give_shape=True)
        # Create the matrix
        self.M = np.array(
            [[components[0], components[1], components[2], components[3]],
             [components[4], components[5], components[6], components[7]],
             [components[8], components[9], components[10], components[11]],
             [components[12], components[13], components[14], components[15]]])
        # Rest of operations
        self.no_rotation = False
        # self.size = components[0].size
        self.shape, _ = select_shape(self,
                                             shape_var=new_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)
        self.set_global_phase(phase=global_phase)
        self.no_rotation = False
        return self

    def from_matrix(self,
                    M,
                    global_phase=default_phase,
                    length=1,
                    shape_like=None,
                    shape=None):
        """Create a Mueller object from an external array.

        Parameters:
            M (numpy.ndarray): New matrix. At least two dimensions must be of size 4.
            global_phase (numpy.ndarray): Adds a global phase to the Stokes object. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Check if the matrix is of the correct Size
        M = np.array(M)
        s = M.size
        # 1D and 2D
        if M.ndim == 1 or M.ndim == 2:
            if M.size % 16 == 0:
                M = np.reshape(M, (4, 4, int(M.size / 16)))
            else:
                raise ValueError(
                    'M must have a number of elements multiple of 16.')
            if M.size == 16:
                sh = None
            else:
                sh = [int(M.size / 16)]
        # 3D or more
        elif M.ndim > 2:
            sh = np.array(M.shape)
            N = np.sum(sh == 4)
            if N > 1:
                # Find the matrix indices and the final shape
                ind1 = np.argmin(~(sh == 4))
                sh = np.delete(sh, ind1)
                ind2 = np.argmin(~(sh == 4))
                sh = np.delete(sh, ind2)
                ind2 = ind2 + 1
                # Calculate the components and construct the matrix from them
                M = np.array([[
                    multitake(M, [0, 0], [ind1, ind2]).flatten(),
                    multitake(M, [0, 1], [ind1, ind2]).flatten(),
                    multitake(M, [0, 2], [ind1, ind2]).flatten(),
                    multitake(M, [0, 3], [ind1, ind2]).flatten()
                ],
                              [
                                  multitake(M, [1, 0], [ind1, ind2]).flatten(),
                                  multitake(M, [1, 1], [ind1, ind2]).flatten(),
                                  multitake(M, [1, 2], [ind1, ind2]).flatten(),
                                  multitake(M, [1, 3], [ind1, ind2]).flatten()
                              ],
                              [
                                  multitake(M, [2, 0], [ind1, ind2]).flatten(),
                                  multitake(M, [2, 1], [ind1, ind2]).flatten(),
                                  multitake(M, [2, 2], [ind1, ind2]).flatten(),
                                  multitake(M, [2, 3], [ind1, ind2]).flatten()
                              ],
                              [
                                  multitake(M, [3, 0], [ind1, ind2]).flatten(),
                                  multitake(M, [3, 1], [ind1, ind2]).flatten(),
                                  multitake(M, [3, 2], [ind1, ind2]).flatten(),
                                  multitake(M, [3, 3], [ind1, ind2]).flatten()
                              ]])

            else:
                raise ValueError(
                    'M must have four elements in at least two dimensions. Instead, it has shape = {}'
                    .format(M.shape))
        else:
            raise ValueError('M can not be empty')

        # Increase length if required
        if M.size == 16 and length > 1:
            M = np.multiply.outer(np.squeeze(M), np.ones(length))
        # End operations
        # self.size = int(M.size / 16)
        self.M = M
        self.no_rotation = False
        self.shape, _ = select_shape(self,
                                             shape_var=sh,
                                             shape_fun=shape,
                                             shape_like=shape_like)
        self.set_global_phase(phase=global_phase)
        return self

    def from_normalized(self,
                        m,
                        M00=None,
                        global_phase=default_phase,
                        shape_like=None,
                        shape=None):
        """Creates a Mueller object directly from the normalized matrix $m = M/M_{00}$, and $M_{00}$.

        Parameters:
            m (4x4 numpy.matrix): Mueller matrix
            M00 (float): [0, 1] Mean transmission coefficient. Default: maximum possible (None).
            global_phase (numpy.ndarray): Adds a global phase to the Stokes object. Default: 0.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Easy case, M00 is a number
        if type(M00) in number_types:
            M = m * M00
            self.from_matrix(M,
                             global_phase=global_phase,
                             shape=shape,
                             shape_like=shape_like)
        # Complicated case, M00 is an array
        else:
            # Check that the variables can be used
            if m.size > 16 and m.size / 16 != M00.size:
                raise ValueError(
                    'M00 of {} elements is incompatible with m of {} elements'.
                    format(M00.size, m.size / 16))
            # Create the object as if the matrix were not normalized
            self.from_matrix(M, shape=False)
            components = self.parameters.components(shape=False)
            # Denormalize
            for ind in range(16):
                components[ind] = components[ind] * M00
            # Create the correct object
            self.from_components(components,
                                 global_phase=global_phase,
                                 shape=shape,
                                 shape_like=shape_like)
        return self

    def from_blocks(self,
                    Dv=zero_D,
                    Pv=zero_D,
                    m=zero_m,
                    M00=1,
                    global_phase=default_phase,
                    length=1,
                    shape_like=None,
                    shape=None):
        """Method that creates a Mueller object from the block components of its matrix.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            Dv (numpy.ndarray): Diattenuation vector 3xN array.
            Pv (numpy.ndarray): Polarizance vector 3xN array.
            m (numpy.ndarray): Small matrix m 3x3xN array.
            M00 (numpy.ndarray): Parameter of average intensity array of size N. Default: 1
            global_phase (numpy.ndarray): Adds a global phase to the Stokes object. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Prepare the variables
        M00, Dv, Pv, m, new_shape = prepare_variables_blocks(
            M00,
            Dv,
            Pv,
            m,
            length=length,
            give_shape=True,
            multiply_by_M00=True)
        # Build the object
        components = [
            M00, Dv[0], Dv[1], Dv[2], Pv[0], m[0, 0], m[0, 1], m[0, 2],
            Pv[1], m[1, 0], m[1, 1], m[1, 2], Pv[2], m[2, 0], m[2, 1], m[2, 2]
        ]
        self.from_components(components, global_phase=global_phase)
        # Update object parameters
        self.shape, _ = select_shape(self,
                                             shape_var=new_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)
        # self.M00, self.D, self.P, self.m = (M00, D, P, m)
        return self


    # @_actualize_
    def from_Jones(self, J, length=1, shape_like=None, shape=None):
        """Takes a Jones Matrix and converts into Mueller Matrix

        .. math:: M(J)=\left[\begin{array}{cccc}
                    1 & 0 & 0 & 1\\
                    1 & 0 & 0 & -1\\
                    0 & 1 & 1 & 0\\
                    0 & i & -i & 0
                    \end{array}\right]\left(J\otimes J^{*}\right)\left[\begin{array}{cccc}
                    1 & 0 & 0 & 1\\
                    1 & 0 & 0 & -1\\
                    0 & 1 & 1 & 0\\
                    0 & i & -i & 0
                    \end{array}\right]^{-1}

        References:
            "Polarized light and the Mueller Matrix approach", 2nd Ed., J. J. Gil and E. Ossikowski, pp 107.

        Parameters:
            J (Jones_matrix): Jones matrix object.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Extract components from the Jones object and derivatives
        J00, J01, J10, J11 = J.parameters.components(shape=False)
        J00m = np.abs(J00)**2
        J01m = np.abs(J01)**2
        J10m = np.abs(J10)**2
        J11m = np.abs(J11)**2
        J00c = np.conj(J00)
        J01c = np.conj(J01)
        J10c = np.conj(J10)
        J11c = np.conj(J11)
        # Create the components of the Mueller object
        components = [0] * 16
        components[0] = (J00m + J01m + J10m + J11m) / 2
        components[1] = (J00m - J01m + J10m - J11m) / 2
        components[2] = (J00 * J01c + J10 * J11c).real
        components[3] = (J00 * J01c + J10 * J11c).imag
        components[4] = (J00m + J01m - J10m - J11m) / 2
        components[5] = (J00m - J01m - J10m + J11m) / 2
        components[6] = (J00 * J01c - J10 * J11c).real
        components[7] = (J00 * J01c + J10c * J11).imag
        components[8] = (J00 * J10c + J01 * J11c).real
        components[9] = (J00 * J10c - J01 * J11c).real
        components[10] = (J00 * J11c + J01 * J10c).real
        components[11] = (J10 * J01c + J11c * J00).imag
        components[12] = (J10 * J00c + J01c * J11).imag
        components[13] = (J10 * J00c + J11c * J01).imag
        components[14] = (J11 * J00c + J01c * J10).imag
        components[15] = (J00 * J11c - J01 * J10c).real
        phase = J.parameters.global_phase()
        # Create the object
        self.from_components(components,
                             global_phase=phase,
                             length=length,
                             shape=shape,
                             shape_like=shape_like)
        return self

    # @_actualize_
    def from_covariance(self,
                        H,
                        global_phase=default_phase,
                        length=1,
                        shape_like=None,
                        shape=None):
        """Calculates the Mueller matrix from the covariance matrix:

        $M_{ij}=Trace\left[\left(\sigma_{i}\otimes\sigma_{j}^{*}\right)H\right]$

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            H (Mueller or numpy.ndarray): Covariance matrix.
            global_phase (numpy.ndarray): Adds a global phase to the Stokes object. Default: 0.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Check if H is a py_pol object or an array
        if isinstance(H, np.ndarray):
            obj = Mueller(self.name)
            obj.from_matrix(H)
        else:
            obj = H
        old_shape = obj.shape
        obj.shape = None
        # Initialize the matrix
        M = np.zeros((4, 4, obj.size), dtype=complex)
        # Loop in elements
        components = []
        elem = Mueller()
        for indI in range(4):
            for indJ in range(4):
                # Calculate the Sij matrix
                Sij = np.multiply.outer(np.kron(S[indI], np.conj(S[indJ])),
                                        np.ones(obj.size))
                elem.from_matrix(Sij)
                # Multiply it by H
                elem = elem * obj
                # Save the trace as the new component
                components += [
                    np.array(elem.parameters.trace(shape=False), dtype=float)
                ]
        # Create the new object
        self.from_components(components, global_phase=global_phase)
        # Reshape if necessary
        self.shape, _ = select_shape(obj=self,
                                             shape_var=old_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)

        return self

    # @_actualize_
    # def from_inverse(self, M):
    #     """Calculates the Mueller matrix from the inverse matrix.
    #
    #     Parameters:
    #         M (numpy.matrix 4x4 or Mueller object): Inverse matrix.
    #
    #     Returns:
    #         (numpy.matrix): 4x4 matrix.
    #     """
    #     try:
    #         if M.type == 'Mueller':
    #             self.from_matrix(M.M.I)
    #         else:
    #             self.from_matrix(M.I)
    #     except:
    #         self.from_matrix(M.I)
    #     return self.M

    def vacuum(self, global_phase=0, length=1, shape_like=None, shape=None):
        """Creates the matrix for vacuum i.e., an optically neutral element.

        Parameters:
            global_phase (float or numpy.ndarray): Adds a global phase to the Jones matrix. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        self.filter_amplifier(D=1,
                              length=length,
                              global_phase=global_phase,
                              shape=shape,
                              shape_like=shape_like)
        return self

    def filter_amplifier(self,
                         D=1,
                         global_phase=0,
                         length=1,
                         shape_like=None,
                         shape=None):
        """Creates the Mueller object of neutral filters or amplifiers.

        Parameters:
            D (float or numpy.ndarray): Attenuation (gain if > 1). Default: 1.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Calculate
        components = [D] + [0] * 4 + [D] + \
            [0] * 4 + [D] + [0] * 4 + [D]
        self.from_components(components=components,
                             global_phase=global_phase,
                             length=length,
                             shape=shape,
                             shape_like=shape_like)
        return self

    # @_actualize_
    def mirror(self,
               ref=1,
               ref_field=None,
               global_phase=0,
               length=1,
               shape_like=None,
               shape=None):
        """Mueller matrix of a mirror.

        Parameters:
            ref (float or numpy.ndarray): Intensity reflectivity of the mirror. Default: 1.
            ref_field (float or numpy.ndarray): Electric field reflectivity coefficient. If not None, it overrides REF. Default: None.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Use the intensity reflectivity
        if ref_field is not None:
            ref = np.abs(ref_field)**2
        # Prepare variables
        (ref,
         global_phase), new_shape = prepare_variables(vars=[ref, global_phase],
                                                      expand=[True, False],
                                                      length=length,
                                                      give_shape=True)
        # Calculate
        components = [ref] + [0] * 4 + [ref] + \
            [0] * 4 + [-ref] + [0] * 4 + [-ref]
        self.from_components(components=components,
                             global_phase=global_phase,
                             shape=shape,
                             shape_like=shape_like)
        self.no_rotation = True
        return self

    # @_actualize_
    def depolarizer_perfect(self,
                            M00=1,
                            global_phase=0,
                            length=1,
                            shape_like=None,
                            shape=None):
        """Creates a perfect depolarizer:

        .. math:: M_{p}=\left[\begin{array}{cccc}
            M_{00} & 0 & 0 & 0\\
            0 & d_{1} & 0 & 0\\
            0 & 0 & d_{2} & 0\\
            0 & 0 & 0 & d_{3}
            \end{array}\right]

        Parameters:
            M00 (float, default 1): Parameter of average intensity. Default: 1.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Calculate
        components = [M00] + [0] * 15
        self.from_components(components=components,
                             length=length,
                             global_phase=global_phase,
                             shape=shape,
                             shape_like=shape_like)
        return self

    # @_actualize_
    def depolarizer_diagonal(self,
                             d,
                             M00=1,
                             global_phase=0,
                             length=1,
                             shape_like=None,
                             shape=None):
        """Creates a diagonal depolarizer:

        .. math:: M_{p}=\left[\begin{array}{cccc}
            M_{00} & 0 & 0 & 0\\
            0 & d_{1} & 0 & 0\\
            0 & 0 & d_{2} & 0\\
            0 & 0 & 0 & d_{3}
            \end{array}\right]

        Parameters:
            d (list, float or numpy.ndarray): Absorption coefficients. If list, it must contain three float or numpy arrays, one for each diagonal value. If float or numpy.ndarray, $d_1 = d_2 = d_3$.
            M00 (float, default 1): Parameter of average intensity. Default: 1.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Calculate
        if isinstance(d, list):
            components = [M00] + [0] * 4 + [d[0]] + [0] * 4 + [
                d[1]
            ] + [0] * 4 + [d[2]]
        else:
            components = [M00] + [0] * 4 + [d] + [0] * 4 + [d] + [0] * 4 + [d]
        self.from_components(components=components,
                             length=length,
                             global_phase=global_phase,
                             shape=shape,
                             shape_like=shape_like)
        return self

    # @_actualize_
    def depolarizer_states(self,
                           d,
                           S,
                           Dv=zero_D,
                           Pv=zero_D,
                           M00=1,
                           global_phase=0,
                           length=1,
                           shape_like=None,
                           shape=None):
        """Creates a general depolarizer form its three eigenstates and their depolarization factors (eigenvalues), plus its polarization or polarizance vector.

        Parameters:
            d (list, float or numpy.ndarray): Depolarization factors (eigenvalues of m). If list, it must contain three float or numpy arrays, one for each diagonal value. If float or numpy.ndarray, $d_1 = d_2 = d_3$.
            S (list or Stokes): Principal states. If list, it must contain three Stokes objects. If Stokes, at least one dimension must have dimension 3.
            Dv (numpy.ndarray): Diattenuation vector. If None, the polarizance vector is used instead. Default: None.
            Pv (numpy.ndarray): Polarizance vector. Used only if Dv is None. If None, the depolarizer will have zero diattenuation and polarizance. Default: None.
            M00 (float, default 1): Parameter of average intensity. Default: 1.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Prepare variables. First the extinction
        if isinstance(d, list):
            d_list = d
        else:
            sh = np.array(d.shape)
            ind = np.argmin(~(sh == 3))
            d_list = [
                np.take(d, 0, axis=ind),
                np.take(d, 1, axis=ind),
                np.take(d, 2, axis=ind)
            ]
        (d1, d2, d3), new_shape1 = prepare_variables(vars=d,
                                                     expand=[True] * 3,
                                                     length=length,
                                                     give_shape=True)
        d1 = np.sqrt(d1.flatten())
        d2 = np.sqrt(d2.flatten())
        d3 = np.sqrt(d3.flatten())
        # Now, the states
        if isinstance(S, list):
            S1, S2, S3 = S
        else:
            S1, S2, S3 = create_Stokes(N=3)
            sh = np.array(S.shape)
            ind = np.argmin(~(sh == 3))
            M = S.parameters.matrix()
            S1.from_matrix(np.take(M, 0, axis=ind + 1))
            S2.from_matrix(np.take(M, 1, axis=ind + 1))
            S3.from_matrix(np.take(M, 2, axis=ind + 1))
        S1.normalize()
        S2.normalize()
        S3.normalize()
        comp = [
            S1.M[1, :], S1.M[2, :], S1.M[3, :], S2.M[1, :], S2.M[2, :],
            S2.M[3, :], S3.M[1, :], S3.M[2, :], S3.M[3, :]
        ]
        comp, new_shape2 = prepare_variables(vars=comp,
                                             expand=[True] * 9,
                                             length=length,
                                             give_shape=True)
        for ind in range(9):
            comp[ind] = comp[ind].flatten()
        new_shape = take_shape([new_shape1, new_shape2])

        # Create the small matrix of the depolarizer
        v1 = np.array([d1 * comp[0], d1 * comp[1], d1 * comp[2]])
        v2 = np.array([d2 * comp[3], d2 * comp[4], d2 * comp[5]])
        v3 = np.array([d3 * comp[6], d3 * comp[7], d3 * comp[8]])
        m1 = kron_axis(v1, v1, axis=0)
        m2 = kron_axis(v2, v2, axis=0)
        m3 = kron_axis(v3, v3, axis=0)
        m = m1 + m2 + m3

        # Create the object
        self.from_blocks(M00=M00, Dv=Dv, Pv=Pv, m=m)
        self.shape, _ = select_shape(obj=self,
                                             shape_var=new_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)

        return self

    # @_actualize_
    def diattenuator_perfect(self,
                             azimuth=0,
                             global_phase=0,
                             length=1,
                             shape_like=None,
                             shape=None):
        """Mueller 4x4 matrix for a perfect diattenuator (polarizer).

        Parameters:
            azimuth (float or numpy.ndarray): Rotation angle of the high transmission polarizer axis. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        self.diattenuator_linear(p1=1,
                                 p2=0,
                                 azimuth=azimuth,
                                 global_phase=global_phase,
                                 shape=shape,
                                 shape_like=shape_like)
        return self

    # @_actualize_
    def diattenuator_linear(self,
                            p1=1,
                            p2=0,
                            Tmax=None,
                            Tmin=None,
                            azimuth=0,
                            global_phase=0,
                            length=1,
                            shape_like=None,
                            shape=None):
        """Mueller matrices of pure linear homogeneous diattenuators.

        .. math:: M\left(\theta=0\right)=\frac{1}{2}\left[\begin{array}{cccc}
                            p_{1}^{2}+p_{2}^{2} & p_{1}^{2}-p_{2}^{2} & 0 & 0\\
                            p_{1}^{2}-p_{2}^{2} & p_{1}^{2}+p_{2}^{2} & 0 & 0\\
                            0 & 0 & 2p_{1}p_{2} & 0\\
                            0 & 0 & 0 & 2p_{1}p_{2}
                            \end{array}\right]

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), (4.79) - p. 143.
            Handbook of Optics vol 2. 22.16 (Table 1).

        Parameters:
            p1 (float or numpy.ndarray): Field transmission of the transmission axis. Default: 1.
            p2 (float or numpy.ndarray): Field transmission of the attenuation axis. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            azimuth (float or numpy.ndarray): Rotation angle of the high transmission polarizer axis. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Use field transmission coefficients
        if Tmax is None:
            Tmax = p1**2
        if Tmin is None:
            Tmin = p2**2
        # Prepare variables
        (p1, p2,
         azimuth), new_shape = prepare_variables(vars=[p1, p2, azimuth],
                                                 expand=[False, False, False],
                                                 length=length,
                                                 give_shape=True)
        # Calculate intensity transmission coefficients
        a = (Tmax + Tmin) / 2
        b = (Tmax - Tmin) / 2
        c = np.sqrt(Tmax * Tmin)
        # Calculate the matrix
        components = [a] + [b] + [0] * 2 + [b] + \
            [a] + [0] * 4 + [c] + [0] * 4 + [c]
        self.from_components(components=components,
                             global_phase=global_phase,
                             length=length)

        self.rotate(azimuth)
        self.shape, _ = select_shape(obj=self,
                                             shape_var=new_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)
        return self
    
    def diattenuator_circular(self,
                            p1=1,
                            p2=0,
                            Tmax=None,
                            Tmin=None,
                            kind='d',
                            global_phase=0,
                            length=1,
                            shape_like=None,
                            shape=None):
        """Creates a real polarizer with circular eigenstates.

        Parameters:
            p1 (float or numpy.ndarray): Electric field transmission coefficient of the transmission eigenstate. Default: 1.
            p2 (float or numpy.ndarray): Electric field transmission coefficient of the extinction eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            kind (str): Transmission eigenstate is 'd','r', 'right', 'dextro' or 'l', 'i', 'left', 'levo'.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        ellipticity = 45*degrees if kind.lower() in ["r", "d", "right", "dextro"] else -45*degrees
        return self.diattenuator_azimuth_ellipticity(p1=p1,
                                                     p2=p2,
                                                     Tmax=Tmax,
                                                     Tmin=Tmin,
                                                     azimuth=0,
                                                     ellipticity=ellipticity,
                                                     global_phase=global_phase,
                                                     length=length,
                                                     shape_like=shape_like,
                                                     shape=shape)

    # @_actualize_
    def diattenuator_general(self,
                            azimuth=0,
                            ellipticity=0,
                            alpha=0,
                            delay=0,
                            *args,
                            **kwargs):
        """Method that calculates the most general homogenous diattenuator.

        Parameters:
            azimuth (float): [0, pi]: Azimuth of the transmission eigenstate. Only used if alpha and delay are None.  Default: 0.
            ellipticity (float): [-pi/4, pi/]: Ellipticity angle of the transmission eigenstate. Only used if alpha and delay are None.  Default: 0.
            alpha (float or numpy.ndarray): [0, pi/2]: tan(alpha) is the ratio between field amplitudes of X and Y components of the transmission eigenstate. Overrides azimuth and ellipticity if not None. Default: 0.
            delay (float or numpy.ndarray): [0, 2*pi]: phase difference between X and Y field components of the transmission eigenstate. Overrides azimuth and ellipticity if not None. Default: 0.
            p1 (float or numpy.ndarray): Electric field transmission coefficient of the transmission eigenstate. Default: 1.
            p2 (float or numpy.ndarray): Electric field transmission coefficient of the extinction eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of azimuth and ellipticity or alpha and delay are forced to be in their correct range. Default: True.

        Returns:
            (Mueller): Created object.
        """
        if alpha is None or delay is None:
            return self.diattenuator_azimuth_ellipticity(
                azimuth=azimuth,
                ellipticity=ellipticity,
                *args,
                **kwargs)
        else:
            return self.diattenuator_charac_angles(
                alpha=alpha,
                delay=delay,
                *args,
                **kwargs)

    # @_actualize_
    def diattenuator_azimuth_ellipticity(self,
                                         p1=1,
                                         p2=0,
                                         Tmax=None,
                                         Tmin=None,
                                         azimuth=0,
                                         ellipticity=0,
                                         global_phase=0,
                                         length=1,
                                         shape_like=None,
                                         shape=None,
                                         force_limits=False):
        """Method that calculates the most general homogenous diattenuator from
        diattenuator parameters with the intermediate step of calculating the
        diattenuation vector.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 142.

        Parameters:
            p1 (float or numpy.ndarray): Electric field transmission coefficient of the transmission eigenstate. Default: 1.
            p2 (float or numpy.ndarray): Electric field transmission coefficient of the extinction eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            azimuth (float): [0, pi]: Azimuth. Defaut: 0
            ellipticity (float): [-pi/4, pi/4]: Ellipticity angle. Default: 0
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of azimuth and ellipticity are forced to be in the range [0, pi) and [-pi/4, pi/4] respectively. Default: False.

        Returns:
            (Mueller): Created object.
        """
        # Use field transmission coefficients
        if Tmax is None:
            Tmax = p1**2
        if Tmin is None:
            Tmin = p2**2
        # Prepare variables
        (Tmax, Tmin, azimuth, ellipticity), new_shape = prepare_variables(
            vars=[Tmax, Tmin, azimuth, ellipticity],
            expand=[True, False, False, False],
            length=length,
            give_shape=True)
        # Restrict parameter values to the correct interval
        if force_limits:
            azimuth = put_in_limits(azimuth, "azimuth")
            ellipticity = put_in_limits(ellipticity, "ellipticity")
        # Calculate the diattenuation vector and M00
        cte = (Tmax - Tmin) / (Tmax + Tmin)
        cond = np.isnan(cte)
        if np.any(cond):
            cte[cond] = 0
        M00 = (Tmax + Tmin) / 2
        Dv = [
            cte * np.cos(2 * azimuth) * np.cos(2 * ellipticity),
            cte * np.sin(2 * azimuth) * np.cos(2 * ellipticity),
            cte * np.sin(2 * ellipticity)
        ]
        # Create the element
        self.diattenuator_vector(Dv=Dv,
                                 M00=M00,
                                 global_phase=global_phase,
                                 shape=shape,
                                 shape_like=shape_like)
        self.shape, _ = select_shape(obj=self, shape_var=new_shape)
        return self

    # @_actualize_
    def diattenuator_charac_angles(self,
                                   p1=1,
                                   p2=0,
                                   Tmax=None,
                                   Tmin=None,
                                   alpha=0,
                                   delay=0,
                                   global_phase=0,
                                   length=1,
                                   shape_like=None,
                                   shape=None,
                                   force_limits=False):
        """Method that calculates the most general homogenous diattenuator from diattenuator parameters with the intermediate step of calculating the diattenuation vector.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 142.

        Parameters:
            p1 (float or numpy.ndarray): Electric field transmission coefficient of the transmission eigenstate Default: 1
            p2 (float or numpy.ndarray): Electric field transmission coefficient of the extinction eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            alpha (float or numpy.ndarray): [0, pi/2]: tan(alpha) is the ratio between field amplitudes of X and Y components. Default: 0.
            delay (float or numpy.ndarray): [0, 2*pi]: phase difference between X and Y field components. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of alpha and delay are forced to be in the range [0, pi/2] and [0, 2*pi) respectively. Default: False.

        Returns:
            (Mueller): Created object.
        """
        # Use field transmission coefficients
        if Tmax is None:
            Tmax = p1**2
        if Tmin is None:
            Tmin = p2**2
        # Prepare variables
        (Tmax, Tmin, alpha, delay), new_shape = prepare_variables(
            vars=[Tmax, Tmin, alpha, delay],
            expand=[True, False, False, False],
            length=length,
            give_shape=True)
        # Restrict parameter values to the correct interval
        if force_limits:
            alpha = put_in_limits(alpha, "alpha")
            delay = put_in_limits(delay, "delay")
        # Calculate diattenuation vector and M00
        cte = (Tmax - Tmin) / (Tmax + Tmin)
        cond = np.isnan(cte)
        if np.any(cond):
            cte[cond] = 0
        M00 = (Tmax + Tmin) / 2
        Dv = [
            cte * np.cos(2 * alpha), cte * np.sin(2 * alpha) * np.cos(delay),
            cte * np.sin(2 * alpha) * np.sin(delay)
        ]
        # Create the element
        self.diattenuator_vector(Dv=Dv,
                                 M00=M00,
                                 global_phase=global_phase,
                                 shape=shape,
                                 shape_like=shape_like)
        self.shape, _ = select_shape(obj=self, shape_var=new_shape)
        return self

    # @_actualize_
    def diattenuator_vector(self,
                            Dv,
                            M00=None,
                            global_phase=0,
                            length=1,
                            shape_like=None,
                            shape=None):
        """Method that calculates the most general homogenous diattenuator from the
        Diattenuation or Polarizance vector.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 142.

        Parameters:
            Dv (3xN numpy.ndarray): Diattenuation or Polarizance vectors.
            M00 (float or numpy.ndarray): Parameter of average intensity. If None, the maximum possible value is used. Default: None.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Prepare the variables
        _, Dv, _, _, new_shape = prepare_variables_blocks(Dv=Dv,
                                                          give_shape=True)
        # Calculate the diattenuation and make it physically realizable
        d = np.linalg.norm(Dv, axis=0)
        cond = d > 1
        if np.any(cond):
            Dv[0, cond] = Dv[0, cond] / d[cond]
            Dv[1, cond] = Dv[1, cond] / d[cond]
            Dv[2, cond] = Dv[2, cond] / d[cond]
            d[cond] = 1
        # Calculate maximum achievable m00
        if M00 is None:
            M00 = 1 / (1 + d)
        # Calculate the m small matrix
        skd = np.sqrt(1 - d**2)
        m1 = np.multiply.outer(np.diag([1, 1, 1]), skd)
        cte = (1 - skd) / d**2
        cond = np.isnan(cte)
        if np.any(cond):
            cte[cond] = 0
        m2 = kron_axis(Dv, Dv, axis=0) * np.multiply.outer(
            np.ones((3, 3)), cte)
        m = m1 + m2
        # This equation fails if d=0
        cond = d < tol_default
        if np.any(cond):
            m = np.array([np.eye(3)])
        # Now we have all the necessary blocks
        self.from_blocks(Dv,
                         Dv,
                         m,
                         M00,
                         global_phase=global_phase,
                         shape=shape,
                         shape_like=shape_like)
        self.shape, _ = select_shape(obj=self, shape_var=new_shape)
        return self

    # @_actualize_
    def quarter_waveplate(self,
                          azimuth=0,
                          global_phase=0,
                          length=1,
                          shape_like=None,
                          shape=None):
        """Mueller matrices of ideal quarter-wave retarder.

        .. math:: M\left(\theta=0\right)=\left[\begin{array}{cccc}
                        1 & 0 & 0 & 0\\
                        0 & 1 & 0 & 0\\
                        0 & 0 & 0 & 1\\
                        0 & 0 & -1 & 0
                        \end{array}\right]

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), (4.32) - p. 132
            Handbook of Optics vol 2. 22.16 (Table 1).

        Parameters:
            azimuth (float or numpy.ndarray): rotation angle of the high transmission polarizer axis. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Calculate the matrix components
        components = [1] + [0] * 4 + [1] + [0] * 5 + \
            [1] + [0] * 2 + [-1] + [0]
        # Create the object
        self.from_components(components,
                             global_phase=global_phase,
                             length=length,
                             shape=shape,
                             shape_like=shape_like)
        self.rotate(azimuth)
        return self

    # @_actualize_
    def half_waveplate(self,
                       azimuth=0,
                       global_phase=0,
                       length=1,
                       shape_like=None,
                       shape=None):
        """Mueller matrices of ideal half-wave retarders.

        .. math:: M\left(\theta=0\right)=\left[\begin{array}{cccc}
                        1 & 0 & 0 & 0\\
                        0 & 1 & 0 & 0\\
                        0 & 0 & -1 & 0\\
                        0 & 0 & 0 & -1
                        \end{array}\right]

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), (4.32) - p. 132
            Handbook of Optics vol 2. 22.16 (Table 1).

        Parameters:
            azimuth (float or numpy.ndarray): rotation angle of the high transmission polarizer axis. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Calculate the matrix components
        components = [1] + [0] * 4 + [1] + [0] * 4 + [-1] + [0] * 4 + [-1]
        # Create the object
        self.from_components(components,
                             global_phase=global_phase,
                             length=length,
                             shape=shape,
                             shape_like=shape_like)
        self.rotate(azimuth)
        return self

    # @_actualize_
    def retarder_general(self,
                        azimuth=0,
                        ellipticity=0,
                        alpha=None,
                        delay=None,
                        *args,
                        **kwargs):
        """Creates the most general retarder.

        Parameters:
            azimuth (float): [0, pi]: Azimuth of the fast eigenstate. Only used if alpha and delay are None. Default: 0.
            ellipticity (float): [-pi/4, pi/4]: Ellipticity angle of the fast eigenstate. Only used if alpha and delay are None. Default: 0.
            alpha (float): [0, pi/2]: tan(alpha) is the ratio between amplitudes of the eigenstates in Jones formalism. Overrides azimuth and ellipticity. Default: None.
            delay (float): [0, 2*pi]: phase difference between both components of the eigenstates in Jones formalism. Overrides azimuth and ellipticity. Default: None.
            ret (float or numpy.ndarray): [0, pi] Retardance introduced to the slow eigenstate respect to the fast eigenstate. Default: 90 degrees.
            M00 (float or numpy.ndarray): Parameter of average intensity. Default: 1.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of R, and azimuth and ellipticity or alpha and delay are forced to be in their correct range. Default: False.
            R (float or None): Same as ret, overrides it if it is not None. Maintained for retrocompatibility. Default: None.

        Returns:
            (Mueller): Created object.
        """
        if alpha is None or delay is None:
            return self.retarder_azimuth_ellipticity(
                azimuth=azimuth,
                ellipticity=ellipticity,
                *args,
                **kwargs)
        else:
            return self.retarder_charac_angles(
                alpha=alpha,
                delay=delay,
                *args,
                **kwargs)


    # @_actualize_
    def retarder_azimuth_ellipticity(self,
                                     ret=90*degrees,
                                     azimuth=0,
                                     ellipticity=0,
                                     M00=1,
                                     global_phase=0,
                                     length=1,
                                     shape_like=None,
                                     shape=None,
                                     force_limits=False,
                                     R=None):
        """Method that calculates the most general homogeneous retarder from azimuth and ellipticity of the fast eigenstate. The method calculates first the retardance vector, and uses it to calculate the Mueler matrix.

        References:
            "Polarized light and the Mueller Matrix approach", J. J. Gil, pp 125.

        Parameters:
            R (float or numpy.ndarray): [0, pi] Retardance introduced to the slow eigenstate respect to the fast eigenstate. Default: 90 degrees.
            azimuth (float or numpy.ndarray): [0, pi]: Azimuth. Default: 0.
            ellipticity (float or numpy.ndarray): [-pi/4, pi/4]: Ellipticity angle. Default: 0.
            M00 (float or numpy.ndarray): Parameter of average intensity. Default: 1.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of ret, azimuth and ellipticity are forced to be in the range [0,pi], [0, pi) and [-pi/4, pi/4] respectively. Default: False.
            R (float or None): Same as ret, overrides it if it is not None. Maintained for retrocompatibility. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Prepare variables
        R = R if R is not None else ret
        (R, azimuth, ellipticity), new_shape = prepare_variables(
            vars=[R, azimuth, ellipticity],
            expand=[False, True, True],
            give_shape=True)
        # Restrict parameter values to the correct interval
        azimuth[np.isnan(azimuth)] = 0  # Avoid nan values
        if force_limits:
            azimuth = put_in_limits(azimuth, "azimuth")
            ellipticity = put_in_limits(ellipticity, "ellipticity")
            R = put_in_limits(R, "Retardance")

        # Calculate the normalized retardance vector
        Rv = np.array([
            np.cos(2 * azimuth) * np.cos(2 * ellipticity),
            np.sin(2 * azimuth) * np.cos(2 * ellipticity),
            np.sin(2 * ellipticity)
        ])
        # Create the object
        self.retarder_vector(Rv=Rv,
                             ret=R,
                             kind='normalized',
                             M00=M00,
                             global_phase=global_phase,
                             shape=shape,
                             shape_like=shape_like)
        self.shape, _ = select_shape(obj=self, shape_var=new_shape)
        return self

    # @_actualize_
    def retarder_linear(self,
                        ret=90*degrees,
                        azimuth=0,
                        global_phase=0,
                        length=1,
                        shape_like=None,
                        shape=None,
                        force_limits=False,
                        R=None):
        """Mueller matrix of homogeneous linear retarders.

        .. math:: M\left(\theta=0\right)=\left[\begin{array}{cccc}
                            1 & 0 & 0 & 0\\
                            0 & 1 & 0 & 0\\
                            0 & 0 & \cos(\Delta) & \sin(\Delta)\\
                            0 & 0 & -\sin(\Delta) & \cos(\Delta)
                            \end{array}\right]

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), (4.31) - p. 132
            Handbook of Optics vol 2. 22.16 (Table 1).

        Parameters:
            ret (float or numpy.ndarray): [0, pi] Retardance introduced to the slow eigenstate respect to the fast eigenstate. Default: 90 degrees.
            azimuth (float or numpy.ndarray): rotation angle of the high transmission polarizer axis. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of ret are forced to be in the range [0, pi]. Default: False.
            R (float or None): Same as ret, overrides it if it is not None. Maintained for retrocompatibility. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Prepare variables
        R = R if R is not None else ret
        if force_limits:
            R = put_in_limits(R, 'Retardance')
        # Calculate the matrix components
        components = [1] + [0] * 4 + [1] + [0] * 4 + \
            [np.cos(R)] + [np.sin(R)] + [0] * 2 + [-np.sin(R)] + [np.cos(R)]
        # Create the object
        self.from_components(components,
                             global_phase=global_phase,
                             length=length,
                             shape=shape,
                             shape_like=shape_like)
        self.rotate(azimuth)
        return self
    
    def retarder_circular(self,
                        ret=90 * degrees,
                        kind="d",
                        global_phase=0,
                        length=1,
                        shape_like=None,
                        shape=None,
                        force_limits=False,
                        R=None):
        """Creates a linear retarder.

        Parameters:
            ret (float): [0, pi] Retardance (delay between components). Default: 90 degrees.
            kind (str): Fast eigenstate is 'd','r' (right, dextro) or 'l', 'i' (left, levo).
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of R are forced to be in the range [0, pi]. Default: False.
            R (float or None): Same as ret, overrides it if it is not None. Maintained for retrocompatibility. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        ellipticity = 45*degrees if kind in "rdRD" else -45*degrees
        return self.retarder_azimuth_ellipticity(ret=ret,
                                                azimuth=0,
                                                ellipticity=ellipticity,
                                                global_phase=global_phase,
                                                length=length,
                                                shape_like=shape_like,
                                                shape=shape,
                                                force_limits=force_limits,
                                                R=R)

    # @_actualize_
    def retarder_charac_angles(self,
                               ret=90*degrees,
                               alpha=0,
                               delay=0,
                               M00=1,
                               global_phase=0,
                               length=1,
                               shape_like=None,
                               shape=None,
                               force_limits=False,
                               R=None):
        """Method that calculates the most general homogeneous retarder from the characteristic angles of the fast eigenstate. The method calculates first the retardance vector, and uses it to calculate the Mueler matrix.

        References:
            "Polarized light and the Mueller Matrix approach", J. J. Gil, pp 125.

        Parameters:
            ret (float or numpy.ndarray): [0, pi] Retardance introduced to the slow eigenstate respect to the fast eigenstate. Default: 90 degrees.
            alpha (float or numpy.ndarray): [0, pi]: tan(alpha) is the ratio between amplitudes of the electric field of the fast eigenstate. Default: 0.
            delay (float or numpy.ndarray): [0, 2*pi]: phase difference between both components of the electric field of the fast eigenstate. Default: 0.
            M00 (float or numpy.ndarray): Parameter of average intensity. Default: 1
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values ofret,  alpha and delay are forced to be in the range [0,pi], [0, pi/2] and [0, 2*pi) respectively. Default: False.
            R (float or None): Same as ret, overrides it if it is not None. Maintained for retrocompatibility. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Prepare variables
        R = R if R is not None else ret
        (R, alpha,
         delay), new_shape = prepare_variables(vars=[R, alpha, delay],
                                               expand=[False, True, True],
                                               give_shape=True)
        # Restrict parameter values to the correct interval
        if force_limits:
            alpha = put_in_limits(alpha, "alpha")
            delay = put_in_limits(delay, "delay")
        # Calculate the normalized retardance vector
        Rv = np.array([
            np.cos(2 * alpha),
            np.sin(2 * alpha) * np.cos(delay),
            np.sin(2 * alpha) * np.sin(delay)
        ])
        # Create the object
        self.retarder_vector(Rv=Rv,
                             ret=R,
                             kind='normalized',
                             M00=M00,
                             global_phase=global_phase,
                             shape=shape,
                             shape_like=shape_like)
        self.shape, _ = select_shape(obj=self, shape_var=new_shape)
        return self

    # @_actualize_
    def retarder_vector(self,
                        Rv,
                        ret=90 * degrees,
                        kind='normalized',
                        M00=1,
                        global_phase=0,
                        length=1,
                        shape_like=None,
                        shape=None,
                        force_limits=False,
                        R=None):
        """Method that calculates the most general homogeneous retarder from the retardance vector.

        References:
            "Polarized light and the Mueller Matrix approach", J. J. Gil, pp 125.

        Parameters:
            Rv (3xN numpy.ndarray): Retardance vector.
            R (float or numpy.ndarray): [0, pi] Retardance introduced to the slow eigenstate respect to the fast eigenstate. Default: 90 degrees.
            kind (string): Identifies the type of retardance vector. There are three possibilities: NORMALIZED (also called Pauli vector), STRAIGHT or COMPLETE. Default: 'normalized'.
            M00 (float or numpy.ndarray): Mean transmission coefficient. If different than 1, the object won't be a pure retarder. Default: 1.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of ret are forced to be in the range [0, pi]. Only applicable to NORALIZED. Default: False.
            R (float or None): Same as ret, overrides it if it is not None. Maintained for retrocompatibility. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Prepare the variables
        R = R if R is not None else ret
        _, Rv, _, _, new_shape = prepare_variables_blocks(Dv=Rv,
                                                          give_shape=True)
        # Choose the right retardance
        if kind in ('complete', 'COMPLETE', 'Complete'):
            # Complete retardance vectors
            R = np.linalg.norm(Rv, axis=0)
            aux = np.multiply.outer(np.ones(3), R)
            cond = R != 0
            if np.any(cond):
                Rv[:, cond] = Rv[:, cond] / aux[:, cond]
        elif kind in ('straight', 'STRAIGHT', 'Straight'):
            # Straight retardance vectors
            R = np.linalg.norm(Rv, axis=0) * np.pi
            aux = np.multiply.outer(np.ones(3), R / np.pi)
            cond = R != 0
            if np.any(cond):
                Rv[:, cond] = Rv[:, cond] / aux[:, cond]
        else:
            # Normalized (Pauli) retardance vector
            if force_limits:
                R = put_in_limits(R, 'Retardance')
            norm = np.linalg.norm(Rv, axis=0)
            aux = np.multiply.outer(np.ones(3), norm)
            cond = (norm != 0) + (norm != 1)
            if np.any(cond):
                Rv[:, cond] = Rv[:, cond] / aux[:, cond]
        # Reshape R
        (R, M00), new_shape2 = prepare_variables(vars=[R, M00],
                                                 expand=[True, False],
                                                 length=Rv.shape[1],
                                                 give_shape=True)
        new_shape = take_shape((new_shape, new_shape2))
        # Calculate small m matrix
        sR, cR = (np.sin(R), np.cos(R))
        m = np.multiply.outer(np.eye(3), cR)
        for indI in range(3):
            for indJ in range(3):
                m[indI, indJ, :] += Rv[indI, :] * Rv[indJ, :] * (1 - cR)
                for indK in range(3):
                    m[indI, indJ, :] = m[indI, indJ, :] + \
                        Eijk(indI, indJ, indK) * Rv[indK, :] * sR
        # Create the object
        self.from_blocks(Dv=np.zeros(3),
                         Pv=np.zeros(3),
                         m=m,
                         M00=M00,
                         global_phase=global_phase,
                         length=length,
                         shape=shape,
                         shape_like=shape_like)
        self.shape, _ = select_shape(obj=self, shape_var=new_shape)
        return self

    # @_actualize_
    def diattenuator_retarder_linear(self,
                                     p1=1,
                                     p2=0,
                                     Tmax=None,
                                     Tmin=None,
                                     ret=90 * degrees,
                                     azimuth=0,
                                     global_phase=0,
                                     length=1,
                                     shape_like=None,
                                     shape=None,
                                     R=None):
        """Creates the Mueller matrices of linear diattenuator retarders with the same eigenstates for diattenuation and retardance.

        .. math:: M\left(\theta=0\right)=\frac{1}{2}\left[\begin{array}{cccc}
            p_{1}^{2}+p_{2}^{2} & p_{1}^{2}-p_{2}^{2} & 0 & 0\\
            p_{1}^{2}-p_{2}^{2} & p_{1}^{2}+p_{2}^{2} & 0 & 0\\
            0 & 0 & 2p_{1}p_{2}\cos(\varDelta) & 2p_{1}p_{2}\sin(\varDelta)\\
            0 & 0 & -2p_{1}p_{2}\sin(\varDelta) & 2p_{1}p_{2}\cos(\varDelta)
            \end{array}\right]

        References:
            Handbook of Optics vol 2. 22.16 (Table 1).

        Parameters:
            p1 (float or numpy.ndarray): Field transmission of the transmission axis. Default: 1.
            p2 (float or numpy.ndarray): Field transmission of the attenuation axis. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            ret (float or numpy.ndarray): [0, pi] Retardance introduced to the slow eigenstate respect to the fast eigenstate. Default: 90 degrees.
            azimuth (float or numpy.ndarray): rotation angle of the high transmission polarizer axis. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            R (float or None): Same as ret, overrides it if it is not None. Maintained for retrocompatibility. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Use field transmission coefficients
        if Tmax is None:
            Tmax = p1**2
        if Tmin is None:
            Tmin = p2**2
        # Prepare variables
        (p1, p2,
         azimuth), new_shape = prepare_variables(vars=[p1, p2, azimuth],
                                                 expand=[False, False, False],
                                                 length=length,
                                                 give_shape=True)
        # Calculate intensity transmission coefficients
        a = (Tmax + Tmin) / 2
        b = (Tmax - Tmin) / 2
        c = np.sqrt(Tmax * Tmin)
        # Calculate the matrix
        components = [a] + [b] + [0] * 2 + [b] + [a] + [0] * 4 + \
            [c * np.cos(R)] + [c * np.sin(R)] + [0] * 2 + \
            [-c * np.sin(R)] + [c * np.cos(R)]
        self.from_components(components=components,
                             global_phase=global_phase,
                             shape=shape,
                             shape_like=shape_like)
        self.rotate(azimuth)
        self.shape, _ = select_shape(obj=self, shape_var=new_shape)
        return self

    def diattenuator_retarder_general(self,
                                    azimuth=0,
                                    ellipticity=0,
                                    alpha=None,
                                    delay=None,
                                    *args,
                                    **kwargs):
        """Creates the most general homogenous diattenuator retarder.

        Parameters:
            azimuth (float or numpy.ndarray): rotation angle of the high transmission and fast eigenstate. Only used if alpha or delay are None. Default: 0.
            ellipticity (float): [-pi/4, pi/]: Ellipticity angle of the high transmission and fast eigenstate. Only used if alpha or delay are None. Default: 0.
            alpha (float or numpy.ndarray): [0, pi/2]: tan(alpha) is the ratio between field amplitudes of X and Y components of the high transmission and fast eigenstate. Overrides azimuth and ellipticity if not None. Default: None.
            delay (float or numpy.ndarray): [0, 2*pi]: Phase difference between X and Y field components of the high transmission and fast eigenstate. Overrides azimuth and ellipticity if not None. Default: None.
            p1 (float or numpy.ndarray): Field transmission of the fast axis. Default: 1.
            p2 (float or numpy.ndarray): Electric field transmission coefficient of the extinction eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            ret (float or numpy.ndarray): Retardance. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of azimuth and ellipticity or alpha and delay (but not ret) are forced to be in their correct range. Default: False.
            R (float or None): Same as ret, overrides it if it is not None. Maintained for retrocompatibility. Default: None.

        Returns:
            (Mueller): Created object.
        """
        if alpha is None or delay is None:
            Jr = self.retarder_azimuth_ellipticity(
                azimuth=azimuth,
                ellipticity=ellipticity,
                *args,
                **kwargs)
            Jd = self.diattenuator_azimuth_ellipticity(
                azimuth=azimuth,
                ellipticity=ellipticity,
                *args,
                **kwargs)
        else:
            Jr = self.retarder_charac_angles(
                alpha=alpha,
                delay=delay,
                *args,
                **kwargs)
            Jd = self.diattenuator_charac_angles(
                alpha=alpha,
                delay=delay,
                *args,
                **kwargs)
            
        return Jd * Jr

    def diattenuator_retarder_azimuth_ellipticity(self,
                                                  p1=1,
                                                  p2=0,
                                                  Tmax=None,
                                                  Tmin=None,
                                                  ret=0,
                                                  azimuth=0,
                                                  ellipticity=0,
                                                  global_phase=0,
                                                  length=1,
                                                  shape_like=None,
                                                  shape=None,
                                                  force_limits=False,
                                                  R=None):
        """Creates the most general homogenous diattenuator retarder from the azimuth and ellipticity of the fast eigenstate.

        Parameters:
            p1 (float or numpy.ndarray): Field transmission of the fast axis. Default: 1.
            p2 (float or numpy.ndarray): Electric field transmission coefficient of the extinction eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            R (float or numpy.ndarray): Retardance. Default: 0.
            azimuth (float or numpy.ndarray): rotation angle of the high transmission polarizer axis. Default: 0.
            ellipticity (float): [-pi/4, pi/]: Ellipticity angle.  Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of azimuth and ellipticity (but not ret) are forced to be in the range [0,pi) and [-pi/4, pi/4] respectively. Default: False.
            R (float or None): Same as ret, overrides it if it is not None. Maintained for retrocompatibility. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Use field transmission coefficients
        if Tmax is not None:
            p1 = np.sqrt(Tmax)
        if Tmin is not None:
            p2 = np.sqrt(Tmin)
        # Prepare variables
        R = R if R is not None else ret
        (p1, p2, R,
         azimuth, ellipticity), new_shape = prepare_variables(vars=[p1, p2, R, azimuth, ellipticity],
                                                 expand=[False, False, False, False, False],
                                                 length=length,
                                                 give_shape=True)
        if force_limits:
            azimuth = put_in_limits(azimuth, "azimuth")
            ellipticity = put_in_limits(ellipticity, "ellipticity")
        # Restrict retardance between 0 and 360 degrees
        R = put_in_limits(R, "delay", out_number=False)
        # Create the two objects
        name = self.name
        E1 = Mueller().diattenuator_azimuth_ellipticity(p1=p1,
                                            p2=p2,
                                            azimuth=azimuth,
                                            ellipticity=ellipticity,
                                            shape=shape,
                                            shape_like=shape_like,
                                            length=length)
        # Orthogonal state for R > 180º
        cond = R > np.pi
        if cond.size > 0:
            azimuth[cond] = azimuth[cond] + np.pi/2
            ellipticity[cond] = -ellipticity[cond]
        E2 = Mueller().retarder_azimuth_ellipticity(R=R,
                                        azimuth=azimuth,
                                        ellipticity=ellipticity,
                                        global_phase=global_phase,
                                        shape=shape,
                                        shape_like=shape_like,
                                        length=length)
        self = E1 * E2
        self.name = name
        return self

    def diattenuator_retarder_charac_angles(self,
                                            p1=1,
                                            p2=0,
                                            Tmax=None,
                                            Tmin=None,
                                            ret=0,
                                            alpha=0,
                                            delay=0,
                                            global_phase=0,
                                            length=1,
                                            shape_like=None,
                                            shape=None,
                                            force_limits=False,
                                            R=None):
        """Creates the most general homogenous diattenuator retarder from the characteristic angles of the fast eigenstate.

        Parameters:
            p1 (float or numpy.ndarray): Field transmission of the fast axis. Default: 1.
            p2 (float or numpy.ndarray): Electric field transmission coefficient of the extinction eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            ret (float or numpy.ndarray): Retardance. Default: 0.
            alpha (float or numpy.ndarray): [0, pi/2]: tan(alpha) is the ratio between field amplitudes of X and Y components. Default: 0.
            delay (float or numpy.ndarray): [0, 2*pi]: phase difference between X and Y field components. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of alpha and delay (but not ret) are forced to be in the range [0, pi/2] and [0, 2*pi) respectively. Default: False.
            R (float or None): Same as ret, overrides it if it is not None. Maintained for retrocompatibility. Default: None.

        Returns:
            (Mueller): Created object.
        """
        # Use field transmission coefficients
        if Tmax is not None:
            p1 = np.sqrt(Tmax)
        if Tmin is not None:
            p2 = np.sqrt(Tmin)
        # Prepare variables
        R = R if R is not None else ret
        (p1, p2, R,
         alpha, delay), new_shape = prepare_variables(vars=[p1, p2, R, alpha, delay],
                                                 expand=[False, False, False, False, False],
                                                 length=length,
                                                 give_shape=True)
        if force_limits:
            alpha = put_in_limits(alpha, "alpha")
            delay = put_in_limits(delay, "delay")
        # Restrict retardance between 0 and 360 degrees
        R = put_in_limits(R, "delay", out_number=False)
        # Create the two objects
        name = self.name
        E1 = Mueller().diattenuator_charac_angles(p1=p1,
                                      p2=p2,
                                      alpha=alpha,
                                      delay=delay,
                                      shape=shape,
                                      shape_like=shape_like,
                                      length=length)
        # Orthogonal state for R > 180º
        cond = R > np.pi
        if cond.size > 0:
            alpha[cond] = np.pi/2 - alpha[cond]
            delay[cond] = delay[cond] + np.pi
        E2 = Mueller().retarder_charac_angles(R=R,
                                  alpha=alpha,
                                  delay=delay,
                                  global_phase=global_phase,
                                  shape=shape,
                                  shape_like=shape_like,
                                  length=length)
        self = E1 * E2
        self.name = name
        return self
    
    # @_actualize_
    def biaxial_material(self,
                          ne=1,
                          no=1,
                          thickness=1,
                          wavelength=0.6328,
                          azimuth=0,
                          rotation=0,
                          length=1,
                          shape_like=None,
                          shape=None):
        """Creates a material using the physical properties of an anisotropic material. The material behaves as a retardaer if the real part of the refractive indices are different and, as a polarizer if the imaginary parts are different.

        Parameters:
            ne (float or numpy.ndarray): Extraordinary refractive index. Coincides with the refractive index for x component when azimuth is 0. Default: 1.
            no (float or numpy.ndarray): Ordinary refractive index. Coincides with the efractive index for y component when azimuth is 0. Default: 1.
            thickness (float or numpy.ndarray): Thickness of the sheet in microns. Default: 1 um.
            wavelength (float or numpy.ndarray): Wavelength of the illumination. Default: 0.6328 um.
            azimuth (float or numpy.ndarray): Rotation angle of the extraordinary axis. Default: 0.
            rotation (float or numpy.ndarray): Rotation of the element in the X-Z plane. The origin is X axis. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Mueller): Created object.
        """
        J = Jones_matrix().biaxial_material(ne=ne,
                                            no=no,
                                            thickness=thickness,
                                            wavelength=wavelength,
                                            azimuth=azimuth,
                                            rotation=rotation,
                                            length=length,
                                            shape_like=shape_like,
                                            shape=shape)
        return self.from_Jones(J)

    def general_eigenstates(self,
                            E1,
                            E2=None,
                            p1=1,
                            p2=0,
                            Tmax=None,
                            Tmin=None,
                            ret=0,
                            global_phase=0,
                            length=1,
                            shape_like=None,
                            shape=None,
                            force_limits=False,
                            R=None):
        """Creates the most general pure optical element from its eigenstates.

        Parameters:
            E1 (Jones_vector): First eigenstate.
            E2 (Jones_vector): Second eigenstate. If None, E2 is taken as the perpendicular state to E1, so the optical object is homogenous. Default: None
            p1 (float or numpy.ndarray): Field transmission of the fast axis. Default: 1.
            p2 (float or numpy.ndarray): Electric field transmission coefficient of the extinction eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            ret (float or numpy.ndarray): Retardance. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            force_limits (bool): If True, the values of ret are forced to be in the range [0, pi]. Default: False.
            R (float or None): Same as ret, overrides it if it is not None. Maintained for retrocompatibility. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Use field transmission coefficients
        if Tmax is not None:
            p1 = np.sqrt(Tmax)
        if Tmin is not None:
            p2 = np.sqrt(Tmin)
        # Main calculation
        if E2 is None:
            # Simple case: homogenous case
            az, el = E1.parameters.azimuth_ellipticity()
            self.diattenuator_retarder_azimuth_ellipticity(
                p1=p1,
                p2=p2,
                ret=ret,
                azimuth=az,
                ellipticity=el,
                shape=shape,
                shape_like=shape_like,
                length=length,
                global_phase=global_phase,
                force_limits=force_limits,
                R=R)
        else:
            # Complicated case: inhomogeneous case. It must be done from Jones
            E = Jones_vector(self.name)
            E.general_eigenstates(E1=E1,
                                  E2=E2,
                                  p1=p1,
                                  p2=p2,
                                  ret=ret,
                                  global_phase=global_phase,
                                  length=length,
                                  shape_like=shape_like,
                                  shape=shape,
                                  force_limits=force_limits,
                                  R=R)
            self.from_Jones(E)
        return self



########################################################################
# Parameters
########################################################################


class Parameters_Mueller(object):
    """Class for Mueller Matrix Parameters.

    Parameters:
        self.parent (Mueller_matrix): Parent object.
    """
    def __init__(self, parent):
        self.parent = parent

    def __repr__(self):
        """print all parameters
        TODO: print all as mueller_matrix"""
        dict = self.get_all(verbose=True, draw=True)
        return ''

    def get_all(self, verbose=False, draw=False):
        """Creates a dictionary with all the parameters of Mueller matrix.

        Parameters:
            verbose (bool): If True, print all parameters. Default: False.
            draw (bool): If True, draw all plots/images of the parameters. Default: False.
        """
        dict_params = {}
        dict_params['Mean transmission'] = self.mean_transmission(
            verbose=verbose, draw=draw)
        dict_params['Inhomogeneity'] = self.inhomogeneity(verbose=verbose,
                                                               draw=draw)
        dict_params['Diattenuation'] = self.diattenuation(verbose=verbose,
                                                               draw=draw)
        dict_params['Linear diattenuation'] = self.diattenuation_linear(
            verbose=verbose, draw=draw)
        dict_params[
            'Circular diattenuation'] = self.diattenuation_circular(
                verbose=verbose, draw=draw)
        dict_params['Polarizance'] = self.polarizance(verbose=verbose,
                                                           draw=draw)
        dict_params['Linear polarizance'] = self.polarizance_linear(
            verbose=verbose, draw=draw)
        dict_params['Circular polarizance'] = self.polarizance_circular(
            verbose=verbose, draw=draw)
        dict_params['Spheric purity'] = self.spheric_purity(
            verbose=verbose, draw=draw)
        dict_params['Retardance'] = self.retardance(verbose=verbose,
                                                         draw=draw)
        dict_params['Polarimetric purity'] = self.polarimetric_purity(
            verbose=verbose, draw=draw)
        dict_params['Depolarization degree'] = self.depolarization_index(
            verbose=verbose, draw=draw)
        dict_params[
            'Polarimetric purity indices'] = self.polarimetric_purity_indices(
                verbose=verbose, draw=draw)
        dict_params['Transmissions'] = self.transmissions(kind='all',
                                                               verbose=verbose,
                                                               draw=draw)
        dict_params['Retardance'] = self.retardance(verbose=verbose,
                                                         draw=draw)
        dict_params['Eigenstates'], dict_params[
            'Eigenvectors'] = self.eig(verbose=verbose, draw=draw)
        dict_params['Determinant'] = self.det(verbose=verbose, draw=draw)
        dict_params['Trace'] = self.trace(verbose=verbose, draw=draw)
        dict_params['Norm'] = self.norm(verbose=verbose, draw=draw)

        return dict_params

    def matrix(self, shape=None, shape_like=None):
        """Returns the numpy array of Mueller matrices.

        Parameters:
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.


        Returns:
            (numpy.array) 4x4xN numpy array.
        """
        shape, _ = select_shape(obj=self.parent,
                                shape_fun=shape,
                                shape_like=shape_like)
        if shape is not None and len(shape) > 1:
            shape = tuple([4, 4] + list(shape))
            M = np.reshape(self.parent.M, shape)
        else:
            M = self.parent.M
        return M

    def components(self,
                   out_number=True,
                   shape_like=None,
                   shape=None,
                   verbose=False,
                   draw=False):
        """Extracts the matrix components of the Mueller matrix.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            M00 (numpy.ndarray): array of the 0, 0 element of the matrix.
            M01 (numpy.ndarray): array of the 0, 1 element of the matrix.
            ...
            M32 (numpy.ndarray): array of the 3, 2 element of the matrix.
            M33 (numpy.ndarray): array of the 3, 3 element of the matrix.
        """
        # Calculate the components
        components = []
        for ind1 in range(4):
            for ind2 in range(4):
                comp = self.parent.M[ind1, ind2, :]
                if out_number and comp.size == 1:
                    comp = comp[0]
                components.append(comp)
        # Reshape if required
        components = reshape(components,
                             shape_like=shape_like,
                             shape_fun=shape,
                             obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The matrix components of {} are:'.format(
                self.parent.name)
            title = ('M00', 'M01', 'M02', 'M03', 'M10', 'M11', 'M12', 'M13',
                     'M20', 'M21', 'M22', 'M23', 'M30', 'M31', 'M32', 'M33')
            PrintParam(param=components,
                       shape=self.parent.shape,
                       title=title,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return components

    def mean_transmission(self,
                          out_number=True,
                          shape_like=None,
                          shape=None,
                          verbose=False,
                          draw=False):
        """Gives the mean transmission coefficients.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        M00 = self.parent.M[0, 0, :]
        # If the result is a number and the user asks for it, return a float
        if out_number and M00.size == 1:
            M00 = M00[0]
        # Reshape if neccessary
        M00 = reshape([M00],
                      shape_like=shape_like,
                      shape_fun=shape,
                      obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The mean transmission of {} is:'.format(
                self.parent.name)
            PrintParam(param=M00,
                       shape=self.parent.shape,
                       title='Mean transmission',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return M00

    def diattenuation_vector(self,
                             normalize=True,
                             as_Stokes=False,
                             shape_like=None,
                             shape=None,
                             verbose=False,
                             draw=False):
        """Gives the diattenuation vector. The first dimension will always have size 3.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            normalize (bool): If True, normalizes the diattenuation vector to M00. Default: True.
            as_Stokes (bool): If True, the vectors are inside a normlized Stokes object. Default: False.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or Stokes): Result.
        """
        # Extract components
        # TODO: No idea why deepcopy is required here
        comp = deepcopy(
            self.components(shape=shape,
                            shape_like=shape_like,
                            out_number=False))
        # Normalize (avoiding dividing by 0)
        if normalize:
            cond = comp[0] > tol_default
            if np.any(cond):
                for ind in range(1, 4):
                    comp[ind][cond] = comp[ind][cond] / comp[0][cond]
        # Transform in Stokes if needed
        if as_Stokes:
            S0 = np.sqrt(comp[1]**2 + comp[2]**2 + comp[3]**2)
            D = Stokes().from_components([S0, comp[1], comp[2], comp[3]])
            D.normalize()
        else:
            D = np.array([comp[1], comp[2], comp[3]])
        # Print the result if required
        if verbose or draw:
            if as_Stokes:
                D.parameters.components(verbose=verbose, draw=draw)
            else:
                heading = 'The diattenuation vector of {} is:'.format(
                    self.parent.name)
                title = ('D[0]', 'D[1]', 'D[2]')
                PrintParam(param=[D[0, :], D[1, :], D[2, :]],
                           shape=self.parent.shape,
                           title=title,
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
        return D

    def polarizance_vector(self,
                           normalize=True,
                           shape_like=None,
                           shape=None,
                           verbose=False,
                           draw=False):
        """Gives the polarizance vector. The first dimension will always have size 3.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            normalize (bool): If True, normalizes the diattenuation vector to M00. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray): Result.
        """
        # Extract components
        # TODO: No idea why deepcopy is required here
        comp = deepcopy(
            self.components(shape=shape,
                            shape_like=shape_like,
                            out_number=False))
        # Normalize (avoiding dividing by 0)
        if normalize:
            cond = comp[0] > tol_default
            if np.any(cond):
                for ind in (4, 8, 12):
                    comp[ind][cond] = comp[ind][cond] / comp[0][cond]
        P = np.array([comp[4], comp[8], comp[12]])
        # Print the result if required
        if verbose or draw:
            heading = 'The polarizance vector of {} is:'.format(
                self.parent.name)
            title = ('P[0]', 'P[1]', 'P[2]')
            PrintParam(param=[P[0, :], P[1, :], P[2, :]],
                       shape=self.parent.shape,
                       title=title,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return P

    def small_matrix(self,
                     normalize=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """Gives the small matrix m. The first two dimensions will always have size 3.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            normalize (bool): If True, normalizes the diattenuation vector to M00. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray): Result.
        """
        # Extract components
        # TODO: No idea why deepcopy is required here
        comp = deepcopy(
            self.components(shape=shape,
                            shape_like=shape_like,
                            out_number=False))
        # Normalize (avoiding dividing by 0)
        if normalize:
            cond = comp[0] > tol_default
            for ind in (5, 6, 7, 9, 10, 11, 13, 14, 15):
                comp[ind][cond] = comp[ind][cond] / comp[0][cond]
        m = np.array([[comp[5], comp[6], comp[7]],
                      [comp[9], comp[10], comp[11]],
                      [comp[13], comp[14], comp[15]]])
        # Print the result if required
        if verbose or draw:
            heading = 'The small matrix of {} is:'.format(self.parent.name)
            title = ('m[0,0]', 'm[0,1]', 'm[0,2]', 'm[1,0]', 'm[1,1]',
                     'm[1,2]', 'm[2,0]', 'm[2,1]', 'm[2,2]')
            PrintParam(param=[
                m[0, 0, :], m[0, 1, :], m[0, 2, :], m[1, 0, :], m[1, 1, :],
                m[1, 2, :], m[2, 0, :], m[2, 1, :], m[2, 2, :]
            ],
                       shape=self.parent.shape,
                       title=title,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return m

    def blocks(self,
               normalize=True,
               out_number=True,
               shape_like=None,
               shape=None,
               verbose=False,
               draw=False):
        """Method that gives the Mueller matrix block components: $M_{00}$ (mean transmission), $D$ (diattenuation vector), $P$ (polarizance vector) and $m$ (small matrix).

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            normalize (bool): If True, normalizes the diattenuation vector to M00. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            m00 (float or numpy.ndarray): Average intensity.
            D (numpy.ndarray): Diattenuation vectors 3xN.
            P (numpy.ndarray): Polarizance vector 3xN.
            m (numpy.ndarray): Small m matrix 3x3xN.
        """
        # Extract the variables from the parent object
        M00 = self.mean_transmission(shape=shape,
                                     shape_like=shape_like,
                                     out_number=out_number)
        D = self.diattenuation_vector(shape=shape, shape_like=shape_like)
        P = self.polarizance_vector(shape=shape, shape_like=shape_like)
        m = self.small_matrix(shape=shape, shape_like=shape_like)
        # Print the result if required
        if verbose or draw:
            print('The block components of {} are:'.format(self.parent.name))
            heading = '  - M_00 of {} is:'.format(self.parent.name)
            title = ('M00')
            PrintParam(param=[M00],
                       shape=self.parent.shape,
                       title=title,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
            heading = '  - Diattenuation vector of {} is:'.format(
                self.parent.name)
            title = ('D[0]', 'D[1]', 'D[2]')
            PrintParam(param=[D[0, :], D[1, :], D[2, :]],
                       shape=self.parent.shape,
                       title=title,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
            heading = '  - Polarizance vector of {} is:'.format(
                self.parent.name)
            title = ('P[0]', 'P[1]', 'P[2]')
            PrintParam(param=[P[0, :], P[1, :], P[2, :]],
                       shape=self.parent.shape,
                       title=title,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
            heading = '  - Small matrix of {} is:'.format(self.parent.name)
            title = ('m[0,0]', 'm[0,1]', 'm[0,2]', 'm[1,0]', 'm[1,1]',
                     'm[1,2]', 'm[2,0]', 'm[2,1]', 'm[2,2]')
            PrintParam(param=[
                m[0, 0, :], m[0, 1, :], m[0, 2, :], m[1, 0, :], m[1, 1, :],
                m[1, 2, :], m[2, 0, :], m[2, 1, :], m[2, 2, :]
            ],
                       shape=self.parent.shape,
                       title=title,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return M00, D, P, m

    def global_phase(self,
                     give_nan=True,
                     out_number=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """Calculates the phase of J00 (which is the reference for global phase in py_pol model).

        Parameters:
            give_nan(bool): If False, NaN values are transformed into 0. Default: True.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
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

    def inhomogeneity(self,
                      out_number=True,
                      shape_like=None,
                      shape=None,
                      verbose=False,
                      draw=False):
        """Calculates the inhomogeneity parameter. This parameter is 0 for homogeneous optical elements and 1 for totally inhomogeneous (degenerate) elements.

        Note: The equation of the reference shows at least an incorrect result in the diattenuator retarders.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 119.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        # Calculate the components
        c = self.components(shape=False, out_number=False)
        det = self.det(shape=False, out_number=False)
        tr = self.trace(shape=False, out_number=False)
        # Calculate the parameter
        cte = (tr + c[1] + c[4] + 1j * (c[11] + c[14])) / \
            np.sqrt(2 * (c[0] + c[5] + c[1] + c[4]))
        num = 4 * c[0] - np.abs(cte)**2 - np.abs(cte**2 - 4 * det**0.25)
        den = 4 * c[0] - np.abs(cte)**2 + np.abs(cte**2 - 4 * det**0.25)
        cond = np.abs(num) < tol_default
        if np.any(cond):
            num[cond] = 0  # Safety for -0 numeric error
        eta = np.sqrt(num / den)
        # Safety: 0/0 must be changed to 0
        cond = (num < tol_default) and (den < tol_default)
        if np.any(cond):
            eta[cond] = 0
        # If the result is a number and the user asks for it, return a float
        if out_number and eta.size == 1:
            eta = eta[0]
        # Reshape if neccessary
        eta = reshape([eta],
                      shape_like=shape_like,
                      shape_fun=shape,
                      obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The inhomogeneity parameter of {} is:'.format(
                self.parent.name)
            PrintParam(param=eta,
                       shape=self.parent.shape,
                       title='Inhomogeneity parameter',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return eta

    def diattenuation(self,
                      out_number=True,
                      shape_like=None,
                      shape=None,
                      verbose=False,
                      draw=False):
        """Calculates the diattenuation of the Mueller matrices.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", pp. 200, CRC Press (2016)

        Parameters:
            out_number (bool): if True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        # Calculate the diattenuation
        D = self.diattenuation_vector(shape=shape, shape_like=shape_like)
        D = np.linalg.norm(D, axis=0)
        # If the result is a number and the user asks for it, return a float
        if out_number and D.size == 1:
            D = D[0]
        # Print the result if required
        if verbose or draw:
            heading = 'The diattenuation of {} is:'.format(self.parent.name)
            PrintParam(param=D,
                       shape=self.parent.shape,
                       title='Diattenuation',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return D

    def diattenuation_linear(self,
                             out_number=True,
                             shape_like=None,
                             shape=None,
                             verbose=False,
                             draw=False):
        """Calculates the linear diattenuation of the Mueller matrices.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        # Calculate the diattenuation
        D = self.diattenuation_vector(shape=shape, shape_like=shape_like)
        D = np.linalg.norm(D[0:2, :], axis=0)
        # If the result is a number and the user asks for it, return a float
        if out_number and D.size == 1:
            D = D[0]
        # Print the result if required
        if verbose or draw:
            heading = 'The linear diattenuation of {} is:'.format(
                self.parent.name)
            PrintParam(param=D,
                       shape=self.parent.shape,
                       title='Linear diattenuation',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return D

    def diattenuation_circular(self,
                               out_number=True,
                               shape_like=None,
                               shape=None,
                               verbose=False,
                               draw=False):
        """Calculates the circular diattenuation of the Mueller matrices.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        # Calculate the diattenuation
        D = self.diattenuation_vector(shape=shape, shape_like=shape_like)
        D = np.abs(D[2, :])
        # If the result is a number and the user asks for it, return a float
        if out_number and D.size == 1:
            D = D[0]
        # Print the result if required
        if verbose or draw:
            heading = 'The circular diattenuation of {} is:'.format(
                self.parent.name)
            PrintParam(param=D,
                       shape=self.parent.shape,
                       title='Circular diattenuation',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return D

    def polarizance(self,
                    out_number=True,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """Calculates the polarizance of the Mueller matrices.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        # Calculate the polarizance
        P = self.polarizance_vector(shape=shape, shape_like=shape_like)
        P = np.linalg.norm(P, axis=0)
        # If the result is a number and the user asks for it, return a float
        if out_number and P.size == 1:
            P = P[0]
        # Reshape if neccessary
        P = reshape([P],
                    shape_like=shape_like,
                    shape_fun=shape,
                    obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The polarizance of {} is:'.format(self.parent.name)
            PrintParam(param=P,
                       shape=self.parent.shape,
                       title='Polarizance',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return P

    def polarizance_linear(self,
                           out_number=True,
                           shape_like=None,
                           shape=None,
                           verbose=False,
                           draw=False):
        """Calculates the linear polarizance of the Mueller matrices.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        # Calculate the polarizance
        P = self.polarizance_vector(shape=shape, shape_like=shape_like)
        P = np.linalg.norm(P[0:2, :], axis=0)
        # If the result is a number and the user asks for it, return a float
        if out_number and P.size == 1:
            P = P[0]
        # Print the result if required
        if verbose or draw:
            heading = 'The linear polarizance of {} is:'.format(
                self.parent.name)
            PrintParam(param=P,
                       shape=self.parent.shape,
                       title='Linear polarizance',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return P

    def polarizance_circular(self,
                             out_number=True,
                             shape_like=None,
                             shape=None,
                             verbose=False,
                             draw=False):
        """Calculates the linear polarizance of the Mueller matrices.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            out_number (bool): if True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        # Calculate the polarizance
        P = self.polarizance_vector(shape=shape, shape_like=shape_like)
        P = np.abs(P[2, :])
        # If the result is a number and the user asks for it, return a float
        if out_number and P.size == 1:
            P = P[0]
        # Print the result if required
        if verbose or draw:
            heading = 'The circular polarizance of {} is:'.format(
                self.parent.name)
            PrintParam(param=P,
                       shape=self.parent.shape,
                       title='Circular polarizance',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return P

    def degree_polarizance(self,
                           out_number=True,
                           shape_like=None,
                           shape=None,
                           verbose=False,
                           draw=False):
        """Calculates the degree of polarizance of the Mueller matrices.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        # Calculate the degree
        P = self.polarizance(out_number=out_number,
                             shape=shape,
                             shape_like=shape_like)
        D = self.diattenuation(out_number=out_number,
                               shape=shape,
                               shape_like=shape_like)
        Dp = sqrt((P**2 + D**2) / 2)
        # Print the result if required
        if verbose or draw:
            heading = 'The degree of polarizance of {} is:'.format(
                self.parent.name)
            PrintParam(param=Dp,
                       shape=self.parent.shape,
                       title='Degree of polarizance',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return Dp

    def spheric_purity(self,
                       out_number=True,
                       shape_like=None,
                       shape=None,
                       verbose=False,
                       draw=False):
        """Calculates the degree of spheric purity of the Mueller matrices.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016) pp 204.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        # Calculate the degree
        m = self.small_matrix(shape=shape, shape_like=shape_like)
        SP = np.linalg.norm(m, axis=(0, 1)) / sqrt(3)
        # If the result is a number and the user asks for it, return a float
        if out_number and SP.size == 1:
            SP = SP[0]
        # Print the result if required
        if verbose or draw:
            heading = 'The degree of spherical purity of {} is:'.format(
                self.parent.name)
            PrintParam(param=SP,
                       shape=self.parent.shape,
                       title='Spherical purity',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return SP

    # Similar to purity grades

    def retardance(self,
                   out_number=True,
                   shape_like=None,
                   shape=None,
                   verbose=False,
                   draw=False):
        """Calculates the retardance of the Mueller matrix of a pure retarder.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016) pp 129.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        # TODO: (Jesus) We have to find a way to know if we are in the 180 -> 360 degrees region. Clue: In that case, opper triangular part is negative (except for ellipticity negative and phi lower than 90 deg).

        # Calculate the retardance
        m = self.small_matrix(shape=shape, shape_like=shape_like)
        cosR = (np.trace(m) - 1) / 2
        R = np.arccos(cosR)
        # If the result is a number and the user asks for it, return a float
        if out_number and R.size == 1:
            R = R[0]
        # Print the result if required
        if verbose or draw:
            heading = 'The retardance of {} is (deg):'.format(self.parent.name)
            PrintParam(param=R / degrees,
                       shape=self.parent.shape,
                       title='Retardance (deg)',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return R

    # Polarization or despolarization
    def polarimetric_purity(self,
                            out_number=True,
                            shape_like=None,
                            shape=None,
                            verbose=False,
                            draw=False):
        """Calculates the degree of polarimetric purity of the Mueller matrices.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        # Calculate the degree
        Pp = self.degree_polarizance(out_number=out_number,
                                     shape=shape,
                                     shape_like=shape_like)
        Ps = self.spheric_purity(out_number=out_number,
                                 shape=shape,
                                 shape_like=shape_like)
        PP = sqrt(2. / 3. * Pp**2 + Ps**2)
        # Print the result if required
        if verbose or draw:
            heading = 'The degree of polarimetric purity of {} is:'.format(
                self.parent.name)
            PrintParam(param=PP,
                       shape=self.parent.shape,
                       title='Polarimetric purity',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return PP

    def depolarization_index(self,
                             out_number=True,
                             shape_like=None,
                             shape=None,
                             verbose=False,
                             draw=False):
        """Calculates the depolarization index of the Mueller matrices.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        PP = self.polarimetric_purity(out_number=out_number,
                                      shape=shape,
                                      shape_like=shape_like)
        DI = sqrt(1. - PP**2)
        # Print the result if required
        if verbose or draw:
            heading = 'The depolarization index of {} is:'.format(
                self.parent.name)
            PrintParam(param=DI,
                       shape=self.parent.shape,
                       title='Depolarization index',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return DI

    # def depolarization_factors(self):
    #     """Calculates the Euclidean distance and depolarization factor
    #
    #     References:
    #         Handbook of Optics vol 2. 22.49 (46 and 47)
    #
    #     Parameters:
    #         M (4x4 numpy.matrix): Mueller matrix
    #
    #     Returns:
    #         (float): Euclidean distance of the normalized Mueller matrix from an ideal depolarizer
    #         (float): Dep(M) depolarization of the matrix
    #     """
    #     # TODO: (Jesus) Check if Mnorm must be used instead of M
    #
    #     M = self.M
    #     quadratic_sum = (array(M)**2).sum()
    #     euclidean_distance = sqrt(quadratic_sum - M[0, 0]**2) / M[0, 0]
    #     depolarization = 1 - euclidean_distance / sqrt(3)
    #
    #     return euclidean_distance, depolarization

    # Polarimetric purity

    def polarimetric_purity_indices(self,
                                    out_number=True,
                                    shape_like=None,
                                    shape=None,
                                    verbose=False,
                                    draw=False):
        """Calculates the polarimetric purity indices of the Mueller matrices.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 208.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            P1, P2, P3 (numpy.ndarray or float): Results.
        """
        # Calculate the covariance matrix
        H = self.parent.covariance_matrix(keep=True)
        # Calculate parameters of covariance matrix
        th = np.abs(H.parameters.trace(shape=False, out_number=out_number))
        v1, v2, v3, v4 = H.parameters.eigenvalues(shape=False,
                                                  out_number=False)
        # Order the eigenvalues
        vals = np.array([np.abs(v1), np.abs(v2), np.abs(v3), np.abs(v4)])
        vals = np.sort(vals, axis=0)
        # Calculate indices
        P1 = (vals[3, :] - vals[2, :]) / th
        P2 = (vals[3, :] + vals[2, :] - 2 * vals[1, :]) / th
        P3 = (vals[3, :] + vals[2, :] + vals[1, :] - 3 * vals[0, :]) / th
        # Size 1 objects need some care
        if P1.size == 1 and P1.ndim == 1:
            P1, P2, P3 = (P1[0], P2[0], P3[0])
        # Reshape if neccessary
        P1, P2, P3 = reshape([P1, P2, P3],
                             shape_like=shape_like,
                             shape_fun=shape,
                             obj=self.parent)
        # Print the result if required
        if verbose or draw:
            # Eigenvalues
            heading = 'The polarimetric purity indices of {} are:'.format(
                self.parent.name)
            PrintParam(param=(P1, P2, P3),
                       shape=self.parent.shape,
                       title=('P1', 'P2', 'P3'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return P1, P2, P3

    def transmissions(self,
                      kind='Intensity',
                      out_number=True,
                      shape_like=None,
                      shape=None,
                      verbose=False,
                      draw=False):
        """Calculate the maximum and minimum transmitance of an optical element.

        References:
            Handbook of Optics vol 2. 22.32 (eq.38)

        Parameters:
            kind (str): There are three options, FIELD, INTENSITY or ALL.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            T_max (numpy.ndarray): Maximum intensity transmission.
            T_min (numpy.ndarray): Minimum intensity transmission.
            p1 (numpy.ndarray): Maximum field transmission.
            p2 (numpy.ndarray): Minimum field transmission.
        """
        # Take the parameters
        M00 = self.mean_transmission(shape=shape,
                                     shape_like=shape_like,
                                     out_number=out_number)
        D = self.diattenuation(shape=shape,
                               shape_like=shape_like,
                               out_number=out_number)
        # Calculate the transmissions
        T_max = M00 * (1 + D)
        T_min = M00 * (1 - D)
        if kind.upper() in ('FIELD', 'ALL'):
            p1 = np.sqrt(T_max)
            p2 = np.sqrt(T_min)
        # Print the result if required
        if verbose or draw:
            # Intensity
            if kind.upper() in ('INTENSITY', 'ALL'):
                heading = 'The intensity transmissions of {} are:'.format(
                    self.parent.name)
                PrintParam(param=(T_max, T_min),
                           shape=self.parent.shape,
                           title=('Maximum (int.)', 'Minimum (int.)'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
            # Field
            if kind.upper() in ('FIELD', 'ALL'):
                heading = 'The field transmissions of {} are:'.format(
                    self.parent.name)
                PrintParam(param=(p1, p2),
                           shape=self.parent.shape,
                           title=('Maximum (int.)', 'Minimum (int.)'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
        # Return
        ret = []
        if kind.upper() in ('INTENSITY', 'ALL'):
            ret += [T_max, T_min]
        if kind.upper() in ('FIELD', 'ALL'):
            ret += [p1, p2]
        return ret

    def retardance_vector(self,
                          kind='norm',
                          shape_like=None,
                          shape=None,
                          verbose=False,
                          draw=False):
        """Calculate the retardance of a vector.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 128-130.

        Args:
            kind (string): Identifies the type of retardance vector. There are three possibilities: NORM (also called Pauli vector), STRAIGHT or COMPLETE. Default: 'norm'.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray): 3xN array of the result.
        """
        # Extract the necessary parameters
        R = self.retardance(shape=False, out_number=False)
        m = self.small_matrix(shape=False)
        Rv = np.zeros((3, self.parent.size))
        # Transform depending on the type of vector
        if kind in ('complete', 'COMPLETE', 'Complete'):
            cte = R
        elif kind in ('straight', 'STRAIGHT', 'Straight'):
            cte = R / np.pi
        else:
            cte = np.ones_like(R)
        # General case: retardance different than 0 or 180º
        cond_G = (R >= tol_default) * (R <= 180 * degrees - tol_default)
        if np.any(cond_G):
            sR = 2 * np.sin(R[cond_G]) / cte[cond_G]
            Rv[0, cond_G] = (m[1, 2, cond_G] - m[2, 1, cond_G]) / sR
            Rv[1, cond_G] = (m[2, 0, cond_G] - m[0, 2, cond_G]) / sR
            Rv[2, cond_G] = (m[0, 1, cond_G] - m[1, 0, cond_G]) / sR
        # Particular case: R = 0
        cond_Z = R < tol_default
        if np.any(cond_Z):
            # If R = 0, we don't have a retarder after all, so any vector is valid
            Rv[0, cond_Z] = cte[cond_Z]
        # Particular case: R = 180º
        cond_P = R > 180 * degrees - tol_default
        if np.any(cond_P):
            # This case is more tricky, we have to calculate Rv from its eigenstates (supposing that it is a pure retarder)
            S1, _ = self.eigenstates(shape=False)
            alpha, delay = S1.parameters.charac_angles(shape=False,
                                                       out_number=False)
            Rv[0, cond_P] = cte[cond_P] * np.cos(2 * alpha[cond_P])
            Rv[1, cond_P] = cte[cond_P] * \
                np.sin(2 * alpha[cond_P]) * np.cos(2 * delay[cond_P])
            Rv[2, cond_P] = cte[cond_P] * \
                np.sin(2 * alpha[cond_P]) * np.sin(2 * delay[cond_P])
        # Print the result if required
        if verbose or draw:
            # Extract the components
            r0, r1, r2 = (Rv[0, :], Rv[1, :], Rv[2, :])
            r0, r1, r2 = reshape([r0, r1, r2],
                                 shape_like=shape_like,
                                 shape_fun=shape,
                                 obj=self.parent)
            # Print
            heading = 'The retardance vector ({}) components components of {} are:'.format(
                kind, self.parent.name)
            PrintParam(param=[r0, r1, r2],
                       shape=self.parent.shape,
                       title=('r0', 'r1', 'r2'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Reshape the vector if neccessary
        shape = take_shape([self.parent, shape_like, shape])
        if shape is not None and len(shape) > 1:
            shape = np.insert(shape, 0, 3)
            Rv = np.reshape(Rv, shape)
        return Rv

    def eig(self,
            values_as_matrix=False,
            vectors_as_matrix=False,
            out_number=True,
            shape_like=None,
            shape=None,
            verbose=False,
            draw=False):
        """
        Calculates the eigenvalues and eigenvectors of the Mueller matrices.

        Parameters:
            values_as_matrix (bool): If True, the eigenvalues output is a numpy.ndarray instead of a list of arrays. Default: False.
            vectors_as_matrix (bool): If True, the eigenvectors output is a numpy.ndarray instead of a list of arrays. Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            eigenvalues (list or numpy.ndarray): List with the four eigenvalues (if values_as_matrix is False) or 4xN array (if values_as_matrix is True).
            eigenvectors (list or numpy.ndarray): List with the four eigenvectors (if vectors_as_matrix is False) or 4x4xN array (if vectors_as_matrix is True).
        """
        # Calculate the eig
        M = np.moveaxis(self.parent.M, -1, 0)
        val, vect = np.linalg.eig(M)
        # Order the eigenvalues in the py_pol way
        if ~values_as_matrix or verbose or draw:
            v1, v2, v3, v4 = (val[:, 0], val[:, 1], val[:, 2], val[:, 3])
            # Size 1 objects need some care
            if v1.size == 1 and v1.ndim > 1:
                v1, v2, v3, v4 = (v1[0], v2[0], v3[0], v4[0])
            if out_number and v1.size == 1:
                v1, v2, v3, v4 = (v1[0], v2[0], v3[0], v4[0])
            # Reshape if neccessary
            eigenvalues = reshape([v1, v2, v3, v4],
                                  shape_like=shape_like,
                                  shape_fun=shape,
                                  obj=self.parent)
        # Reshape eigenvalues if output as matrix
        if values_as_matrix:
            val = np.moveaxis(val, -1, 0)
            new_shape, _ = select_shape(obj=self.parent,
                                        shape_fun=shape,
                                        shape_like=shape_like)
            if new_shape is not None and len(new_shape) > 1:
                new_shape = [4] + list(new_shape)
                val = np.reshape(val, new_shape)

        # Reshape eigenvectors in pypol way
        if ~vectors_as_matrix or verbose or draw:
            e1 = np.array(
                [vect[:, 0, 0], vect[:, 1, 0], vect[:, 2, 0], vect[:, 3, 0]])
            e2 = np.array(
                [vect[:, 0, 1], vect[:, 1, 1], vect[:, 2, 1], vect[:, 3, 1]])
            e3 = np.array(
                [vect[:, 0, 2], vect[:, 1, 2], vect[:, 2, 2], vect[:, 3, 2]])
            e4 = np.array(
                [vect[:, 0, 3], vect[:, 1, 3], vect[:, 2, 3], vect[:, 3, 3]])
            eigenvectors = []
            for elem in (e1, e2, e3, e4):
                for ind in range(4):
                    eigenvectors.append(elem[ind, :])
            eigenvectors = reshape(eigenvectors,
                                   shape_like=shape_like,
                                   shape_fun=shape,
                                   obj=self.parent)
            new_shape = [4] + list(eigenvectors[0].shape)
            if len(new_shape) > 2:
                e1 = np.reshape(e1, new_shape)
                e2 = np.reshape(e2, new_shape)
                e3 = np.reshape(e3, new_shape)
                e4 = np.reshape(e4, new_shape)
        # Reshape eigenvectors if output as matrix
        if vectors_as_matrix:
            vect = np.moveaxis(vect, -2, 0)
            vect = np.moveaxis(vect, -1, 1)
            new_shape, _ = select_shape(obj=self.parent,
                                        shape_fun=shape,
                                        shape_like=shape_like)
            if new_shape is not None and len(new_shape) > 1:
                new_shape = [4, 4] + list(new_shape)
                vect = np.reshape(vect, new_shape)

        # Print the result if required
        if verbose or draw:
            # Eigenvalues
            heading = 'The eigenvalues of {} are:'.format(self.parent.name)
            PrintParam(param=eigenvalues,
                       shape=self.parent.shape,
                       title=('v1', 'v2', 'v3', 'v4'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
            # Eigenvectors
            heading = 'The eigenvectors of {} are:'.format(self.parent.name)
            PrintParam(param=eigenvectors,
                       shape=self.parent.shape,
                       title=('e1 I', 'e1 Q', 'e1 U', 'e1 V', 'e2 I', 'e2 Q',
                              'e2 U', 'e2 V', 'e3 I', 'e3 Q', 'e3 U', 'e3 V',
                              'e4 I', 'e4 Q', 'e4 U', 'e4 V'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        # Return
        if values_as_matrix:
            eigenvalues = val
        if vectors_as_matrix:
            eigenvectors = vect
        else:
            eigenvectors = (e1, e2, e3, e4)
        return eigenvalues, eigenvectors

    def eigenvectors(self,
                     vectors_as_matrix=False,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """
        Calculates the eigenvectors of the Mueller matrices.

        Parameters:
            vectors_as_matrix (bool): If True, the eigenvectors output is a numpy.ndarray instead of a list of arrays. Default: False.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray): 4x4xN eigenvector matrix (if vectors_as_matrix is True).
            e1, e2, e3, e4 (numpy.ndarray): 4xN eigenvector matrices (if vectors_as_matrix is False).
        """
        # Calculate
        M = np.moveaxis(self.parent.M, -1, 0)
        _, vect = np.linalg.eig(M)

        # Reshape eigenvectors in pypol way
        if ~vectors_as_matrix or verbose or draw:
            e1 = np.array(
                [vect[:, 0, 0], vect[:, 1, 0], vect[:, 2, 0], vect[:, 3, 0]])
            e2 = np.array(
                [vect[:, 0, 1], vect[:, 1, 1], vect[:, 2, 1], vect[:, 3, 1]])
            e3 = np.array(
                [vect[:, 0, 2], vect[:, 1, 2], vect[:, 2, 2], vect[:, 3, 2]])
            e4 = np.array(
                [vect[:, 0, 3], vect[:, 1, 3], vect[:, 2, 3], vect[:, 3, 3]])
            eigenvectors = []
            for elem in (e1, e2, e3, e4):
                for ind in range(4):
                    eigenvectors.append(elem[ind, :])
            eigenvectors = reshape(eigenvectors,
                                   shape_like=shape_like,
                                   shape_fun=shape,
                                   obj=self.parent)
            # new_shape = [4] + list(eigenvectors[0].shape)
            # if len(new_shape) > 2:
            #     e1 = np.reshape(e1, new_shape)
            #     e2 = np.reshape(e2, new_shape)
            #     e3 = np.reshape(e3, new_shape)
            #     e4 = np.reshape(e4, new_shape)
        # Reshape eigenvectors if output as matrix
        if vectors_as_matrix:
            vect = np.moveaxis(vect, -2, 0)
            vect = np.moveaxis(vect, -1, 1)
            new_shape, _ = select_shape(obj=self.parent,
                                        shape_fun=shape,
                                        shape_like=shape_like)
            if new_shape is not None and len(new_shape) > 1:
                new_shape = [4, 4] + list(new_shape)
                vect = np.reshape(vect, new_shape)

        # Print the result if required
        if verbose or draw:
            # Eigenvectors
            # print_vect = []
            heading = 'The eigenvectors of {} are:'.format(self.parent.name)
            # for elem in eigenvectors:
            #     for ind in range(4):
            #         print_vect.append(elem[ind, :])
            # print_vect = reshape(
            #     print_vect,
            #     shape_like=shape_like,
            #     shape_fun=shape,
            #     obj=self.parent)
            PrintParam(param=eigenvectors,
                       shape=self.parent.shape,
                       title=('e1 I', 'e1 Q', 'e1 U', 'e1 V', 'e2 I', 'e2 Q',
                              'e2 U', 'e2 V', 'e3 I', 'e3 Q', 'e3 U', 'e3 V',
                              'e4 I', 'e4 Q', 'e4 U', 'e4 V'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        # Return
        if vectors_as_matrix:
            eigenvectors = vect
        return eigenvectors

    def eigenstates(self,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """
        Calculates the eigenstates of the optical object. It must be done in Jones formalism, so it is only valid for pure Mueller matrices.

        Parameters:
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            S1, S2, S3, S4 (Stokes): eigenstates.
        """
        # Calculate the eigenstates in Jones
        J = Jones_matrix(self.parent.name)
        J.from_Mueller(self.parent, shape_like=self.parent)
        E1, E2 = J.parameters.eigenstates(shape=shape, shape_like=shape_like)
        # Transfrom to Stokes
        S1, S2 = create_Stokes(('1st eigenstate', '2nd eigenstate'))
        S1.from_Jones(E1)
        S2.from_Jones(E2)
        # Print the result if required
        if verbose or draw:
            # Extract the info
            heading = 'The eigenstates of {} are:'.format(self.parent.name)
            c1 = S1.parameters.components()
            c2 = S2.parameters.components()
            components = list(c1) + list(c2)
            # Print
            PrintParam(param=components,
                       shape=self.parent.shape,
                       title=('S1 I', 'S1 Q', 'S1 U', 'S1 V', 'S2 I', 'S2 Q',
                              'S2 U', 'S2 V'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return S1, S2

    def eigenvalues(self,
                    values_as_matrix=False,
                    out_number=True,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """
        Calculates the eigenvalues and eigenstates of the Jones object.

        Parameters:
            values_as_matrix (bool): If True, the eigenvalues output is a numpy.ndarray instead of a list of arrays. Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            v (numpy.ndarray): 4xN eigenvalues matrix (if values_as_matrix is True).
            v1, v2, v3, v4 (numpy.ndarray or float): Individual eigenvalues (if values_as_matrix is False).
        """
        # Calculate the eigenvalues
        M = np.moveaxis(self.parent.M, -1, 0)
        val = np.linalg.eigvals(M)
        # Order the eigenvalues in the py_pol way
        if ~values_as_matrix or verbose or draw:
            v1, v2, v3, v4 = (val[:, 0], val[:, 1], val[:, 2], val[:, 3])
            # Size 1 objects need some care
            if v1.size == 1 and v1.ndim > 1:
                v1, v2, v3, v4 = (v1[0], v2[0], v3[0], v4[0])
            if out_number and v1.size == 1:
                v1, v2, v3, v4 = (v1[0], v2[0], v3[0], v4[0])
            # Reshape if neccessary
            eigenvalues = reshape([v1, v2, v3, v4],
                                  shape_like=shape_like,
                                  shape_fun=shape,
                                  obj=self.parent)
        # Reshape eigenvalues if output as matrix
        if values_as_matrix:
            val = np.moveaxis(val, -1, 0)
            new_shape, _ = select_shape(obj=self.parent,
                                        shape_fun=shape,
                                        shape_like=shape_like)
            if new_shape is not None and len(new_shape) > 1:
                new_shape = [4] + list(new_shape)
                val = np.reshape(val, new_shape)
        else:
            v1, v2, v3, v4 = reshape([v1, v2, v3, v4],
                                     shape_like=shape_like,
                                     shape_fun=shape,
                                     obj=self.parent)
        # Print the result if required
        if verbose or draw:
            # Eigenvalues
            heading = 'The eigenvalues of {} are:'.format(self.parent.name)
            PrintParam(param=(v1, v2, v3, v4),
                       shape=self.parent.shape,
                       title=('v1', 'v2', 'v3', 'v4'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        # Return
        if values_as_matrix:
            eigenvalues = val
        else:
            eigenvalues = (v1, v2, v3, v4)
        return eigenvalues

    def det(self,
            out_number=True,
            shape_like=None,
            shape=None,
            verbose=False,
            draw=False):
        """
        Calculates the determinants of the Mueller matrices.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray, float or complex): Result.
        """
        # Calculate the eigenstates
        M = np.moveaxis(self.parent.M, -1, 0)
        det = np.linalg.det(M)
        # If the result is a number and the user asks for it, return a float
        if out_number and det.size == 1:
            det = det[0]
        # Reshape if neccessary
        det = reshape([det],
                      shape_like=shape_like,
                      shape_fun=shape,
                      obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The determinant of {} is:'.format(self.parent.name)
            PrintParam(param=det,
                       shape=self.parent.shape,
                       title='Determinant',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return det

    def trace(self,
              out_number=True,
              shape_like=None,
              shape=None,
              verbose=False,
              draw=False):
        """
        Calculates the trace of the Mueller matrices.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or float): Result.
        """
        # Calculate the eigenstates
        trace = np.trace(self.parent.M)
        # If the result is a number and the user asks for it, return a float
        if out_number and trace.size == 1:
            trace = trace[0]
        # Reshape if neccessary
        trace = reshape([trace],
                        shape_like=shape_like,
                        shape_fun=shape,
                        obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The trace of {} is:'.format(self.parent.name)
            PrintParam(param=trace,
                       shape=self.parent.shape,
                       title='Trace',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return trace

    def norm(self,
             out_number=True,
             shape_like=None,
             shape=None,
             verbose=False,
             draw=False):
        """
        Calculates the Frobenius norm of the Mueller matrices.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray, float or complex): Result.
        """
        # Calculate the eigenstates
        norm = np.linalg.norm(self.parent.M, axis=(0, 1))
        # If the result is a number and the user asks for it, return a float
        if out_number and norm.size == 1:
            norm = norm[0]
        # Reshape if neccessary
        norm = reshape([norm],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The norm of {} is:'.format(self.parent.name)
            PrintParam(param=norm,
                       shape=self.parent.shape,
                       title='Norm',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return norm


class Analysis_Mueller(object):
    """Class for Analysis of Mueller Analysis

    Parameters:
        mueller_matrix (Mueller_matrix): Mueller Matrix

    Attributes:
        self.marent (Mueller): parent object.
    """
    def __init__(self, parent):
        self.parent = parent

    def depolarizer(self,
                    transmissions='all',
                    angles="all",
                    depolarization='all',
                    out_number=True,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """Calculates some of the parameters from the Mueller matrix of a diattenuator.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            transmissions (string): Determines the type of transmission output, FIELD, INTENSITY or ALL. Default: All.
            angles (string): Determines the type of angles output, CHARAC (characteristic angles), AZIMUTH (azimuth and ellipticity) or ALL. Default: All.
            depolarization (string): Determines the type of depolarization information: INDEX, FACTORS or ALL. Default: All.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            trans_D, transP (list). Transmissions calculated from the diattenuation and polarizance vectors (in that order). Depending on the input parameter transmissions, it can contain the field transmissions, the intensity transmissions or both.
            angles_D, angles_P (list). Angles of the transmission eigenstate calculated from the diattenuation and polarizance vectors respectively. Depending on the input parameter angles, it can contain the characteristic angles, the azimuth and ellipticity, or all of them.
            depol (numpy.ndarray or float): Depolarization index.
            S (list): List with the three principal states.
        """
        # Extract the information
        D = self.parent.parameters.diattenuation(shape=shape,
                                                 shape_like=shape_like,
                                                 out_number=out_number)
        Dv = self.parent.parameters.diattenuation_vector(shape=shape,
                                                         shape_like=shape_like)
        P = self.parent.parameters.polarizance(shape=shape,
                                               shape_like=shape_like,
                                               out_number=out_number)
        Pv = self.parent.parameters.polarizance_vector(shape=shape,
                                                       shape_like=shape_like)
        M00 = self.parent.parameters.mean_transmission(shape=shape,
                                                       shape_like=shape_like,
                                                       out_number=out_number)
        m = self.parent.parameters.small_matrix(shape=False)
        depol = self.parent.parameters.depolarization_index(
            shape=shape, shape_like=shape_like, out_number=out_number)

        # Calculate transmissions
        Tmax = M00 * (1 + D)
        Tmin = M00 * (1 - D)
        p1, p2 = (np.sqrt(Tmax), np.sqrt(Tmin))
        trans_D, title_trans = ([], [])
        if transmissions in ('ALL', 'All', 'all', 'INTENSITY', 'Intensity',
                             'intensity'):
            trans_D += [Tmax, Tmin]
            title_trans += ['Max. transmission', 'Min. transmission']
        if transmissions in ('ALL', 'All', 'all', 'FIELD', 'Field', 'field'):
            trans_D += [p1, p2]
            title_trans += ['p1', 'p2']

        Tmax = M00 * (1 + P)
        Tmin = M00 * (1 - P)
        p1, p2 = (np.sqrt(Tmax), np.sqrt(Tmin))
        trans_P = []
        if transmissions in ('ALL', 'All', 'all', 'INTENSITY', 'Intensity',
                             'intensity'):
            trans_P += [Tmax, Tmin]
        if transmissions in ('ALL', 'All', 'all', 'FIELD', 'Field', 'field'):
            trans_P += [p1, p2]

        # Calculate angles
        azimuth, ellipticity = extract_azimuth_elipt(Dv, use_nan=False)
        alpha, delay = azimuth_elipt_2_charac_angles(azimuth, ellipticity)
        ang_D, title_ang = ([], [])
        if angles in ('ALL', 'All', 'all', 'CHARAC', 'Charac', 'charac'):
            ang_D += [alpha, delay]
            title_ang += ['Alpha', 'Delay']
        if angles in ('ALL', 'All', 'all', 'AZIMUTH', 'Azimuth', 'azimuth'):
            ang_D += [azimuth, ellipticity]
            title_ang += ['Azimuth', 'Ellipticity angle']

        azimuth, ellipticity = extract_azimuth_elipt(Pv, use_nan=False)
        alpha, delay = azimuth_elipt_2_charac_angles(azimuth, ellipticity)
        ang_P = []
        if angles in ('ALL', 'All', 'all', 'CHARAC', 'Charac', 'charac'):
            ang_P += [alpha, delay]
        if angles in ('ALL', 'All', 'all', 'AZIMUTH', 'Azimuth', 'azimuth'):
            ang_P += [azimuth, ellipticity]

        # Calculate depolarization factors and eigenstates
        m = np.moveaxis(m, -1, 0)
        val, vect = np.linalg.eig(m)
        # val, vect = order_eig(val, vect, 'reverse')
        d1, d2, d3 = (val[:, 0], val[:, 1], val[:, 2])
        S1, S2, S3 = create_Stokes(
            ('First principal state', 'Second principal state',
             'Third principal state'))
        S1.from_components([1, vect[:, 0, 0], vect[:, 1, 0], vect[:, 2, 0]])
        S2.from_components([1, vect[:, 0, 1], vect[:, 1, 1], vect[:, 2, 1]])
        S3.from_components([1, vect[:, 0, 2], vect[:, 1, 2], vect[:, 2, 2]])
        # Reshaoe them
        d1, d2, d3 = reshape([d1, d2, d3],
                             shape_like=shape_like,
                             shape_fun=shape,
                             obj=self.parent)
        S1.shape, _ = select_shape(obj=self.parent,
                                         shape_fun=shape,
                                         shape_like=shape_like)
        S2.shape, _ = select_shape(obj=self.parent,
                                         shape_fun=shape,
                                         shape_like=shape_like)
        S3.shape, _ = select_shape(obj=self.parent,
                                         shape_fun=shape,
                                         shape_like=shape_like)
        # Save them
        principal_states = []
        depolar = []
        if depolarization.upper() in ('ALL', 'INDEX'):
            depolar += [depol]
        if depolarization.upper() in ('ALL', 'FACTORS'):
            depolar = [d1, d2, d3]
            principal_states += [S1, S2, S3]

        # Print the result if required
        if verbose or draw:
            # Transform angles to degrees for representation
            angles_rep_D = []
            for a in ang_D:
                angles_rep_D.append(a / degrees)
            angles_rep_P = []
            for a in ang_P:
                angles_rep_P.append(a / degrees)

            print('\nAnalysis of {} as depolarizer:'.format(self.parent.name))

            # Depolarization index
            if depolarization.upper() in ('ALL', 'INDEX'):
                heading = '- Depolarization index of {} is:'.format(
                    self.parent.name)
                PrintParam(param=[depol],
                           shape=self.parent.shape,
                           title=('Depolarization index'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
            # Depolarization factors and eigenstates
            if depolarization.upper() in ('ALL', 'FACTORS'):
                heading = '- First depolarization factor of {} is:'.format(
                    self.parent.name)
                PrintParam(param=[d1],
                           shape=self.parent.shape,
                           title=('1st depolarization factor'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
                if angles in ('ALL', 'All', 'all', 'CHARAC', 'Charac',
                              'charac'):
                    _ = S1.parameters.charac_angles(verbose=verbose, draw=draw)
                if angles in ('ALL', 'All', 'all', 'AZIMUTH', 'Azimuth',
                              'azimuth'):
                    _ = S1.parameters.azimuth_ellipticity(verbose=verbose,
                                                          draw=draw)

                heading = '- Second depolarization factor of {} is:'.format(
                    self.parent.name)
                PrintParam(param=[d2],
                           shape=self.parent.shape,
                           title=('2nd depolarization factor'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
                if angles in ('ALL', 'All', 'all', 'CHARAC', 'Charac',
                              'charac'):
                    _ = S2.parameters.charac_angles(verbose=verbose, draw=draw)
                if angles in ('ALL', 'All', 'all', 'AZIMUTH', 'Azimuth',
                              'azimuth'):
                    _ = S2.parameters.azimuth_ellipticity(verbose=verbose,
                                                          draw=draw)

                heading = '- Third depolarization factor of {} is:'.format(
                    self.parent.name)
                PrintParam(param=[d3],
                           shape=self.parent.shape,
                           title=('3rd depolarization factor'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
                if angles in ('ALL', 'All', 'all', 'CHARAC', 'Charac',
                              'charac'):
                    _ = S3.parameters.charac_angles(verbose=verbose, draw=draw)
                if angles in ('ALL', 'All', 'all', 'AZIMUTH', 'Azimuth',
                              'azimuth'):
                    _ = S3.parameters.azimuth_ellipticity(verbose=verbose,
                                                          draw=draw)

            # Depolarizers usually have only diattenuation or polarizance.
            if np.any(D > tol_default):
                heading = '- Transmissions of {} from diattenuation are:'.format(
                    self.parent.name)
                PrintParam(param=trans_D,
                           shape=self.parent.shape,
                           title=title_trans,
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
                heading = '- Angles of {} from polarizance are:'.format(
                    self.parent.name)
                PrintParam(param=angles_rep_P,
                           shape=self.parent.shape,
                           title=title_ang,
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
            else:
                print('- {} has no diattenuation.'.format(self.parent.name))

            if np.any(P > tol_default):
                heading = '- Transmissions of {} from polarizance are:'.format(
                    self.parent.name)
                PrintParam(param=trans_P,
                           shape=self.parent.shape,
                           title=title_trans,
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
                heading = '- Angles of {} from diattenuation are:'.format(
                    self.parent.name)
                PrintParam(param=angles_rep_D,
                           shape=self.parent.shape,
                           title=title_ang,
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
            else:
                print('- {} has no polarizance.'.format(self.parent.name))

        # Output
        return trans_D, trans_P, ang_D, ang_P, depolar, principal_states

    def diattenuator(self,
                     transmissions='all',
                     angles="all",
                     out_number=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """Calculates all the parameters from the Mueller Matrix of a pure homogeneous diattenuator (using the diattenuance vector). If the object is not a pure homogenous diattenuator, some parameters may be wrong.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            transmissions (string): Determines the type of transmission output, FIELD, INTENSITY or ALL. Default: All.
            angles (string): Determines the type of angles output, CHARAC (characteristic angles), AZIMUTH (azimuth and ellipticity) or ALL. Default: All.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            transmissions (list). Transmissions of the diattenuator. Depending on the input parameter transmissions, it can contain the field transmissions, the intensity transmissions or both.
            angles (list). Angles of the transmission eigenstate. Depending on the input parameter angles, it can contain the characteristic angles, the azimuth and ellipticity, or all of them.
        """
        # Extract the diattenuation and diattenuation vector
        D = self.parent.parameters.diattenuation(shape=shape,
                                                 shape_like=shape_like,
                                                 out_number=out_number)
        Dv = self.parent.parameters.diattenuation_vector(shape=shape,
                                                         shape_like=shape_like)
        M00 = self.parent.parameters.mean_transmission(shape=shape,
                                                       shape_like=shape_like,
                                                       out_number=out_number)

        # Calculate transmissions
        Tmax = M00 * (1 + D)
        Tmin = M00 * (1 - D)
        p1, p2 = (np.sqrt(Tmax), np.sqrt(Tmin))
        trans, title_trans = ([], [])
        if transmissions in ('ALL', 'All', 'all', 'INTENSITY', 'Intensity',
                             'intensity'):
            trans += [Tmax, Tmin]
            title_trans += ['Max. transmission', 'Min. transmission']
        if transmissions in ('ALL', 'All', 'all', 'FIELD', 'Field', 'field'):
            trans += [p1, p2]
            title_trans += ['p1', 'p2']

        # Calculate angles
        azimuth, ellipticity = extract_azimuth_elipt(Dv, use_nan=False)
        alpha, delay = azimuth_elipt_2_charac_angles(azimuth, ellipticity)
        ang, title_ang = ([], [])
        if angles in ('ALL', 'All', 'all', 'CHARAC', 'Charac', 'charac'):
            ang += [alpha, delay]
            title_ang += ['Alpha', 'Delay']
        if angles in ('ALL', 'All', 'all', 'AZIMUTH', 'Azimuth', 'azimuth'):
            ang += [azimuth, ellipticity]
            title_ang += ['Azimuth', 'Ellipticity angle']

        # Print the result if required
        if verbose or draw:
            # Transform angles to degrees for representation
            angles_rep = []
            for a in ang:
                angles_rep.append(a / degrees)

            print('\nAnalysis of {} as diattenuator:\n'.format(
                self.parent.name))
            heading = '- Transmissions of {} are:'.format(self.parent.name)
            PrintParam(param=trans,
                       shape=self.parent.shape,
                       title=title_trans,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
            heading = '- Angles of {} are:'.format(self.parent.name)

            PrintParam(param=angles_rep,
                       shape=self.parent.shape,
                       title=title_ang,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        # Output
        return trans, ang

    def polarizer(self,
                  transmissions='all',
                  angles="all",
                  out_number=True,
                  shape_like=None,
                  shape=None,
                  verbose=False,
                  draw=False):
        """Calculates all the parameters from the Mueller Matrix of a pure homogeneous diattenuator (using the polarizance vector). If the object is not a pure homogenous diattenuator, some parameters may be wrong.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            transmissions (string): Determines the type of transmission output, FIELD, INTENSITY or ALL. Default: All.
            angles (string): Determines the type of angles output, CHARAC (characteristic angles), AZIMUTH (azimuth and ellipticity) or ALL. Default: All.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            transmissions (list). Transmissions of the diattenuator. Depending on the input parameter transmissions, it can contain the field transmissions, the intensity transmissions or both.
            angles (list). Angles of the transmission eigenstate. Depending on the input parameter angles, it can contain the characteristic angles, the azimuth and ellipticity, or all of them.
        """
        # Extract the diattenuation and diattenuation vector
        P = self.parent.parameters.polarizance(shape=shape,
                                               shape_like=shape_like,
                                               out_number=out_number)
        Pv = self.parent.parameters.polarizance_vector(shape=shape,
                                                       shape_like=shape_like)
        M00 = self.parent.parameters.mean_transmission(shape=shape,
                                                       shape_like=shape_like,
                                                       out_number=out_number)

        # Calculate transmissions
        Tmax = M00 * (1 + P)
        Tmin = M00 * (1 - P)
        p1, p2 = (np.sqrt(Tmax), np.sqrt(Tmin))
        trans, title_trans = ([], [])
        if transmissions in ('ALL', 'All', 'all', 'INTENSITY', 'Intensity',
                             'intensity'):
            trans += [Tmax, Tmin]
            title_trans += ['Max. transmission', 'Min. transmission']
        if transmissions in ('ALL', 'All', 'all', 'FIELD', 'Field', 'field'):
            trans += [p1, p2]
            title_trans += ['p1', 'p2']

        # Calculate angles
        azimuth, ellipticity = extract_azimuth_elipt(Pv)
        alpha, delay = azimuth_elipt_2_charac_angles(azimuth, ellipticity)
        ang, title_ang = ([], [])
        if angles in ('ALL', 'All', 'all', 'CHARAC', 'Charac', 'charac'):
            ang += [alpha, delay]
            title_ang += ['Alpha', 'Delay']
        if angles in ('ALL', 'All', 'all', 'AZIMUTH', 'Azimuth', 'azimuth'):
            ang += [azimuth, ellipticity]
            title_ang += ['Azimuth', 'Ellipticity angle']

        # Print the result if required
        if verbose or draw:
            # Transform angles to degrees for representation
            angles_rep = []
            for a in ang:
                angles_rep.append(a / degrees)

            print('\nAnalysis of {} as polarizer:\n'.format(self.parent.name))
            heading = '- Transmissions of {} are:'.format(self.parent.name)
            PrintParam(param=trans,
                       shape=self.parent.shape,
                       title=title_trans,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
            heading = '- Angles of {} are:'.format(self.parent.name)

            PrintParam(param=angles_rep,
                       shape=self.parent.shape,
                       title=title_ang,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        # Output
        return trans, ang

    def retarder(self,
                 angles="all",
                 out_number=True,
                 shape_like=None,
                 shape=None,
                 verbose=False,
                 draw=False):
        """Calculates all the parameters from the Mueller Matrix of a pure homogeneous retarder. If the object is not a pure homogenous retarder, some parameters may be wrong.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016)

        Parameters:
            angles (string): Determines the type of angles output, CHARAC (characteristic angles), AZIMUTH (azimuth and ellipticity) or ALL. Default: All.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            R (np.ndarray). Retardance.
            angles (list). Angles of the transmission eigenstate. Depending on the input parameter angles, it can contain the characteristic angles, the azimuth and ellipticity, or all of them.
        """
        # Extract the diattenuation and diattenuation vector
        R = self.parent.parameters.retardance(shape=shape,
                                              shape_like=shape_like,
                                              out_number=out_number)
        Rv = self.parent.parameters.retardance_vector(shape=shape,
                                                      shape_like=shape_like)

        # Calculate angles
        azimuth, ellipticity = extract_azimuth_elipt(Rv)
        alpha, delay = azimuth_elipt_2_charac_angles(azimuth, ellipticity)
        ang, title_ang = ([], [])
        if angles in ('ALL', 'All', 'all', 'CHARAC', 'Charac', 'charac'):
            ang += [alpha, delay]
            title_ang += ['Alpha', 'Delay']
        if angles in ('ALL', 'All', 'all', 'AZIMUTH', 'Azimuth', 'azimuth'):
            ang += [azimuth, ellipticity]
            title_ang += ['Azimuth', 'Ellipticity angle']

        # Print the result if required
        if verbose or draw:
            # Transform angles to degrees for representation
            angles_rep = []
            for a in ang:
                angles_rep.append(a / degrees)

            print('\nAnalysis of {} as retarder:\n'.format(self.parent.name))
            heading = '- Retardance of {} is:'.format(self.parent.name)
            PrintParam(param=(R / degrees),
                       shape=self.parent.shape,
                       title=('Retardance'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

            heading = '- Angles of {} are:'.format(self.parent.name)
            PrintParam(param=angles_rep,
                       shape=self.parent.shape,
                       title=title_ang,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        # Output
        return R, ang

    # # Matrix filtering
    def filter_purify_number(self, Neig=3, keep=False):
        """Method that filters experimental errors by making zero a certain number of eigenvalues of the covariance matrix.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016) pp 226.

        Parameters:
            Neig (int): Number of eigenvalues (1-3) of the covariant matrix to be made 0. Default: 3.
            keep (bool): If True, the original object won't be altered. Default: False.

        Returns:
            (Mueller): Result.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.parent.copy()
        else:
            new_obj = self.parent
        old_shape = new_obj.shape

        # Calculate covariance matrix eigenvalues
        new_obj.covariance_matrix()
        val, vect = new_obj.parameters.eig(values_as_matrix=True,
                                           vectors_as_matrix=True,
                                           shape=False)
        # Find the order of eigenvalues
        order = np.argsort(val, axis=0)
        # Make 0 the desired eigenvalues
        val[order < Neig] = 0
        # Recompose the matrix
        H, diag, Ht = create_Mueller(N=3)
        diag.from_components([val[0, :]] + 4 * [0] + [val[1, :]] + 4 * [0] +
                             [val[2, :]] + 4 * [0] + [val[3, :]])
        H.from_matrix(vect)
        Ht.from_matrix(np.conj(np.transpose(vect, axes=(1, 0, 2))))
        result = H * diag * Ht
        new_obj.from_covariance(result, shape=old_shape)

        return new_obj

    # # Matrix filtering

    def filter_purify_threshold(self, threshold=0.01, keep=False):
        """Method that filters experimental errors by making zero a certain number of eigenvalues of the covariance matrix.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016) pp 226.

        Parameters:
            threshold (float): If eigenvalues are lower than threshold, they will be make 0. Default: 0.01.
            keep (bool): If True, the original object won't be altered. Default: False.

        Returns:
            (Mueller): Result.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.parent.copy()
        else:
            new_obj = self.parent
        old_shape = new_obj.shape

        # Calculate covariance matrix eigenvalues
        new_obj.covariance_matrix()
        val, vect = new_obj.parameters.eig(values_as_matrix=True,
                                           vectors_as_matrix=True,
                                           shape=False)
        # Make 0 the desired eigenvalues
        val[val < threshold] = 0
        # Recompose the matrix
        H, diag, Ht = create_Mueller(N=3)
        diag.from_components([val[0, :]] + 4 * [0] + [val[1, :]] + 4 * [0] +
                             [val[2, :]] + 4 * [0] + [val[3, :]])
        H.from_matrix(vect)
        Ht.from_matrix(np.conj(np.transpose(vect, axes=(1, 0, 2))))
        result = H * diag * Ht
        new_obj.from_covariance(result, shape=old_shape)

        return new_obj

    def filter_physical_conditions(self,
                                   tol=tol_default_filter,
                                   ignore_cond=None,
                                   keep=False,
                                   _counter=0):
        """Method that filters experimental errors by forcing the Mueller matrix to fulfill the conditions necessary for a matrix to be a real optical component.

        Parameters:
            tol (float): Tolerance in equalities.
            ignore_cond (list): Conditions to ignore. If False or None, no condition is ignored. Default: None.
            keep (bool): If True, the object is updated to the filtered result. If false, a new fresh copy is created. Default: True.
            _counter (int): Auxiliar variable that shouldn't be used when calling this function from outside.

        Returns:
            Mf (4x4 matrix): Filtered Mueller matrix.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.parent.copy()
        else:
            new_obj = self.parent
        old_shape = new_obj.shape

        # Start by calculating if the object is physically realizable
        cond, partial = new_obj.checks.is_physical(give_all=True,
                                                   tol=tol,
                                                   ignore_cond=ignore_cond,
                                                   shape=False,
                                                   out_number=False)

        # Act only if there is some violations or max numebr of iter is reached
        if np.any(~cond) and _counter <= counter_max:
            # Start by calculating the required information
            comp = new_obj.parameters.components(shape=False, out_number=False)
            D = new_obj.parameters.diattenuation(shape=False, out_number=False)
            P = new_obj.parameters.polarizance(shape=False, out_number=False)

            # Now solve the conditions one by one. However, in order to avoid problems, once we have acted in one matrix of the object, we won't modify it again until next iterations, except if the conditions are complementary
            used = np.zeros(new_obj.size, dtype=bool)

            # Condition 1: Elements must be real
            cond = ~partial[0]
            if np.any(cond):
                for ind, elem in enumerate(comp):
                    comp[ind] = np.array(elem, dtype=float)
                used = used + ~partial[0]

            # Condition 2a: M00 must be greater than 0
            cond = ~partial[1]
            if np.any(cond):
                comp[0][cond] = 0
                used = used + cond

            # Condition 2b: M00 must be lower than 1
            cond = ~partial[2]
            if np.any(cond):
                M00 = comp[0][cond]
                for ind, elem in enumerate(comp):
                    comp[ind][cond] = comp[ind][cond] / M00
                used = used + cond

            # Condition 3: All elements must be lower than M00
            cond = ~partial[3]
            if np.any(cond):
                M00 = comp[0]
                for ind, elem in enumerate(comp):
                    cond2 = np.abs(comp[ind]) > M00
                    comp[ind][cond * cond2] = np.sign(
                        comp[ind][cond * cond2]) * M00[cond2]
                used = used + cond

            # Condition 4a: Diattenuation can't be greater than 1
            cond = ~partial[4] * ~used
            if np.any(cond):
                comp[1][cond] = comp[1][cond] / D[cond]
                comp[2][cond] = comp[2][cond] / D[cond]
                comp[3][cond] = comp[3][cond] / D[cond]
                used = used + cond

            # Condition 4b: Polarizance can't be greater than 1
            cond = ~partial[5] * ~used
            if np.any(cond):
                comp[4][cond] = comp[4][cond] / P[cond]
                comp[8][cond] = comp[8][cond] / P[cond]
                comp[12][cond] = comp[12][cond] / P[cond]
                used = used + cond

            # Condition 5a: Total transmission can't be greater than 1
            cond = ~partial[6] * ~used
            if np.any(cond):
                aux = 1 / comp[0][cond] - 1
                comp[1][cond] = comp[1][cond] * aux
                comp[2][cond] = comp[2][cond] * aux
                comp[3][cond] = comp[3][cond] * aux
                used = used + cond

            # Condition 5b: Total reciproc transmission can't be greater than 1
            cond = ~partial[7] * ~used
            if np.any(cond):
                aux = 1 / comp[0][cond] - 1
                comp[4][cond] = comp[4][cond] * aux
                comp[8][cond] = comp[8][cond] * aux
                comp[12][cond] = comp[12][cond] * aux
                used = used + cond

            # Condition 6: Tr(M*M^T) <= 4*M00. Leave away M00, P and D components, as they should be fixed now
            cond = ~partial[8] * ~used
            if np.any(cond):
                # Leave away M00, P and D components
                cte1 = 2 * (comp[1][cond] * comp[4][cond] + comp[2][cond] *
                            comp[8][cond] + comp[3][cond] * comp[12][cond])
                cte2 = comp[5][cond]**2 + comp[10][cond]**2 + comp[15][
                    cond]**2 + 2 * (comp[6][cond] * comp[9][cond] +
                                    comp[7][cond] * comp[13][cond] +
                                    comp[11][cond] * comp[14][cond])
                aux = np.sqrt((3 * comp[0][cond]**2 - cte1) / cte2)
                for ind, elem in enumerate(comp):
                    if not ind in (0, 1, 2, 3, 4, 8, 12):
                        comp[ind][cond] = comp[ind][cond] * aux
                used = used + cond

            # Cond 7a: Condition in m matrix
            cond = ~partial[9] * ~used
            if np.any(cond):
                M00 = comp[0][cond]
                # Calculate the row weights and signs
                k1 = comp[5][cond] + comp[6][cond] + comp[7][cond]
                k2 = comp[9][cond] + comp[10][cond] + comp[11][cond]
                k3 = comp[13][cond] + comp[14][cond] + comp[15][cond]
                s1 = np.sign(1 - k1 / (M00 * D[cond]))
                s2 = np.sign(1 - k2 / (M00 * D[cond]))
                s3 = np.sign(1 - k3 / (M00 * D[cond]))
                kT = np.abs(k1) + np.abs(k2) + np.abs(k3)
                # Fix
                cond2 = (k1 != 0) * (comp[1][cond] != 0)
                if np.any(cond2):
                    aux = (M00[cond2] * D[cond][cond2] / k1[cond2]) * (
                        1 - s1[cond2] * M00[cond2] * (1 - D[cond][cond2]) /
                        (np.abs(comp[1][cond][cond2] * k1[cond2] / kT[cond2])))
                    comp[5][cond][cond2] *= aux
                    comp[6][cond][cond2] *= aux
                    comp[7][cond][cond2] *= aux

                cond2 = (k2 != 0) * (comp[2][cond] != 0)
                if np.any(cond2):
                    aux = (M00[cond2] * D[cond][cond2] / k2[cond2]) * (
                        1 - s2[cond2] * M00[cond2] * (1 - D[cond][cond2]) /
                        (np.abs(comp[2][cond][cond2] * k2[cond2] / kT[cond2])))
                    comp[9][cond][cond2] *= aux
                    comp[10][cond][cond2] *= aux
                    comp[11][cond][cond2] *= aux

                cond2 = (k3 != 0) * (comp[3][cond] != 0)
                if np.any(cond2):
                    aux = (M00[cond2] * D[cond][cond2] / k3[cond2]) * (
                        1 - s3[cond2] * M00[cond2] * (1 - D[cond][cond2]) /
                        (np.abs(comp[3][cond][cond2] * k3[cond2] / kT[cond2])))
                    comp[13][cond][cond2] *= aux
                    comp[14][cond][cond2] *= aux
                    comp[15][cond][cond2] *= aux
                used = used + cond

            # Cond 7b: Condition in reciprocal m matrix
            cond = ~partial[10] * ~used
            if np.any(cond):
                M00 = comp[0][cond]
                # Calculate the row weights and signs
                k1 = comp[5][cond] + comp[9][cond] + comp[13][cond]
                k2 = comp[6][cond] + comp[10][cond] + comp[14][cond]
                k3 = comp[7][cond] + comp[11][cond] + comp[15][cond]
                s1 = np.sign(1 - k1 / (M00 * P[cond]))
                s2 = np.sign(1 - k2 / (M00 * P[cond]))
                s3 = np.sign(1 - k3 / (M00 * P[cond]))
                kT = np.abs(k1) + np.abs(k2) + np.abs(k3)
                # Fix
                cond2 = (k1 != 0) * (comp[4][cond] != 0)
                if np.any(cond2):
                    aux = (M00[cond2] * D[cond][cond2] / k1[cond2]) * (
                        1 - s1[cond2] * M00[cond2] * (1 - D[cond][cond2]) /
                        (np.abs(comp[4][cond][cond2] * k1[cond2] / kT[cond2])))
                    comp[5][cond][cond2] *= aux
                    comp[9][cond][cond2] *= aux
                    comp[13][cond][cond2] *= aux

                cond2 = (k2 != 0) * (comp[2][cond] != 0)
                if np.any(cond2):
                    aux = (M00[cond2] * D[cond][cond2] / k2[cond2]) * (
                        1 - s2[cond2] * M00[cond2] * (1 - D[cond][cond2]) /
                        (np.abs(comp[7][cond][cond2] * k2[cond2] / kT[cond2])))
                    comp[6][cond][cond2] *= aux
                    comp[10][cond][cond2] *= aux
                    comp[14][cond][cond2] *= aux

                cond2 = (k3 != 0) * (comp[3][cond] != 0)
                if np.any(cond2):
                    aux = (M00[cond2] * D[cond][cond2] / k3[cond2]) * (
                        1 - s3[cond2] * M00[cond2] * (1 - D[cond][cond2]) /
                        (np.abs(comp[10][cond][cond2] * k3[cond2] / kT[cond2]))
                    )
                    comp[7][cond][cond2] *= aux
                    comp[11][cond][cond2] *= aux
                    comp[15][cond][cond2] *= aux
                used = used + cond

            # Cond 8: Conditions in the eigenvalues of the covariance matrix
            cond = (~partial[11] + ~partial[12] + ~partial[13]) * ~used
            if np.any(cond):
                # Calculate the covariance matrix and its eigenvalues
                obj = new_obj.covariance_matrix(keep=True)
                val, vect = obj.parameters.eig(values_as_matrix=True,
                                               vectors_as_matrix=True,
                                               shape=False)

                # Condition 8a: Eigenvalues must be real
                cond2 = ~partial[11] * ~used
                if np.any(cond2):
                    val = np.real(val)
                used = used + cond2

                # Condition 8a: Eigenvalues must be greater than 0
                cond2 = ~partial[12] * ~used
                if np.any(cond2):
                    val[val < 0] = 0

                # Condition 8a: Eigenvalues must be lower than 1
                cond2 = ~partial[13] * ~used
                if np.any(cond2):
                    val[val > 1] = 1

                # Now, lets merge the fix in the last condition with the rest
                H, diag, Ht = create_Mueller(N=3)
                diag.from_components([val[0, :]] + 4 * [0] + [val[1, :]] +
                                     4 * [0] + [val[2, :]] + 4 * [0] +
                                     [val[3, :]])
                H.from_matrix(vect)
                Ht.from_matrix(np.conj(np.transpose(vect, axes=(1, 0, 2))))
                result = H * diag * Ht
                result.from_covariance(result)
                comp2 = result.parameters.components(shape=False,
                                                     out_number=False)
                for ind, elem in enumerate(comp2):
                    comp[ind][cond] = elem[cond]

            # Finally, use the corrected values
            new_obj.from_components(comp, shape=old_shape)

            # Call again this function to solve all the problems iteratively through recurrence
            new_obj.analysis.filter_physical_conditions(
                tol=tol,
                ignore_cond=ignore_cond,
                keep=keep,
                _counter=_counter + 1)

        return new_obj

    ####################################################################
    # Matrix decomposition
    ####################################################################

    def decompose_pure(
            self,
            decomposition='RP',
            give_all=False,
            tol=tol_default,
            # filter=False,
            out_number=True,
            shape_like=None,
            shape=None,
            verbose=False,
            draw=False,
            transmissions='all',
            angles="all"):
        """Polar decomposition of a pure Mueller matrix in a retarder and a diattenuator.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 151.
            S. Y. Lu, R. A. Chipman; "Interpretation of Mueller matrices based on polar decomposition"; J. Opt. Soc. Am. A/Vol. 13, No. 5 (1996)

        Parameters:
            decomposition (string): string with the order of the elements: retarder (R) or diattenuator/polarizer (D or P). Default: RP.
            give_all (bool): If True, the dictionary of parameters will be given in the returned. Default: False.
            tol (float): Tolerance in equalities.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (float): If true, the function prints out some information about the matrices.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.
            transmissions (string): Determines the type of transmission output, FIELD, INTENSITY or ALL. Default: All.
            angles (string): Determines the type of angles output, CHARAC (characteristic angles), AZIMUTH (azimuth and ellipticity) or ALL. Default: All.

        Returns:
            Mr (Mueller): Mueller matrix of the retarder.
            Md (Mueller): Mueller matrix of the diattenuator.
            param (dictionary, optional): Dictionary with all the parameters of the decomposed elements.
        """
        # In order to be efficient, start by printing soem things
        Md = self.parent.copy()
        Md.name = 'Diattenuator of ' + self.parent.name
        if verbose or draw:
            if decomposition[0] in ('Rr'):
                decomposition = 'Mr * Md'
            else:
                decomposition = 'Md * Mr'
            print("\n------------------------------------------------------")
            print('Pure decomposition of {} as M = {}.'.format(
                self.parent.name, decomposition))
        # Filter the matrix if required TODO

        # First step, extract the diattenuator parameters
        if decomposition[0] in ('Rr'):
            trans, ang = Md.analysis.diattenuator(transmissions='all',
                                                  angles='all',
                                                  out_number=out_number,
                                                  shape=shape,
                                                  shape_like=shape_like,
                                                  verbose=verbose,
                                                  draw=draw)
        else:
            trans, ang = Md.analysis.polarizer(transmissions='all',
                                               angles='all',
                                               out_number=out_number,
                                               shape=shape,
                                               shape_like=shape_like,
                                               verbose=verbose,
                                               draw=draw)

        p1, p2 = trans[2:]
        azD, elD = ang[2:]

        # Now, we can create the diatenuator object
        Md.diattenuator_azimuth_ellipticity(p1=p1,
                                            p2=p2,
                                            azimuth=azD,
                                            ellipticity=elD)

        # Now, the retarder. To do that, we have to invert the diattenuator
        Mdi = Mueller()
        Mdi.diattenuator_azimuth_ellipticity(p1=1 / p1,
                                             p2=1 / p2,
                                             azimuth=azD,
                                             ellipticity=elD)
        if decomposition[0] in ('Rr'):
            Mr = self.parent * Mdi
        else:
            Mr = Mdi * self.parent

        # In some cases, the diattenuator is not invertible. Solve those cases
        cond = p2 < tol
        if np.any(cond):
            # If p1 was also 0, the original matrix is a total absorbr, so a retarder does not ahve sense
            cond2 = p1 < tol
            if np.any(cond2):
                Mr[cond2] = self.parent[cond2]

            # For the rest of cases, the retarder is not unique. Let's chose the one with the lowest retardance.
            cond2 = ~cond2 * cond
            if np.any(cond2):
                # First, the retardance
                Dv = self.parent.parameters.diattenuation_vector(
                    shape=shape, shape_like=shape_like)
                Pv = self.parent.parameters.polarizance_vector(
                    shape=shape, shape_like=shape_like)
                R = np.arccos(np.sum(Dv * Pv, axis=0))
                # Now, the retardance vector
                Rv = np.cross(Pv, Dv, axis=0)
                norm = np.linalg.norm(Rv, axis=0)
                Rv[0, :] = Rv[0, :] / norm
                Rv[1, :] = Rv[1, :] / norm
                Rv[2, :] = Rv[2, :] / norm
                # Finally, construct the object
                Mr2 = Mueller()
                Mr2.retarder_vector(Rv=Rv,
                                    R=R,
                                    shape=shape,
                                    shape_like=shape_like)
                Mr[cond2] = Mr2[cond2]

        Mr.name = 'Retarder of ' + self.parent.name

        # Set the global phases
        if self.parent.global_phase is not None:
            Md.set_global_phase(self.parent.global_phase / 2,
                                shape=shape,
                                shape_like=shape_like)
            Mr.set_global_phase(self.parent.global_phase / 2,
                                shape=shape,
                                shape_like=shape_like)

        # Information dictionary and visualization
        if give_all or verbose or draw:
            # Diattenuator
            trans, ang = Md.analysis.diattenuator(transmissions=transmissions,
                                                  angles=angles,
                                                  out_number=out_number,
                                                  verbose=verbose,
                                                  draw=draw)
            parameters = {}
            if transmissions.upper() == 'FIELD':
                parameters['p1'], parameters['p2'] = trans
            elif transmissions.upper() == 'INTENSITY':
                parameters['Tmax'], parameters['Tmin'] = trans
            else:
                parameters['Tmax'], parameters['Tmin'], parameters[
                    'p1'], parameters['p2'] = trans
            if angles.upper() == 'CHARAC':
                parameters['alpha D'], parameters['delay D'] = ang
            elif angles.upper() == 'ALL':
                parameters['alpha D'], parameters['delay D'], parameters[
                    'azimuth D'], parameters['ellipticity D'] = ang
            else:
                parameters['azimuth D'], parameters['ellipticity D'] = ang
            # Retarder
            R, ang = Mr.analysis.retarder(angles=angles,
                                          out_number=out_number,
                                          shape=shape,
                                          shape_like=shape_like,
                                          verbose=verbose,
                                          draw=draw)
            parameters['R'] = R
            if angles.upper() == 'CHARAC':
                parameters['alpha R'], parameters['delay R'] = ang
            elif angles.upper() == 'ALL':
                parameters['alpha R'], parameters['delay R'], parameters[
                    'azimuth R'], parameters['ellipticity R'] = ang
            else:
                parameters['azimuth R'], parameters['ellipticity R'] = ang

            # Error
            if decomposition[0] in ('Rr'):
                Mrec = Mr * Md
            else:
                Mrec = Md * Mr
            error = self.parent.parameters.matrix(
                shape=shape, shape_like=shape_like) - Mrec.parameters.matrix(
                    shape=shape, shape_like=shape_like)
            error = np.linalg.norm(error, axis=(0, 1)) / 16
            if ~out_number and self.parent.size > 1:
                error = np.array([error])
            parameters['Error'] = error
            if verbose or draw:
                heading = '{} decomposition mean square error:'.format(
                    self.parent.name)
                PrintParam(param=[error],
                           shape=self.parent.shape,
                           title=['MSE'],
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
                print(
                    "------------------------------------------------------\n")

        # Return
        if give_all:
            return Mr, Md, parameters
        else:
            return Mr, Md

    def decompose_polar(self,
                        decomposition='PRD',
                        give_all=False,
                        tol=tol_default,
                        out_number=True,
                        shape_like=None,
                        shape=None,
                        verbose=False,
                        draw=False,
                        transmissions='all',
                        angles="all",
                        depolarization='all'):
        """Polar decomposition of a physically realizable Mueller matrix in a partial depolarizer, retarder and a diattenuator.

        TODO: When the depolarizer is singular with 2 or 3 non-zero eigenvalues, the decomposed retarder is often erroneous (60-80% of the time).

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 257.

        Parameters:
            decomposition (string): string with the order of the elements: depolarizer (P), retarder (R) or diattenuator (D). There are six possible decompositions: PRD, RDP, PDR, RDP, DRP and DPR. Default: PRD.
            give_all (bool): If true, the dictionary of parameters will be given in the returned. Default: False.
            tol (float): Tolerance in equalities.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (float): If true, the function prints out some information about the matrices.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.
            transmissions (string): Determines the type of transmission output, FIELD, INTENSITY or ALL. Default: All.
            angles (string): Determines the type of angles output, CHARAC (characteristic angles), AZIMUTH (azimuth and ellipticity) or ALL. Default: All.
            depolarization (string): Determines the type of depolarization information: INDEX, FACTORS or ALL. Default: All.

        Returns:
            Mr (Mueller): Mueller matrix of the retarder.
            Md (Mueller): Mueller matrix of the diattenuator/polarizer.
            Mp (Mueller): Mueller matrix of the depolarizer.
            param (dictionary, optional): Dictionary with all the parameters of the decomposed elements.
        """
        # Some common parameters
        M = self.parent.copy()
        pure = M.checks.is_pure(shape=shape, shape_like=shape_like)
        # # If M is completelly pure, there is no point in continuing in this path, go to the pure decomposition instead
        # if np.all(pure):
        #     Mp = Mueller('Depolarizer of ' + self.parent.name)
        #     # Pure case (no depolarizer).
        #     new_dec = decomposition.replace('P', '')
        #     new_dec.replace('p', '')
        #     Mr, Md = self.decompose_pure(decomposition=new_dec, tol=tol)
        #     Mp.vacuum(length=M.size, shape=Mr.shape)
        # else:
        # Create objects and extract usefull info
        Mp, Md = create_Mueller(N=2)
        M00 = M.parameters.mean_transmission(shape=shape,
                                             shape_like=shape_like)
        singular = M.checks.is_singular(shape=shape, shape_like=shape_like)
        det = self.parent.parameters.det(shape=False)

        # Chhoose decomposition
        if decomposition.upper() == 'PRD':
            # Start by calculating the diattenuator
            Dv = M.parameters.diattenuation_vector(shape=shape,
                                                   shape_like=shape_like)
            Md.diattenuator_vector(Dv=Dv, M00=M00)
            # Now, calculate the inverse of the diattenuator and extract it from the total matrix
            (p1, p2), (alpha,
                       delay) = Md.analysis.diattenuator(transmissions='field',
                                                         angles='charac')
            Mdi = Mueller()
            Mdi.diattenuator_charac_angles(p1=1 / p1,
                                           p2=1 / p2,
                                           alpha=alpha,
                                           delay=delay)
            Mdi.M[np.isnan(Mdi.M)] = 0  # Allow continuing if p2 = 0
            Mdi.M[np.isinf(Mdi.M)] = 0  # Allow continuing if p2 = 0
            Mf = M * Mdi

            # Now, we have the polarizance vector of Md. Extract it
            _, _, Pvf, mf = Mf.parameters.blocks()
            Mf.from_blocks(m=mf)
            # Calculate eigenvalues and eigenvectors of the square matrix
            Mf2 = Mf * Mf.transpose(keep=True)
            val, vect = Mf2.parameters.eig(values_as_matrix=True,
                                           vectors_as_matrix=True,
                                           shape=False)
            # print(val, vect)
            # Recompose the matrix of depolarizer using the square root of the eigenvalues
            H, diag, Ht = create_Mueller(('R2', 'Diag', 'R2T'))
            diag.from_components([np.sqrt(val[0, :])] + 4 * [0] +
                                 [np.sqrt(val[1, :])] + 4 * [0] +
                                 [np.sqrt(val[2, :])] + 4 * [0] +
                                 [np.sign(det) * np.sqrt(val[3, :])])
            H.from_matrix(vect)
            Ht.from_matrix(np.conj(np.transpose(vect, axes=(1, 0, 2))))
            result = H * diag * Ht
            # print(H, diag, Ht)
            # print('result1', result)
            mp = result.parameters.small_matrix()
            Mp.from_blocks(Pv=Pvf, m=mp)
            # Finally, the retarder can be calculated by using the inverse matrix of Mp
            diag.from_components([1 / np.sqrt(val[0, :])] + 4 * [0] +
                                 [1 / np.sqrt(val[1, :])] + 4 * [0] +
                                 [1 / np.sqrt(val[2, :])] + 4 * [0] +
                                 [np.sign(det) / np.sqrt(val[3, :])])
            result = H * diag * Ht
            # print('result2', result)
            Mr = result * Mf
            # print('Mf', Mf, result, Mr)

            # Now, let's see if we have the special cases of singular matrices.
            # First case: Mp is singular and Md not
            cond = singular * (p2 > tol)
            if np.any(cond):
                Mp_aux, Mr_aux = create_Mueller(N=2)
                # We have to wnow how many eigenvalues of Mf are different from 0
                N = np.sum(val > tol, axis=0)
                # print('N = ', N)
                if Mr.ndim > 1:
                    N = np.reshape(N, Mr.shape)
                condN1 = cond * (N == 1)
                condN2 = cond * (N == 2)
                condN3 = cond * (N == 3)
                condN23 = condN2 + condN3
                # Only one eigenvalue, No retarder at all. This is very simple
                if np.any(condN1):
                    Mp_aux.from_blocks(Pv=Pvf, shape=Mp.shape)
                    Mr_aux.vacuum(length=Mr.size, shape=Mr.size)
                    Mr[condN1] = Mr_aux[condN1]
                    Mp[condN1] = Mp_aux[condN1]
                # We start to need the right eigenvectors of Mf^2
                if np.any(condN23):
                    # Left eigenvectors. Make eigenvale 1 = 0 to remove it
                    val_aux = val
                    val_aux[val == 1] = 0
                    val_ord, vect_ord = order_eig(val_aux, vect, 'reverse')
                    v1 = vect_ord[1:, 0, :]
                    # Left eigenvectors. Make eigenvale 1 = 0 to remove it
                    Mf2_b = Mf.transpose(keep=True) * Mf
                    val_b, vect_b = Mf2_b.parameters.eig(
                        values_as_matrix=True,
                        vectors_as_matrix=True,
                        shape=False)
                    val_b[val_b == 1] = 0
                    val_b_ord, vect_b_ord = order_eig(val_b, vect_b, 'reverse')
                    w1 = vect_b_ord[1:, 0, :]
                # Two eigenvalues.
                if np.any(condN2):
                    # TODO: Works 30% of the time
                    # Calculate Mp
                    trace_m2 = Mf2.parameters.trace(
                    ) - Mf2.parameters.mean_transmission()
                    result = Mf2 / np.sqrt(trace_m2)
                    Mp_aux.from_blocks(
                        Pv=Pvf,
                        m=result.parameters.small_matrix(normalize=False),
                        shape=Mp.shape)
                    Mp[condN2] = Mp_aux[condN2]
                    # Calculate retardance
                    trace_m = Mf.parameters.trace(
                    ) - Mf.parameters.mean_transmission()
                    R = np.arccos(trace_m / np.sqrt(trace_m2))
                    # Calculate normalized retardance vector
                    Rv = np.cross(v1, w1, axis=0)
                    norm = np.linalg.norm(Rv, axis=0)
                    Rv[0, :] = Rv[0, :] / norm
                    Rv[1, :] = Rv[1, :] / norm
                    Rv[2, :] = Rv[2, :] / norm
                    # Calculate the retarder
                    Mr_aux.retarder_vector(R=R, Rv=Rv)
                    Mr[condN2] = Mr_aux[condN2]
                # Three eigenvalues
                if np.any(condN3):
                    # TODO: Works 15% of the time
                    # Calculate Mp
                    suma = np.sqrt(val_ord[0, :]) + np.sqrt(val_ord[1, :])
                    producto = np.sqrt(val_ord[0, :] * val_ord[1, :])
                    result = Mf2 + producto * result.vacuum(length=M.size)
                    result.inverse()
                    result = suma * result * Mf2
                    Mp_aux.from_blocks(
                        Pv=Pvf,
                        m=result.parameters.small_matrix(normalize=False),
                        shape=Mp.shape)
                    Mp[condN3] = Mp_aux[condN3]
                    # Calculate the second set of vectors
                    v2 = vect_ord[1:, 1, :]
                    w2 = vect_b_ord[1:, 1, :]
                    prod_v = np.cross(v1, v2, axis=0)
                    norm = np.linalg.norm(prod_v, axis=0)
                    prod_v[0, :] /= norm
                    prod_v[1, :] /= norm
                    prod_v[2, :] /= norm
                    prod_w = np.cross(w1, w2, axis=0)
                    norm = np.linalg.norm(prod_w, axis=0)
                    prod_w[0, :] /= norm
                    prod_w[1, :] /= norm
                    prod_w[2, :] /= norm
                    # Calculate Mr
                    mr = kron_axis(v1, w1, axis=0) + \
                    kron_axis(v2, w2, axis=0) + kron_axis(prod_v, prod_w, axis=0)
                    Mr_aux.from_blocks(m=mr)
                    Mr[condN3] = Mr_aux[condN3]

            # Now, the diattenuator is singular
            cond = p2 <= tol
            if np.any(cond):
                # In this case, both depolarizer and retarder are not unique. We choose a diagonal depolarizer (easy to understand) and the retarder with minimum retardance.
                Mp_aux, Mr_aux = create_Mueller(N=2)
                # Start by the depolarizer
                P = M.parameters.polarizance(shape=shape,
                                             shape_like=shape_like)
                Mp_aux.depolarizer_diagonal(d=P)
                Mp[cond] = Mp_aux[cond]
                # Now, the retarder
                Pv = M.parameters.polarizance_vector(shape=shape,
                                                     shape_like=shape_like)
                Dv = M.parameters.diattenuation_vector(shape=shape,
                                                       shape_like=shape_like)
                R = np.arccos(np.sum(Pv * Dv, axis=0) / P)
                Rv = np.cross(Pv, Dv, axis=0)
                norm = np.linalg.norm(Rv, axis=0)
                Rv[0, :] /= norm
                Rv[1, :] /= norm
                Rv[2, :] /= norm
                Mr_aux.retarder_vector(R=R, Rv=Rv)
                Mr[cond] = Mr_aux[cond]

        elif decomposition.upper() == 'PDR':
            Mr, Md, Mp = M.analysis.decompose_polar(tol=tol,
                                                    shape_like=shape_like,
                                                    shape=shape)
            Md = Mr * Md * Mr.transpose(keep=True)

        elif decomposition.upper() == 'RPD':
            Mr, Md, Mp = M.analysis.decompose_polar(tol=tol,
                                                    shape_like=shape_like,
                                                    shape=shape)
            Mp = Mr.transpose(keep=True) * Mp * Mr

        elif decomposition.upper() == 'DRP':
            # Same as above for the transposed matrix
            M2 = M.transpose(keep=True)
            Mr, Md, Mp = M2.analysis.decompose_polar(tol=tol,
                                                     shape_like=shape_like,
                                                     shape=shape)
            Mr.transpose()
            Mp.transpose()

        elif decomposition.upper() == 'RDP':
            Mr, Md, Mp = M.analysis.decompose_polar(tol=tol,
                                                    shape_like=shape_like,
                                                    shape=shape,
                                                    decomposition='DRP')
            Md = Mr.transpose(keep=True) * Md * Mr

        elif decomposition.upper() == 'DPR':
            Mr, Md, Mp = M.analysis.decompose_polar(tol=tol,
                                                    shape_like=shape_like,
                                                    shape=shape,
                                                    decomposition='DRP')
            Mp = Mr * Mp * Mr.transpose(keep=True)

        else:
            raise ValueError(
                'Decomposition {} is not valid, must be a combination of R, P and D without repetition'
                .format(decomposition))

        # Correct names
        Mp.name = 'Depolarizer of ' + self.parent.name
        Md.name = 'Diattenuator of ' + self.parent.name
        Mr.name = 'Retarder of ' + self.parent.name

        # Set glpbal phase
        if M.global_phase is not None:
            gp = M.parameters.global_phase(shape=shape, shape_like=shape_like)
            Md.set_global_phase(gp / 3)
            Mp.set_global_phase(gp / 3)
            Mr.set_global_phase(gp / 3)

        # Finally, print options and extract the info
        # print the heading
        if verbose or draw or give_all:
            if verbose or draw:
                decomposition = ''
                for elem in decomposition:
                    if elem in 'Rr':
                        decomposition += 'Mr'
                    elif elem in 'Dd':
                        decomposition += 'Md'
                    else:
                        decomposition += 'Mp'
                    if elem != decomposition[-1]:
                        decomposition += ' * '
                print(
                    "\n------------------------------------------------------")
                print('Polar decomposition of {} as M = {}.'.format(
                    self.parent.name, decomposition))
            # Parameters
            parameters = {}
            # Diattenuator
            trans, ang = Md.analysis.diattenuator(transmissions=transmissions,
                                                  angles=angles,
                                                  out_number=out_number,
                                                  verbose=verbose,
                                                  draw=draw)
            if transmissions.upper() == 'FIELD':
                parameters['p1'], parameters['p2'] = trans
            elif transmissions.upper() == 'INTENSITY':
                parameters['Tmax'], parameters['Tmin'] = trans
            else:
                parameters['Tmax'], parameters['Tmin'], parameters[
                    'p1'], parameters['p2'] = trans
            if angles.upper() == 'CHARAC':
                parameters['alpha D'], parameters['delay D'] = ang
            elif angles.upper() == 'ALL':
                parameters['alpha D'], parameters['delay D'], parameters[
                    'azimuth D'], parameters['ellipticity D'] = ang
            else:
                parameters['azimuth D'], parameters['ellipticity D'] = ang
            # Retarder
            R, ang = Mr.analysis.retarder(angles=angles,
                                          out_number=out_number,
                                          shape=shape,
                                          shape_like=shape_like,
                                          verbose=verbose,
                                          draw=draw)
            parameters['R'] = R
            if angles.upper() == 'CHARAC':
                parameters['alpha R'], parameters['delay R'] = ang
            elif angles.upper() == 'ALL':
                parameters['alpha R'], parameters['delay R'], parameters[
                    'azimuth R'], parameters['ellipticity R'] = ang
            else:
                parameters['azimuth R'], parameters['ellipticity R'] = ang
            # Depolarizer
            trans_D, trans_P, angles_D, angles_P, depol, principal_states = Mp.analysis.depolarizer(
                transmissions=transmissions,
                angles=angles,
                depolarization=depolarization,
                out_number=out_number,
                verbose=verbose,
                draw=draw)
            if transmissions.upper() == 'FIELD':
                parameters['p1 Depol D'], parameters['p2 Depol D'] = trans_D
                parameters['p1 Depol P'], parameters['p2 Depol P'] = trans_P
            elif transmissions.upper() == 'INTENSITY':
                parameters['Tmax Depol D'], parameters[
                    'Tmin Depol D'] = trans_D
                parameters['Tmax Depol P'], parameters[
                    'Tmin Depol P'] = trans_P
            else:
                parameters['Tmax Depol D'], parameters[
                    'Tmin Depol D'], parameters['p1 Depol D'], parameters[
                        'p2 Depol D'] = trans_D
                parameters['Tmax Depol P'], parameters[
                    'Tmin Depol P'], parameters['p1 Depol P'], parameters[
                        'p2 Depol P'] = trans_P
            if angles.upper() == 'CHARAC':
                parameters['alpha Depol D'], parameters[
                    'delay Depol D'] = angles_D
                parameters['alpha Depol P'], parameters[
                    'delay Depol P'] = angles_P
            elif angles.upper() == 'ALL':
                parameters['alpha Depol D'], parameters[
                    'delay Depol D'], parameters[
                        'azimuth Depol D'], parameters[
                            'ellipticity Depol D'] = angles_D
                parameters['alpha Depol P'], parameters[
                    'delay Depol P'], parameters[
                        'azimuth Depol P'], parameters[
                            'ellipticity Depol P'] = angles_P
            else:
                parameters['azimuth Depol D'], parameters[
                    'ellipticity Depol D'] = angles_D
                parameters['azimuth Depol P'], parameters[
                    'ellipticity Depol P'] = angles_P
            # Error
            Mrec = Mueller()
            Mrec.vacuum(length=M.size)
            Mrec.shape = Md.shape
            for elem in decomposition:
                if elem.upper() == 'R':
                    Mrec = Mrec * Mr
                elif elem.upper() == 'D':
                    Mrec = Mrec * Md
                else:
                    Mrec = Mrec * Mp
            error = M.parameters.matrix(
                shape=shape, shape_like=shape_like) - Mrec.parameters.matrix(
                    shape=shape, shape_like=shape_like)
            error = np.linalg.norm(error, axis=(0, 1)) / 16
            if out_number and M.size == 1:
                error = error[0]
            parameters['Error'] = error
            if verbose or draw:
                heading = '{} decomposition mean square error:'.format(
                    self.parent.name)
                PrintParam(param=[error],
                           shape=self.parent.shape,
                           title=['MSE'],
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
                print(
                    "------------------------------------------------------\n")

        # Return
        if give_all:
            return Mr, Md, Mp, parameters
        else:
            return Mr, Md, Mp

            # # Calculate the diattenuator/polarizer
            # p1, p2, alphaP, delayP, fiP, chiP = self.diattenuator()
            # Mp.diattenuator_charac_angles_from_vector(
            #     p1, p2, alphaP, delayP)
            # D = Mp.parameters.diattenuation()
            # # Sometimes, due to numeric calculation, D may be slightly higher than 1. Fix it.
            # if D > 1 and D < 1 + eps:
            #     D = 1
            # elif D > 1 + eps:
            #     raise ValueError(
            #         "Mueller matrix is not real, diattenuation is > 1.")
            # # Check if the matrix M is singular or not.
            # singM = parent.checks.is_singular(tol=tol)
            # singMp = Mp.checks.is_singular(tol=tol)
            # nz = 0
            # if singMp:
            #     # We have to determine if only Md is singular or not
            #     P = parent.parameters.polarizance()
            #     cond3 = np.abs(1 - P) <= tol
            #     if cond3:
            #         # Print type of decomposition
            #         if verbose:
            #             print(
            #                 "Both the depolarizer and the polarizer are singular."
            #             )
            #         # Homogeneous case
            #         Md.from_matrix(np.identity(4))
            #         Mr, Mp = decompose_pure(M, decomposition='PR', tol=tol)
            #     else:
            #         # Print type of decomposition
            #         if verbose:
            #             print("The polarizer is singular.")
            #         # Calculate the depolarizer polarizance vector
            #         Pdv = parent.P
            #         Mr.from_matrix(np.identity(4))
            #         cero = np.matrix(np.zeros(3))
            #         ceroM = np.zeros((3, 3))
            #         Md.from_blocks(cero, Pdv, ceroM)
            # else:
            #     # Calculate the depolarizer polarizance vector
            #     # Dv, Pv, m, m00 = divide_in_blocks(M)
            #     # Pdv = (Pv - m * Dv.T) / (1 - D**2)
            #     # For calculating the small matrix m of the depolarizer we need an
            #     # auxiliary matrix mf
            #     Gaux = matrix(np.diag([1, -1, -1, -1]))
            #     if singM:
            #         Mpinv = Gaux * Mp * Gaux * D**(-2)
            #     else:
            #         Mpinv = Gaux * Mp * Gaux * (1 - D**2)**(-1)
            #     Mf = parent * Mpinv
            #     _, Pdv, mf, _ = divide_in_blocks(Mf.M)
            #     md2 = mf * mf.T
            #     qi2, mr2 = np.linalg.eigh(md2)
            #     # check_eig(qi2, mr2, md2)
            #     qi = np.sqrt(qi2)
            #     cero = np.matrix(np.zeros(3))
            #     # Calculation method depends on Md being singular or not
            #     if singM:  # If M is singular and Mp is not => Md is singular
            #         # Calculate the number of eigenvalues that are zero and order them
            #         qi2, mr2 = order_eig(qi2, mr2)
            #         qi = np.sqrt(qi2)
            #         nz = sum(qi < tol)
            #         # Calculate other auxiliary matrices and vectors
            #         md1 = mf.T * mf
            #         qi12, mr1 = np.linalg.eigh(md1)
            #         qi12, mr1 = order_eig(qi12, mr1)
            #         v1, v2, w1, w2 = (mr2[:, 0], mr2[:, 1], mr1[:, 0],
            #                           mr1[:, 1])
            #         if nz == 3:
            #             # Print type of decomposition
            #             if verbose:
            #                 print(
            #                     "Depolarized matrix singular case with three null eigenvalues."
            #                 )
            #             # Trivial case
            #             md = np.zeros([3, 3])
            #             Md.from_blocks(cero, Pdv, md)
            #             Mr.from_matrix(np.eye(4))
            #         elif nz == 2:
            #             # Print type of decomposition
            #             if verbose:
            #                 print(
            #                     "Depolarized matrix singular case with two null eigenvalues."
            #                 )
            #             # Depolarizer
            #             md = mf * mf.T / sqrt(np.trace(mf * mf.T))
            #             Md.from_blocks(cero, Pdv, md)
            #             # Retarder. Note that it is not unique.
            #             # TODO: inexacto
            #             cR = np.trace(mf) / sqrt(np.trace(mf * mf.T))
            #             R = np.arccos(cR)
            #             x1 = np.cross(v1.T, w1.T)
            #             Mr.retarder_from_vector(R,
            #                                     x1[0] / np.linalg.norm(x1))
            #         else:
            #             # Print type of decomposition
            #             if verbose:
            #                 print(
            #                     "Depolarized matrix singular case with one null eigenvalue."
            #                 )
            #             # Depolarizer
            #             mat_aux = mf * mf.T + qi[0] * qi[1] * np.eye(3)
            #             md = (qi[0] + qi[1]) * (mat_aux).I * mf * mf.T
            #             Md.from_blocks(cero, Pdv, md)
            #             # Retarder. Note that it is not unique.
            #             # TODO: No funciona de momento
            #             # mr1 = np.matrix(mr1)
            #             # mr2 = np.matrix(mr2)
            #             # print([qi, qi2])
            #             # print(mf - mr2 * np.matrix(np.diag(qi)) * mr1.T)
            #             # print(mf.T * mf - mr1 * np.diag(qi2) * mr1.T)
            #             # print(mf * mf.T - mr2 * np.diag(qi2) * mr2.T)
            #             # print("")
            #             # print(mr1 * mr1.T)
            #             (y1, y2) = (np.cross(v1.T, v2.T),
            #                         np.cross(w1.T, w2.T))
            #             mr = v1 * w1.T + v2 * w2.T + y1 * y2.T / (
            #                 np.linalg.norm(y1) * np.linalg.norm(y2))
            #             # mr = np.matrix(mr)
            #             # print(mr * mr.T)
            #             Mr.from_blocks(cero, cero.T, mr)
            #     else:
            #         # Print type of decomposition
            #         if verbose:
            #             print("General case.")
            #         # General case
            #         s = np.sign(np.linalg.det(M))
            #         md = np.diag([qi[0], qi[1], s * qi[2]])
            #         md = mr2 * md * mr2.T
            #         Md.from_blocks(cero, Pdv, md)
            #         # Calculate the retarder
            #         mdinv = mr2 * np.diag(
            #             [1 / qi[0], 1 / qi[1], s / qi[2]]) * mr2.T
            #         mr = mdinv * mf
            #         Mr.from_blocks(cero, cero.T, mr)
            # if decomposition in ('DPR', 'dpr'):
            #     Mpure = Mr * Mp
            #     Mr, Mp = Mpure.analysis.decompose_pure(
            #         decomposition='PR', tol=tol)
        #
        #     elif decomposition in ('PRD', 'prd', 'RPD', 'rpd'):
        #         # This procedure is simple, we just have to traspose
        #         Mtras = Mueller(parent.name)
        #         Mtras.from_matrix(M.T)
        #         Md, Mr, Mp = Mtras.analysis.decompose_polar(tol=tol)
        #         Mr.from_matrix(Mr.M.T)
        #         Md.from_matrix(Md.M.T)
        #         # Reverse order of polarizer and retarder if required
        #         if decomposition in ('RPD', 'rpd'):
        #             Mpure = Mp * Mr
        #             Mr, Mp = Mpure.analysis.decompose_pure(
        #                 decomposition='RP', tol=tol)
        #     elif decomposition in ('PDR', 'pdr'):
        #         # We can calculate this one slightly differently
        #         Mp, Mr, Md = parent.analysis.decompose_polar(
        #             decomposition='PRD', tol=tol)
        #         Md_matrix = Mr.M * Md.M * Mr.M.I
        #         Md.from_matrix(Md_matrix)
        #     elif decomposition in ('RDP', 'rdp'):
        #         # We can calculate this one as the traspose of the previous one
        #         Mtras = Mueller(parent.name)
        #         Mtras.from_matrix(M.T)
        #         Mp, Md, Mr = Mtras.analysis.decompose_polar(
        #             decomposition='PDR', tol=tol)
        #         Mr.from_matrix(Mr.M.T)
        #         Md.from_matrix(Md.M.T)
        #     else:
        #         raise ValueError("Decomposition not yet implemented.")
        # # Order the output matrices
        # Mout = [0, 0, 0]
        # for ind in range(3):
        #     if decomposition[ind] == 'D':
        #         Mout[ind] = Md
        #     elif decomposition[ind] == 'P':
        #         Mout[ind] = Mp
        #     else:
        #         Mout[ind] = Mr
        # # Calculate parameters
        # if verbose or give_all:
        #     R, alphaR, delayR, fiR, chiR = Mr.analysis.retarder()
        #     Pd = Md.parameters.polarizance()
        #     Desp = Md.parameters.depolarization_index()
        # # Calculate error
        # if give_all or verbose:
        #     if decomposition == 'DRP':
        #         Mt = Md * Mr * Mp
        #         D = np.abs(Mt.M - M)
        #     MeanErr = np.linalg.norm(D)
        #     MaxErr = D.max()
        # # Print results
        # if verbose:
        #     if decomposition == 'DRP':
        #         print("Polar decomposition of the matrix M = Mdesp * Mr * Mp:")
        #     for ind in range(3):
        #         print("")
        #         if decomposition[ind] == 'D':
        #             print("The depolarizer Mueller matrix is:")
        #             print(Md)
        #             print("Parameters:")
        #             print(("  - Polarizance = {}.".format(Pd)))
        #             print(("  - Depolarization degree = {}.".format(Desp)))
        #         elif decomposition[ind] == 'P':
        #             print("The diatenuator/polarizer Mueller matrix is:")
        #             if singM and nz < 3:
        #                 print(
        #                     "Warning: Retarder matrix may be slightly inaccurate"
        #                 )
        #             print(Mp)
        #             print("Parameters:")
        #             print(("  - p1 = {}; p2 = {}.".format(p1, p2)))
        #             print((
        #                 "  - Angle = {} deg; Delay between components = {} deg."
        #                 .format((alphaP / degrees), (delayP / degrees))))
        #             print(
        #                 ("  - Azimuth = {} deg; Ellipticity = {} deg.".format(
        #                     (fiP / degrees), (chiP / degrees))))
        #             Tmax, Tmin = Mp.parameters.transmissions()
        #             print((
        #                 "  - Max. transmission = {} %; Min. transmission = {} %."
        #                 .format((Tmax * 100), (Tmin * 100))))
        #         else:
        #             print("The retarder Mueller matrix is:")
        #             print(Mr)
        #             print("Parameters:")
        #             print(("  - Delay = {} deg.".format((R / degrees))))
        #             print((
        #                 "  - Angle = {} deg; Delay between components = {} deg."
        #                 .format((alphaR / degrees), (delayR / degrees))))
        #             print(
        #                 ("  - Azimuth = {} deg; Ellipticity = {} deg.".format(
        #                     (fiR / degrees), (chiR / degrees))))
        #     print("")
        #     print(("The mean square error in the decomposition is: {}".format(
        #         MeanErr)))
        #     print(
        #         ("The maximum error in the decomposition is: {}".format(MaxErr)
        #          ))
        #     print("------------------------------------------------------")
        # # Dictionary of parameters
        # if give_all:
        #     param = dict(
        #         Delay=R,
        #         AngleR=alphaR,
        #         AxisDelayR=delayR,
        #         AzimuthR=chiR,
        #         EllipticityR=fiR,
        #         p1=p1,
        #         p2=p2,
        #         AngleP=alphaP,
        #         AxisDelayP=delayP,
        #         AzimuthP=fiP,
        #         EllipticityP=chiP,
        #         DespPolarizance=Pd,
        #         DespDegree=Desp,
        #         MeanError=MeanErr,
        #         MaxError=MaxErr)
        # # Output
        # if give_all:
        #     return Mout[0], Mout[1], Mout[2], param
        # else:
        #     return Mout[0], Mout[1], Mout[2]


class Check_Mueller(object):
    """Class for Check of Mueller Matrices

    Parameters:
        parent (Mueller_matrix): Mueller Matrix

    Attributes:
        self.parent (Mueller): parent object.
    """
    def __init__(self, parent):
        self.parent = parent

    def __repr__(self):
        """Print all parameters."""
        self.get_all(verbose=True, draw=True)
        return ''

    def get_all(self, verbose=False, draw=False):
        """Creates a dictionary with all the checks of Mueller matrix.

        Parameters:
            verbose (bool): If True, print all parameters. Default: False.
            draw (bool): If True, draw all plots/images of the parameters. Default: False.
        """
        dict_params = {}
        dict_params['Physical'] = self.is_physical(verbose=verbose,
                                                        draw=draw)
        dict_params['Pure'] = self.is_pure(verbose=verbose, draw=draw)
        dict_params['Homogenous'] = self.is_homogeneous(verbose=verbose,
                                                             draw=draw)
        dict_params['Retarder'] = self.is_retarder(verbose=verbose,
                                                        draw=draw)
        dict_params['Diattenuator'] = self.is_diattenuator(
            verbose=verbose, draw=draw)
        dict_params['Depolarizer'] = self.is_depolarizer(verbose=verbose,
                                                              draw=draw)
        dict_params['Singular'] = self.is_singular(verbose=verbose,
                                                        draw=draw)
        dict_params['Symmetric'] = self.is_symmetric(verbose=verbose,
                                                          draw=draw)

        return dict_params

    def is_physical(self,
                    tol=tol_default,
                    give_all=False,
                    ignore_cond=None,
                    out_number=True,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """A Mueller matrix must fulfill several conditions in order to be physically realizable:

        1. $M_{ij}\in\mathbb{R}$ for i, j = 0, 1, 2 and 3.
        2a. $M_{00} \geq 0$.
        2b. $M_{00} \leq 1$ (except in active media).
        3. $M_{00}\geq abs(M_{ij})$ for i, j = 0, 1, 2 and 3.
        4a. $D \leq 1$
        4a. $P \leq 1$.
        5a. $M_{00} (1 + D) \leq 1$ (except in active media).
        5b. $M_{00} (1 + P) \leq 1$ (except in active media).
        6. $Tr(M*M^T)\leq 4(M_{00})^2$.
        7a. $M_{00}^{2}\left(1-D\right)^{2}\geq\mathop{\sum_{i=1}^{3}M_{0i}^{2}\left(1-\sum_{j=1}^{3}\frac{M_{ij}}{M_{00}D}\right)}^{2}$.
        7b. $M_{00}^{2}\left(1-P\right)^{2}\geq\mathop{\sum_{i=1}^{3}M_{i0}^{2}\left(1-\sum_{j=1}^{3}\frac{M_{ji}}{M_{00}P}\right)}^{2}$.
        8a. $\lambda_{i}\in\mathbb{R}$
        8b. $\lambda_{i}  \geq  0$.
        8c. $\lambda_{i}  \leq  1$ (except in active media).

        Being D the diattenuation, P the polarizance and $\lambda_{i}$ the eigenvalues of the covariance matrix.

        References:
            Handbook of Optics vol 2. 22.34 (There is an errata in the equation equivalent to condition 7).
            "Polarized light and the Mueller Matrix approach", J. J. Gil, pp 187.

        Parameters:
            tol (float): Tolerance in conditions. Default: tol_default.
            give_all (bool): If True, the method also gives a list with the individual conditions. Default: False.
            ignore_cond (list): Conditions to ignore. If False or None, no condition is ignored. Default: None.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            cond (numpy.ndarray or bool): Result.
            partial_conditions (list): List with the partial conditions.

        """
        # Prepare to ignore conditions
        if ignore_cond is None or ignore_cond is False:
            use_cond = [True] * 8
        else:
            use_cond = [True] * 8
            for ind in range(8):
                if ind in ignore_cond:
                    use_cond[ind] = False

        # Calculate the parameters
        M00 = self.parent.parameters.mean_transmission(shape=shape,
                                                       shape_like=shape_like,
                                                       out_number=False)
        D = self.parent.parameters.diattenuation(shape=shape,
                                                 shape_like=shape_like,
                                                 out_number=False)
        P = self.parent.parameters.polarizance(shape=shape,
                                               shape_like=shape_like,
                                               out_number=False)
        comp = self.parent.parameters.components(shape=shape,
                                                 shape_like=shape_like,
                                                 out_number=False)
        H = self.parent.covariance_matrix(keep=True)
        vals = H.parameters.eigenvalues(shape=shape,
                                        shape_like=shape_like,
                                        out_number=False)

        # Calculate the conditions
        if use_cond[0]:
            cond_1 = np.isreal(comp[0])
            for ind in range(1, 16):
                cond_1 *= np.isreal(comp[ind])
        else:
            cond_1 = np.ones_like(D, dtype=bool)

        if use_cond[1]:
            cond_2a = M00 >= -tol
            cond_2b = M00 <= 1 + tol
        else:
            cond_2a = np.ones_like(D, dtype=bool)
            cond_2b = cond_2a

        if use_cond[2]:
            cond_3 = np.ones_like(D, dtype=bool)
            for ind in range(1, 16):
                cond_3 *= comp[0] >= np.abs(comp[ind])
        else:
            cond_3 = np.ones_like(D, dtype=bool)

        if use_cond[3]:
            cond_4a = D <= 1 + tol
            cond_4b = P <= 1 + tol
        else:
            cond_4a = np.ones_like(D, dtype=bool)
            cond_4b = cond_4a

        if use_cond[4]:
            cond_5a = M00 * (1 + D) <= 1 + tol
            cond_5b = M00 * (1 + P) <= 1 + tol
        else:
            cond_5a = np.ones_like(D, dtype=bool)
            cond_5b = cond_5a

        if use_cond[5]:
            sum = comp[5]**2 + comp[10]**2 + comp[15]**2 + 2 * (
                comp[1] * comp[4] + comp[2] * comp[8] + comp[3] * comp[12] +
                comp[6] * comp[9] + comp[7] * comp[13] + comp[11] * comp[14])
            cond_6 = sum <= 3 * M00**2 + tol
        else:
            cond_6 = np.ones_like(D, dtype=bool)

        if use_cond[6]:
            aux1 = M00**2 + (1 - D)**2
            aux2 = (comp[1]**2 * (1 - (comp[5] + comp[6] + comp[7]) /
                                  (M00 * D))**2 + comp[2]**2 *
                    (1 - (comp[9] + comp[10] + comp[11]) / (M00 * D))**2 +
                    comp[3]**2 * (1 - (comp[13] + comp[14] + comp[15]) /
                                  (M00 * D))**2)
            cond_7a = aux1 >= aux2
            aux1 = M00**2 + (1 - P)**2
            aux2 = (comp[4]**2 * (1 - (comp[5] + comp[9] + comp[13]) /
                                  (M00 * P))**2 + comp[8]**2 *
                    (1 - (comp[6] + comp[10] + comp[14]) / (M00 * P))**2 +
                    comp[12]**2 * (1 - (comp[7] + comp[11] + comp[15]) /
                                   (M00 * P))**2)
            cond_7b = aux1 >= aux2
        else:
            cond_7a = np.ones_like(D, dtype=bool)
            cond_7b = cond_5a

        if use_cond[7]:
            cond_8a = np.ones_like(D, dtype=bool)
            cond_8b = np.ones_like(D, dtype=bool)
            cond_8c = np.ones_like(D, dtype=bool)
            for elem in vals:
                # print('elements', elem)
                cond_8a *= np.abs(np.imag(elem)) < tol
                # print('result', cond_8a)
                cond_8b *= elem >= -tol
                cond_8c *= elem <= 1 + tol
        else:
            cond_8a = np.ones_like(D, dtype=bool)
            cond_8b = cond_8a
            cond_8c = cond_8a

        # Merge de conditions and multiply
        cond_partial = [
            cond_1, cond_2a, cond_2b, cond_3, cond_4a, cond_4b, cond_5a,
            cond_5b, cond_6, cond_7a, cond_7b, cond_8a, cond_8b, cond_8c
        ]

        cond = np.ones_like(D, dtype=bool)
        for elem in cond_partial:
            cond *= elem
            # If the result is a number and the user asks for it, return a float
            if out_number and elem.size == 1:
                elem = elem[0]
        if out_number and cond.size == 1:
            cond = cond[0]

        # Print the result if required
        if verbose or draw:
            if give_all:
                cond_plot = [cond] + cond_partial
                titles = [
                    'Physical', 'Real elements', 'M00 >= 0', 'M00 <= 1',
                    'abs(Mij) <= M00', 'D <= 1', 'P <= 1', 'Tmax <= 1',
                    'Recip Tmax <= 1', 'Tr(M*M^T) <= 4*M00^2', 'm cond (D)',
                    'm cond (P)', 'Real eigenvalues', 'Eigenvalues >= 0',
                    'Eigenvalues <= 1'
                ]
            else:
                cond_plot = [cond]
                titles = ['Physical']
            heading = '{} is physically realizable:'.format(self.parent.name)
            PrintParam(param=cond_plot,
                       shape=self.parent.shape,
                       title=titles,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        if give_all:
            return cond, cond_partial
        else:
            return cond

    def is_non_depolarizing(self,
                            out_number=True,
                            shape_like=None,
                            shape=None,
                            verbose=False,
                            draw=False):
        """Checks if matrix is pure, i.e., is non-depolarizing (the degree of polarimetric purity must be 1).

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Calculate the parameters
        PP = self.parent.parameters.polarimetric_purity(shape=shape,
                                                        shape_like=shape_like,
                                                        out_number=out_number)
        cond = 1 - PP <= tol_default
        # Print the result if required
        if verbose or draw:
            heading = '{} is pure (non-depolarizing):'.format(self.parent.name)
            PrintParam(param=cond,
                       shape=self.parent.shape,
                       title='Pure',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return cond

    def is_pure(self,
                out_number=True,
                shape_like=None,
                shape=None,
                verbose=False,
                draw=False):
        """Checks if matrix is pure, i.e., is non-depolarizing (the degree of polarimetric purity must be 1).

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Calculate the parameters
        PP = self.parent.parameters.polarimetric_purity(shape=shape,
                                                        shape_like=shape_like,
                                                        out_number=out_number)
        cond = 1 - PP <= tol_default
        # Print the result if required
        if verbose or draw:
            heading = '{} is pure (non-depolarizing):'.format(self.parent.name)
            PrintParam(param=cond,
                       shape=self.parent.shape,
                       title='Pure',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return cond

    def is_homogeneous(self,
                       out_number=True,
                       shape_like=None,
                       shape=None,
                       verbose=False,
                       draw=False):
        """Checks if the matrix is homogeneous, i.e., its two eigenstates are perpendicular. If true, the inhomogeneity parameter must be 0.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 119.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Calculate the parameters
        par = self.parent.parameters.inhomogeneity(shape=shape,
                                                   shape_like=shape_like,
                                                   out_number=out_number)
        cond = par <= tol_default
        # Print the result if required
        if verbose or draw:
            heading = '{} is homogeneous:'.format(self.parent.name)
            PrintParam(param=cond,
                       shape=self.parent.shape,
                       title='Homogeneous',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return cond

    def is_retarder(self,
                    give_all=False,
                    out_number=True,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """Checks if the matrix M corresponds to a pure retarder.There are three
        conditions:

        1. Is physical
        2. Is pure
        3. Diatteunation = 0.
        4. Polarizance = 0.
        5. M must be unitary ($M^{T}=M^{-1}$).

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 129.

        Parameters:
            give_all (bool): If True, the method also gives a list with the individual conditions. Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            cond (numpy.ndarray or bool): Result.
            partial_conditions (list): List with the partial conditions (if give_all is True).
        """
        # Reality and purity
        cond1 = self.is_physical()
        cond2 = self.is_pure()

        # Diattenuation and polarizance
        par = self.parent.parameters.diattenuation(shape=shape,
                                                   shape_like=shape_like,
                                                   out_number=out_number)
        cond3 = par <= tol_default

        par = self.parent.parameters.polarizance(shape=shape,
                                                 shape_like=shape_like,
                                                 out_number=out_number)
        cond4 = par <= tol_default

        # Unitary
        obj_T = self.parent.transpose(keep=True)
        obj_inv = self.parent.inverse(keep=True)
        obj = obj_T - obj_inv
        cond5 = norm_1a = obj_error(obj_T,
                                    obj_inv,
                                    shape=shape,
                                    shape_like=shape_like,
                                    out_number=out_number) < tol_default

        # Reshape
        cond1, cond2, cond3, cond4, cond5 = reshape([cond1, cond2, cond3, cond4, cond5],
                                      shape_like=shape_like,
                                      shape_fun=shape,
                                      obj=self.parent)
        cond = cond1 * cond2 * cond3 * cond4 * cond5

        # Print the result if required
        if verbose or draw:
            if give_all:
                cond_plot = [cond, cond1, cond2, cond3, cond4, cond5]
                titles = ['Retarder', "Physical", "Pure", 'D = 0', 'P = 0', 'Unitary']
            else:
                cond_plot = [cond]
                titles = ['Retarder']
            heading = '{} is a retarder:'.format(self.parent.name)
            PrintParam(param=cond_plot,
                       shape=self.parent.shape,
                       title=titles,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        if give_all:
            return cond, [cond1, cond2, cond3, cond4, cond5]
        else:
            return cond

    def is_diattenuator(self,
                        give_all=False,
                        out_number=True,
                        shape_like=None,
                        shape=None,
                        verbose=False,
                        draw=False):
        """Checks if the matrix M corresponds to a pure homogeneous diattenuator. It must fullfill several conditions:

        1. Is physical
        2. Is pure
        3. Diattenuation > 0.
        4. $M = M^T$.
        5. The eigenstates of M are the Stokes vectors (1, D) and (1, -D).

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 142.

        Parameters:
            give_all (bool): If True, the method also gives a list with the individual conditions. Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            cond (numpy.ndarray or bool): Result.
            partial_conditions (list): List with the partial conditions (if give_all is True).
        """
        # Reality and purity
        cond1 = self.is_physical()
        cond2 = self.is_pure()

        # Calculate the first two conditions
        par = self.parent.parameters.diattenuation(shape=False,
                                                   out_number=False)
        cond3 = par > tol_default
        Dv = self.parent.parameters.diattenuation_vector(shape=False)

        obj_T = self.parent.transpose(keep=True)
        cond4 = obj_error(self.parent,
                          obj_T,
                          shape=shape,
                          shape_like=shape_like,
                          out_number=out_number) < tol_default

        # The third one is more complicated. Start by calculating the diattenuation vector
        D = self.parent.parameters.diattenuation_vector(shape=False, as_Stokes=True)
        # Check if it is eigenstate
        condA = self.is_eigenstate(D)
        D.M[1:,:] = -D.M[1:,:]
        condB = self.is_eigenstate(D)
        cond5 = condA * condB

        # Reshape
        cond1, cond2, cond3, cond4, cond5 = reshape([cond1, cond2, cond3, cond4, cond5],
                                      shape_like=shape_like,
                                      shape_fun=shape,
                                      obj=self.parent)
        # Total condition
        cond = cond1 * cond2 * cond3 * cond4 * cond5

        # Print the result if required
        if verbose or draw:
            if give_all:
                cond_plot = [cond, cond1, cond2, cond3, cond4, cond5]
                titles = [
                    'Diattenuator', "Physical", "Pure", 'D > 0', 'Symetric', 'Correct eigenstates'
                ]
            else:
                cond_plot = [cond]
                titles = ['Diattenuator']
            heading = '{} is a diattenuator:'.format(self.parent.name)
            PrintParam(param=cond_plot,
                       shape=self.parent.shape,
                       title=titles,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        if give_all:
            return cond, [cond1, cond2, cond3, cond4, cond5]
        else:
            return cond

    def is_polarizer(self, *args, **kwargs):
        """Checks if the matrix M corresponds to a pure homogeneous polarizer (diattenuator). It must fullfill several conditions:

        1. Is physical
        2. Is pure
        3. Polarizance > 0.
        4. $M = M^T$.
        5. The eigenstates of M are the Stokes vectors (1, P) and (1, -P).

        Note: This method is the same as is_diattenuator.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 142.

        Parameters:
            give_all (bool): If True, the method also gives a list with the individual conditions. Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            cond (numpy.ndarray or bool): Result.
            partial_conditions (list): List with the partial conditions.
        """
        return self.transpose(keep=True).is_diattenuator(*args, **kwargs)

    def is_depolarizer(self,
                       give_all=False,
                       out_number=True,
                       shape_like=None,
                       shape=None,
                       verbose=False,
                       draw=False):
        """Checks if the matrix M corresponds to a depolarizer. It must fullfill several conditions:

        1. Is physical
        2. Depolarization index > 0.
        3. $m = m^T$ (m being the small m matrix).

        References:
            S. Y. Lu, R. A. Chipman; "Interpretation of Mueller matrices based on polar decomposition"; J. Opt. Soc. Am. A/Vol. 13, No. 5 (1996)

        Parameters:
            give_all (bool): If True, the method also gives a list with the individual conditions. Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            cond (numpy.ndarray or bool): Result.
            partial_conditions (list): List with the partial conditions.
        """
        # Reality
        cond1 = self.is_physical()
        # Calculate the parameters
        inhom = self.parent.parameters.depolarization_index(
            shape=shape, shape_like=shape_like, out_number=False)
        m = self.parent.parameters.small_matrix(shape=False)
        # First condition
        cond2 = inhom > tol_default
        # Second condition
        mT = np.transpose(m, axes=(1, 0, 2))
        cond3 = np.linalg.norm(m - mT, axis=(0, 1)) < tol_default
        # Reshape
        cond1, cond2, cond3 = reshape([cond1, cond2, cond3],
                                      shape_like=shape_like,
                                      shape_fun=shape,
                                      obj=self.parent)
        # Total condition
        cond = cond1 * cond2 * cond3
        # Draw
        if verbose or draw:
            if give_all:
                cond_plot = [cond, cond1, cond2, cond3]
                titles = ['Depolarizer', "Physical", 'Depol. index > 0', 'm is symetric']
            else:
                cond_plot = [cond]
                titles = ['Depolarizer']
            heading = '{} is a depolarizer:'.format(self.parent.name)
            PrintParam(param=cond_plot,
                       shape=self.parent.shape,
                       title=titles,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        if give_all:
            return cond, [cond1, cond2, cond3]
        else:
            return cond

    def is_singular(self,
                    out_number=True,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """Checks if the matrix is singular (det(M) = 0).

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 282.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Calculate the parameters
        par = self.parent.parameters.det(shape=shape,
                                         shape_like=shape_like,
                                         out_number=out_number)
        cond = par <= tol_default
        # Print the result if required
        if verbose or draw:
            heading = '{} is singular:'.format(self.parent.name)
            PrintParam(param=cond,
                       shape=self.parent.shape,
                       title='Singular',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return cond

    def is_symmetric(self,
                     out_number=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """
        Determines if the object matrix is symmetric (i.e. $$M = M^T$$).

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Calculate the condition
        obj_T = self.parent.transpose(keep=True)
        cond = obj_error(self.parent,
                         obj_T,
                         shape=shape,
                         shape_like=shape_like,
                         out_number=out_number) < tol_default
        # Reshape if neccessary
        cond = reshape([cond],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is symmetric:'.format(self.parent.name)
            PrintParam(param=cond,
                       shape=self.parent.shape,
                       title='Symmetric',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return cond

    def is_eigenstate(self, S, out_number=True, shape_like=None, shape=None, verbose=False, draw=False):
        """
        Determines if the vector S is an eigenstate of the object.

        Parameters:
            S (Stokes or Jones_vector): State to test.
            out_number (bool): if True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Multiply
        Sout = self.parent * S
        # Check if S and Sout are proportional
        prop = Sout.M[0,:] / S.M[0,:]
        cond1 = np.abs(Sout.M[1,:] / S.M[1,:] - prop) < tol_default
        cond2 = np.abs(Sout.M[2,:] / S.M[2,:] - prop) < tol_default
        cond3 = np.abs(Sout.M[3,:] / S.M[3,:] - prop) < tol_default
        cond = cond1 * cond2 * cond3
        # Reshape if neccessary
        cond = reshape([cond],
              shape_like=shape_like,
              shape_fun=shape,
              obj=self.parent)
        # Print the result if required
        if verbose or draw:
           heading = '{} is an eigenstate of {}:'.format(S.name, self.parent.name)
           PrintParam(param=cond,
                      shape=self.parent.shape,
                      title='Eigenstate',
                      heading=heading,
                      verbose=verbose,
                      draw=draw)

        return cond
