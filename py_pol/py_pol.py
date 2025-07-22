# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------
# Authors:    Luis Miguel Sanchez Brea and Jesus del Hoyo
# Date:       2019/01/09 (version 1.0)
# License:    GPL
# -------------------------------------
"""
This is the file with the base classes all py_pol main classes will inherit from. They will have some basic methods common to all classes. These classes is not intended to use except for inheritance.

**Class fields**
    * **M**: Array containing the physical information of the object.
    * **name**: Name of the object for print purposes.
    * **shape**: Shape desired for the outputs.
    * **size**: Number of stores Jones vectors.
    * **type**: Type of the object. This is used for determining the object class as using isinstance may throw unexpected results in .ipynb files.

**Manipulation methods**
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
"""

from copy import deepcopy

from .__init__ import np, degrees, shapes, sizes

#############################
## DEFAULTS (TODO)
#############################

global options
options = {}
options["change_names"] = False
options["keep"] = False

def get_options():
    """TODO"""
    return options

def set_option(key, value):
    """TODO"""
    options[key] = value


############################
## CLASSES
###########################

class Py_pol(object):
    """Basic class where all py_pol main classes will inherit from."""

    #######################
    ## RESERVED METHODS
    #######################

    def __init__(self, name="", _class="Py_pol"):
        """Triggers during the Stokes inicialization..

        Parameters:
            name (string): Name of the object for representation purposes. Default: "".

        Returns:
            (Py_pol):
        """
        self._type = _class
        self.name = name
        self.M = np.zeros(shapes[_class] + [1])
        self.shape = None


    def __iter__(self):
        """Call to iterator class."""
        return Py_pol_Iterator(self)

    def __len__(self):
        """
        Gives the size of the object.
        """
        return self.size

    ########################
    ## PROPERTIES
    #######################

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def type(self):
        return self._type

    # @type.setter
    # def type(self, value):
    #     raise ValueError("This prop")

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        # Safety checks
        if self.type in ("Jones_vector", "Stokes") and value.ndim != 2:
            if value.ndim == 1 and value.size == sizes[self.type]:
                value = np.reshape(value, shapes[self.type])
            else:
                raise ValueError("M must be a matrix of 2 dimensions ({} used)".format(value.ndim))
        elif self.type in ("Jones_matrix", "Mueller") and value.ndim != 3:
            if value.ndim == 2 and value.size == sizes[self.type]:
                value = np.reshape(value, shapes[self.type])
            else:
                raise ValueError("M must be a matrix of 3 dimensions ({} used)".format(value.ndim))
        if self.type == "Jones_vector" and value.shape[0] != 2:
            raise ValueError("Jones_vector M must be of shape 2xN ({} used)".format(value.shape))
        elif self.type == "Stokes" and value.shape[0] != 4:
            raise ValueError("Stokes M must be of shape 4xN ({} used)".format(value.shape))
        elif self.type == "Jones_matrix" and (value.shape[0] != 2 or value.shape[1] != 2):
            raise ValueError("Jones_matrix M must be of shape 2x2xN ({} used)".format(value.shape))
        elif self.type == "Mueller" and (value.shape[0] != 4 or value.shape[1] != 4):
            raise ValueError("Mueller M must be of shape 4x4xN ({} used)".format(value.shape))

        # Store old shapein case it could be used
        try:
            old_shape = self._shape
            old_size = self._size
        except:
            old_shape = None
            old_size = 0

        # Set values
        if self.type in ("Jones_vector", "Jones_matrix"):
            self._M = np.array(value, dtype=complex)
        else:
            self._M = value
        self._size = value.shape[-1]
        if self.size == old_size:
            self.shape = old_shape
        else:
            self.shape = [self.size]
            # self._ndim = 1


    @property
    def size(self):
        return self._size

    # @size.setter
    # def size(self, value):
    #     N = self.M.shape[-1]
    #     if value != N:
    #         raise ValueError("Size {} different than number of elements {}".format(value, N))
    #     self._size = value


    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if value is None:
            self._shape = [self.size]
            self._ndim = 0
        else:
            N = np.prod(value)
            if N != self.size:
                raise ValueError("Shape {} can't be applied to object of size {}.".format(value, self.size))
            self._shape = list(value)
            if self.size <=1:
                self._ndim = 0
            else:
                self._ndim = len(value)

    @property
    def ndim(self):
        return self._ndim

    # @ndim.setter
    # def ndim(self, value):
    #     if self.size == 1 or self.shape is None:
    #         self._ndim = 0
    #     else:
    #         N = len(self.shape)
    #         if value != N:
    #             raise ValueError("Dimension {} different than corresponding to shape {}".format(value, self.shape))
    #         self._ndim = value

    ###################
    ## MANIPULATION
    ###################
    def get_list(self, out_number=True, shape_vectors=True):
        """Returns a list of np.ndarrays. Each array is a vector or matrix corresponding with one element or light source.

        Parameters:
            out_number (bool): if True and the object is size 1, return an array instead of a list. Default: True.
            shape_vectors (bool): If True and the object is Jones_vector or Stokes, the output arrays will have dimension 2x1 and 4x1 instead 2 and 4 respectively. Default: True.

        Returns:
            (numpy.ndarray or list): Created object.
        """
        # Calculate array shape
        shape = shapes[self.type]
        if shape_vectors and self.type in ("Jones_vector", "Stokes"):
            shape = shape + [1]
        # Make the list
        list = []
        components = self.parameters.components(shape=False, out_number=False)
        for indL in range(self.size):
            a = np.zeros(sizes[self.type], dtype=self.M.dtype)
            for indA, comp in enumerate(components):
                a[indA] = comp[indL]
            list.append(np.resize(a, shape))
        # Return
        if out_number and self.size == 1:
            list = list[0]
        return list

    def from_list(self, l, shape_like=None, shape=None):
        """Create a Py_pol object from a list of numpy arrays of the correct size.

        Parameters:
            l (list): list of np.ndarrays, lists, tuples or Py_pol objects.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Py_pol): Created object.
        """
        Saux = self.copy()
        for ind, elem in enumerate(l):
            # Get the data in an ordered way from any shape
            if isinstance(elem, (np.ndarray, tuple, list)):
                Saux.from_matrix(elem)
            elif self.type == elem.type:
                Saux = elem
            else:
                raise ValueError("New element of type {} can't be added to an object of type {}.".format(elem.type, self.type))
            # Concatenate
            if ind == 0:
                M = Saux.M
            else:
                M = np.hstack((M, Saux.M))


        # # Preallocate memory
        # if isinstance(l[0], (np.ndarray, tuple, list)):
        #     M = np.array(l[0])
        #     if M.ndim == 1:
        #         M = M.reshape((M.size, 1))
        # else:
        #     M = l[0].M
        # # Fill it
        # for elem in l[1:]:
        #     if isinstance(elem, (np.ndarray, tuple, list)):
        #         Maux = np.array(elem)
        #         if Maux.ndim == 1:
        #             Maux = Maux.reshape((Maux.size, 1))
        #         M = np.hstack((M, Maux))
        #     elif self.type == elem.type:
        #         M = np.hstack((M, elem.M))
        #     else:
        #         raise ValueError("New element of type {} can't be added to an object of type {}.".format(elem.type, self.type))
        # Update
        self.from_matrix(M, shape=shape, shape_like=shape_like)
        return self

    def concatenate(self, objs, shape_like=None, shape=None, keep=False, change_name=options["change_names"]):
        """Create a Py_pol object from an iterable of Py_pol objects.

        Parameters:
            objs (iterable): iterable of Py_pol objects.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            keep (bool): if True, the original element is not updated. Default: False.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Py_pol): Created object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Preallocate
        M = new_obj.M
        axis = 1 if new_obj.type in ("Jones_vector", "Stokes") else 2

        # Concatenate
        for elem in objs:
            # Safety check
            if elem.type != new_obj.type:
                raise ValueError("Object {} is type {} instead of {}.".format(elem.name, elem.type, new_obj.type))
            # Get data
            M = np.concatenate((M, elem.M), axis=axis)

        # Create
        new_obj.from_matrix(M, shape=shape, shape_like=shape_like)

        return new_obj


    def flip(self, axis=None, keep=options["keep"], change_name=options["change_names"]):
        """Flips the order of the elements stored in the object.

        Parameters:
            axis (int, list or tuple): Axes along which the flip is performed. If None, the object is flipped as flattened. Default: None.
            keep (bool): if True, the original element is not updated. Default: False.
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
            new_list = new_obj.get_list(out_number=False)
            new_list.reverse()
            new_obj.from_list(new_list)
        else:
            # Divide in components
            components = new_obj.parameters.components()
            # Flip each one individually
            for ind in range(len(components)):
                components[ind] = np.flip(components[ind], axis=axis)
            # Use them to create the new object
            new_obj.from_components(components)
        # End operations
        if change_name:
            new_obj.name = 'Flip of ' + new_obj.name
        new_obj.shape = self.shape
        return new_obj

    def stretch(self, length, keep=options["keep"]):
        """Function that stretches an object with a single element to have a higher number of equal elements.

        Parameters:
            length (int): Number of elements.
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Jones_vector): Recalculated Jones vector.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Act only if neccessary
        if new_obj.size == 1 and length > 1:
            # Get components
            components = new_obj.parameters.components(out_number=True)
            for ind in range(len(components)):
                components[ind] = components[ind] * np.ones(length)
            # Use them to create the new object
            if new_obj._type in ("Stokes", "Mueller"):
                new_obj.from_components(components, global_phase=new_obj.parameters.global_phase())
            else:
                new_obj.from_components(components)
        # Return
        return new_obj

    def copy(self, N=1):
        """Creates a copy of the object.

        Parameters:
            N (int): Number of copies. Default: 1.

        Returns:
            (Py_pol): Copied object.
        """
        if N <= 1:
            return deepcopy(self)
        else:
            E = []
            for ind in range(N):
                E.append(deepcopy(self))
            return E

    def reshape(self, shape):
        """Changes the shape of the object.

        Parameter:
            shape (tuple, list or 1-D np.ndarray): New shape.
        """
        self.shape = shape
        return self

    def shape_like(self, obj):
        """Takes the shape of another object.

        Parameter:
            obj (Py_pol or nd.array): Object to take the shape.
        """
        # Check that the new shape can be used
        if obj.shape is not None:
            if np.prod(obj.shape) != self.size:
                raise ValueError(
                    'The number of elements of {} and object are not the same'.
                    format(self.name))
        self.shape = obj.shape
        return self

    def flatten(self, keep=False):
        """Method that flattens the objcet (transforms N-D objects in 1D objects if N>=1).

        Parameters:
            keep (bool): If True, self is not updated. Default: False.

        Returns:
            (Py_pol): Flattened object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Flatten
        new_obj.shape = [new_obj.size]
        return new_obj

    def draw(self, verbose=True, shape_like=None, shape=None, kind='jones'):
        """Draw the components of the object. This is a wrap of parameters.components.

        Parameters:
            verbose (bool): if True prints the parameter. Default: False.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
        """
        if self.type in ("Jones_vector", "Jones_matrix"):
            self.parameters.components(draw=True, verbose=verbose, shape=shape, shape_like=shape_like,kind=kind)
        else:
            self.parameters.components(draw=True, verbose=verbose, shape=shape, shape_like=shape_like)


    def clear(self):
        """Removes data and name form the object.
        """
        self.__init__(self.name)


class Py_pol_Iterator:
    """Iterator of the Py_pol classes."""
    def __init__(self, object):
        """Inicialize the instance."""
        self._object = object
        self._index = 0

    def __next__(self):
        """Returns the next value of the iteration."""
        # Calculate length
        if self._object.shape is None:
            length = self._object.size
        else:
            length = self._object.shape[0]

        # Pick the result
        if self._index >= length:
            raise StopIteration
        else:
            self._index += 1
            return self._object[self._index-1,...]
    
    def __iter__(self):
        """Returns the iterator."""
        return self
