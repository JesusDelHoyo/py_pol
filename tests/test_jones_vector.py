# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Jones_vector module"""

import sys

import numpy as np
from numpy import array

from py_pol import degrees, eps
from py_pol.jones_vector import Jones_vector, create_Jones_vectors
from py_pol.stokes import Stokes
from py_pol.utils import comparison


class TestJonesVector(object):
    def test_sum(self):

        solution = array([[1.], [1.]])
        j1, j2 = create_Jones_vectors(('j1', 'j2'))
        print(j1, j2)
        j1.linear_light(azimuth=0 * degrees)
        j2.linear_light(azimuth=90 * degrees)
        j3 = j1 + j2
        proposal = j3.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: + 0D"

        solution = array([[1., 1., 1.], [1., 1., 1.]])
        j1, j2 = create_Jones_vectors(('j1', 'j2'))
        j1.linear_light(azimuth=0 * degrees)
        j2.linear_light(azimuth=90 * degrees, length=3)
        j3 = j1 + j2
        proposal = j3.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D + 1D"

        solution = array([[1., 1., 1.], [1., 1., 1.]])
        j1, j2 = create_Jones_vectors(('j1', 'j2'))
        j1.linear_light(azimuth=0 * degrees, length=3)
        j2.linear_light(azimuth=90 * degrees, length=3)
        j3 = j1 + j2
        proposal = j3.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D + 1D"

        solution = (2, 2)
        j1, j2 = create_Jones_vectors(('j1', 'j2'))
        j1.linear_light(azimuth=0 * degrees, length=4)
        j2.linear_light(azimuth=90 * degrees, length=4, shape=(2, 2))
        j3 = j1 + j2
        proposal = j3.shape
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 1D + 1D shape"

    def test_substraction(self):

        solution = array([[1.], [-1.]])
        j1, j2 = create_Jones_vectors(('j1', 'j2'))
        j1.linear_light(azimuth=0 * degrees)
        j2.linear_light(azimuth=90 * degrees)
        j3 = j1 - j2
        proposal = j3.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: - 0D"

        solution = array([[1., 1., 1.], [-1., -1., -1.]])
        j1, j2 = create_Jones_vectors(('j1', 'j2'))
        j1.linear_light(azimuth=0 * degrees)
        j2.linear_light(azimuth=90 * degrees, length=3)
        j3 = j1 - j2
        proposal = j3.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D - 1D"

        solution = array([[1., 1., 1.], [-1., -1., -1.]])
        j1, j2 = create_Jones_vectors(('j1', 'j2'))
        j1.linear_light(azimuth=0 * degrees, length=3)
        j2.linear_light(azimuth=90 * degrees, length=3)
        j3 = j1 - j2
        proposal = j3.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D - 1D"

        solution = (2, 2)
        j1, j2 = create_Jones_vectors(('j1', 'j2'))
        j1.linear_light(azimuth=0 * degrees, length=4)
        j2.linear_light(azimuth=90 * degrees, length=4, shape=(2, 2))
        j3 = j1 - j2
        proposal = j3.shape
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 1D - 1D shape"

    def test_multiplication(self):

        solution = array([[2], [0.]])
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=0 * degrees)
        j2 = 2 * j1
        proposal = j2.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2*J"

        solution = array([[2], [0.]])
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=0 * degrees)
        j2 = j1 * 2
        proposal = j2.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: J*2"

        solution = array([[3], [2]])
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=0 * degrees)
        j2 = Jones_vector('j2')
        j2.linear_light(azimuth=90 * degrees)
        j3 = 3 * j1 + 2 * j2
        proposal = j3.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 3 * j1 + 2 * j2"

        solution = array([[2, 2, 2], [0, 0, 0]])
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=0 * degrees, length=3)
        j2 = 2 * j1
        proposal = j2.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: * 1D"

        solution = array([[2, 2, 2], [0, 0, 0]])
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=0 * degrees)
        a = 2 * np.ones(3)
        j2 = a * j1
        proposal = j2.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D * J"

        solution = (2, 2)
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=90 * degrees, length=4)
        a = 2 * np.ones((2, 2))
        j2 = a * j1
        proposal = j2.shape
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 2D * 2D shape"

    def test_division(self):

        solution = array([[0.5], [0.]])
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=0 * degrees)
        j2 = j1 / 2
        proposal = j2.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: J/2"

        solution = array([[0.5, 0.5, 0.5], [0, 0, 0]])
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=0 * degrees, length=3)
        j2 = j1 / 2
        proposal = j2.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: / 1D"

        solution = array([[0.5, 0.5, 0.5], [0, 0, 0]])
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=0 * degrees)
        a = 2 * np.ones(3)
        j2 = j1 / 2
        proposal = j2.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: J / 1D"

        solution = (2, 2)
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=90 * degrees, length=4)
        a = 2 * np.ones((2, 2))
        j2 = j1 / a
        proposal = j2.shape
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 2D / 2D shape"

    def test_rotate(self):

        solution = array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]])
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=0 * degrees)
        j1.rotate(angle=45 * degrees)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 45*degrees"

        r2 = 1 / np.sqrt(2)
        solution = array([[r2, r2, r2], [r2, r2, r2]])
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=0 * degrees, length=3)
        j1.rotate(angle=45 * degrees)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 45*degrees 1D"

        solution = (2, 2)
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=0 * degrees, length=4, shape=(2, 2))
        j1.rotate(angle=45 * degrees)
        proposal = j1.shape
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 45*degrees shape"

    def test_clear(self):

        solution = array([[0.], [0.]])
        j1 = Jones_vector('j1')
        j1.linear_light(azimuth=0 * degrees, length=5)
        j1.clear()
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: clear"

    def test_from_components(self):

        solution = array([[1.], [0.]])
        j1 = Jones_vector()
        j1.from_components(1, 0)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (1,0)"

        solution = array([[2], [2 + 2j]])
        j1 = Jones_vector()
        j1.from_components(2, 2 + 2j)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (2, 2j)"

        solution = array([np.ones(5), np.zeros(5)])
        j1 = Jones_vector()
        j1.from_components(np.ones(5), 0)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (1D, 0)"

        solution = (2, 2)
        j1 = Jones_vector()
        j1.from_components(np.ones((2, 2)), np.ones((2, 2)))
        proposal = j1.shape
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (1D, 0)"

    def test_from_matrix(self):

        solution = array([[1.], [0.]])
        J1 = np.array([[1.], [0.]])
        j1 = Jones_vector()
        j1.from_matrix(J1)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (1,0)"

        solution = array([np.ones(5), np.zeros(5)])
        J1 = [np.ones(5), np.zeros(5)]
        j1 = Jones_vector()
        j1.from_matrix(J1)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (1D,1D)"

    def test_from_Stokes(self):
        pass  # TODO:

        # solution = array([[1.], [0.]])
        # S = Stokes()
        # S.from_components(1, 1, 0, 0)
        # j1 = Jones_vector('J1')
        # j1.from_Stokes(S)
        # proposal = j1.parameters.matrix()
        # assert comparison(
        #     proposal, solution, eps
        # ), sys._getframe().f_code.co_name + ".py --> example: (1, 1, 0, 0)"
        #
        # solution = array([[0.], [1.]])
        # S = Stokes()
        # S.from_components(1, -1, 0, 0)
        # j1 = Jones_vector('J1')
        # j1.from_Stokes(S)
        # proposal = j1.parameters.matrix()
        # assert comparison(
        #     proposal, solution, eps
        # ), sys._getframe().f_code.co_name + ".py --> example: (1, -1, 0, 0)"
        #
        # solution = array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]])
        # S = Stokes()
        # S.from_components(1, 0, 1, 0)
        # j1 = Jones_vector('J1')
        # j1.from_Stokes(S)
        # proposal = j1.parameters.matrix()
        # assert comparison(
        #     proposal, solution, eps
        # ), sys._getframe().f_code.co_name + ".py --> example: (1, 0, 1, 0)"
        #
        # solution = array([[-1 / np.sqrt(2)], [1 / np.sqrt(2)]])
        # # signed changed for assert, but it works like this.
        # S = Stokes()
        # S.from_components(1, 0, -1, 0)
        # j1 = Jones_vector('J1')
        # j1.from_Stokes(S)
        # proposal = j1.parameters.matrix()
        # assert comparison(
        #     proposal, solution, eps
        # ), sys._getframe().f_code.co_name + ".py --> example: (1, 0, -1, 0)"
        #
        # solution = array([[1. / np.sqrt(2)], [1j / np.sqrt(2)]])
        # S = Stokes()
        # S.from_components(1, 0, 0, 1)
        # j1 = Jones_vector('J1')
        # j1.from_Stokes(S)
        # proposal = j1.parameters.matrix()
        # assert comparison(
        #     proposal, solution, eps
        # ), sys._getframe().f_code.co_name + ".py --> example: (1, 0, 0, 1)"
        #
        # solution = array([[1. / np.sqrt(2)], [-1j / np.sqrt(2)]])
        # S = Stokes()
        # S.from_components(1, 0, 0, -1)
        # j1 = Jones_vector('J1')
        # j1.from_Stokes(S)
        # proposal = j1.parameters.matrix()
        # assert comparison(
        #     proposal, solution, eps
        # ), sys._getframe().f_code.co_name + ".py --> example: (1, 0, 0, -1)"

    def test_linear_light(self):

        solution = array([[1.], [0.]])
        j1 = Jones_vector()
        j1.linear_light(azimuth=0 * degrees)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0*degrees"

        solution = array([[1, 1 / np.sqrt(2), 0], [0, 1 / np.sqrt(2), 1]])
        j1 = Jones_vector()
        j1.linear_light(azimuth=[0 * degrees, 45 * degrees, 90 * degrees])
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D"

        solution = (2, 2)
        j1 = Jones_vector()
        j1.linear_light(azimuth=np.ones((2, 2)))
        proposal = j1.shape
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: shape"

    def test_circular_light(self):

        solution = array([[1.], [1j]])
        j1 = Jones_vector()
        j1.circular_light(amplitude=np.sqrt(2), kind='r')
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 'r'"

        solution = array([[1.], [-1j]])
        j1 = Jones_vector()
        j1.circular_light(amplitude=np.sqrt(2), kind='l')
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 'l'"

        solution = array([[1, 2, 3], [1j, 2j, 3j]])
        j1 = Jones_vector()
        j1.circular_light(amplitude=np.sqrt(2) * np.array([1, 2, 3]), kind='r')
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D"

        solution = (2, 2)
        j1 = Jones_vector()
        j1.circular_light(amplitude=np.ones((2, 2)))
        proposal = j1.shape
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: shape"

    def test_elliptical_light(self):

        solution = array([[1.], [1j]])
        j1 = Jones_vector()
        j1.elliptical_light(a=1, b=1, azimuth=0 * degrees)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (1,1,0)"

        solution = array([[1.], [1j]])
        j1 = Jones_vector()
        j1.elliptical_light(a=1, b=1, azimuth=90 * degrees)
        j1.remove_global_phase()
        proposal = j1.parameters.matrix()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + ".py --> example: (1,1,90*degrees)"

        solution = array([[1], [-1j]])
        j1 = Jones_vector()
        j1.elliptical_light(kind='l')
        proposal = j1.parameters.matrix()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + ".py --> example: (1, 1, 0, left)"

        solution = array([[1j], [1j]])
        j1 = Jones_vector()
        j1.elliptical_light(
            a=np.sqrt(2), b=0, global_phase=90 * degrees, azimuth=45 * degrees)
        proposal = j1.parameters.matrix()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + ".py --> example: sqrt(2),0, 90 * degrees, 45 * degrees"

        a = np.array([1, 2, 3])
        solution = array([[1, 2, 3], [1j, 2j, 3j]])
        j1 = Jones_vector()
        j1.elliptical_light(
            a=a, b=a, azimuth=0 * degrees)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D"

        solution = (2, 2)
        j1 = Jones_vector()
        j1.elliptical_light(a=np.ones((2, 2)))
        proposal = j1.shape
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: shape"

    def test_general_azimuth_ellipticity(self):

        solution = array([[1.], [0.]])
        j1 = Jones_vector()
        j1.general_azimuth_ellipticity(
            azimuth=0 * degrees, ellipticity=0, amplitude=1)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (0,0,1)"

        solution = array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]])
        j1 = Jones_vector()
        j1.general_azimuth_ellipticity(
            azimuth=45 * degrees, ellipticity=0, amplitude=1)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (45,0,1)"

        solution = array([[0.5 - 0.5j], [0.5 + 0.5j]])
        j1 = Jones_vector()
        j1.general_azimuth_ellipticity(
            azimuth=45 * degrees, ellipticity=45 * degrees, amplitude=1)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (45,45,1)"

        solution = array([[0], [1j]])
        j1 = Jones_vector()
        j1.general_azimuth_ellipticity(
            azimuth=0 * degrees, ellipticity=90 * degrees, amplitude=1)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (0,90,1)"

        solution = array([[1., 1 / np.sqrt(2), 0.5 - 0.5j, 0],
                          [0., 1 / np.sqrt(2), 0.5 + 0.5j, 1j]])
        j1 = Jones_vector()
        j1.general_azimuth_ellipticity(
            azimuth=np.array([0, 45 * degrees, 45 * degrees, 0]),
            ellipticity=np.array([0, 0, 45 * degrees, 90 * degrees]),
            amplitude=1)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D"

        solution = (2, 2)
        j1 = Jones_vector()
        j1.general_azimuth_ellipticity(azimuth=np.ones((2, 2)))
        proposal = j1.shape
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: shape"

    def test_general_charac_angles(self):

        solution = array([[1.], [0.]])
        j1 = Jones_vector()
        j1.general_charac_angles(alpha=0, delay=0, amplitude=1)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (0,0,1)"

        solution = array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]])
        j1 = Jones_vector()
        j1.general_charac_angles(alpha=45 * degrees, delay=0, amplitude=1)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (45,0,1)"

        solution = array([[0.5 - 0.5j], [0.5 + 0.5j]])
        j1 = Jones_vector()
        j1.general_charac_angles(
            alpha=45 * degrees, delay=90 * degrees, amplitude=1)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (45,90,1)"

        solution = array([[1., 1 / np.sqrt(2), 0.5 - 0.5j],
                          [0., 1 / np.sqrt(2), 0.5 + 0.5j]])
        j1 = Jones_vector()
        j1.general_charac_angles(
            alpha=np.array([0, 45 * degrees, 45 * degrees]),
            delay=np.array([0, 0, 90 * degrees]),
            amplitude=1)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D"

        solution = (2, 2)
        j1 = Jones_vector()
        j1.general_charac_angles(alpha=np.ones((2, 2)))
        proposal = j1.shape
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: shape"

    def test_copy(self):

        j1 = Jones_vector()
        j1.general_charac_angles(
            alpha=np.array([0, 45 * degrees, 45 * degrees]),
            delay=np.array([0, 0, 90 * degrees]),
            amplitude=1)
        solution = j1.parameters.matrix()
        j2 = j1.copy()
        proposal = j2.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D"

    def test_normalize(self):

        solution = np.ones(5)
        j1 = Jones_vector()
        j1.linear_light(amplitude=np.random.rand(5))
        j1.normalize()
        proposal = j1.parameters.intensity()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D"

    def test_rotate_to_azimuth(self):

        j1 = Jones_vector()
        j1.linear_light(length=5)
        solution = j1.parameters.matrix()
        j1.linear_light(azimuth=np.random.rand(5))
        j1.rotate_to_azimuth(0)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: rand 2 X"

        angle = np.random.rand(5)
        j1 = Jones_vector()
        j1.linear_light(azimuth=angle)
        solution = j1.parameters.matrix()
        j1.linear_light(length=5)
        j1.rotate_to_azimuth(azimuth=angle)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: X 2 rand"

    def test_add_global_phase(self):

        phase = np.random.rand(5)
        j1 = Jones_vector()
        j1.circular_light(length=5, global_phase=phase)
        solution = j1.parameters.matrix()
        j1.circular_light(length=5)
        j1.add_global_phase(phase)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D"

    def test_remove_global_phase(self):

        phase = np.random.rand(5)
        j1 = Jones_vector()
        j1.circular_light(length=5)
        solution = j1.parameters.matrix()
        j1.circular_light(length=5, global_phase=phase)
        j1.remove_global_phase()
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D"

    def test_add_global_phase(self):

        alpha = np.random.rand(5)
        delay = np.random.rand(5)
        j1 = Jones_vector()
        j1.general_charac_angles(alpha=alpha, delay=delay)
        j1.add_global_phase(phase=delay / 2)
        solution = j1.parameters.matrix()
        j1.general_charac_angles(alpha=alpha)
        j1.add_delay(delay=delay)
        proposal = j1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D"
