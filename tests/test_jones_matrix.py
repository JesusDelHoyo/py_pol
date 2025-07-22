# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Jones_matrix module"""

import sys

import numpy as np
from numpy import matrix

from py_pol import degrees, eps, um
from py_pol.jones_matrix import Jones_matrix
from py_pol.jones_vector import Jones_vector
from py_pol.utils import comparison


class TestJonesMatrix(object):
    def test_multiplication(self):

        solution = matrix([[2, 0], [0, 0]])
        M1 = Jones_matrix('M1')
        M1.from_elements(1, 1, 0, 0)
        M2 = Jones_matrix('M2')
        M2.from_elements(1, 0, 1, 0)
        M3 = M1 * M2
        proposal = M3.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: M1*M2"

        solution = matrix([[1, 1], [1, 1]])
        M1 = Jones_matrix('M1')
        M1.from_elements(1, 1, 0, 0)
        M2 = Jones_matrix('M2')
        M2.from_elements(1, 0, 1, 0)
        M3 = M2 * M1
        proposal = M3.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: M2*M1"

        solution = matrix([[3, 3], [0, 0]])
        M1 = Jones_matrix('M1')
        M1.from_elements(1, 1, 0, 0)
        M4 = 3 * M1
        proposal = M4.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 3 * M1"

        solution = matrix([[3, 3], [0, 0]])
        M1 = Jones_matrix('M1')
        M1.from_elements(1, 1, 0, 0)
        M5 = M1 * 3
        proposal = M5.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: M1*3"

        solution = matrix([[5, 2], [3, 0]])
        M1 = Jones_matrix('M1')
        M1.from_elements(1, 1, 0, 0)
        M2 = Jones_matrix('M2')
        M2.from_elements(1, 0, 1, 0)
        M6 = 2 * M1 + 3 * M2
        proposal = M6.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2*M1+3*M2"

    def test_multiplication_Matrix_vector(self):

        solution = matrix([[1 + 1j], [0]])

        M1 = Jones_matrix('M1')
        M1.from_elements(1, 1, 0, 0)

        J1 = Jones_vector('J1')
        J1.from_elements(1, 1j)
        J2 = M1 * J1
        proposal = J2.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2*M1+3*M2"

    def test_rotate(self):

        solution = matrix([[0.5, 0.5], [0.5, 0.5]])
        M1 = Jones_matrix('M_rot')
        M1.diattenuator_perfect(angle=0 * degrees)
        M1.rotate(45 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 45*degrees"

    def test_from_elements(self):

        solution = matrix([[1, 0], [0, 0]])
        M1 = Jones_matrix()
        M1.from_elements(1, 0, 0, 0)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: (1, 0, 0 ,0)"

        solution = matrix([[1, 0], [0, 1j]])
        M1 = Jones_matrix()
        M1.from_elements(1, 0, 0, 1j)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: (1, 0, 0, 1j)"

    def test_from_matrix(self):

        solution = matrix([[1, 2], [3, 4j]])
        M1 = Jones_matrix()
        M1.from_matrix(solution)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: (1,2,3,4j)"

    def test_neutral_element(self):

        solution = matrix([[0.25, 0], [0, 0.25]])
        M1 = Jones_matrix('M_neutral')
        M1.filter_amplifier(D=.25)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0.25"

    def test_linear_polarizer(self):

        solution = matrix([[1, 0], [0, 0]])
        M1 = Jones_matrix('M_linear')
        M1.diattenuator_perfect(angle=0 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0*degrees"

        solution = matrix([[0.5, 0.5], [0.5, 0.5]])
        M1 = Jones_matrix('M_linear')
        M1.diattenuator_perfect(angle=45 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 45*degrees"

        solution = matrix([[0, 0], [0, 1]])
        M1 = Jones_matrix('M_linear')
        M1.diattenuator_perfect(angle=90 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 90*degrees"

    def test_diattenuator_linear(self):

        solution = matrix([[1, 0], [0, 0]])
        M1 = Jones_matrix('M_linear')
        M1.diattenuator_linear(p1=1, p2=0, angle=0)
        proposal = M1.parameters.matrix()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + ".py --> example: p1=1, p2=0, angle=0"

        solution = matrix([[0.75, 0.0], [0.0, 0.25]])
        M1 = Jones_matrix('M_linear')
        M1.diattenuator_linear(p1=.75, p2=0.25, angle=0 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + ".py --> p1=.75, p2=0.25, angle=0 * degrees"

        solution = matrix([[0.25, 0.0], [0.0, 0.75]])
        M1 = Jones_matrix('M_linear')
        M1.diattenuator_linear(p1=.75, p2=0.25, angle=90 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + ".py --> p1=.75, p2=0.25, angle=90 * degrees"

        solution = matrix([[0.5, 0.25], [0.25, 0.5]])
        M1 = Jones_matrix('M_linear')
        M1.diattenuator_linear(p1=.75, p2=0.25, angle=45 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + ".py --> p1=.75, p2=0.25, angle=45 * degrees"

    def test_retarder_linear(self):

        solution = matrix([[1, 0], [0, 1]])
        M1 = Jones_matrix('retarder_linear')
        M1.retarder_linear(D=0, angle=0)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: no retarder"

        solution = matrix([[1, 0], [0, 1 / np.sqrt(2) + 1j / np.sqrt(2)]])
        M1 = Jones_matrix('retarder_linear')
        M1.retarder_linear(D=45 * degrees, angle=0)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 45*degrees"

        solution = matrix([[1, 0], [0, 1j]])
        M1 = Jones_matrix('retarder_linear')
        M1.retarder_linear(D=90 * degrees, angle=0)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 90*degrees"

        solution = matrix([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])
        M1 = Jones_matrix('retarder_linear')
        M1.retarder_linear(D=90 * degrees, angle=45 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + ".py --> example: D = 90*degrees, angle=45"

    def test_retarder_material(self):

        solution = matrix([[1, 0], [0, 1]])
        M1 = Jones_matrix('retarder_material')
        M1.retarder_material(
            ne=1.5, no=1.5, d=250 * um, wavelength=0.6328 * um, angle=0.0)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: no retarder"

        wavelength = 0.5 * um
        solution = matrix([[1, 0], [0, 1j]])
        M1 = Jones_matrix('retarder_material')
        M1.retarder_material(
            ne=2, no=1, d=wavelength / 4, wavelength=wavelength, angle=0.0)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: lambda/4"

        solution = matrix([[1, 0], [0, -1]])
        M1 = Jones_matrix('retarder_material')
        M1.retarder_material(
            ne=2, no=1, d=wavelength / 2, wavelength=wavelength, angle=0.0)

        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: no retarder"

    def test_diattenuator_retarder_linear(self):

        solution = matrix([[1, 0], [0, 1j]])
        M1 = Jones_matrix('M_diat')
        M1.diattenuator_retarder_linear(delay=np.pi / 2, p1=1, p2=1, angle=0)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: pi/2"

        solution = matrix([[1, 0], [0, -1]])
        M1 = Jones_matrix('M_diat')
        M1.diattenuator_retarder_linear(delay=np.pi, p1=1, p2=1, angle=0)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: pi"

        solution = matrix([[0.5 + 0.25j, 0.5 - 0.25j],
                           [0.5 - 0.25j, 0.5 + 0.25j]])
        M1 = Jones_matrix('M_diat')
        M1.diattenuator_retarder_linear(
            delay=np.pi / 2, p1=1, p2=.5, angle=45 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + ".py --> example: delay=np.pi/2, p1=1, p2=.5, angle=45*degrees"

    def test_diattenuator_azimuth_ellipticity(self):

        solution = matrix([[0.75, 0], [0, 0.25]])
        M1 = Jones_matrix('diat_az_el')
        M1.diattenuator_azimuth_ellipticity(
            p1=.75, p2=0.25, azimuth=0 * degrees, ellipticity=0 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(proposal, solution,
                          eps), sys._getframe().f_code.co_name + ".py --> 1"

        solution = matrix([[0.75, 0], [0, 0.25]])
        M1 = Jones_matrix('diat_az_el')
        M1.diattenuator_azimuth_ellipticity(
            p1=0.75, p2=0.25, azimuth=0 * degrees, ellipticity=90 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(proposal, solution,
                          eps), sys._getframe().f_code.co_name + ".py --> 2"

        solution = matrix([[0.5, -0.5j], [0.5j, 0.5]])
        M1 = Jones_matrix('diat_az_el')
        M1.diattenuator_azimuth_ellipticity(
            p1=1, p2=0, azimuth=0 * degrees, ellipticity=45 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(proposal, solution,
                          eps), sys._getframe().f_code.co_name + ".py --> 3"

        solution = matrix([[0.5, 0.25], [0.25, 0.5]])
        M1 = Jones_matrix('diat_az_el')
        M1.diattenuator_azimuth_ellipticity(
            p1=0.75, p2=0.25, azimuth=45 * degrees, ellipticity=0 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(proposal, solution,
                          eps), sys._getframe().f_code.co_name + ".py --> 4"

        solution = matrix([[0.5, -0.5j], [0.5j, 0.5]])
        M1 = Jones_matrix('diat_az_el')
        M1.diattenuator_azimuth_ellipticity(
            p1=1, p2=0, azimuth=45 * degrees, ellipticity=45 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(proposal, solution,
                          eps), sys._getframe().f_code.co_name + ".py --> 5"

        solution = matrix([[0.5, 0.25], [0.25, 0.5]])
        M1 = Jones_matrix('diat_az_el')
        M1.diattenuator_azimuth_ellipticity(
            p1=0.75, p2=0.25, azimuth=45 * degrees, ellipticity=90 * degrees)
        proposal = M1.parameters.matrix()
        assert comparison(proposal, solution,
                          eps), sys._getframe().f_code.co_name + ".py --> 6"
