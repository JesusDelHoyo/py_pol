# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for stokes module"""

import sys

import numpy as np
from numpy import matrix

from py_pol import degrees, eps
from py_pol.stokes import Stokes
from py_pol.utils import comparison


class TestStokes(object):
    def test_from_elements(self):

        solution = matrix([[1], [0], [0], [0]])
        M1 = Stokes('from_elements')
        M1.from_elements(1, 0, 0, 0)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: (1, 0, 0, 0)"

        solution = matrix([[1], [0.5], [0.25], [0.1]])
        M1 = Stokes('from_elements')
        M1.from_elements(1, 0.5, 0.25, 0.1)
        proposal = M1.parameters.matrix()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + ".py --> example: (1, 0.5, 0.25, 0.1)"

    def test_E(self):
        num_points = 1001
        E = np.zeros((num_points, 3), dtype=complex)
        t = np.linspace(0, 2 * np.pi, num_points)
        E[:, 0] = np.sin(t)
        E[:, 1] = np.sin(t)
        # E[:, 2] = np.zeros_like(E[:, 2])

        solution = matrix([[1], [0], [1], [0]])

        M1 = Stokes('from_elements')
        M1.from_distribution(E, is_normalized=True)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: (1, 0, 1, 0)"

    def test_sum(self):

        solution = matrix([[2], [2], [0], [0]])
        M1 = Stokes('j1')
        M1.linear_light(angle=0, intensity=1)
        M2 = Stokes('j2')
        M2.linear_light(angle=0, intensity=1)
        M3 = M1 + M2
        proposal = M3.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 0-0*degrees"

        solution = matrix([[2], [1], [1], [0]])
        M1 = Stokes('j1')
        M1.linear_light(angle=0, intensity=1)
        M2 = Stokes('j2')
        M2.linear_light(angle=45 * degrees, intensity=1)
        M3 = M1 + M2
        proposal = M3.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 0-45*degrees"

        solution = matrix([[2], [0], [0], [0]])
        M1 = Stokes('j1')
        M1.linear_light(angle=0, intensity=1)
        M2 = Stokes('j2')
        M2.linear_light(angle=90 * degrees, intensity=1)
        M3 = M1 + M2
        proposal = M3.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 0-45*degrees"

    def test_product(self):
        solution = matrix([[5], [2], [3], [0]])
        M1 = Stokes('j1')
        M1.linear_light(angle=0, intensity=1)
        M2 = Stokes('j2')
        M2.linear_light(angle=45 * degrees, intensity=1)
        M3 = 2 * M1 + 3 * M2
        proposal = M3.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example:2*M1+3*M2"

    def test_general_charac_angles(self):

        solution = matrix([[1], [0], [0], [0]])
        M1 = Stokes('l_gen')
        M1.general_charac_angles(
            alpha=0 * degrees, delay=0 * degrees, intensity=1, pol_degree=0)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (0,0,1,0)"

        solution = matrix([[1], [1], [0], [0]])
        M1 = Stokes('l_gen')
        M1.general_charac_angles(
            alpha=0 * degrees, delay=0 * degrees, intensity=1, pol_degree=1)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: (0,0,1,1)"

        solution = matrix([[1], [0], [0], [0.5]])
        M1 = Stokes('l_gen')
        M1.general_charac_angles(
            alpha=45 * degrees, delay=90 * degrees, intensity=1, pol_degree=.5)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: (45,90,1,0.5)"

        solution = matrix([[1], [-0.5], [0], [0]])
        M1 = Stokes('l_gen')
        M1.general_charac_angles(
            alpha=90 * degrees, delay=45 * degrees, intensity=1, pol_degree=.5)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: (90,45,1,0.5)"

    def test_circular_light(self):

        solution = matrix([[1], [0], [0], [1]])
        M1 = Stokes('l_c')
        M1.circular_light(kind='r', intensity=1)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: kind='r'"

        solution = matrix([[1], [0], [0], [-1]])
        M1 = Stokes('l_c')
        M1.circular_light(kind='l', intensity=1)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: kind='l'"

    def test_linear_light(self):

        solution = matrix([[1], [1], [0], [0]])
        M1 = Stokes('l_linear')
        M1.linear_light(angle=0 * degrees, intensity=1)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0*degrees"

        solution = matrix([[1], [0], [1], [0]])
        M1 = Stokes('l_linear')
        M1.linear_light(angle=45 * degrees, intensity=1)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 45*degrees"

        solution = matrix([[1], [-1], [0], [0]])
        M1 = Stokes('l_linear')
        M1.linear_light(angle=90 * degrees, intensity=1)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 90*degrees"

        solution = matrix([[2], [0], [-2], [0]])
        M1 = Stokes('l_linear')
        M1.linear_light(angle=135 * degrees, intensity=2)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 135*degrees"

        solution = matrix([[1], [1], [0], [0]])
        M1 = Stokes('l_linear')
        M1.linear_light(angle=180 * degrees, intensity=1)
        proposal = M1.parameters.matrix()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + ".py --> example: 180*degrees"
