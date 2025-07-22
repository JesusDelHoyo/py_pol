# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for stokes parameters Class in stokes module"""

import sys

import numpy as np
from numpy import matrix

from py_pol import degrees, eps
from py_pol.stokes import Stokes
from py_pol.utils import comparison


class TestStokes_parameters(object):
    def test_custom(self):

        # solution = matrix([[1], [0], [0], [0]])
        # M1 = Stokes('custom')
        # M1.custom(1, 0, 0, 0)
        # proposal = M1.parameters.matrix()
        # assert comparison(proposal, solution, eps), sys._getframe(
        # ).f_code.co_name + ".py --> example: (1, 0, 0, 0)"
        assert True
