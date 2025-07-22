# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Jones_matrix parameters class"""

from py_pol import degrees, eps, np
from py_pol.jones_matrix import Jones_matrix
from py_pol.utils import comparison


class TestJonesMatrixParameters(object):
    def test_vacuum(self):

        solution = (True, 0, 0)
        M1 = Jones_matrix('M1')
        M1.from_elements(1, 0, 0, 1)

        is_hom = M1.parameters.is_homogeneous()
        delay = M1.parameters.retardance()
        diat = M1.parameters.diattenuation()

        assert np.linalg.norm(is_hom - solution[0]) < eps
        assert np.linalg.norm(delay - solution[1]) < eps
        assert np.linalg.norm(diat - solution[2]) < eps

    def test_linear_polarizer(self):

        solution = (True, 0, 1)
        M1 = Jones_matrix('M1')
        M1.diattenuator_perfect(angle=0 * degrees)

        is_hom = M1.parameters.is_homogeneous()
        delay = M1.parameters.retardance()
        diat = M1.parameters.diattenuation()

        assert np.linalg.norm(is_hom - solution[0]) < eps
        assert np.linalg.norm(delay - solution[1]) < eps
        assert np.linalg.norm(diat - solution[2]) < eps

    def test_diattenuator_lineal(self):

        solution = (True, 0, 0.8)
        M1 = Jones_matrix('M1')
        M1.diattenuator_linear(p1=.75, p2=0.25, angle=0 * degrees)

        is_hom = M1.parameters.is_homogeneous()
        delay = M1.parameters.retardance()
        diat = M1.parameters.diattenuation()

        assert np.linalg.norm(is_hom - solution[0]) < eps
        assert np.linalg.norm(delay - solution[1]) < eps
        assert np.linalg.norm(diat - solution[2]) < eps

    def test_retarder_linear(self):

        solution = (True, 90 * degrees, 0)
        M1 = Jones_matrix('M1')
        M1.retarder_linear(D=90 * degrees, angle=0)

        is_hom = M1.parameters.is_homogeneous()
        delay = M1.parameters.retardance()
        diat = M1.parameters.diattenuation()

        assert np.linalg.norm(is_hom - solution[0]) < eps
        assert np.linalg.norm(delay - solution[1]) < eps
        assert np.linalg.norm(diat - solution[2]) < eps

    def test_diattenuator_azimuth_ellipticity(self):
        # TODO - LM: check

        solution = (True, 0, 0.8)
        M1 = Jones_matrix('M1')
        M1.diattenuator_azimuth_ellipticity(
            p1=0.75, p2=0.25, azimuth=45 * degrees, ellipticity=90 * degrees)
        is_hom = M1.parameters.is_homogeneous()
        delay = M1.parameters.retardance()
        diat = M1.parameters.diattenuation()

        assert np.linalg.norm(is_hom - solution[0]) < eps
        assert np.linalg.norm(delay - solution[1]) < eps
        assert np.linalg.norm(diat - solution[2]) < eps

    def test_inhomogenous(self):
        # TODO - LM: check
        assert True
