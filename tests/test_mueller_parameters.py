# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for mueller parameters Class in mueller module"""

import sys

import numpy as np
from numpy import sqrt

from py_pol import degrees, eps
from py_pol.mueller import Mueller
from py_pol.utils import comparison


class TestMueller_parameters(object):
    def test_print(self):
        try:
            (a, b) = np.random.rand(2)
            M1 = Mueller('M1')
            M1.diattenuator_linear(p1=sqrt(a), p2=sqrt(b), angle=0 * degrees)
            print(M1.parameters)
            assert True, sys._getframe().f_code.co_name + "@ Print"
        except:
            assert False, sys._getframe().f_code.co_name + "@ Print"

    def test_mean_transmission(self):
        (a, b) = np.random.rand(2)
        solution = (a + b) / 2
        M1 = Mueller('M1')
        M1.diattenuator_linear(p1=sqrt(a), p2=sqrt(b), angle=0 * degrees)
        proposal = M1.parameters.mean_transmission()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ Mean trans of random diattenuator"

    def test_inhomogeneity(self):
        solution = 0
        P1 = Mueller('P1')
        P1.diattenuator_linear(p1=1, p2=0, angle=0 * degrees)
        proposal = P1.parameters.inhomogeneity()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + "@ polarizer + Retarder @45 deg"

        solution = sqrt(2) / 2
        P1 = Mueller('P1')
        P1.diattenuator_linear(p1=1, p2=0, angle=0 * degrees)
        R1 = Mueller('R1')
        R1.quarter_waveplate(angle=45 * degrees)
        M1 = (P1 * R1)
        proposal = M1.parameters.inhomogeneity()
        assert comparison(
            proposal, solution, eps
        ), sys._getframe().f_code.co_name + "@ polarizer + Retarder @45 deg"

        solution = 1
        P1 = Mueller('P1')
        P1.diattenuator_linear(p1=1, p2=0, angle=0 * degrees)
        P2 = Mueller('P2')
        P2.diattenuator_linear(p1=1, p2=0, angle=90 * degrees)
        R1 = Mueller('R1')
        R1.quarter_waveplate(angle=45 * degrees)
        M1 = (P1 * R1) * P2
        proposal = M1.parameters.inhomogeneity()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ polarizer + Retarder @45 deg + polarizer @90 deg"

    def test_diattenuation(self):
        D = np.matrix(2 * (np.matrix(np.random.rand(3)) - 0.5) / 3)
        solution = np.linalg.norm(D)
        M1 = Mueller('M1')
        M1.diattenuator_from_vector(D)
        proposal = M1.parameters.diattenuation()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ polarizer with random diatenuation vector"

    def test_diattenuation_linear(self):
        D = np.matrix(2 * (np.matrix(np.random.rand(3)) - 0.5) / 3)
        solution = np.linalg.norm(D[0:2])  # Check this if fails
        M1 = Mueller('M1')
        M1.diattenuator_from_vector(D)
        proposal = M1.parameters.diattenuation_linear()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ polarizer with random diatenuation vector"

    def test_diattenuation_circular(self):
        D = np.matrix(2 * (np.matrix(np.random.rand(3)) - 0.5) / 3)
        solution = np.linalg.norm(D[0, 2])  # Check this if fails
        M1 = Mueller('M1')
        M1.diattenuator_from_vector(D)
        proposal = M1.parameters.diattenuation_circular()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ polarizer with random diatenuation vector"

    def test_polarizance(self):
        D = np.matrix(2 * (np.matrix(np.random.rand(3)) - 0.5) / 3)
        solution = np.linalg.norm(D)
        M1 = Mueller('M1')
        M1.diattenuator_from_vector(D)
        proposal = M1.parameters.polarizance()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ polarizer with random diatenuation vector"

    def test_polarizance_linear(self):
        D = np.matrix(2 * (np.matrix(np.random.rand(3)) - 0.5) / 3)
        solution = np.linalg.norm(D[0, 0:2])  # Check this if fails
        M1 = Mueller('M1')
        M1.diattenuator_from_vector(D)
        proposal = M1.parameters.polarizance_linear()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ polarizer with random diatenuation vector"

    def test_polarizance_circular(self):
        D = np.matrix(2 * (np.matrix(np.random.rand(3)) - 0.5) / 3)
        solution = np.linalg.norm(D[0, 2])  # Check this if fails
        M1 = Mueller('M1')
        M1.diattenuator_from_vector(D)
        proposal = M1.parameters.polarizance_circular()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ polarizer with random diatenuation vector"

    def test_polarizance_degree(self):
        D = np.matrix(2 * (np.matrix(np.random.rand(3)) - 0.5) / 3)
        solution = np.linalg.norm(D)
        M1 = Mueller('M1')
        M1.diattenuator_from_vector(D)
        proposal = M1.parameters.polarizance_degree()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ polarizer with random diatenuation vector"

    def test_spheric_purity(self):
        a = np.random.rand(1)
        solution = 1 / sqrt(3)
        M1 = Mueller('M1')
        M1.diattenuator_linear(p1=1, p2=0, angle=a[0] * 180 * degrees)
        proposal = M1.parameters.spheric_purity()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ Ideal polarizer with random angle"

        solution = 1
        (D, alpha, delta) = np.random.rand(3)
        M1 = Mueller('M1')
        M1.retarder_charac_angles_from_Jones(
            D * 360 * degrees, alpha * 90 * degrees, delta * 360 * degrees)
        proposal = M1.parameters.spheric_purity()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + "@ Random retarder"

    def test_retardance(self):
        (D, alpha, delta) = np.random.rand(3)
        solution = D * np.pi
        M1 = Mueller('M1')
        M1.retarder_charac_angles_from_vector(
            D * 180 * degrees, alpha * 90 * degrees, delta * 360 * degrees)
        proposal = M1.parameters.retardance()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + "@ Random retarder"

    def test_polarimetric_purity(self):
        """Test for polarimetric_purity and depolarization_index."""
        solution = 1
        (p1, p2, alpha, delta) = np.random.rand(4)
        M1 = Mueller('M1')
        M1.diattenuator_charac_angles_from_Jones(p1, p2, alpha * 90 * degrees,
                                                delta * 360 * degrees)
        proposal = M1.parameters.polarimetric_purity()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ Polarimetric purity for random polarizer"
        solution = 0
        proposal = M1.parameters.depolarization_index()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ Depolarization degree for random polarizer"

        solution = 1
        (D, alpha, delta) = np.random.rand(3)
        M1 = Mueller('M1')
        M1.retarder_charac_angles_from_Jones(
            D * 180 * degrees, alpha * 90 * degrees, delta * 360 * degrees)
        proposal = M1.parameters.polarimetric_purity()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ Polarimetric purity for random retarder"
        solution = 0
        proposal = M1.parameters.depolarization_index()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ Depolarization degree for random retarder"

        dep = np.random.rand(1)
        solution = dep[0]
        M1 = Mueller('M1')
        M1.depolarizer(dep[0])
        proposal = M1.parameters.polarimetric_purity()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ Polarimetric purity for random depolarizer"
        solution = sqrt(1 - dep[0]**2)
        proposal = M1.parameters.depolarization_index()
        assert comparison(proposal, solution, eps), sys._getframe(
        ).f_code.co_name + "@ Depolarization degree for random depolarizer"

    def test_depolarization_factors(self):
        # TODO: Es poco importante
        assert True

    def test_polarimetric_purity_indices(self):
        # TODO: Es poco importante y encima costara encontrar ejemplos
        assert True
