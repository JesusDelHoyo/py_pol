# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Jones_vector module"""

import sys

import numpy as np

from py_pol import degrees, eps
from py_pol.jones_vector import Jones_vector
from py_pol.utils import comparison, params_to_list


class TestJonesVectorParameters(object):
    def test_components(self):

        (solutionX, solutionY) = (1, 1j)
        j1 = Jones_vector()
        j1.circular_light(amplitude=np.sqrt(2))
        proposalX, proposalY = j1.parameters.components()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D X"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D Y"

        E = np.random.rand(5)
        (solutionX, solutionY) = (E, 1j * E)
        j1 = Jones_vector()
        j1.circular_light(amplitude=np.sqrt(2) * E)
        proposalX, proposalY = j1.parameters.components()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D X"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D Y"

        E = np.random.rand(3, 3)
        (solutionX, solutionY) = (E, 1j * E)
        j1 = Jones_vector()
        j1.circular_light(amplitude=np.sqrt(2) * E)
        proposalX, proposalY = j1.parameters.components()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2D X"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2D Y"

    def test_amplitudes(self):

        (solutionX, solutionY) = (1, 1)
        j1 = Jones_vector()
        j1.circular_light(amplitude=np.sqrt(2))
        proposalX, proposalY = j1.parameters.amplitudes()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D X"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D Y"

        E = np.random.rand(5)
        (solutionX, solutionY) = (E, E)
        j1 = Jones_vector()
        j1.circular_light(amplitude=np.sqrt(2) * E)
        proposalX, proposalY = j1.parameters.amplitudes()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D X"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D Y"

        E = np.random.rand(3, 3)
        (solutionX, solutionY) = (E, E)
        j1 = Jones_vector()
        j1.circular_light(amplitude=np.sqrt(2) * E)
        proposalX, proposalY = j1.parameters.amplitudes()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2D X"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2D Y"

    def test_intensity(self):

        solution = 2
        j1 = Jones_vector()
        j1.circular_light(intensity=2)
        proposal = j1.parameters.intensity()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D"

        E = np.random.rand(3, 3)
        solution = E**2
        j1 = Jones_vector()
        j1.circular_light(amplitude=E)
        proposal = j1.parameters.intensity()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2D"

    def test_charac_angles(self):

        alpha = np.random.rand(1) * 90 * degrees
        delay = np.random.rand(1) * 360 * degrees
        (solutionX, solutionY) = (alpha, delay)
        j1 = Jones_vector()
        j1.general_charac_angles(alpha=alpha, delay=delay)
        proposalX, proposalY = j1.parameters.charac_angles()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D alpha"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D delay"

        alpha = np.random.rand(5) * 90 * degrees
        delay = np.random.rand(5) * 360 * degrees
        (solutionX, solutionY) = (alpha, delay)
        j1 = Jones_vector()
        j1.general_charac_angles(alpha=alpha, delay=delay)
        proposalX, proposalY = j1.parameters.charac_angles()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D alpha"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D delay"

        alpha = np.random.rand(3, 3) * 90 * degrees
        delay = np.random.rand(3, 3) * 360 * degrees
        (solutionX, solutionY) = (alpha, delay)
        j1 = Jones_vector()
        j1.general_charac_angles(alpha=alpha, delay=delay)
        proposalX, proposalY = j1.parameters.charac_angles()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2D alpha"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2D delay"

    def test_azimuth_ellipticity(self):

        azimuth = np.random.rand(1) * 180 * degrees
        ellipticity = (2 * np.random.rand(1) - 1) * 45 * degrees
        (solutionX, solutionY) = (azimuth, ellipticity)
        j1 = Jones_vector()
        j1.general_azimuth_ellipticity(
            azimuth=azimuth, ellipticity=ellipticity)
        proposalX, proposalY = j1.parameters.azimuth_ellipticity()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D azimuth"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D ellipticity"

        azimuth = np.random.rand(5) * 180 * degrees
        ellipticity = (2 * np.random.rand(5) - 1) * 45 * degrees
        (solutionX, solutionY) = (azimuth, ellipticity)
        j1 = Jones_vector()
        j1.general_azimuth_ellipticity(
            azimuth=azimuth, ellipticity=ellipticity)
        proposalX, proposalY = j1.parameters.azimuth_ellipticity()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D azimuth"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D ellipticity"

        azimuth = np.random.rand(3, 3) * 180 * degrees
        ellipticity = (2 * np.random.rand(3, 3) - 1) * 45 * degrees
        (solutionX, solutionY) = (azimuth, ellipticity)
        j1 = Jones_vector()
        j1.general_azimuth_ellipticity(
            azimuth=azimuth, ellipticity=ellipticity)
        proposalX, proposalY = j1.parameters.azimuth_ellipticity()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2D azimuth"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2D ellipticity"

    def test_global_phase(self):

        phase = np.random.rand(1) * 360 * degrees
        azimuth = np.random.rand(1) * 180 * degrees
        ellipticity = (2 * np.random.rand(1) - 1) * 45 * degrees
        solution = phase
        j1 = Jones_vector()
        j1.general_azimuth_ellipticity(
            azimuth=azimuth, ellipticity=ellipticity)
        j1.remove_global_phase()
        j1.add_global_phase(phase)
        proposal = j1.parameters.global_phase()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D"

        phase = np.random.rand(5) * 360 * degrees
        azimuth = np.random.rand(1) * 180 * degrees
        ellipticity = (2 * np.random.rand(1) - 1) * 45 * degrees
        solution = phase
        j1 = Jones_vector()
        j1.general_azimuth_ellipticity(
            azimuth=azimuth, ellipticity=ellipticity)
        j1.remove_global_phase()
        j1.add_global_phase(phase)
        proposal = j1.parameters.global_phase()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D"

        phase = np.random.rand(3, 3) * 360 * degrees
        azimuth = np.random.rand(1) * 180 * degrees
        ellipticity = (2 * np.random.rand(1) - 1) * 45 * degrees
        solution = phase
        j1 = Jones_vector()
        j1.general_azimuth_ellipticity(
            azimuth=azimuth, ellipticity=ellipticity)
        j1.remove_global_phase()
        j1.add_global_phase(phase)
        proposal = j1.parameters.global_phase()
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2D"

    def test_ellipse_axes(self):

        azimuth = np.random.rand(1) * 180 * degrees
        a = 7
        b = 5
        (solutionX, solutionY) = (a, b)
        j1 = Jones_vector()
        j1.elliptical_light(
            azimuth=azimuth, a=a, b=b)
        proposalX, proposalY = j1.parameters.ellipse_axes()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D a"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 0D b"

        azimuth = np.random.rand(5) * 180 * degrees
        a = 7
        b = 5
        (solutionX, solutionY) = (a, b)
        j1 = Jones_vector()
        j1.elliptical_light(
            azimuth=azimuth, a=a, b=b)
        proposalX, proposalY = j1.parameters.ellipse_axes()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D a"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 1D b"

        azimuth = np.random.rand(3, 3) * 180 * degrees
        a = 7
        b = 5
        (solutionX, solutionY) = (a, b)
        j1 = Jones_vector()
        j1.elliptical_light(
            azimuth=azimuth, a=a, b=b)
        proposalX, proposalY = j1.parameters.ellipse_axes()
        assert comparison(
            proposalX, solutionX,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2D a"
        assert comparison(
            proposalY, solutionY,
            eps), sys._getframe().f_code.co_name + ".py --> example: 2D b"
