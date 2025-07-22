# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for stokes module"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from py_pol import degrees, eps
from py_pol.utils import (azimuth_elipt_2_charac_angles,
                          charac_angles_2_azimuth_elipt, comparison, fit_sine,
                          rotation_matrix_Mueller)


class Test_Utils(object):
    def test_fit_sine(self):
        N = 100  # number of data points
        t = np.linspace(0, 4 * np.pi, N)
        f = 1.15247  # Optional!! Advised not to use

        data_good = 3.0 * (1 + np.sin(f * t + 0.001))
        data = data_good + 0.25 * np.random.randn(N)

        solution = data_good

        guess_std, guess_freq, guess_phase, guess_mean = fit_sine(
            t, data, has_draw=True)
        proposal = guess_std * np.sin(guess_freq * t +
                                      guess_phase) + guess_mean

        plt.figure()
        plt.plot(t, proposal, 'k')
        plt.plot(t, solution, 'r')

        assert comparison(proposal, solution,
                          2e-2 * len(proposal)), sys._getframe().f_code.co_name

    def test_rotation_matrix_Mueller(self):
        solution = np.matrix(
            np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0,
                                                                  1]]))
        proposal = rotation_matrix_Mueller(45 * degrees)
        assert comparison(
            proposal, solution,
            eps), sys._getframe().f_code.co_name + "@ Rotation 45 deg"

    def test_transformation_coordinates(self):
        N = 20
        errors = np.zeros(N)
        for ind in range(N):
            (alpha, delta) = np.random.rand(2)
            alpha = alpha * 90 * degrees
            delta = delta * 360 * degrees
            solution = np.array([alpha, delta])
            (fi, chi) = charac_angles_2_azimuth_elipt(alpha, delta)
            proposal = azimuth_elipt_2_charac_angles(fi, chi)
            proposal = np.array(proposal)
            errors[ind] = comparison(proposal, solution, eps)
        assert all(
            errors), sys._getframe().f_code.co_name + "@ Start in charac angles"

        for ind in range(N):
            (azimuth, ellipticity) = np.random.rand(2)
            azimuth = azimuth * 180 * degrees
            ellipticity = (1 - 2 * ellipticity) * 45 * degrees
            solution = np.array([azimuth, ellipticity])
            (alpha, delay) = azimuth_elipt_2_charac_angles(azimuth, ellipticity)
            proposal = charac_angles_2_azimuth_elipt(alpha, delay)
            proposal = np.array(proposal)
            errors[ind] = comparison(proposal, solution, eps)
        assert all(errors), sys._getframe().f_code.co_name + "@ Start in azimuth-ellipticity"
