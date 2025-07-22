# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for drawing module"""

import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from py_pol import degrees
from py_pol.drawings import (draw_ellipse_jones, draw_ellipse_stokes,
                             draw_ellipses_jones, draw_on_poincare,
                             draw_poincare_sphere)
from py_pol.jones_vector import Jones_vector
from py_pol.stokes import Stokes

newpath = "drawings"
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d_%H_%M_%S")
newpath = "{}_{}/".format(newpath, date)

if not os.path.exists(newpath):
    os.makedirs(newpath)


class TestDrawing(object):
    def test_stokes_poincare_inclass(self):
        s01 = Stokes('s0')
        s01.linear_light(angle=0, intensity=1)
        filename = '{}{}.png'.format(newpath, sys._getframe().f_code.co_name)
        s01.draw_poincare(filename=filename)
        plt.legend()
        assert True

    def test_stokes_poincare_1_point(self):

        s01 = Stokes('s0')
        s01.linear_light(angle=0, intensity=1)

        ax, fig = draw_poincare_sphere(
            stokes_points=s01,
            angle_view=[45 * degrees, 45 * degrees],
            kind='scatter',
            color='r',
            label='rotation')  # 'line', 'scatter'

        plt.legend()
        filename = '{}{}.png'.format(newpath, sys._getframe().f_code.co_name)
        fig.savefig(filename)
        assert True

    def test_stokes_poincare_list_classes(self):

        Stokes_points1 = []

        s01 = Stokes('s0')
        s01.linear_light(angle=0, intensity=1)
        print(s01)
        # Stokes_points.append(s01.M)

        angles = np.linspace(0, 90 * degrees, 90)

        for i, angle in enumerate(angles):
            s_rot = s01.rotate(angle=angle, keep=False, returns_matrix=False)
            Stokes_points1.append(s_rot)

        ax, fig = draw_poincare_sphere(
            stokes_points=Stokes_points1,
            angle_view=[45 * degrees, 45 * degrees],
            kind='line',
            color='r',
            label='rotation')  # 'line', 'scatter'

        plt.legend()
        filename = '{}{}.png'.format(newpath, sys._getframe().f_code.co_name)
        fig.savefig(filename)
        assert True

    def test_stokes_poincare_list_matrices(self):

        Stokes_points2 = []

        s02 = Stokes('s0')
        s02.general_charac_angles(
            alpha=45 * degrees,
            delay=45 * degrees,
            intensity=1,
            pol_degree=0.75)
        print(s02)
        # Stokes_points.append(s01.M)

        angles = np.linspace(0, 90 * degrees, 90)

        for i, angle in enumerate(angles):
            s_rot = s02.rotate(angle=angle, keep=False, returns_matrix=True)
            Stokes_points2.append(s_rot)

        ax, fig = draw_poincare_sphere(
            stokes_points=Stokes_points2,
            angle_view=[45 * degrees, 45 * degrees],
            kind='line',
            color='r',
            label='rotation')  # 'line', 'scatter'

        plt.legend()
        filename = '{}{}.png'.format(newpath, sys._getframe().f_code.co_name)
        fig.savefig(filename)
        assert True

    def test_Stokes_on_poincare(self):
        Stokes_points2 = []

        s02 = Stokes('s0')
        s02.general_charac_angles(
            alpha=45 * degrees,
            delay=45 * degrees,
            intensity=1,
            pol_degree=0.75)
        print(s02)
        # Stokes_points.append(s01.M)

        angles = np.linspace(0, 90 * degrees, 90)

        for i, angle in enumerate(angles):
            s_rot = s02.rotate(angle=angle, keep=False, returns_matrix=True)
            Stokes_points2.append(s_rot)

        ax, fig = draw_poincare_sphere(
            stokes_points=None,
            angle_view=[45 * degrees, 45 * degrees],
            kind='line',
            color='r',
            label='rotation')  # 'line', 'scatter'
        ax, fig = draw_poincare_sphere(
            stokes_points=None,
            angle_view=[45 * degrees, 45 * degrees],
            kind='line',
            color='r',
            label='rotation')  # 'line', 'scatter'
        draw_on_poincare(
            ax, Stokes_points2, kind='line', color='r', label='rotation')  #
        plt.legend()
        filename = '{}{}.png'.format(newpath, sys._getframe().f_code.co_name)
        fig.savefig(filename)
        assert True

    def test_ellipse_stokes_line(self):
        s01 = Stokes('s0')
        s01.elliptical_light(
            a=2, b=1, phase=45 * degrees, angle=45 * degrees, pol_degree=0.8)
        filename = '{}{}.png'.format(newpath, sys._getframe().f_code.co_name)
        s01.draw_ellipse(
            kind='line', limit='', has_line=True, filename=filename)
        plt.legend()
        assert True

    def test_ellipse_stokes_probabilities(self):
        s01 = Stokes('s0')
        s01.elliptical_light(
            a=2, b=1, phase=45 * degrees, angle=45 * degrees, pol_degree=0.8)
        filename = '{}{}.png'.format(newpath, sys._getframe().f_code.co_name)
        s01.draw_ellipse(
            kind='probabilities', limit='', has_line=True, filename=filename)
        plt.legend()
        assert True

    def test_ellipse_jones(self):
        s01 = Jones_vector('s0')
        s01.elliptical_light(a=2, b=1, phase=45 * degrees, angle=45 * degrees)
        filename = '{}{}.png'.format(newpath, sys._getframe().f_code.co_name)
        draw_ellipse_jones(j0=s01, filename=filename)
        assert True

    def test_ellipse_jones_inclass(self):
        s01 = Jones_vector('s0')
        s01.elliptical_light(a=2, b=1, phase=45 * degrees, angle=45 * degrees)
        filename = '{}{}.png'.format(newpath, sys._getframe().f_code.co_name)
        s01.draw_ellipse(filename=filename)
        assert True

    def test_ellipse_jones_several1(self):

        Jones_vectors_0 = []

        j0 = Jones_vector('j0')
        j0.linear_light(amplitude=1, angle=0 * degrees)

        angles = np.linspace(0, 180 * degrees, 6, endpoint=False)

        for i, angle in enumerate(angles):
            ji = j0.rotate(angle=angle, keep=False)
            Jones_vectors_0.append(ji)
            print(ji)

        ax, fig = draw_ellipses_jones(Jones_vectors_0, filename='')
        plt.legend()
        filename = '{}{}.png'.format(newpath, sys._getframe().f_code.co_name)
        fig.savefig(filename)
        assert True

    def test_ellipse_jones_several2(self):

        Jones_vectors_1 = []

        j0 = Jones_vector('j0')
        j0.elliptical_light(a=1, b=1, phase=45 * degrees, angle=0 * degrees)

        angles = np.linspace(0, 180 * degrees, 6, endpoint=False)

        for i, angle in enumerate(angles):
            ji = j0.rotate(angle=angle, keep=False)
            Jones_vectors_1.append(ji)

        ax, fig = draw_ellipses_jones(Jones_vectors_1, filename='')
        plt.legend()
        filename = '{}{}.png'.format(newpath, sys._getframe().f_code.co_name)
        fig.savefig(filename)
        assert True
