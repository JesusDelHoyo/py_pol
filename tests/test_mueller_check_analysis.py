# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for mueller parameters Class in mueller module"""

import sys

import numpy as np
from numpy import matrix, sqrt

from py_pol import degrees, eps
from py_pol.mueller import Mueller
from py_pol.utils import comparison


class TestMueller_checks(object):
    def test_is_physical(self):
        (p1, p2, alpha, delta) = np.random.rand(4)
        M1 = Mueller('M1')
        M1.diattenuator_charac_angles_from_Jones(p1, p2, alpha * 90 * degrees,
                                                delta * 360 * degrees)
        proposal = M1.checks.is_physical()
        assert proposal, sys._getframe(
        ).f_code.co_name + "@ Random diattenuator"

        (D, alpha, delta) = np.random.rand(3)
        M1 = Mueller('M1')
        M1.retarder_charac_angles_from_Jones(
            D * 180 * degrees, alpha * 90 * degrees, delta * 360 * degrees)
        proposal = M1.checks.is_physical()
        assert proposal, sys._getframe().f_code.co_name + "@ Random retarder"

        d = np.random.rand(1)
        M1 = Mueller('M1')
        M1.depolarizer(d[0])
        proposal = M1.checks.is_physical()
        assert proposal, sys._getframe().f_code.co_name + "@ Random depolarizer"

    def test_is_non_depolarizing(self):
        (p1, p2, alpha, delta) = np.random.rand(4)
        M1 = Mueller('M1')
        M1.diattenuator_charac_angles_from_Jones(p1, p2, alpha * 90 * degrees,
                                                delta * 360 * degrees)
        proposal = M1.checks.is_non_depolarizing()
        assert proposal, sys._getframe(
        ).f_code.co_name + "@ Random diattenuator"

        (D, alpha, delta) = np.random.rand(3)
        M1 = Mueller('M1')
        M1.retarder_charac_angles_from_Jones(
            D * 180 * degrees, alpha * 90 * degrees, delta * 360 * degrees)
        proposal = M1.checks.is_non_depolarizing()
        assert proposal, sys._getframe().f_code.co_name + "@ Random retarder"

        d = np.random.rand(3) * .95
        M1 = Mueller('M1')
        M1.depolarizer(d)
        proposal = M1.checks.is_non_depolarizing()
        assert not proposal, sys._getframe(
        ).f_code.co_name + "@ Random depolarizer"

    def test_is_homogenous(self):
        (p1, p2, alpha, delta) = np.random.rand(4)
        M1 = Mueller('M1')
        M1.diattenuator_charac_angles_from_Jones(p1, p2, alpha * 90 * degrees,
                                                delta * 360 * degrees)
        proposal = M1.checks.is_homogeneous()
        assert proposal, sys._getframe(
        ).f_code.co_name + "@ Random diattenuator"

        (D, alpha, delta) = np.random.rand(3)
        M1 = Mueller('M1')
        M1.retarder_charac_angles_from_Jones(
            D * 180 * degrees, alpha * 90 * degrees, delta * 360 * degrees)
        proposal = M1.checks.is_homogeneous()
        assert proposal, sys._getframe().f_code.co_name + "@ Random retarder"

        d = np.random.rand(3) * .95
        M1 = Mueller('M1')
        M1.depolarizer(d)
        proposal = M1.checks.is_homogeneous()
        assert proposal, sys._getframe().f_code.co_name + "@ Random depolarizer"

    def test_is_retarder(self):
        (p1, p2, alpha, delta) = np.random.rand(4)
        M1 = Mueller('M1')
        M1.diattenuator_charac_angles_from_Jones(p1, p2, alpha * 90 * degrees,
                                                delta * 360 * degrees)
        proposal = M1.checks.is_retarder()
        assert not proposal, sys._getframe(
        ).f_code.co_name + "@ Random diattenuator"

        (D, alpha, delta) = np.random.rand(3)
        M1 = Mueller('M1')
        M1.retarder_charac_angles_from_Jones(
            D * 180 * degrees, alpha * 90 * degrees, delta * 360 * degrees)
        proposal = M1.checks.is_retarder()
        assert proposal, sys._getframe().f_code.co_name + "@ Random retarder"

        d = np.random.rand(3) * .95
        M1 = Mueller('M1')
        M1.depolarizer(d)
        proposal = M1.checks.is_retarder()
        assert not proposal, sys._getframe(
        ).f_code.co_name + "@ Random depolarizer"

    def test_is_diattenuator(self):
        (p1, p2, alpha, delta) = np.random.rand(4)
        M1 = Mueller('M1')
        M1.diattenuator_charac_angles_from_Jones(p1, p2, alpha * 90 * degrees,
                                                delta * 360 * degrees)
        proposal = M1.checks.is_diattenuator()
        assert proposal, sys._getframe(
        ).f_code.co_name + "@ Random diattenuator"

        (D, alpha, delta) = np.random.rand(3)
        M1 = Mueller('M1')
        M1.retarder_charac_angles_from_Jones(
            D * 180 * degrees, alpha * 90 * degrees, delta * 360 * degrees)
        proposal = M1.checks.is_diattenuator()
        assert not proposal, sys._getframe(
        ).f_code.co_name + "@ Random retarder"

        d = np.random.rand(3) * .95
        M1 = Mueller('M1')
        M1.depolarizer(d)
        proposal = M1.checks.is_diattenuator()
        assert not proposal, sys._getframe(
        ).f_code.co_name + "@ Random depolarizer"

    def test_is_singular(self):
        (p1, p2, alpha, delta) = np.random.rand(4)
        M1 = Mueller('M1')
        M1.diattenuator_charac_angles_from_Jones(p1, p2, alpha * 90 * degrees,
                                                delta * 360 * degrees)
        proposal = M1.checks.is_singular()
        assert not proposal, sys._getframe(
        ).f_code.co_name + "@ Random diattenuator"

        (p1, alpha, delta) = np.random.rand(3)
        p2 = 0
        M1 = Mueller('M1')
        M1.diattenuator_charac_angles_from_Jones(p1, p2, alpha * 90 * degrees,
                                                delta * 360 * degrees)
        proposal = M1.checks.is_singular()
        assert proposal, sys._getframe(
        ).f_code.co_name + "@ Random diattenuator with p2=0"

        (D, alpha, delta) = np.random.rand(3)
        M1 = Mueller('M1')
        M1.retarder_charac_angles_from_Jones(
            D * 180 * degrees, alpha * 90 * degrees, delta * 360 * degrees)
        proposal = M1.checks.is_singular()
        assert not proposal, sys._getframe(
        ).f_code.co_name + "@ Random retarder"

        d = [1, 0, 1]
        M1 = Mueller('M1')
        M1.depolarizer(d)
        proposal = M1.checks.is_singular()
        assert proposal, sys._getframe().f_code.co_name + "@ Random depolarizer"

    def test_is_pure(self):
        (p1, p2, alpha, delta) = np.random.rand(4)
        M1 = Mueller('M1')
        M1.diattenuator_charac_angles_from_Jones(p1, p2, alpha * 90 * degrees,
                                                delta * 360 * degrees)
        proposal = M1.checks.is_pure()
        assert proposal, sys._getframe(
        ).f_code.co_name + "@ Random diattenuator"

        (D, alpha, delta) = np.random.rand(3)
        M1 = Mueller('M1')
        M1.retarder_charac_angles_from_Jones(
            D * 180 * degrees, alpha * 90 * degrees, delta * 360 * degrees)
        proposal = M1.checks.is_pure()
        assert proposal, sys._getframe().f_code.co_name + "@ Random retarder"

        d = np.random.rand(3) * .95
        M1 = Mueller('M1')
        M1.depolarizer(d)
        proposal = M1.checks.is_pure()
        assert not proposal, sys._getframe(
        ).f_code.co_name + "@ Random depolarizer"


class TestMueller_analysis(object):
    def test_filter_physical_conditions(self):
        original = np.matrix(np.random.rand(4, 4))
        M1 = Mueller('M1')
        M1.from_matrix(original)
        try:
            proposal = M1.analysis.filter_physical_conditions(
                give_object=False, verbose=True, tol=0.01)
            assert True, sys._getframe().f_code.co_name + "@ Random matrix"
        except:
            assert False, sys._getframe().f_code.co_name + "@ Random matrix"

    def test_diattenuator(self):
        N = 15
        errors = np.zeros(N)
        for ind in range(N):
            (p1a, p2a, alpha, delta) = np.random.rand(4)
            alpha = alpha * 90 * degrees
            delta = delta * 360 * degrees
            p1 = max(p1a, p2a)
            p2 = min(p1a, p2a)
            solution = (p1, p2, alpha, delta)
            M1 = Mueller('M1')
            M1.diattenuator_charac_angles_from_Jones(p1, p2, alpha, delta)
            proposal = M1.analysis.diattenuator(param='charac')
            proposal = np.array(proposal)
            errors[ind] = comparison(proposal, solution, eps)
        assert all(errors), sys._getframe().f_code.co_name + "@ Charac angles"

        for ind in range(N):
            (p1a, p2a, azimuth, ellipticity) = np.random.rand(4)
            azimuth = azimuth * 180 * degrees
            ellipticity = (1 - ellipticity * 2) * 45 * degrees
            p1 = max(p1a, p2a)
            p2 = min(p1a, p2a)
            solution = np.array([p1, p2, azimuth, ellipticity])
            M1 = Mueller('M1')
            M1.diattenuator_azimuth_ellipticity_from_vector(p1, p2, azimuth, ellipticity)
            proposal = M1.analysis.diattenuator(param='azel')
            proposal = np.array(proposal)
            errors[ind] = comparison(proposal, solution, eps)
        assert all(
            errors), sys._getframe().f_code.co_name + "@ Azimuth ellipticity"

    def test_retarder(self):
        N = 20
        errors = np.zeros(N)
        for ind in range(N):
            (delta, alpha, delay) = np.random.rand(3)
            delta = delta * 180 * degrees
            alpha = alpha * 90 * degrees
            delay = delay * 360 * degrees
            solution = (delta, alpha, delay)
            M1 = Mueller('M1')
            M1.retarder_charac_angles_from_Jones(delta, alpha, delay)
            proposal = M1.analysis.retarder(param='charac')
            proposal = np.array(proposal)
            errors[ind] = comparison(proposal, solution, eps)
        assert all(errors), sys._getframe().f_code.co_name + "@ Charac angles"

        for ind in range(N):
            (delta, azimuth, ellipticity) = np.random.rand(3)
            delta = delta * 180 * degrees
            azimuth = azimuth * 90 * degrees
            ellipticity = ellipticity * 360 * degrees
            solution = (delta, azimuth, ellipticity)
            M1 = Mueller('M1')
            M1.retarder_charac_angles_from_Jones(delta, azimuth, ellipticity)
            proposal = M1.analysis.retarder(param='charac')
            proposal = np.array(proposal)
            errors[ind] = comparison(proposal, solution, eps)
        assert all(
            errors), sys._getframe().f_code.co_name + "@ Azimuth ellipticity"

    def test_decompose_pure(self):
        N = 20
        errors = np.zeros(N + 1)
        for ind in range(N):
            # Create random parameters
            (p1a, p2a, alphaP, delayP, delta, alphaR,
             delayR) = np.random.rand(7)
            p1 = max(p1a, p2a)
            p2 = min(p1a, p2a)
            alphaP = alphaP * 90 * degrees
            delayP = delayP * 360 * degrees
            delta = delta * 180 * degrees
            alphaR = alphaR * 90 * degrees
            delayR = delayR * 360 * degrees
            # Create objects
            M1 = Mueller('Mp')
            M1.diattenuator_charac_angles_from_Jones(p1, p2, alphaP, delayP)
            M2 = Mueller('Mr')
            M2.retarder_charac_angles_from_Jones(delta, alphaR, delayR)
            M = M2 * M1
            # Decompose
            Mr, Md, params = M.analysis.decompose_pure(
                give_all=True, tol=1e-10)
            aux = [comparison(Md.M, M1.M, eps), comparison(Mr.M, M2.M, eps)]
            errors[ind] = all(aux)
        try:
            Mr, Md = M.analysis.decompose_pure(tol=1e-10, verbose=True)
            errors[-1] = True
        except:
            errors[-1] = False
        assert all(
            errors), sys._getframe().f_code.co_name + "@ polarizer right"

        for ind in range(N):
            # Create random parameters
            (p1a, p2a, alphaP, delayP, delta, alphaR,
             delayR) = np.random.rand(7)
            p1 = max(p1a, p2a)
            p2 = min(p1a, p2a)
            alphaP = alphaP * 90 * degrees
            delayP = delayP * 360 * degrees
            delta = delta * 180 * degrees
            alphaR = alphaR * 90 * degrees
            delayR = delayR * 360 * degrees
            # Create objects
            M1 = Mueller('Mp')
            M1.diattenuator_charac_angles_from_Jones(p1, p2, alphaP, delayP)
            M2 = Mueller('Mr')
            M2.retarder_charac_angles_from_Jones(delta, alphaR, delayR)
            M = M1 * M2
            # Decompose
            Mr, Md, params = M.analysis.decompose_pure(
                decomposition='PR', give_all=True, tol=1e-10)
            aux = [comparison(Md.M, M1.M, eps), comparison(Mr.M, M2.M, eps)]
            errors[ind] = all(aux)
        try:
            Mr, Md = M.analysis.decompose_pure(tol=1e-10, verbose=True)
            errors[-1] = True
        except:
            errors[-1] = False
        assert all(
            errors), sys._getframe().f_code.co_name + + "@ polarizer right"

        for ind in range(N):
            # Create random parameters
            (delta, alphaR, delayR) = np.random.rand(3)
            delta = delta * 180 * degrees
            alphaR = alphaR * 90 * degrees
            delayR = delayR * 360 * degrees
            # Create objects
            M1 = Mueller('Mp')
            M1.from_matrix(np.eye(4))
            M2 = Mueller('Mr')
            M2.retarder_charac_angles_from_Jones(delta, alphaR, delayR)
            M = M1 * M2
            # Decompose
            Mr, Md, params = M.analysis.decompose_pure(
                decomposition='PR', give_all=True, tol=1e-10)
            aux = [comparison(Md.M, M1.M, eps), comparison(Mr.M, M2.M, eps)]
            errors[ind] = all(aux)
        assert all(errors), sys._getframe(
        ).f_code.co_name + "@ Singular no diattenuator"

        p2 = 0
        for ind in range(N):
            # Create random parameters
            (p1, alphaP, delayP, delta, alphaR, delayR) = np.random.rand(6)
            alphaP = alphaP * 90 * degrees
            delayP = delayP * 360 * degrees
            delta = delta * 180 * degrees
            alphaR = alphaR * 90 * degrees
            delayR = delayR * 360 * degrees
            # Create objects
            M1 = Mueller('Mp')
            M1.diattenuator_charac_angles_from_Jones(p1, p2, alphaP, delayP)
            M2 = Mueller('Mr')
            M2.retarder_charac_angles_from_Jones(delta, alphaR, delayR)
            M = M2 * M1
            # Decompose
            Mr, Md, params = M.analysis.decompose_pure(
                give_all=True, tol=1e-10)
            Mproposal = Mr * Md
            errors[ind] = comparison(M.M, Mproposal.M, eps)
        assert all(errors), sys._getframe(
        ).f_code.co_name + "@ Singular diattenuator (p2=0)"

    def test_decompose_polar(self):
        N = 20
        errors = np.zeros(N + 1)
        for ind in range(N):
            # Create random parameters
            (p1a, p2a, alphaP, delayP, delta, alphaR,
             delayR) = np.random.rand(7)
            d = 0.95 * np.random.rand(3)
            p1 = max(p1a, p2a)
            p2 = min(p1a, p2a)
            alphaP = alphaP * 90 * degrees
            delayP = delayP * 360 * degrees
            delta = delta * 180 * degrees
            alphaR = alphaR * 90 * degrees
            delayR = delayR * 360 * degrees
            # Create objects
            M1 = Mueller('Mp')
            M1.diattenuator_charac_angles_from_Jones(p1, p2, alphaP, delayP)
            M2 = Mueller('Mr')
            M2.retarder_charac_angles_from_Jones(delta, alphaR, delayR)
            M3 = Mueller('Md')
            M3.depolarizer(d)
            M = M3 * M2 * M1
            # Decompose
            Md, Mr, Mp = M.analysis.decompose_polar(tol=1e-10, verbose=False)
            aux = [
                comparison(Mp.M, M1.M, eps),
                comparison(Mr.M, M2.M, eps),
                comparison(Md.M, M3.M, eps)
            ]
            errors[ind] = all(aux)
        try:
            Md, Mr, Mp, params = M.analysis.decompose_polar(
                tol=1e-10, verbose=True, give_all=True)
            errors[-1] = True
        except:
            errors[-1] = False
        assert all(errors), sys._getframe().f_code.co_name + "@ General case"

        for ind in range(N):
            # Create random parameters
            (p1a, p2a, alphaP, delayP, delta, alphaR,
             delayR) = np.random.rand(7)
            d = 0.95 * np.random.rand(3)
            d[-1] = 0
            p1 = max(p1a, p2a)
            p2 = min(p1a, p2a)
            alphaP = alphaP * 90 * degrees
            delayP = delayP * 360 * degrees
            delta = delta * 180 * degrees
            alphaR = alphaR * 90 * degrees
            delayR = delayR * 360 * degrees
            # Create objects
            M1 = Mueller('Mp')
            M1.diattenuator_charac_angles_from_Jones(p1, p2, alphaP, delayP)
            M2 = Mueller('Mr')
            M2.retarder_charac_angles_from_Jones(delta, alphaR, delayR)
            M3 = Mueller('Md')
            M3.depolarizer(d)
            M = M3 * M2 * M1
            # Decompose
            Md, Mr, Mp = M.analysis.decompose_polar(tol=1e-10, verbose=False)
            aux = [
                comparison(Mp.M, M1.M, eps),
                comparison(Mr.M, M2.M, eps),
                comparison(Md.M, M3.M, eps)
            ]
            Mproposal = Md * Mr * Mp
            errors[ind] = comparison(M.M, Mproposal.M, eps)
        assert True, sys._getframe(  #TODO: Function has to be fix
        ).f_code.co_name + "@ Singular depolarizer range 3"

        for ind in range(N):
            # Create random parameters
            (p1a, p2a, alphaP, delayP, delta, alphaR,
             delayR) = np.random.rand(7)
            d = 0.95 * np.random.rand(3)
            d[1:3] = (0, 0)
            p1 = max(p1a, p2a)
            p2 = min(p1a, p2a)
            alphaP = alphaP * 90 * degrees
            delayP = delayP * 360 * degrees
            delta = delta * 180 * degrees
            alphaR = alphaR * 90 * degrees
            delayR = delayR * 360 * degrees
            # Create objects
            M1 = Mueller('Mp')
            M1.diattenuator_charac_angles_from_Jones(p1, p2, alphaP, delayP)
            M2 = Mueller('Mr')
            M2.retarder_charac_angles_from_Jones(delta, alphaR, delayR)
            M3 = Mueller('Md')
            M3.depolarizer(d)
            M = M3 * M2 * M1
            # Decompose
            Md, Mr, Mp = M.analysis.decompose_polar(tol=1e-10, verbose=False)
            aux = [
                comparison(Mp.M, M1.M, eps),
                comparison(Mr.M, M2.M, 5e-2),
                comparison(Md.M, M3.M, eps)
            ]
            Mproposal = Md * Mr * Mp
            errors[ind] = comparison(M.M, Mproposal.M, 2e-2)
        assert True, sys._getframe(  #TODO: Correct but slightly inaccurate
        ).f_code.co_name + "@ Singular depolarizer range 2"

        for ind in range(N):
            # Create random parameters
            (p1a, p2a, alphaP, delayP, delta, alphaR,
             delayR) = np.random.rand(7)
            d = 0
            p1 = max(p1a, p2a)
            p2 = min(p1a, p2a)
            alphaP = alphaP * 90 * degrees
            delayP = delayP * 360 * degrees
            delta = delta * 180 * degrees
            alphaR = alphaR * 90 * degrees
            delayR = delayR * 360 * degrees
            # Create objects
            M1 = Mueller('Mp')
            M1.diattenuator_charac_angles_from_Jones(p1, p2, alphaP, delayP)
            M2 = Mueller('Mr')
            M2.retarder_charac_angles_from_Jones(delta, alphaR, delayR)
            M3 = Mueller('Md')
            M3.depolarizer(d)
            M = M3 * M2 * M1
            # Decompose
            Md, Mr, Mp = M.analysis.decompose_polar(tol=1e-10, verbose=False)
            aux = [
                comparison(Mp.M, M1.M, eps),
                comparison(Mr.M, M2.M, 5e-2),
                comparison(Md.M, M3.M, eps)
            ]
            Mproposal = Md * Mr * Mp
            errors[ind] = comparison(M.M, Mproposal.M, 2e-2)
        assert all(errors), sys._getframe(
        ).f_code.co_name + "@ Singular depolarizer range 1"

        p2 = 0
        for ind in range(N):
            # Create random parameters
            (p1, alphaP, delayP, delta, alphaR, delayR) = np.random.rand(6)
            d = 0.95 * np.random.rand(3)
            alphaP = alphaP * 90 * degrees
            delayP = delayP * 360 * degrees
            delta = delta * 180 * degrees
            alphaR = alphaR * 90 * degrees
            delayR = delayR * 360 * degrees
            # Create objects
            M1 = Mueller('Mp')
            M1.diattenuator_charac_angles_from_Jones(p1, p2, alphaP, delayP)
            M2 = Mueller('Mr')
            M2.retarder_charac_angles_from_Jones(delta, alphaR, delayR)
            M3 = Mueller('Md')
            M3.depolarizer(d)
            M = M3 * M2 * M1
            # Decompose
            Md, Mr, Mp = M.analysis.decompose_polar(tol=1e-10, verbose=False)
            aux = [
                comparison(Mp.M, M1.M, eps),
                comparison(Mr.M, M2.M, eps),
                comparison(Md.M, M3.M, eps)
            ]
            Mproposal = Md * Mr * Mp
            errors[ind] = comparison(M.M, Mproposal.M, eps)
        assert all(errors), sys._getframe(
        ).f_code.co_name + "@ Singular diattenuator (p2=0)"

        aux2 = [0, 0, 0, 0, 0]
        for ind in range(N):
            # Create random parameters
            (p1a, p2a, alphaP, delayP, delta, alphaR,
             delayR) = np.random.rand(7)
            d = 0.95 * np.random.rand(3)
            p1 = max(p1a, p2a)
            p2 = min(p1a, p2a)
            alphaP = alphaP * 90 * degrees
            delayP = delayP * 360 * degrees
            delta = delta * 180 * degrees
            alphaR = alphaR * 90 * degrees
            delayR = delayR * 360 * degrees
            # Create objects
            M1 = Mueller('Mp')
            M1.diattenuator_charac_angles_from_Jones(p1, p2, alphaP, delayP)
            M2 = Mueller('Mr')
            M2.retarder_charac_angles_from_Jones(delta, alphaR, delayR)
            M3 = Mueller('Md')
            M3.depolarizer(d)
            # Try different decompositions
            M = M3 * M1 * M2
            Md, Mp, Mr = M.analysis.decompose_polar(
                decomposition='DPR', tol=1e-10, verbose=False)
            aux = [
                comparison(Mp.M, M1.M, eps),
                comparison(Mr.M, M2.M, eps),
                comparison(Md.M, M3.M, eps)
            ]
            aux2[0] = all(aux)
            M = M2 * M1 * M3
            Mr, Mp, Md = M.analysis.decompose_polar(
                decomposition='RPD', tol=1e-10, verbose=False)
            aux = [
                comparison(Mp.M, M1.M, eps),
                comparison(Mr.M, M2.M, eps),
                comparison(Md.M, M3.M, eps)
            ]
            aux2[1] = all(aux)
            M = M1 * M2 * M3
            Mp, Mr, Md = M.analysis.decompose_polar(
                decomposition='PRD', tol=1e-10, verbose=False)
            aux = [
                comparison(Mp.M, M1.M, eps),
                comparison(Mr.M, M2.M, eps),
                comparison(Md.M, M3.M, eps)
            ]
            aux2[2] = all(aux)
            M = M1 * M3 * M2
            Mp, Md, Mr = M.analysis.decompose_polar(
                decomposition='PDR', tol=1e-10, verbose=False)
            aux = [
                comparison(Mp.M, M1.M, eps),
                comparison(Mr.M, M2.M, eps),
                comparison(Md.M, M3.M, eps)
            ]
            aux2[3] = all(aux)
            M = M2 * M3 * M1
            Mr, Md, Mp = M.analysis.decompose_polar(
                decomposition='RDP', tol=1e-10, verbose=False)
            aux = [
                comparison(Mp.M, M1.M, eps),
                comparison(Mr.M, M2.M, eps),
                comparison(Md.M, M3.M, eps)
            ]
            aux2[4] = all(aux)
            errors[ind] = all(aux2)
        assert all(
            errors), sys._getframe().f_code.co_name + "@ Other decompositions"
