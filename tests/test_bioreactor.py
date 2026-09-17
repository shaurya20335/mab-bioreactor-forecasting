"""Numerical, conservation, and domain checks for the bioreactor model."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from bioreactor import (
    CSV_COLUMNS, DEFAULT_INITIAL_STATE, ModelDomainError, Operation, Parameters,
    export_csv, observations, rhs, simulate,
)


class BioreactorTests(unittest.TestCase):
    def test_published_lactate_equation_signed_glucose_and_grouping(self):
        # Hand-computable case: mu=.5, q_glucose=-.35,
        # q_lactate=(.5/4 + .4*(-.35))*(1-2/10) + .01*(1-2/5) = -.006.
        parameters = replace(Parameters(), mumax=1, ks=0, kp=0, k1=0,
                             kl=2, yxs=2, ms=.1, yxl=4, yls=.4,
                             lacmax1=10, lacmax2=5, mlac=.01)
        derivative = rhs(0, [1000, 3, 0, 1, 2, 0], parameters)
        self.assertAlmostEqual(derivative[3], -1.05)
        self.assertAlmostEqual(derivative[4], -.018)

    def test_closed_batch_death_conserves_total_cells(self):
        parameters = replace(Parameters(), mumax=0, ms=0, mlac=0, mp=0)
        times = np.linspace(0, 1000, 21)
        states = simulate(times, parameters=parameters)
        death_rate = parameters.mud * np.exp(-parameters.k2 / 308)
        assert_allclose(states[:, 1], .7 * np.exp(-death_rate * times), rtol=1e-9)
        assert_allclose(states[:, 1] + states[:, 5], .7 + .7 / 99, rtol=1e-10)
        self.assertLess(observations(states)[-1, 5], 99)

    def test_flows_obey_extensive_balances(self):
        state = np.array([5000., 2., 20., 5., 1., .3])
        base = rhs(0, state)
        operation = Operation(inlet_flow=10, sample_flow=2, perfusion_flow=3)
        derivative = rhs(0, state, operation=operation)
        volume_rate = derivative[0]
        # Net rates of concentration*volume due to flows alone.
        flows = (derivative[1:] - base[1:]) * state[0] + state[1:] * volume_rate
        assert_allclose(flows, [-4, -40, 10 * 18 - 5 * 5, -5, -.6])

    def test_sample_only_removal_keeps_concentrations_constant(self):
        parameters = replace(Parameters(), mumax=0, mud=0, ms=0, mlac=0, mp=0)
        times = np.linspace(0, 100, 11)
        states = simulate(times, parameters=parameters, operation=Operation(sample_flow=2))
        assert_allclose(states[:, 0], 5650 - 2 * times)
        assert_allclose(states[:, 1:], np.tile(DEFAULT_INITIAL_STATE[1:], (11, 1)))

    def test_cell_free_fed_batch_matches_analytic_mixing(self):
        initial = [1000., 0., 20., 5., 0., 0.]
        operation = Operation(inlet_flow=10)
        times = np.linspace(0, 100, 21)
        states = simulate(times, initial, operation=operation)
        volumes = 1000 + 10 * times
        assert_allclose(states[:, 0], volumes)
        assert_allclose(states[:, 2], 20000 / volumes, rtol=1e-8)
        assert_allclose(states[:, 3], (5000 + 180 * times) / volumes, rtol=1e-8)
        assert_allclose(observations(states)[:, 5], 0)

    def test_no_live_cells_with_dead_cells_has_zero_viability(self):
        states = simulate([0, 100], [1000., 0., 0., 0., 0., 1.])
        assert_allclose(observations(states)[:, 5], 0)
        assert_allclose(states[:, 5], 1)

    def test_default_run_is_physical_and_tolerance_stable(self):
        times = np.linspace(0, 5000, 101)
        states = simulate(times)
        tighter = simulate(times, rtol=1e-10, atol=1e-12)
        assert_allclose(states, tighter, rtol=1e-7, atol=1e-9)
        self.assertTrue(np.all(states >= 0))
        assert_allclose(states[:, 0], 5650)
        viability = observations(states)[:, 5]
        self.assertTrue(np.all((viability >= 0) & (viability <= 100)))
        self.assertGreater(states[-1, 2], 0)
        self.assertLess(states[-1, 3], states[0, 3])

    def test_long_batch_stops_at_glucose_depletion(self):
        with self.assertRaisesRegex(ModelDomainError, r"glucose depletion at t="):
            simulate([0, 15000])

    def test_zero_glucose_with_live_cells_is_rejected(self):
        initial = list(DEFAULT_INITIAL_STATE)
        initial[3] = 0
        with self.assertRaisesRegex(ModelDomainError, "glucose depletion"):
            simulate([0, 100], initial)

    def test_lactate_depletion_is_detected(self):
        parameters = replace(Parameters(), mumax=0, mlac=0, ms=.01, yls=1)
        with self.assertRaisesRegex(ModelDomainError, "lactate depletion"):
            simulate([0, 100], parameters=parameters)

    def test_product_inhibition_limit_is_detected(self):
        initial = list(DEFAULT_INITIAL_STATE)
        initial[2] = 1 / Parameters().kp - .001
        with self.assertRaisesRegex(ModelDomainError, "inhibition limit"):
            simulate([0, 100], initial)

    def test_emptying_reactor_is_rejected(self):
        with self.assertRaisesRegex(ModelDomainError, "empty the reactor"):
            simulate([0, 1000], operation=Operation(sample_flow=10))

    def test_invalid_inputs_are_rejected(self):
        for times in ([0], [1, 0], [0, 0], [0, np.nan], [[0, 1]]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                simulate(times)
        for initial in ([0, 1, 0, 1, 1, 0], [1, -1, 0, 1, 1, 0], [1, 2]):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                simulate([0, 100], initial)
        for kwargs in ({"temperature": 0}, {"inlet_flow": -1}, {"feed_glucose": np.nan}):
            with self.subTest(operation=kwargs), self.assertRaises(ValueError):
                Operation(**kwargs)
        for kwargs in ({"yxs": 0}, {"kl": 0}, {"ms": -1}, {"kp": np.inf}):
            with self.subTest(parameters=kwargs), self.assertRaises(ValueError):
                Parameters(**kwargs)

    def test_solver_failure_is_reported(self):
        with patch("bioreactor.solve_ivp") as solver:
            solver.return_value.success = False
            solver.return_value.message = "test solver failure"
            with self.assertRaisesRegex(RuntimeError, "test solver failure"):
                simulate([0, 100])

    def test_export_retains_original_schema_and_viability(self):
        times = np.array([0, 10, 20])
        states = simulate(times)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            export_csv(path, times, states)
            self.assertEqual(path.read_text().splitlines()[0], ",".join(CSV_COLUMNS))
            data = np.loadtxt(path, delimiter=",", skiprows=1)
        assert_allclose(data[:, 0], times)
        assert_allclose(data[:, 1:], observations(states), rtol=1e-10)
        self.assertEqual(data[0, -1], 99)


if __name__ == "__main__":
    unittest.main()
