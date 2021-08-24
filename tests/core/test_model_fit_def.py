from src.pyIsoFit.core.model_fit_def import get_guess_params, get_fit_tuples
import numpy as np
import pandas as pd
import unittest

class TestModelFitDef(unittest.TestCase):
    def test_get_guess_params(self):
        df1 = pd.read_csv('../Datasets for testing/Computational Data (EPFL) CO2.csv')
        key_uptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
        key_pressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']

        result1 = get_guess_params("langmuir", df1, key_uptakes, key_pressures)

        guess_result_test1 = {'b': [155.72513521595482, 23.867342457899817, 1.964459201401032],
                              'q': [7.926677561000001, 7.2472832828, 5.755511305500001]}

        for key in guess_result_test1:
            np.testing.assert_array_almost_equal(guess_result_test1[key], result1[key])

        guess_result_test2 = {
            'b1': [62.29005408638193, 9.546936983159927, 0.7857836805604128],
             'b2': [93.4350811295729, 14.32040547473989, 1.1786755208406192],
             'q1': [3.9633387805000004, 3.6236416414, 2.8777556527500003],
             'q2': [3.9633387805000004, 3.6236416414, 2.8777556527500003]}

        result2 = get_guess_params("dsl", df1, key_uptakes, key_pressures)

        for key in guess_result_test2:
            np.testing.assert_array_almost_equal(guess_result_test2[key], result2[key])


    def test_get_fit_tuples(self):
        model = "langmuir"
        guess = {'b': [155.7, 23.8, 1.9],
                 'q': [7.92, 7.25, 5.75]}
        temps = [10, 40, 100]
        cond = True
        cust_bounds = None
        henry_constants = [1234, 172, 11]
        henry_off = False
        q_fix = 7.92

        result1 = (('q', 7.92, True, 0, None), ('delta', 1234, False), ('b', None, None, None, None, 'delta/q'),
                   ('q', 7.92, True, 7.92, 7.921), ('delta', 172, False), ('b', None, None, None, None, 'delta/q'),
                   ('q', 7.92, True, 7.92, 7.921), ('delta', 11, False), ('b', None, None, None, None, 'delta/q'))


        assert False
