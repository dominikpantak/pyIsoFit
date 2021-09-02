from src.pyIsoFit.core.model_fit_def import get_guess_params, get_fit_tuples
import numpy as np
import pandas as pd
import unittest

class TestModelFitDef(unittest.TestCase):
    def test_get_guess_params(self):
        df1 = pd.read_csv('../Datasets for testing/Computational Data (EPFL) CO2.csv')
        key_uptakes = ['Uptake (mmol/g)_13X_10 (°C)']
        key_pressures = ['Pressure (bar)']

        result1 = get_guess_params("langmuir", df1, key_uptakes, key_pressures)

        guess_result_test1 = {'b': [155.72513521595482],
                              'q': [7.926677561000001]}

        for key in guess_result_test1:
            np.testing.assert_array_almost_equal(guess_result_test1[key], result1[key])

        guess_result_test2 = {
             'b1': [62.29005408638193],
             'b2': [93.4350811295729],
             'q1': [3.9633387805000004],
             'q2': [3.9633387805000004]}

        result2 = get_guess_params("dsl", df1, key_uptakes, key_pressures)
        result2_d = get_guess_params("dsl nc", df1, key_uptakes, key_pressures)

        for key in guess_result_test2:
            np.testing.assert_array_almost_equal(guess_result_test2[key], result2[key])
            np.testing.assert_array_almost_equal(guess_result_test2[key], result2_d[key])

        guess_result_test3 = {
            'b0': [155.72513521595482],
            'q': [7.926677561000001],
            'h': [-5000]
        }

        result3 = get_guess_params("langmuir td", df1, key_uptakes, key_pressures)

        for key in guess_result_test3:
            np.testing.assert_array_almost_equal(guess_result_test3[key], result3[key])

        guess_result_test4 = {
            'n': [1.585335512],
            'ka': [1],
            'ca': [0.45]
        }

        result4 = get_guess_params("gab", df1, key_uptakes, key_pressures)

        for key in guess_result_test4:
            np.testing.assert_array_almost_equal(guess_result_test4[key], result4[key])

        guess_result_test5 = {
            'n0': [7.926677561000001],
            'n1': [155.72513521595482],
            'a': [15.57251352],
            'c': [1557.251352]
        }

        result5 = get_guess_params("mdr", df1, key_uptakes, key_pressures)

        for key in guess_result_test5:
            np.testing.assert_array_almost_equal(guess_result_test5[key], result5[key])

        guess_result_test6 = {
            'b': [155.72513521595482],
            'q': [7.926677561000001],
            'n': [1]
        }

        result6 = get_guess_params("sips", df1, key_uptakes, key_pressures)

        for key in guess_result_test6:
            np.testing.assert_array_almost_equal(guess_result_test6[key], result6[key])

        guess_result_test7 = {
            'b': [155.72513521595482],
            'q': [7.926677561000001],
            't': [0.5]
        }

        result7 = get_guess_params("toth", df1, key_uptakes, key_pressures)

        for key in guess_result_test7:
            np.testing.assert_array_almost_equal(guess_result_test7[key], result7[key])

        guess_result_test8 = {
            "c": [155.72513521595482],
            "n": [10],
            "g": [100],
            "q": [3.963338781]
        }

        result8 = get_guess_params("bddt", df1, key_uptakes, key_pressures)

        for key in guess_result_test8:
            np.testing.assert_array_almost_equal(guess_result_test8[key], result8[key])

        guess_result_test9 = {
            "ns": [7.926677561000001],
            "kf": [155.72513521595482],
            "nu": [79.26677561000001],
            "ku": [155.72513521595482],
            "m": [5]
        }

        result9 = get_guess_params("dodo", df1, key_uptakes, key_pressures)

        for key in guess_result_test9:
            np.testing.assert_array_almost_equal(guess_result_test9[key], result9[key])

        guess_result_test10 = {
            'c': [155.72513521595482],
            'n': [7.926677561000001]
        }

        result10 = get_guess_params("bet", df1, key_uptakes, key_pressures)

        for key in guess_result_test10:
            np.testing.assert_array_almost_equal(guess_result_test10[key], result10[key])


    def test_get_fit_tuples(self):
        guess1 = {'b': [155.7, 23.8, 1.9],
                 'q': [7, 7.25, 5.75]}

        std_data = {
            'temps': [10, 40, 100],
            'cust_bounds': None,
            'henry_constants': [1234, 172, 11],
            'q_fix': 7
        }

        def assert_tuples(left, **kwargs):
            for i, tuple_set in enumerate(left):
                test = get_fit_tuples(i=i, **kwargs, **std_data)

                for j, tup in enumerate(tuple_set):
                    tup_test = test[j]
                    for k, item in enumerate(tup):
                        self.assertEqual(item, tup_test[k])


        result1 = (('q', 7, True, 0, None), ('delta', 1234, False), ('b', None, None, None, None, 'delta/q')), \
                  (('q', 7, True, 7, 7.001), ('delta', 172, False), ('b', None, None, None, None, 'delta/q')),\
                  (('q', 7, True, 7, 7.001), ('delta', 11, False), ('b', None, None, None, None, 'delta/q'))

        test1_kwargs = {
            'model': "langmuir",
            'guess': guess1,
            'cond': True,
            'henry_off': False}

        assert_tuples(result1, **test1_kwargs)

        result2 = (('q', 7, True, 0, None), ('b', 155.7, True, 0, None)), \
                  (('q', 7, True, 7, 7.001), ('b', 23.8, True, 0, None)), \
                  (('q', 7, True, 7, 7.001), ('b', 1.9, True, 0, None))

        test2_kwargs = {
            'model': "langmuir",
            'guess': guess1,
            'cond': True,
            'henry_off': True}

        assert_tuples(result2, **test2_kwargs)

        result3 = (('q', 7, True, 0, None), ('b', 155.7, True, 0, None)), \
                  (('q', 7.25, True, 0, None), ('b', 23.8, True, 0, None)), \
                  (('q', 5.75, True, 0, None), ('b', 1.9, True, 0, None))

        test3_kwargs = {
            'model': "langmuir",
            'guess': guess1,
            'cond': False,
            'henry_off': False}

        assert_tuples(result3, **test3_kwargs)

        guess2 = {
             'q1': [3, 2],
             'q2': [3, 2],
            'b1': [62, 32],
            'b2': [93, 32]
        }

        result4 = (('q1', 3, True, 0, None), ('q2', 3, True, 0, None), ('b1', 62, True, 0, None), ('b2', 93, True, 0, None)), \
                  (('q1', 2, True, 0, None), ('q2', 2, True, 0, None), ('b1', 32, True, 0, None),\
                   ('b2', 32, True, 0, None))

        test4_kwargs = {
            'model': "dsl",
            'guess': guess2
        }

        assert_tuples(result4, **test4_kwargs)

        guess3 = {
            'b0': [155, 140],
            'q': [7, 6],
            'h': [-5000, -5000]
        }

        result5 = (('t', 10, False), ('q', 7, True, 0, None), ('h', -5000, True, None, None), ('b0', 155, True, 0, None)),\
                  (('t', 40, False), ('q', 7, True, 7, 7.001), ('h', -5000, True, None, None), ('b0', 140, True, 0, None))


        test5_kwargs = {
            'model': "langmuir td",
            'guess': guess3,
            'cond': True,
            'henry_off': False}

        assert_tuples(result5, **test5_kwargs)

        result6 = (('t', 10, False), ('q', 7, True, 0, None), ('h', -5000, True, None, None), ('b0', 155, True, 0, None)),\
                  (('t', 40, False), ('q', 6, True, 0, None), ('h', -5000, True, None, None), ('b0', 140, True, 0, None))

        test6_kwargs = {
            'model': "langmuir td",
            'guess': guess3,
            'cond': False,
            'henry_off': False}

        assert_tuples(result6, **test6_kwargs)

        guess4 = {
            'n': [15, 14],
            'ka': [7, 6],
            'ca': [1, 2]
        }

        result7 = (('n', 15, True, 0, None), ('ka', 7, True, 0, None), ('ca', 1, True, 0, None)), \
                  (('n', 14, True, 0, None), ('ka', 6, True, 0, None), ('ca', 2, True, 0, None))

        test7_kwargs = {
            'model': "gab",
            'guess': guess4}

        assert_tuples(result7, **test7_kwargs)

        guess5 = {
            'n0': [7, 6],
            'n1': [100, 120],
            'a': [1, 2],
            'c': [3, 4]
        }

        result8 = (('n0', 7, True, 0, None), ('n1', 100, True, 0, None), ('a', 1, True, 0, None),
                   ('c', 3, True, 0, None)), \
                  (('n0', 6, True, 0, None), ('n1', 120, True, 0, None), ('a', 2, True, 0, None), \
                   ('c', 4, True, 0, None))

        test8_kwargs = {
            'model': "mdr",
            'guess': guess5,
            'cond': False,
            'henry_off': False}

        assert_tuples(result8, **test8_kwargs)

        result9 = (('n0', 7, True, 0, None), ('n1', 100, True, 0, None), ('a', 1, True, 0, None),
                   ('c', 3, True, 0, None)), \
                  (('n0', 7, True, 7, 7.001), ('n1', 120, True, 0, None), ('a', 2, True, 0, None), \
                   ('c', 4, True, 0, None))

        test9_kwargs = {
            'model': "mdr",
            'guess': guess5,
            'cond': True,
            'henry_off': True}

        assert_tuples(result9, **test9_kwargs)

        result10 = (('n0', 7, True, 0, None),('delta', 1234, False), ('n1', None, None, None, None, 'delta/n0'), ('a', 1, True, 0, None),
                   ('c', 3, True, 0, None)), \
                  (('n0', 7, True, 7, 7.001),('delta', 172, False), ('n1', None, None, None, None, 'delta/n0'), ('a', 2, True, 0, None), \
                   ('c', 4, True, 0, None))

        test10_kwargs = {
            'model': "mdr",
            'guess': guess5,
            'cond': True,
            'henry_off': False}

        assert_tuples(result10, **test10_kwargs)

        guess6 = {
            'q': [5, 4],
            'b': [100, 90],
            'n': [1, 0.9]
        }

        result11 = (('q', 5, True, 0, None), ('b', 100, True, 0, None), ('n', 1, True, 0, None)), \
                  (('q', 4, True, 0, None), ('b', 90, True, 0, None), ('n', 0.9, True, 0, None))

        test11_kwargs = {
            'model': "sips",
            'guess': guess6
        }

        assert_tuples(result11, **test11_kwargs)

        guess7 = {
            'q': [5, 4],
            'b': [100, 90],
            't': [1, 0.9]
        }

        result12 = (('q', 5, True, 0, None), ('b', 100, True, 0, None), ('t', 1, True, 0, None)), \
                   (('q', 4, True, 0, None), ('b', 90, True, 0, None), ('t', 0.9, True, 0, None))

        test12_kwargs = {
            'model': "toth",
            'guess': guess7
        }

        assert_tuples(result12, **test12_kwargs)

        guess8 = {
            'c': [0.1, 1],
            'n': [10, 20],
            'g': [99, 100],
            'q': [6, 5]
        }

        result13 = (('c', 0.1, True, 0, None), ('n', 10, True, 0, None), ('g', 99, True, 0, None),
                   ('q', 6, True, 0, None)), \
                  (('c', 1, True, 0, None), ('n', 20, True, 0, None), ('g', 100, True, 0, None), \
                   ('q', 5, True, 0, None))

        test13_kwargs = {
            'model': "bddt",
            'guess': guess8
        }

        assert_tuples(result13, **test13_kwargs)

        result14 = (('c', 0.1, True, 0, 1), ('n', 10, True, 0, None), ('g', 99, True, 0, None),
                   ('q', 6, True, 0, None)), \
                  (('c', 1, True, 0, 1), ('n', 20, True, 0, None), ('g', 100, True, 0, None), \
                   ('q', 5, True, 0, None))

        test14_kwargs = {
            'model': "bddt",
            'guess': guess8,
            'cond': True
        }

        assert_tuples(result14, **test14_kwargs)

        guess9 = {
            'ns': [1, 2],
            'kf': [3, 4],
            'nu': [5, 6],
            'ku': [7, 8],
            'm': [9, 10]
        }

        result15 = (('ns', 1, True, 0, None), ('kf', 3, True, 0, None), ('nu', 5, True, 0, None),
                    ('ku', 7, True, 0, None), ('m', 9, True, 0, None)), \
                   (('ns', 2, True, 0, None), ('kf', 4, True, 0, None), ('nu', 6, True, 0, None),
                    ('ku', 8, True, 0, None), ('m', 10, True, 0, None))

        test15_kwargs = {
            'model': "dodo",
            'guess': guess9
        }

        assert_tuples(result15, **test15_kwargs)

        guess10 = {
            'n': [10, 11],
            'c': [12, 13]
        }

        result16 = (('n', 10, True, 0, None), ('c', 12, True, 0, None)), \
                   (('n', 11, True, 0, None), ('c', 13, True, 0, None))

        test16_kwargs = {
            'model': "bet",
            'guess': guess10
        }

        assert_tuples(result16, **test16_kwargs)




