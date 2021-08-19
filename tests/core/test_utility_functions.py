import unittest
from src.pyIsoFit.core.utility_functions import get_xy, plot_settings, \
    get_subplot_size, get_sorted_results, heat_calc, bounds_check, save_func
from src.pyIsoFit.models.henry import henry_approx
import pandas as pd
import numpy as np


class TestUtilityFunctions(unittest.TestCase):
    if __name__ == '__main__':
        unittest.main()

    def test_get_xy(self):
        model1 = "dsl"
        model2 = "dsl"
        model3 = "dsl"
        model4 = "linear langmuir 1"
        model5 = "linear langmuir 2"

        # Testing basic functionality
        x_test1 = [[1, 2, 3, 4], [0.5, 1, 2]]
        y_test1 = [[xi * 2 for xi in x] for x in x_test1]

        # Testing basic functionality with different number of datasets
        x_test2 = [[1., 2., 3., 4.], [0.5, 1., 2.], [5, 5.5, 6, 6.5]]
        y_test2 = [[xi * 2 for xi in x] for x in x_test2]

        # # Testing NaN filter
        x_test3 = [[1., 2., 3., 4.], [0.5, 1., 1.5, 2.], [5, 5.5, 6.5]]
        y_test3 = [[xi * 2 for xi in x] for x in x_test3]

        # Testing langmuir linear 1
        x_test4 = [[1, 2, 3, 4], [0.5, 1, 1.5, 2]]
        y_test4 = [[2, 4, 6, 8], [1, 2, 3, 4]]

        # # Testing langmuir linear 2
        x_test5 = [[1, 2, 3, 4], [0.5, 1, 1.5, 2]]
        y_test5 = [[2, 2, 2, 2], [2, 2, 2, 2]]

        df1 = pd.read_csv('../testing library/test_get_xy1.csv')
        df2 = pd.read_csv('../testing library/test_get_xy2.csv')
        df3 = pd.read_csv('../testing library/test_get_xy3.csv')
        df4 = pd.read_csv('../testing library/test_get_xy4.csv')
        df5 = pd.read_csv('../testing library/test_get_xy5.csv')

        key_x1 = ['x1', 'x2']
        key_x2 = ['x1', 'x2', 'x3']
        key_x3 = ['x1', 'x2', 'x3']
        key_x4 = ['x1', 'x2']
        key_x5 = ['x1', 'x2']

        key_y1 = ['y1', 'y2']
        key_y2 = ['y1', 'y2', 'y3']
        key_y3 = ['y1', 'y2', 'y3']
        key_y4 = ['y1', 'y2']
        key_y5 = ['y1', 'y2']

        x1, y1 = get_xy(df1, key_x1, key_y1, model1, False)
        x2, y2 = get_xy(df2, key_x2, key_y2, model2, False)
        x3, y3 = get_xy(df3, key_x3, key_y3, model3, False)
        x4, y4 = get_xy(df4, key_x4, key_y4, model4, False)
        x5, y5 = get_xy(df5, key_x5, key_y5, model5, False)

        for i in range(len(x_test1)):
            # for j in range(len(x_test1[i])):
            np.testing.assert_array_equal(x1[i], x_test1[i])
            np.testing.assert_array_equal(y1[i], y_test1[i])

            np.testing.assert_array_equal(x2[i], x_test2[i])
            np.testing.assert_array_equal(y2[i], y_test2[i])

            np.testing.assert_array_equal(x3[i], x_test3[i])
            np.testing.assert_array_equal(y3[i], y_test3[i])

            np.testing.assert_array_equal(x4[i], x_test4[i])
            np.testing.assert_array_equal(y4[i], y_test4[i])

            np.testing.assert_array_equal(x5[i], x_test5[i])
            np.testing.assert_array_equal(y5[i], y_test5[i])

    def test_plot_settings(self):
        model1 = "dsl"
        model2 = "langmuir linear 1"
        model3 = "langmuir linear 2"
        model4 = "mdr"


        xtitle_test1, ytitle_test1, logx_test1, logy_test1 = plot_settings(True, model1, False, True)
        xtitle_test2, ytitle_test2, logx_test2, logy_test2 = plot_settings(False, model2, False, True)
        xtitle_test3, ytitle_test3, logx_test3, logy_test3 = plot_settings((False, True), model3, False, True)
        xtitle_test4, ytitle_test4, logx_test4, logy_test4 = plot_settings(False, model4, True, True)

        self.assertEqual(xtitle_test1, "Pressure [bar]")
        self.assertEqual(ytitle_test1, "Uptake [mmol/g]")
        self.assertEqual(logx_test1, True)
        self.assertEqual(logy_test1, True)

        self.assertEqual(xtitle_test2, "1/Pressure [1/bar]")
        self.assertEqual(ytitle_test2, "1/Uptake [g/mmol]")
        self.assertEqual(logx_test2, False)
        self.assertEqual(logy_test2, False)

        self.assertEqual(xtitle_test3, "Pressure [bar]")
        self.assertEqual(ytitle_test3, "Pressure/uptake [(bar mmol)/g]")
        self.assertEqual(logx_test3, False)
        self.assertEqual(logy_test3, True)

        self.assertEqual(xtitle_test4, "Relative pressure [P/P]")
        self.assertEqual(ytitle_test4, "Uptake [mmol/g]")
        self.assertEqual(logx_test4, False)
        self.assertEqual(logy_test4, False)

    def test_get_subplot_size(self):
        lenx_test1 = 3
        lenx_test2 = 6
        lenx_test3 = 12

        i_test1 = 2
        i_test2 = 4
        i_test3 = 10

        result1 = 2, 2, 3
        result2 = 3, 3, 5
        result3 = 4, 4, 11

        self.assertEqual(result1, get_subplot_size(lenx_test1, i_test1))
        self.assertEqual(result2, get_subplot_size(lenx_test2, i_test2))
        self.assertEqual(result3, get_subplot_size(lenx_test3, i_test3))

    def test_get_sorted_results(self):
        values_dict_test1 = {
            0: {'q': 3, 'delta': 1822, 'b': 607},
            1: {'q': 6, 'delta': 281, 'b': 93},
            2: {'q': 9, 'delta': 156, 'b': 52},
            3: {'q': 12, 'delta': 5, 'b': 1}
        }

        result_c_list_test1 = [
            [3, 607],
            [6, 93],
            [9, 52],
            [12, 1]]

        result_dict_test1 = {
            'T (K)': [298, 323, 348, 373],
            'q (mmol/g)': [3, 6, 9, 12],
            'b (1/bar)': [607, 93, 52, 1]
        }

        result_dict1, c_list1 = get_sorted_results(values_dict_test1, "langmuir", [298, 323, 348, 373])

        self.assertDictEqual(result_dict_test1, result_dict1)
        self.assertListEqual(result_c_list_test1, c_list1)

    def test_heat_calc(self):
        temps_test1 = [273, 293, 313]
        model = "langmuir"
        param_dict = {'b (1/bar)': [80.1737834, 81.52710876, 82.7074852]}
        heat1_test = heat_calc(model, temps_test1, param_dict, testing=True)
        heat1 = 0.5526914176979378

        self.assertEqual(heat1, heat1_test)

    def test_bounds_check(self):
        result_test1 = {
                'q': [(0, None), (0, None)],
                'b': [(0, None), (0, None)]
            }

        test1 = bounds_check("langmuir", None, 2)

        self.assertEqual(test1, result_test1)

        test2 = bounds_check("langmuir", result_test1, 2)

        self.assertEqual(test2, result_test1)

    # def test_save_func(self):
    #     directory = '../testing library/'
    #     fit_name = 'test_save_func'
    #     filetype = '.csv'
    #     d = {'col1': [1, 2], 'col2': [3, 4]}
    #     df = pd.DataFrame.from_dict(d)
    #     save_func(directory, fit_name, filetype, df)
    #     df_test = pd.read_csv('../testing library/test_save_func.csv')
    #
    #     pd.testing.assert_frame_equal(df, df_test)





















