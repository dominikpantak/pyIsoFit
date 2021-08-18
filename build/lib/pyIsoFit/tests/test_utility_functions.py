import unittest
from src.pyIsoFit.core.utility_functions import get_xy, henry_approx
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
        x_test4 = [[1, 0.5, 0.333333333, 0.25], [2, 1, 0.666666667, 0.5]]
        y_test4 = [[0.5, 0.25, 0.166666667, 0.125], [1, 0.5, 0.333333333, 0.25]]

        # # Testing langmuir linear 2
        # x_test5 = [[1, 2, 3, 4], [0.5, 1, 1.5, 2]]
        # y_test5 = [[2, 2, 2, 2], [2, 2, 2, 2]]

        df1 = pd.read_csv('testing library/test_get_xy1.csv')
        df2 = pd.read_csv('testing library/test_get_xy2.csv')
        df3 = pd.read_csv('testing library/test_get_xy3.csv')
        # df4 = pd.read_csv('../testing library/test_get_xy4.csv')
        # df5 = pd.read_csv('../testing library/test_get_xy5.csv')

        key_x1 = ['x1', 'x2']
        key_x2 = ['x1', 'x2', 'x3']
        key_x3 = ['x1', 'x2', 'x3']
        # key_x4 = ['x1', 'x2']
        # key_x5 = ['x1', 'x2']

        key_y1 = ['y1', 'y2']
        key_y2 = ['y1', 'y2', 'y3']
        key_y3 = ['y1', 'y2', 'y3']
        # key_y4 = ['y1', 'y2']
        # key_y5 = ['y1', 'y2']

        x1, y1 = get_xy(df1, key_x1, key_y1, model1, False)
        x2, y2 = get_xy(df2, key_x2, key_y2, model2, False)
        x3, y3 = get_xy(df3, key_x3, key_y3, model3, False)
        # x4, y4 = get_xy(df4, key_x4, key_y4, model4, False)
        # x5, y5 = get_xy(df5, key_x5, key_y5, model5, False)

        for i in range(len(x_test1)):
            # for j in range(len(x_test1[i])):
            np.testing.assert_array_equal(x1[i], x_test1[i])
            np.testing.assert_array_equal(y1[i], y_test1[i])

            np.testing.assert_array_equal(x2[i], x_test2[i])
            np.testing.assert_array_equal(y2[i], y_test2[i])

            np.testing.assert_array_equal(x3[i], x_test3[i])
            np.testing.assert_array_equal(y3[i], y_test3[i])

            # self.assertEqual(x4[i][j], x_test4[i][j])
            # self.assertEqual(y4[i][j], y_test4[i][j])

            # self.assertEqual(x5[i][j], x_test5[i][j])
            # self.assertEqual(y5[i][j], y_test5[i][j])

            # self.assertEqual((x2, y2), (x_test2, y_test2))
            # self.assertEqual((x3, y3), (x_test3, y_test3))
            # self.assertEqual((x4, y4), (x_test4, y_test4))
            # self.assertEqual((x5, y5), (x_test5, y_test5))

    def test_henry_approx(self):

        df1 = pd.read_csv('Datasets for testing/Computational Data (EPFL) CO2.csv')
        key_uptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
        key_pressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']

        df_test1 = pd.read_csv('testing library/test_henry1.csv')
        h_const_test1 = [1135.4699824999998, 178.25943, 11.188025249999999]

        h_const1, df_henry1, xy_dict1 = henry_approx(df1, key_pressures, key_uptakes)
        pd.testing.assert_frame_equal(df_henry1, df_test1)
        self.assertListEqual(h_const_test1, h_const1)
        xy_dict_test1 = {'x': [[0., 0.0001, 0.0002],
                               [0., 0.0001, 0.0002],
                               [0., 0.0001, 0.0002, 0.0004, 0.0007, 0.001, 0.002]],
                         'y': [[0., 0.1258678, 0.24687659], [0., 0.01779353, 0.03529044],
                               [0., 0.00126096, 0.00231325, 0.0046578, 0.00791453, 0.01116082, 0.02275687]]}

        for key in xy_dict_test1:
            for i in range(len(xy_dict_test1[key])):
                np.testing.assert_array_almost_equal(xy_dict_test1[key][i], xy_dict1[key][i])

        h_const2, df_henry2, xy_dict2 = henry_approx(df1, key_pressures, key_uptakes, tol='')


