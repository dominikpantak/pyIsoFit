import pandas as pd
import numpy as np
import unittest
from src.pyIsoFit.models.henry import henry_approx

class HenryTest(unittest.TestCase):
    def test_henry_approx(self):

        df1 = pd.read_csv('../Datasets for testing/Computational Data (EPFL) CO2.csv')
        key_uptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
        key_pressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']

        df_test1 = pd.read_csv('../testing library/test_henry1.csv')
        h_const_test1 = [1234.382935, 172.973392, 11.306467]

        h_const1, df_henry1, xy_dict1 = henry_approx(df1, key_pressures, key_uptakes)
        pd.testing.assert_frame_equal(df_henry1, df_test1)

        np.testing.assert_array_almost_equal(h_const_test1, h_const1)

        xy_dict_test1 = {'x': [[0., 0.0001, 0.0002], [0., 0.0001, 0.0002, 0.0004, 0.0007, 0.001],
                               [0., 0.0001, 0.0002, 0.0004, 0.0007]],
                         'y': [[0., 0.1258678, 0.24687659], [0., 0.01779353, 0.03529044, 0.07130377,
                                                             0.11906816, 0.17297339],
                               [0., 0.00126096, 0.00231325, 0.0046578, 0.00791453]]}

        for key in xy_dict_test1:
            for i in range(len(xy_dict_test1[key])):
                np.testing.assert_array_almost_equal(xy_dict_test1[key][i], xy_dict1[key][i])

        df_test2 = pd.read_csv('../testing library/test_henry2.csv')
        h_const_test2 = [1009.113834, 161.477952, 11.306467]

        h_const2, df_henry2, xy_dict2 = henry_approx(df1, key_pressures, key_uptakes, tol=[0.0008, 0.004, 0.001])
        pd.testing.assert_frame_equal(df_henry2, df_test2)
        np.testing.assert_array_almost_equal(h_const_test2, h_const2)
