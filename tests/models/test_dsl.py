import pandas as pd
import unittest
from pyIsoFit.models.dsl import dsl_fit
import ast
import numpy as np


class TestDSL(unittest.TestCase):
    def test_dsl_fit(self):

        df1 = pd.read_csv('../Datasets for testing/Computational Data (EPFL) CO2.csv')
        df2 = pd.read_csv('../Datasets for testing/Computational Data (EPFL) N2.csv')
        df1_test = pd.read_csv('../testing library/test_dsl_CO2.csv')
        df2_test = pd.read_csv('../testing library/test_dsl_N2.csv')

        df_res_dict1 = {'CO2': df1_test, 'N2': df2_test}

        df_list = [df1, df2]
        key_uptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
        key_pressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']
        temps = [283, 313, 373]
        compnames = ['CO2', 'N2']
        meth = 'tnc'
        guess = None
        hentol = 0.999
        show_hen = True
        henry_off = False
        dsl_comp_a = None

        params_dict1 = {'CO2':
            {
                'q1': [3.731195104671767, 3.731195104671767, 3.731195104671767],
                'q2': [3.351730760420434, 3.351730760420434, 3.351730760420434],
                'h1': [-45822.53714881158, -45822.002733165406, -45822.736145063514],
                'h2': [-36142.880987485754, -36142.74581059616, -36143.2175196681],
                'b01': [9.66575455452201e-07, 1.0006643971216533e-06, 1.0459065349355257e-06],
                'b02': [9.298406538071902e-07, 8.154135935356521e-07, 9.270610581424421e-07]},
            'N2': {
                'q1': [3.731195104671767, 3.731195104671767, 3.731195104671767],
                'q2': [3.351730760420434, 3.351730760420434, 3.351730760420434],
                'h1': [-15809.239894595705, -15809.377336371223, -15809.5801474471],
                'h2': [-15809.239894595705, -15809.377336371223, -15809.5801474471],
                'b01': [5.072976049169675e-05, 5.305097686614246e-05, 5.769571321101452e-05],
                'b02': [5.072976049169675e-05, 5.305097686614246e-05, 5.769571321101452e-05]
            }
        }

        file = open("xy_dict.txt", "r")
        contents = file.read()
        xy_dict1 = ast.literal_eval(contents)
        file.close()

        result_test1 = dsl_fit(df_list, key_pressures, key_uptakes, temps, compnames, meth,
                               guess, hentol, show_hen, henry_off, dsl_comp_a)

        xy_dict_test1, results_dict_test1, df_res_dict_test1, params_dict_test1 = result_test1

        for key in xy_dict1:
            x_test = xy_dict_test1[key][0]
            y_test = xy_dict_test1[key][1]
            x = xy_dict1[key][0]
            y = xy_dict1[key][1]
            for j in range(len(x)):
                np.testing.assert_array_almost_equal(x[j], x_test[j])
                np.testing.assert_array_almost_equal(y[j], y_test[j])

        for key in df_res_dict_test1:
            pd.testing.assert_frame_equal(df_res_dict1[key], df_res_dict_test1[key], atol=1e-3)

        for comp in params_dict1:
            self.assertDictEqual(params_dict1[comp], params_dict_test1[comp])