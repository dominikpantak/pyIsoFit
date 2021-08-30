import ast
import unittest
from pyIsoFit.ext_models.ext_dsl import ext_dsl
import numpy as np

class TestExtDSL:
    def test_ext_dsl(self):
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
        temps = [283, 313, 373]
        compnames = ['CO2', 'N2']

        file = open("xy_dict.txt", "r")
        contents = file.read()
        xy_dict1 = ast.literal_eval(contents)
        file.close()

        file = open("ext_dsl_test_result.txt", "r")
        contents = file.read()
        test_result1 = ast.literal_eval(contents)
        file.close()

        yfracs = [0.25, 0.75]

        result_ext_dsl1 = ext_dsl(params_dict1, temps, xy_dict1, compnames, yfracs)

        for key in test_result1:
            comp_qsets = test_result1[key]
            comp_qsets_result = result_ext_dsl1[key]
            for i, qset in enumerate(comp_qsets):
                np.testing.assert_array_almost_equal(comp_qsets_result[i], qset)
