import unittest

from pyIsoFit.models.generic import generic_fit


class TestGeneric(unittest.TestCase):
    def test_generic_fit(self):
        weights = [[3.40899000e-04, 5.62373000e-04, 7.43061000e-04, 1.15223800e-03,
                    1.03874520e-02, 1.01321447e-01, 2.02892257e-01, 3.03936713e-01,
                    4.05375814e-01, 5.06639453e-01, 6.08122354e-01, 7.09517545e-01,
                    8.10868903e-01, 9.12351874e-01, 1.01379105e+00]]

        y = [[0.89433671, 1.02467938, 1.09665765, 1.21675622, 1.83611977,
              2.49497702, 2.7098335, 2.85048766, 2.95980297, 3.04755176,
              3.13044807, 3.20074146, 3.26968693, 3.32791654, 3.37704777]]

        guess = {'b': [156],
                 'q': [7.93]}
        temps = [273]

        cust_bounds = {
            'b': [(0, None)],
            'q': [(0, None)]
        }

        result1_test = generic_fit("langmuir", weights, y, guess, temps,
                                   cond=False, meth='leastsq', cust_bounds=cust_bounds,
                                   fit_report=False, henry_constants=None, henry_off=False)
        result1_test_values_dict = result1_test[1]

        result1_values_dict = {0: {'q': 2.9932894017518707, 'b': 671.2016155322233}}

        for key in result1_values_dict:
            self.assertDictEqual(result1_test_values_dict[key], result1_values_dict[key])
