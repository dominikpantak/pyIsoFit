"""
Module for functions associated with model definitions.

"""
from src.pyIsofit.core.utility_functions import henry_approx, bounds_check


def get_guess_params(model, df, key_uptakes, key_pressures):
    """
    Function for automatic initial guess value calculation based on a langmuir henry regime approximation.

        - The saturation loading is estimated by obtaining the highest uptake value for each dataset.
        - The henry_approx function (utility_functions.py) is used here to calculate the henry constants
        - The langmuir 'b' parameter is calculated from the henry constants based on { KH = b * q }
        - The heat of adsorption guess value is set to -5000 for all datasets as this is difficult to estimate
        based on the datasets given

    :return:
        Returns dictionary of guess values (as lists) for each parameter corresponding to the chosen model

    """
    saturation_loading = [1.1 * df[key_uptakes[i]].max() for i in range(len(key_pressures))]
    henry_lim = henry_approx(df, key_pressures, key_uptakes, False)[0]
    langmuir_b = [kh / qsat for (kh, qsat) in zip(henry_lim, saturation_loading)]
    h_guess = [-5000 for _ in key_pressures]

    if "langmuir" in model and model != "langmuir td":
        return {
            "b": langmuir_b,
            "q": saturation_loading
        }

    if model == "langmuir td":
        return {
            "b0": langmuir_b,
            "q": saturation_loading,
            "h": h_guess
        }

    if model == "dsl" or model == "dsl nc":
        return {
            "q1": [0.5 * q for q in saturation_loading],
            "b1": [0.4 * b for b in langmuir_b],
            "q2": [0.5 * q for q in saturation_loading],
            "b2": [0.6 * b for b in langmuir_b]
        }

    if model == "gab":
        return {
            "n": [0.2 * q for q in saturation_loading],
            "ka": [1.1 * b for b in langmuir_b],
            "ca": [0.1 * b for b in langmuir_b]
        }

    if model == "mdr":
        return {
            "n0": saturation_loading,
            "n1": langmuir_b,
            "a": [0.1 * b for b in langmuir_b],
            "c": [10 * b for b in langmuir_b]
        }
    if model == "sips":
        return {
            "q": saturation_loading,
            "b": langmuir_b,
            "n": [1 for _ in saturation_loading]
        }
    if model == "toth":
        return {
            "q": saturation_loading,
            "b": langmuir_b,
            "t": [0.5 for _ in saturation_loading]
        }
    if "bddt" in model:
        return {
            "c": langmuir_b,
            "n": [10 for _ in saturation_loading],
            "g": [100 for _ in saturation_loading],
            "q": [q * 0.5 for q in saturation_loading]
        }
    if model == "dodo":
        return {
            "ns": saturation_loading,
            "kf": langmuir_b,
            "nu": [b * 10 for b in saturation_loading],
            "ku": langmuir_b,
            "m": [5 for _ in saturation_loading]
        }
    if model == "bet":
        return {
            "n": saturation_loading,
            "c": langmuir_b,
        }
    if model == "toth td":
        return {
            "q": saturation_loading,
            "b0": [b * 0.1 for b in langmuir_b],
            "t": [0.5 for _ in saturation_loading],
            "h": h_guess
        }


def get_fit_tuples(model, guess, temps, i, cond, cust_bounds, henry_constants, henry_off, q_fix):
    """
    Definitions of the fitting procedures for each model in the form of tuples.

    This function has an iterative element (i) as when iterating over a list of datasets for fitting, each fitting
    will have its own unique guess values and bounds.

    These tuples are formed for the .add_many() method in the lmfit Parameters class. The following definition is
    extracted from https://lmfit.github.io/lmfit-py/parameters.html :
        add_many(*parlist) - Add many parameters, using a sequence of tuples.

        Parameters *parlist (sequence of tuple or Parameter) – A sequence of tuples, or a sequence of Parameter
        instances. If it is a sequence of tuples, then each tuple must contain at least a name. The order in each
        tuple must be (name, value, vary, min, max, expr, brute_step).

    :param q_fix: float
        Used to fix the saturation capacity parameter for langmuir.


    :return:
        Returns a tuple of parameters for the .add_many() method in the lmfit Parameters class.
    """
    bounds = bounds_check(model, cust_bounds, len(temps))

    if model == "dsl" and cond is False:
        return ('q1', guess['q1'][i], True, *bounds['q1'][i]), \
               ('q2', guess['q2'][i], True, *bounds['q2'][i]), \
               ('b1', guess['b1'][i], True, *bounds['b1'][i]), \
               ('b2', guess['b2'][i], True, *bounds['b2'][i])

    if "langmuir" in model and model != "langmuir td":
        q = ('q', guess['q'][i], True, *bounds['q'][i])
        b = ('b', guess['b'][i], True, *bounds['b'][i])
        if cond:
            if i != 0:
                q = ('q', q_fix, True, q_fix, q_fix + 0.001)

            if henry_off:
                return q, b
            else:
                delta = ('delta', henry_constants[i], False)
                b = ('b', None, None, None, None, 'delta/q')
                return q, delta, b
        return q, b

    if model == "langmuir td":
        q = ('q', guess['q'][i], True, *bounds['q'][i])
        if cond and i != 0:
            q = ('q', q_fix, True, q_fix, q_fix + 0.001)

        return ('t', temps[i], False), \
               q, \
               ('h', guess['h'][i], True, *bounds['h'][i]), \
               ('b0', guess['b0'][i], True, *bounds['b0'][i])

    if model == "gab":
        return ('n', guess['n'][i], True, *bounds['n'][i]), \
               ('ka', guess['ka'][i], True, *bounds['ka'][i]), \
               ('ca', guess['ca'][i], True, *bounds['ca'][i])

    if model == "mdr":
        n0 = ('n0', guess['n0'][i], True, *bounds['n0'][i])
        n1 = ('n1', guess['n1'][i], True, *bounds['n1'][i])
        a = ('a', guess['a'][i], True, *bounds['a'][i])
        c = ('c', guess['c'][i], True, *bounds['c'][i])
        if cond:
            if i != 0:
                n0 = ('n0', q_fix, True, q_fix, q_fix + 0.001)
            if henry_off:
                return n0, n1, a, c
            else:
                delta = ('delta', henry_constants[i], False)
                n1 = ('n1', None, None, None, None, 'delta/n0')
                return n0, delta, n1, a, c

        return n0, n1, a, c

    if model == "sips":
        return ('q', guess['q'][i], True, *bounds['q'][i]), \
               ('b', guess['b'][i], True, *bounds['b'][i]), \
               ('n', guess['n'][i], True, *bounds['n'][i])

    if model == "toth":
        return ('q', guess['q'][i], True, *bounds['q'][i]), \
               ('b', guess['b'][i], True, *bounds['b'][i]), \
               ('t', guess['t'][i], True, *bounds['t'][i])

    if "bddt" in model:
        if cond is True:
            c_con = ('c', guess['c'][i], True, 0, 1)
        else:
            c_con = ('c', guess['c'][i], True, *bounds['c'][i])
        return c_con, \
               ('n', guess['n'][i], True, *bounds['n'][i]), \
               ('g', guess['g'][i], True, *bounds['g'][i]), \
               ('q', guess['q'][i], True, *bounds['q'][i])

    if model == "dodo":
        return ('ns', guess['ns'][i], True, *bounds['ns'][i]), \
               ('kf', guess['kf'][i], True, *bounds['kf'][i]), \
               ('nu', guess['nu'][i], True, *bounds['nu'][i]), \
               ('ku', guess['ku'][i], True, *bounds['ku'][i]), \
               ('m', guess['m'][i], True, *bounds['m'][i])

    if model == "bet":
        return ('n', guess['n'][i], True, *bounds['n'][i]), \
               ('c', guess['c'][i], True, *bounds['c'][i]),
