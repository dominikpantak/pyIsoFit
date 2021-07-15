"""
Module for all model definitions.

"""
from pyIsofit.core.utility_functions import henry_approx, bounds_check
from pyIsofit.core.model_equations import *


def get_guess_params(model, df, key_uptakes, key_pressures):
    """
    Function for automatic initial guess value calculation based on a langmuir henry regime approximation.
        - The saturation loading is estimated by obtaining the highest uptake value for each dataset.
        - The henry_approx function (utility_functions.py) is used here to calculate the henry constants
        - The langmuir 'b' parameter is calculated from the henry constants based on { KH = b * q }
        - The heat of adsorption guess value is set to -5000 for all datasets as this is difficult to estimate
        based on the datasets given

    :param model:
    :param df:
    :param key_uptakes:
    :param key_pressures:
    :return:
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
    if model == "mdr td":
        return {
            "n0": saturation_loading,
            "n1": langmuir_b,
            "a": [0.1 * b for b in langmuir_b],
            "b": [10 * b for b in langmuir_b],
            "e": [-1000 for i in saturation_loading]
        }
    if model == "sips":
        return {
            "q": saturation_loading,
            "b": langmuir_b,
            "n": [1 for q in saturation_loading]
        }
    if model == "toth":
        return {
            "q": saturation_loading,
            "b": langmuir_b,
            "t": [0.5 for q in saturation_loading]
        }
    if "bddt" in model:
        return {
            "c": langmuir_b,
            "n": [10 for i in saturation_loading],
            "g": [100 for i in saturation_loading],
            "q": [q * 0.5 for q in saturation_loading]
        }
    if model == "dodo":
        return {
            "ns": saturation_loading,
            "kf": langmuir_b,
            "nu": [b * 10 for b in saturation_loading],
            "ku": langmuir_b,
            "m": [5 for i in saturation_loading]
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
            "t": [0.5 for q in saturation_loading],
            "h": h_guess
        }


def get_fit_tuples(model, guess, temps, i=0, cond=False, cust_bounds=None):
    bounds = bounds_check(model, cust_bounds, len(temps))

    if model == "dsl nc":
        return ('q1', guess['q1'][i], True, *bounds['q1'][i]), \
               ('q2', guess['q2'][i], True, *bounds['q2'][i]), \
               ('b1', guess['b1'][i], True, *bounds['b1'][i]), \
               ('b2', guess['b2'][i], True, *bounds['b2'][i]),

    if model == "gab":
        return ('n', guess['n'][i], True, *bounds['n'][i]), \
               ('ka', guess['ka'][i], True, *bounds['ka'][i]), \
               ('ca', guess['ca'][i], True, *bounds['ca'][i])

    if model == "mdr":
        return ('n0', guess['n0'][i], True, *bounds['n0'][i]), \
               ('n1', guess['n1'][i], True, *bounds['n1'][i]), \
               ('a', guess['a'][i], True, *bounds['a'][i]), \
               ('c', guess['c'][i], True, *bounds['c'][i])

    if model == "sips":
        return ('q', guess['q'][i], True, *bounds['q'][i]), \
               ('b', guess['b'][i], True, *bounds['b'][i]), \
               ('n', guess['n'][i], True, *bounds['n'][i])

    if model == "toth":
        return ('q', guess['q'][i], True, *bounds['q'][i]), \
               ('b', guess['b'][i], True, *bounds['b'][i]), \
               ('t', guess['t'][i], True, *bounds['t'][i])

    if model == "toth td":
        return ('temp', temps[i], False), \
               ('q', guess['q'][i], True, *bounds['q'][i]), \
               ('b0', guess['b0'][i], True, *bounds['b0'][i]), \
               ('t', guess['t'][i], True, *bounds['t'][i]), \
               ('h', guess['h'][i], True, *bounds['h'][i])

    if model == "bddt":
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
               ('nu', guess['nu'][i], True, *bounds['ns'][i]), \
               ('ku', guess['ku'][i], True, *bounds['ku'][i]), \
               ('m', guess['m'][i], True, *bounds['m'][i])

    if model == "mdr td":
        return ('t', temps[i], False), \
               ('n0', guess['n0'][i], True, *bounds['n0'][i]), \
               ('n1', guess['n1'][i], True, *bounds['n1'][i]), \
               ('a', guess['a'][i], True, *bounds['a'][i]), \
               ('b', guess['b'][i], True, *bounds['b'][i]), \
               ('e', guess['e'][i], True, *bounds['e'][i])

    if model == "bet":
        return ('n', guess['n'][i], True, *bounds['n'][i]), \
               ('c', guess['c'][i], True, *bounds['c'][i]),


