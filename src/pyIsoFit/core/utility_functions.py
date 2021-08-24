"""
Utility functions used by the main IsothermFit class and fitting procedure functions
"""
import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from pyIsoFit.core.model_dicts import _MODEL_DF_TITLES, _MODEL_PARAM_LISTS, _MODEL_BOUNDS
from pyIsoFit.core.model_equations import r

logger = logging.getLogger('pyIsoFit-master')

colours = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:olive']

_models_with_b = ['langmuir', 'langmuir linear 1', 'langmuir linear 2', 'sips', 'toth']


def get_xy(df, key_pressures, key_uptakes, model, rel_pres):
    """
    Function that reads dataframe with respect to provided keys and converts it into x and y variables which are
    lists of the x and y co-ordinates for each dataset. This function also converts the x and y axis depending on
    the model used - for 'langmuir linear 1': 1/q vs. 1.p, for 'langmuir linear 2': p/q vs. p, for models
    which use relative pressure: q vs. rel. p

    :param df: pd.DataFrame
                Pure-component isotherm data as a pandas dataframe - must be uptake in mmol/g and pressure in bar or
                equivalent. If datasets at different temperatures are required for fitting, the user must specify
                them in the same dataframe.

    :param key_pressures: list[str]
                List of unique column key(s) which correspond to each dataset's pressure values within the
                dataframe. Can input any number of keys corresponding to any number of datasets in the dataframe.
                If multiple dataframes are specified, make sure keys are identical across each dataframe for each
                temperature. Must be inputted in the same order as key_uptakes and temps.

    :param key_uptakes: list[str]
                List of unique column key(s) which correspond to each dataset's uptake values within the
                dataframe. Can input any number of keys corresponding to any number of datasets in the dataframe.
                If multiple dataframes are specified, make sure keys are identical across each dataframe for each
                temperature. Must be inputted in the same order as key_pressures and temps.

    :param model: str
                Model to be fit to dataset(s).

    :param rel_pres : bool
                Input whether to fit the x axis data to relative pressure instead of absolute

    :return:
        Lists of x and y co-ordinates corresponding to each dataset
    """

    # Importing data and allocating to variables
    x2 = [df[key_pressures[i]].values for i in range(len(key_pressures))]
    y2 = [df[key_uptakes[i]].values for i in range(len(key_pressures))]

    # nan filter for datasets - fixes bug where pandas reads empty cells as nan values
    x_filtr = np.array([np.array(x2[i][~np.isnan(x2[i])]) for i in range(len(x2))])
    y_filtr = np.array([np.array(y2[i][~np.isnan(y2[i])]) for i in range(len(y2))])

    x2 = x_filtr
    y2 = y_filtr

    x = []
    y = []

    # Formatting the x and y axis depending on the model
    if model == "langmuir linear 1":
        for i in range(len(key_pressures)):
            x.append(np.array([1 / p for p in x2[i]]))
            y.append(np.array([1 / q for q in y2[i]]))

    elif model == "langmuir linear 2":
        for i in range(len(key_pressures)):
            y.append(np.array([p / q for (p, q) in zip(x2[i], y2[i])]))
            x.append(np.array(x2[i]))

    elif rel_pres is True:
        for i in range(len(key_pressures)):
            pressure = x2[i]
            x.append(np.array([p / pressure[-1] for p in x2[i]]))
            y.append(np.array(y2[i]))
            del pressure
    else:
        for i in range(len(key_pressures)):
            x.append(np.array(x2[i]))
            y.append(np.array(y2[i]))

    del x2, y2

    return x, y


def plot_settings(log, model=None, rel_pres=False, testing=False):
    """
    Settings for plotting.

    :param log: tuple(bool)
        tuple input of whether to set a log axis for x and/or y in the shape (x,y)

    :param model: str
        Model to be fit to dataset(s).

    :param rel_pres : bool
        Input whether to fit the x axis data to relative pressure instead of absolute

    Formatting such as axis labels, log plot toggle and param tick settings
    """
    if model == "langmuir linear 1":
        xtitle = '1/Pressure [1/bar]'
        ytitle = '1/Uptake [g/mmol]'

    elif model == "langmuir linear 2":
        xtitle = 'Pressure [bar]'
        ytitle = 'Pressure/uptake [(bar mmol)/g]'

    elif rel_pres:
        xtitle = 'Relative pressure [P/P]'
        ytitle = 'Uptake [mmol/g]'
    else:
        xtitle = "Pressure [bar]"
        ytitle = "Uptake [mmol/g]"

    tick_style = {'direction': 'in',
                  'length': 4,
                  'width': 0.7,
                  'colors': 'black'}

    if log is False:
        log = (False, False)

    if log is True:
        log = (True, True)

    logx = log[0]
    logy = log[1]

    plt.figure(figsize=(8, 6))
    if logx:
        plt.xscale("log")

    if logy:
        plt.yscale("log")

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.tick_params(**tick_style)
    plt.grid()

    if testing:
        return xtitle, ytitle, logx, logy


def get_subplot_size(lenx, i):
    """
    Gets the subplot size as a tuple depending on the number of datasets passed.
    This is used for creating henry regime subplots.

    :param lenx: int
        Number of datasets

    :param i:
        Iterator

    :return: tuple of shape (x, y, z)
    """
    if lenx <= 4:
        return 2, 2, i + 1
    if 9 >= lenx > 4:
        return 3, 3, i + 1
    if 16 >= lenx > 9:
        return 4, 4, i + 1


def get_sorted_results(values_dict, model, temps):
    """
    Sorts the values dictionary returned by the generic fitting function from the form:
    {dataset1 : {par_1: value, par_2: value ... par_n: value} ... }

    to the form:
    {par_1: [value(dataset1), value(dataset2), ... value(datasetn)], par_2: ... }

    :param values_dict: dict
        Dictionary of parameters for each dataset to be sorted.

    :param model: str
        Model name for accessing the dictionaries

    """
    # Remove intermediate constraint fitting parameter 'delta'
    for i in range(len(values_dict)):
        if 'delta' in values_dict[i].keys():
            values_dict[i].pop('delta')

    # Create a final result dictionary with the first entry being the the temperatures
    final_results_dict = {'T (K)': temps}

    # Get list of keys for the parameters in the model and create a list of empty lists
    # The number of empty lists is equal to the number of parameters in the model
    param_keys = _MODEL_PARAM_LISTS[model]
    params_list = [[] for _ in range(len(param_keys))]
    for i in range(len(values_dict)):
        for j in range(len(param_keys)):
            params_list[j].append(values_dict[i][param_keys[j]])

    # Get list of titles for the columns within the dataframe - same as _MODEL_PARAM_LISTS but with units
    df_keys = _MODEL_DF_TITLES[model]

    # Allocate the dataframe keys to the fitting results lists and add this to the final results dictionary
    for i in range(len(df_keys)):
        final_results_dict[df_keys[i]] = params_list[i]

    # For error calculations a list of fitting parameters is required for each dataset
    c_list = []
    for i in range(len(values_dict)):
        c_list.append(list(values_dict[i].values()))

    return final_results_dict, c_list


def heat_calc(model, temps, param_dict, x=None, testing=False):
    """
    Calculates heat of adsorption for the GAB, Langmuir and DSL isotherm models using the
    temperature dependent parameters.

    For Langmuir and DSL - Van't Hoff:  ln b vs. 1/T
    For GAB - Clausius–Clapeyron approach: ln(x) vs. 1/T
         Plots of 𝑙𝑛(x) vs 1/𝑇 for x at n(x)T1 = n(x)T2 … = n(x)T𝑛

    :param model: str
        Model to be fit to dataset(s).

    :param temps: list[float]
        List of temperatures for calculation of heat of adsorption - must be in K

    :param param_dict: dict
        Dictionary of fitting result lists - this is used for extracting the parameters for plotting


    """
    site = ['A', 'B', 'C']
    t = [1 / i for i in temps]

    def isosteric_heat(t, yparams, j=0):
        ln_b = np.array([np.log(i) for i in yparams])
        mh, bh, rh, ph, sterrh = stats.linregress(t, ln_b)
        h = -0.001 * mh * r

        print( "_______________________________________________________________________\n" + \
               "Heat of adsorption for site " + site[j] + ":" + str(round(h, 2)) + " kJ/mol. \n" + \
               "R sq of Van't Hoff: " + str(round(rh, 4)))
        return h

    if model == "gab":
        xT = []
        h = []
        for i in range(len(x[0])):
            xTT = []
            for p in x:
                xTT.append(p[i])
            xT.append(xTT)
        for xi in xT:
            ln_x = [np.log(i) for i in xi]
            m, b, r2h, p, sterr = stats.linregress(t, ln_x)
            h.append(-0.001 * m * r)

        avgh = np.average(h)
        sterr = (1.96 * np.std(h) / (len(h)) ** 0.5)
        print("ΔH_is: " + str(round(avgh, 2)) + " (∓ " + str(round(sterr, 2)) + ") kJ/mol. ")
        result = avgh

    elif model in _models_with_b:
        yparams = param_dict['b (1/bar)']
        result = isosteric_heat(t, yparams)

    elif model == "dsl":
        yparams = [param_dict['b1 (1/bar)'], param_dict['b2 (1/bar)']]
        result = []
        for i in range(2):
            h = isosteric_heat(t, yparams[i], i)
            result.append(h)
    else:
        result = None

    if testing:
        return result

def bounds_check(model, cust_bounds, num_comps):
    """
    This function checks whether custom_bounds are inputted into it. If custom_bounds = None, it creates a dictionary
    of default bounds (based on the _MODEL_BOUNDS dictionary) where the key for each parameter is matched with a list
    of tuples which has the same length as the number of datasets being fit.

    If custom bounds are inputted, the above condition is ignored and these are used for fitting

    :param model: str
        Dictionary of parameters for each dataset to be sorted.

    :param cust_bounds: list[tuple[float]]
        List of tuples which include min and max bounds for fitting shape (min, max)

    :param num_comps: int
        Number of components to duplicate the default bounds by

    """
    if cust_bounds is None:
        # Converting bounds to list of tuples corresponding to each dataset
        param_list = _MODEL_PARAM_LISTS[model]
        bounds_dict = _MODEL_BOUNDS[model]
        bounds = {}
        for par in param_list:
            par_list = [bounds_dict[par] for _ in range(num_comps)]
            bounds[par] = par_list
        return bounds
    else:
        bounds = cust_bounds
        return bounds


def save_func(directory, fit_name, filetype, df, comp=''):
    """
    Function that saves fitting results to directory

    :param directory: str
        input directory for the location of the save file

    :param fit_name: str
        input name of file to be saved

    :param filetype: str
        input filetype to be saved

    :param df: pd.DataFrame
        dataframe to be converted into .csv or .json

    :param comp: str
        Used to differentiate files when multiple component fitting results are saved

    """
    fit_string = directory + fit_name + comp + filetype

    file_conv_fit = {
        '.csv': df.to_csv(fit_string),
        '.json': df.to_json(fit_string)
    }
    print("File saved to directory")
    return file_conv_fit[filetype]

import pandas as pd
