"""
Utility functions used by the main IsothermFit class and fitting procedure functions
"""
import numpy as np
from pyIsofit.core.model_equations import r2hen, bold, unbold, henry, r
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from pyIsofit.core.model_dicts import _MODEL_DF_TITLES, _MODEL_PARAM_LISTS, _MODEL_BOUNDS, _TEMP_DEP_MODELS

colours = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:olive']

_models_with_b = ['langmuir', 'linear langmuir 1', 'linear langmuir 2', 'sips', 'toth']


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
    x_filtr = [np.array(x2[i][~np.isnan(x2[i])]) for i in range(len(x2))]
    y_filtr = [np.array(y2[i][~np.isnan(y2[i])]) for i in range(len(y2))]

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
            x.append(x2[i])

    elif rel_pres is True:
        for i in range(len(key_pressures)):
            pressure = x2[i]
            x.append(np.array([p / pressure[-1] for p in x2[i]]))
            y.append(y2[i])
            del pressure
    else:
        x = x2
        y = y2

    del x2, y2

    return x, y


def henry_approx(df, key_pressures, key_uptakes, display_hen=False, tol=0.9999, compname="---", henry_off=False):
    """
        Henry approximation function used to estimate henry constants and the henry regime.

    This function works by first adding a (0,0) co-ordinate to each dataset as this is the behaviour exhibited in the
    henry regime. Next, for each temperature dataset, starting with a line made up of three of the first datapoints (
    including the 0,0 point). A gradient (Henry constant) and R squared value of this line is then found through
    linear regression and the package stores it in a variable. It is important to note that the equation that is used
    for linear regression is the Henry isotherm model: q = KH*p.

    This procedure is iterated, with each run adding an additional data point to the line, with the R squared value
    calculated after each fitting. This loop is repeated for the entire dataset. The function stores every calculated
    R2 value as a list, after which the lowest R squared value within this list that is above the minimum tolerance
    value is chosen.

    value. If there are no henry regime lines with an R2 greater than the tolerance value, the henry regime dataset
    corresponding to the maximum R2 value is chosen. The minimum number of datapoints that can be used for the Henry
    regime determination is 3.

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

    :param display_hen: bool
                Input whether to show the henry regime of the datasets approximated by the package. This is False by
                default.

    :param tol: float or list[float]
                The henry region approximation function calculates the henry region by finding a line with the highest
                R squared value in the low pressure region of the dataset. This is done with a default R squared
                tolerance value (set to 0.999).

                For example, if a float is inputted (a different henry tolerance) this will be the henry tolerance value
                used by the function. i.e if 0.98 is inputted the henry regime will be across a large pressure range
                due to the low tolerance for the R squared value of the henry model fitting.

                This function also supports inputting the henry regimes manually. For this, input each henry regime for
                each dataset as a list i.e [1.2, 2.1, ... ]

    :param compname: str or list[str]
                Name of pure component(s) for results formatting.

    :param henry_off : bool
                Input whether to turn off the henry regime fitting constraint when using the standardised fitting
                constraint to langmuir or dsl - this is usually done when fitting experimental data which has a messy
                low pressure region. Default is False.

    :return: tuple
                Returns list of henry constant for each dataset [0], resulting DataFrame object [1], the x and y
                co-ordinates in the form of a dictionary for each dataset [2]
    """
    if henry_off:
        return None, None, None
    # This section finds the henry region of the datasets
    x = []
    y = []
    for i in range(len(key_pressures)):
        # Reads data from dataframe with respect to keys and adds a 0,0 point
        xi = np.insert(np.array(df[key_pressures[i]].values), 0, 0)
        yi = np.insert(np.array(df[key_uptakes[i]].values), 0, 0)
        x.append(xi)
        y.append(yi)
        del xi
        del yi

    henry_constants = []
    henry_limits = []
    henry_rsq = []
    henry_len = []
    errHen = []
    henidx_lst = []
    i = 0

    # Iterate for all datasets at each temperature
    for dataset in y:
        rsq = 1
        x_i = x[i]
        x_henry = [x_i[0], x_i[1], x_i[2]]  # Starting with a minimum of three datapoints
        j = 3
        # Create inner empty lists
        # rsq_ilst stores all calculated r squared values within the inner loops
        # hen_ilst stores all calculated henry constants within the inner loops
        rsq_ilst = []
        hen_ilst = []
        # This loop adds data points while the points correspond to a henry fit with an R^2 of above 0.9999
        if type(tol) == list:
            # Procedure when a list of henry regime limits are passed
            # Iterate over each datapoint within dataset for a given temperature
            while x_i[j] < tol[i] and j < len(x_i) - 2:
                # Add datapoint
                x_henry.append(x_i[j])
                # Create y axis - same length as x axis
                y_henry = dataset[:len(x_henry)]
                # Calculate henry constant
                hen = y_henry[-1] / x_henry[-1]
                # Find r squared value of fit
                rsq = round(r2hen(x_henry, y_henry, henry, hen), 5)
                # Append results to inner lists
                rsq_ilst.append(rsq)
                hen_ilst.append(hen)
                j += 1

            rsqidx = len(hen_ilst) - 1

        else:
            # Procedure when an r squared tolerance is passed
            while rsq > 0 and j < len(x_i):
                # Within the loop - same procedure as in the above condition
                x_henry.append(x_i[j])
                y_henry = dataset[:len(x_henry)]

                hen = y_henry[-1] / x_henry[-1]
                rsq = round(r2hen(x_henry, y_henry, henry, hen), 5)  # r squared calc.
                rsq_ilst.append(rsq)
                hen_ilst.append(hen)
                j += 1

            # Create empty lists
            # abtol stores list of r squared values above the tolerance value
            # itol stores a list of the index values within rsq_ilst
            abtol = []
            itol = []
            i2 = 0
            for rsq in rsq_ilst:
                if rsq > tol:
                    abtol.append(rsq)
                    itol.append(i2)
                i2 += 1

            # Check for an empty list of r squared values above the tolerance value - this can happen if theres a poor
            # Henry regime fitting or if an r squared tolerance is given which is too high
            if abtol == []:
                # In this case the line with the highest r squared value is chosen
                maxrsq = max(rsq_ilst)
                rsqidx = rsq_ilst.index(maxrsq)
            else:
                # Choose the minimum value from the above tolerance r squared list
                rsqfin = min(abtol)
                minidx = abtol.index(rsqfin)
                rsqidx = itol[minidx]

        # +3 to compensate for the initial three datapoints
        henidx = rsqidx + 3
        henidx_lst.append(henidx)

        try:
            henry_len.append(henidx)
            # Saving Henry region parameters to later display
            henry_constants.append(hen_ilst[rsqidx])
            henry_limits.append(x_henry[henidx])
            henry_rsq.append(rsq_ilst[rsqidx])
            # sometimes data may not have a good henry region fit, which could abort the above while loop after the
            # first iteration. This piece of code warns the user of this
        except IndexError:
            print("ERROR - Increase henry region value of index " + str(i))

        # Record the index of the dataset where a henry regime is made up of less than 4 points
        if henidx + 1 < 4:
            errHen.append(str(i + 1))

        i += 1

    # Create resulting henry regime datasets and make dictionary
    x_result = [x[i][:henidx_lst[i]] for i in range(len(x))]
    y_result = [y[i][:henidx_lst[i]] for i in range(len(y))]
    xy_dict = {"x": x_result, "y": y_result}

    # Creating dataframe for henry constants
    df_henry = pd.DataFrame(list(zip(henry_constants, henry_limits, henry_len, henry_rsq)),
                            columns=['Henry constant (mmol/(bar.g))',
                                     'Upper limit (bar)', 'datapoints', 'R squared'])

    if display_hen:
        print(bold + '\nHenry regime for component ' + compname + ':' + unbold)
        # Print a warning if a dataset contains less than 4 points within the henry regime
        if errHen:
            print(unbold + 'WARNING: Henry region for dataset(s) ' + ', '.join(
                errHen) + ' were found to be made up of less than 4 points.')
            print('         This may affect accuracy of results.')
            print(
                '         Henry region tolerance may be entered after log plot toggle parameter (default = 0.9999).\n')

        print(pd.DataFrame(df_henry))

    return henry_constants, df_henry, xy_dict


def plot_settings(log, model=None, rel_pres=False):
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

def heat_calc(model, temps, param_dict, x):
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

        print("_______________________________________________________________________")
        print("Heat of adsorption for site " + site[j] + ":" + str(round(h, 2)) + " kJ/mol. \n" +
              "R sq of Van't Hoff: " + str(round(rh, 4)))

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

    elif model in _models_with_b:
        yparams = param_dict['b (1/bar)']
        isosteric_heat(t, yparams)

    elif model == "dsl":
        yparams = [param_dict['b1 (1/bar)'], param_dict['b2 (1/bar)']]
        for i in range(2):
            isosteric_heat(t, yparams[i], i)
    else:
        return None


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
