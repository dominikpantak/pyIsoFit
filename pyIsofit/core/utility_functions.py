import numpy as np
from pyIsofit.core.model_equations import r2hen, bold, unbold, henry, r
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from pyIsofit.core.model_dicts import _MODEL_DF_TITLES, _MODEL_PARAM_LISTS, _MODEL_BOUNDS, _TEMP_DEP_MODELS


def get_xy(df, keyPressures, keyUptakes, model, rel_pres):
    # Reading data from dataframe with respect to provided keys
    x = []
    y = []

    x2 = []
    y2 = []
    # Importing data and allocating to variables
    for i in range(len(keyPressures)):
        x2.append(df[keyPressures[i]].values)
        y2.append(df[keyUptakes[i]].values)

    # nan filter for datasets - fixes bug where pandas reads empty cells as nan values
    x_filtr = []
    y_filtr = []
    for i in range(len(x2)):
        x_filtr.append(np.array(x2[i][~np.isnan(x2[i])]))
        y_filtr.append(np.array(y2[i][~np.isnan(y2[i])]))

    x2 = x_filtr
    y2 = y_filtr

    if model == "langmuir linear 1":
        for i in range(len(keyPressures)):
            x.append(np.array([1 / p for p in x2[i]]))
            y.append(np.array([1 / q for q in y2[i]]))

    elif model == "langmuir linear 2":
        for i in range(len(keyPressures)):
            y.append(np.array([p / q for (p, q) in zip(x2[i], y2[i])]))
            x.append(x2[i])

    elif rel_pres is True:
        rel_pres = True
        for i in range(len(keyPressures)):
            pressure = x2[i]
            x.append(np.array([p / pressure[-1] for p in x2[i]]))
            y.append(y2[i])
            del pressure
    else:
        x = x2
        y = y2

    del x2, y2

    return x, y


def henry_approx(df, keyPressures, keyUptakes, display_hen=False, tol_or_customhen=0.9999, compname="---"):
    # This section finds the henry region of the datasets
    x = []
    y = []
    for i in range(len(keyPressures)):
        xi = np.insert(np.array(df[keyPressures[i]].values), 0, 0)
        yi = np.insert(np.array(df[keyUptakes[i]].values), 0, 0)
        x.append(xi)
        y.append(yi)
        del xi
        del yi

    henry_constants = []
    henry_limits = []
    henry_rsq = []
    rsq_lst = []
    hen_lst = []
    henry_len = []
    errHen = []
    henidx_lst = []
    i = 0

    for dataset in y:
        rsq = 1
        x_i = x[i]
        x_henry = [x_i[0], x_i[1], x_i[2]]  # Starting with a minimum of three datapoints
        j = 3
        rsq_ilst = []
        hen_ilst = []
        # This loop adds data points while the points correspond to a henry fit with an R^2 of above 0.9995
        if type(tol_or_customhen) == list:
            while x_i[j] < tol_or_customhen[i] and j < len(x_i) - 2:
                x_henry.append(x_i[j])
                y_henry = dataset[:len(x_henry)]
                hen = y_henry[-1] / x_henry[-1]
                rsq = round(r2hen(x_henry, y_henry, henry, hen), 5)  # r squared calc.
                rsq_ilst.append(rsq)
                hen_ilst.append(hen)
                j += 1
            rsq_lst.append(rsq_ilst)
            hen_lst.append(hen_ilst)

            rsqidx = len(hen_ilst) - 1

        else:
            while rsq > 0 and j < len(x_i):
                x_henry.append(x_i[j])
                y_henry = dataset[:len(x_henry)]

                hen = y_henry[-1] / x_henry[-1]
                rsq = round(r2hen(x_henry, y_henry, henry, hen), 5)  # r squared calc.
                rsq_ilst.append(rsq)
                hen_ilst.append(hen)
                j += 1
            rsq_lst.append(rsq_ilst)
            hen_lst.append(hen_ilst)
            # plt.figure()

            abtol = []
            itol = []
            i2 = 0
            for rsq in rsq_ilst:
                if rsq > tol_or_customhen:
                    abtol.append(rsq)
                    itol.append(i2)
                i2 += 1
            if abtol == []:
                maxrsq = max(rsq_ilst)
                rsqidx = rsq_ilst.index(maxrsq)
            else:
                rsqfin = min(abtol)
                minidx = abtol.index(rsqfin)
                rsqidx = itol[minidx]

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
            print("ERROR - Please increase henry region value of index " + str(i))

        if henidx + 1 < 4:
            errHen.append(str(i + 1))
        i += 1

    x_result = [x[i][:henidx_lst[i]] for i in range(len(x))]
    y_result = [y[i][:henidx_lst[i]] for i in range(len(y))]
    xy_dict = {"x": x_result, "y": y_result}

    # Creating dataframe for henry constants
    df_henry = pd.DataFrame(list(zip(henry_constants, henry_limits, henry_len, henry_rsq)),
                            columns=['Henry constant (mmol/(bar.g))',
                                     'Upper limit (bar)', 'datapoints', 'R squared'])

    if display_hen:
        print(bold + '\nHenry regime for component ' + compname + ':' + unbold)
        if errHen:
            print(unbold + 'WARNING: Henry region for dataset(s) ' + ', '.join(
                errHen) + ' were found to be made up of less than 4 points.')
            print('         This may affect accuracy of results.')
            print(
                '         Henry region tolerance may be entered after log plot toggle parameter (default = 0.9999).\n')

        print(pd.DataFrame(df_henry))

    return henry_constants, df_henry, xy_dict


def plot_settings(log, model="default", rel_pres=False):
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

    plt.figure(figsize=(8, 6))
    if log:
        plt.xscale("log")
        plt.yscale("log")
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.tick_params(**tick_style)
    plt.grid()


def get_subplot_size(x, i):
    if len(x) <= 4:
        return 2, 2, i + 1
    if 9 >= len(x) > 4:
        return 3, 3, i + 1
    if 16 >= len(x) > 9:
        return 4, 4, i + 1


def get_sorted_results(values_dict, model, temps):
    final_results_dict = {'T (K)': temps}

    param_keys = _MODEL_PARAM_LISTS[model]
    params_list = [[] for _ in range(len(param_keys))]
    for i in range(len(values_dict)):
        for j in range(len(param_keys)):
            params_list[j].append(values_dict[i][param_keys[j]])

    df_keys = _MODEL_DF_TITLES[model]

    for i in range(len(df_keys)):
        final_results_dict[df_keys[i]] = params_list[i]

    c_list = []
    for i in range(len(temps)):
        values = values_dict[i]
        c_innerlst = []
        if model in _TEMP_DEP_MODELS:
            c_innerlst.append(temps[i])
        for key in param_keys:
            c_innerlst.append(values[key])
        c_list.append(c_innerlst)

    return final_results_dict, c_list


def heat_calc(model, temps, param_dict, x):
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

    elif "langmuir" in model and model != "langmuir td":
        yparams = param_dict['b (1/bar)']
        isosteric_heat(t, yparams)

    elif model == "dsl nc":
        yparams = [param_dict['b1 (1/bar)'], param_dict['b2 (1/bar)']]
        for i in range(2):
            isosteric_heat(t, yparams[i], i)
    else:
        return None


def bounds_check(model, cust_bounds, num_comps):
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
