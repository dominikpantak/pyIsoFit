import numpy as np
from modelEquations import *
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
import math


def henry_approx(df, keyPressures, keyUptakes, display_hen=False, tol_or_customhen=0.9999, compname="---"):
    # This section finds the henry region of the datasets
    x = []
    y = []
    for i in range(len(keyPressures)):
        xi = np.insert(np.array(df[keyPressures[i]].values), 0, 0)
        yi = np.insert(np.array(df[keyUptakes[i]].values), 0, 0)
        # xi = [x for x in xi if not math.isnan(x)]
        # xi = [y for y in yi if not math.isnan(y)]
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
    i = 0

    for dataset in y:
        rsq = 1
        x_i = x[i]
        x_henry = [x_i[0], x_i[1], x_i[2]]  # Starting with a minimum of two datapoints
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
            henidx = rsqidx + 2


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
                maxrsq = rsq_ilst[rsqidx]

            henidx = rsqidx + 2

        try:
            henry_len.append(henidx + 1)
            # Saving Henry region parameters to later display
            henry_constants.append(hen_ilst[rsqidx])
            henry_limits.append(x_henry[henidx])
            henry_rsq.append(rsq_ilst[rsqidx])
            # sometimes data may not have a good henry region fit, which could abort the above while loop after the first
            # iteration. This piece of code warns the user of this
        except IndexError:
            print("ERROR - Please increase henry region value of index " + str(i))

        if henidx + 1 < 4:
            errHen.append(str(i + 1))
        i += 1

    # Creating dataframe for henry constants
    df_henry = pd.DataFrame(list(zip(henry_constants, henry_limits, henry_len, henry_rsq)),
                            columns=['Henry constant (mmol/(bar.g))',
                                     'Upper limit (bar)', 'datapoints', 'R squared'])

    if display_hen:
        print(bold + '\nHenry regime for component ' + compname + ':' + unbold)
        if errHen != []:
            print(unbold + 'WARNING: Henry region for dataset(s) ' + ', '.join(
                errHen) + ' were found to be made up of less than 4 points.')
            print('         This may affect accuracy of results.')
            print(
                '         Henry region tolerance may be entered after log plot toggle parameter (default = 0.9999).\n')

        display(pd.DataFrame(df_henry))

    return henry_constants, df_henry


def get_model(model):
    if model.lower() == "langmuir":
        return langmuir1
    if model.lower() == "langmuir test":
        return langmuir1
    if model.lower() == "langmuir linear 1":
        return langmuirlin1
    if model.lower() == "langmuir linear 2":
        return langmuirlin2
    if model.lower() == "langmuir td":
        return langmuirTD
    if model.lower() == "dsl":
        return dsl
    if model.lower() == "gab":
        return gab
    if model.lower() == "mdr":
        return mdr


def get_guess_params(model, df, keyUptakes, keyPressures):
    if model.lower() != "bddt 2n" or model.lower() != "bddt 2n-1" or model.lower() != "dodo":
        henry_lim = henry_approx(df, keyPressures, keyUptakes, False)[0]
        saturation_loading = [1.1 * df[keyUptakes[i]].max() for i in range(len(keyPressures))]
        langmuir_b = [kh / qsat for (kh, qsat) in zip(henry_lim, saturation_loading)]
        h_guess = [-5 for i in range(len(keyPressures))]

    if "langmuir" in model.lower() and model.lower() != "langmuir td":
        return {
            "b": langmuir_b,
            "q": saturation_loading
        }

    if model.lower() == "langmuir td":
        return {
            "b0": langmuir_b,
            "q": saturation_loading,
            "h": h_guess
        }

    if model.lower() == "dsl" or model.lower() == "dsl nc":
        return {
            "q1": [0.5 * q for q in saturation_loading],
            "b1": [0.4 * b for b in langmuir_b],
            "q2": [0.5 * q for q in saturation_loading],
            "b2": [0.6 * b for b in langmuir_b]
        }

    if model.lower() == "gab":
        return {
            "n": [0.2 * q for q in saturation_loading],
            "ka": [1.1 * b for b in langmuir_b],
            "ca": [0.1 * b for b in langmuir_b]
        }

    if model.lower() == "mdr":
        return {
            "n0": saturation_loading,
            "n1": langmuir_b,
            "a": [0.1 * b for b in langmuir_b],
            "c": langmuir_b
        }

def get_fit_tuples(model, guess, i=0):
    if model.lower() == "langmuir test":
        return ('q', guess['q'][i], True, 0), \
               ('b', guess['b'][i], True, 0),

    if model.lower() == "dsl nc":
        return ('q1', guess['q1'][i], True, 0), \
               ('q2', guess['q2'][i], True, 0), \
               ('b1', guess['b1'][i], True, 0), \
               ('b2', guess['b2'][i], True, 0),

    if model.lower() == "mdr":
        return ('n0', guess['n0'][i], True, 0), \
               ('n1', guess['n1'][i], True, 0), \
               ('a', guess['a'][i], True, 0), \
               ('c', guess['c'][i], True, 0)

    if model.lower() == "gab":
        return ('n', guess['n'][i], True, 0), \
               ('ka', guess['ka'][i], True, 0), \
               ('ca', guess['ca'][i], True, 0)


_model_param_lists = {
    'mdr': ['n0', 'n1', 'a', 'c'],
    'langmuir': ['q', 'b'],
    'langmuir td': ['q', 'b0', 'h'],
    'dsl nc': ['q1', 'q2', 'b1', 'b2'],
    'gab': ['n', 'ka', 'ca']
}

_model_df_titles = {
    'mdr': ['n0', 'n1', 'a', 'c'],
    'langmuir': ['q (mmol/g)', 'b (1/bar)'],
    'langmuir td': ['q (mmol/g)', 'b0 (1/bar)', 'h (kJ/mol)'],
    'dsl nc': ['q1 (mmol/g)', 'q2 (mmol/g)', 'b1 (1/bar)', 'b2 (1/bar)'],
    'gab': ['n (mmol/g)', 'ka (H2O activity coeff.)', 'ca (GAB const.)']
}

_temp_dep_models = ['langmuir td', 'mdr td']

def plot_settings(log, xtitle='Pressure (bar)', ytitle='Uptake (mmol/g)'):
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
