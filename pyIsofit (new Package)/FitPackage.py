# import FitPackage as mc
# import importlib

# importlib.reload(mc)

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from lmfit import Model, Parameters
from modelEquations import *
from utilityFunctions import *
from utilityFunctions import _model_params, _model_param_lists
from modelProcedures import *

_MODELS = [
    "Langmuir", "Quadratic", "BET", "Henry", "TemkinApprox", "DSLangmuir"
]

colours = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:olive']


def get_fit_tuples(model, guess):
    if model == "MDR":
        return ('n0', guess['n0'], True, 0), \
               ('n1', guess['n1'], True, 0), \
               ('a', guess['a'], True, 0), \
               ('c', guess['c'], True, 0)


class IsothermFit:
    def __init__(self,
                 df,
                 compname,
                 temps,
                 keyPressures,
                 keyUptakes,
                 logplot=False,
                 model=None,
                 guess=None,
                 meth='tnc',
                 x=None,
                 y=None,
                 params=None):

        self.df = df  # Dataframe with uptake vs. pressure data
        self.temps = temps  # Temp in deg C
        self.meth = meth  # Optional to choose mathematical fitting method in lmfit (default is leastsq)
        self.compname = compname  # Name of component

        self.keyPressures = keyPressures  # names of the column headers for pressure and uptakes
        self.keyUptakes = keyUptakes  #
        self.model = model
        if type(self.df) is list and model != "DSL":
            raise Exception("\n\n\n\nEnter one dataframe, not a list of dataframes")

        if model is None:
            raise Exception("Enter a model as a parameter")
        self.logplot = logplot

        self.x = x if x is not None else []
        self.y = x if y is not None else []
        self.params = params if y is not None else []

        # ! Dictionary of parameters as a starting point for data fitting
        if type(self.df) is not list:
            self.guess = get_guess_params(model, df, keyUptakes, keyPressures)
        else:
            self.guess = None

        # Override defaults if user provides param_guess dictionary
        if guess is not None:
            for param, guess_val in guess.items():
                if param not in list(self.guess.keys()):
                    raise Exception("%s is not a valid parameter"
                                    " in the %s model." % (param, model))
                self.guess[param] = guess_val

    def fit(self, cond=True, show_hen=False, hen_tol=0.999):
        # Reading data from dataframe with respect to provided keys
        if self.model != "DSL":
            x2 = []
            y2 = []
            # Importing data and allocating to variables
            for i in range(len(keyPressures)):
                x2.append(self.df[keyPressures[i]].values)
                y2.append(self.df[keyUptakes[i]].values)

            if self.model == "Langmuir linear 1":
                for i in range(len(self.keyPressures)):
                    self.x.append([1 / p for p in x2[i]])
                    self.y.append([1 / q for q in y2[i]])

            elif self.model == "Langmuir linear 2":
                for i in range(len(self.keyPressures)):
                    self.y.append([p / q for (p, q) in zip(x2[i], y2[i])])
                    self.x.append(x2[i])

            elif self.model == "MDR" or self.model == "MDR TD":
                for i in range(len(self.keyPressures)):
                    pressure = x2[i]
                    self.x.append(np.array([p / pressure[-1] for p in x2[i]]))
                    self.y.append([i for i in y2])
                    del pressure
            else:
                self.x = x2
                self.y = y2

            base_params = self.model, self.x, self.y, self.guess, self.temps

            henry_constants = henry_approx(self.df, self.keyPressures, self.keyUptakes, show_hen, hen_tol,
                                           self.compname)

        # SINGLE LANGMUIR FITTING
        if "Langmuir" in self.model and self.model != "Langmuir TD":
            self.params = langmuir_fit(*base_params, cond, henry_constants, self.meth)

        elif self.model == "Langmuir TD":
            self.params = langmuirTD_fit(*base_params, cond, self.meth)

        elif self.model == "GAB":
            self.params = gab_fit(*base_params, meth)

        elif self.model == "DSL":
            dsl_result = dsl_fit(self.df, self.keyPressures, self.keyUptakes,
                                 self.temps, self.compname, self.meth, self.guess, hen_tol)
            df_dict, results_dict = dsl_result
        else:
            isotherm = get_model(self.model)
            gmod = Model(isotherm, nan_policy="omit")

            model_params = get_fit_tuples(self.model, self.guess)

            params = []
            values_dict = {}
            for i in range(len(self.x)):
                pars = Parameters()
                pars.add_many(*model_params)
                results = gmod.fit(self.y[i], pars, x=self.x[i], method=meth)
                params.append(results)
                values_dict[i] = results.values
                del results, pars

            values_dict_sorted = {}
            param_keys = _model_param_lists(self.model)
            params_list = [[] for i in range(len(param_keys))]
            for i in range(len(values_dict)):
                for j in range(len(param_keys)):
                    params_list[j].append(values_dict[i][param_keys[j]])










        if type(self.df) is list:
            self.params = results_dict
            for i in range(len(compname)):
                x_i, y_i = df_dict[compname[i]]
                self.x.append(x_i)
                self.y.append(y_i)

    def plot(self):
        np.linspace(0, 10, 301)

        ##### Plotting results #####

        # plt.title()

        if type(self.df) is list:
            for i in range(len(self.df)):
                plot_settings(self.logplot, 'xaxis', 'yaxis')

                comp_x_params = self.params[compname[i]]
                plt.title(compname[i])
                for j in range(len(self.keyPressures)):
                    plt.plot(self.x[i][j], comp_x_params[j].best_fit, '-', color=colours[j],
                             label="{temps} °C Fit".format(temps=self.temps[j]))
                    plt.plot(self.x[i][j], self.y[i][j], 'ko', color='0.75',
                             label="Data at {temps} °C".format(temps=self.temps[j]))
                plt.legend()
        else:
            plot_settings(self.logplot, 'xaxis', 'yaxis')

            for i in range(len(self.keyPressures)):
                plt.plot(self.x[i], self.params[i].best_fit, '-', color=colours[i],
                         label="{temps} °C Fit".format(temps=self.temps[i]))
                plt.plot(self.x[i], self.y[i], 'ko', color='0.75',
                         label="Data at {temps} °C".format(temps=self.temps[i]))

        plt.show()


df1 = pd.read_csv('Computational Data (EPFL) CO2.csv')
# df2 = pd.read_csv('Computational Data (EPFL) N2.csv')
# df_list = [df1, df2]
compname = 'CO2'
temps = [10, 40, 100]
meth = 'tnc'
keyUptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
keyPressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']

# keyPressures = ['Pressure1', 'Pressure2', 'Pressure3']
# keyUptakes = ['Uptake1', 'Uptake2', 'Uptake3']
tolerance = 0.9999  # set minimum r squared value

langmuir = IsothermFit(df1, compname, temps, keyPressures, keyUptakes, True, "MDR")
langmuir.fit(False, True)
langmuir.plot()
