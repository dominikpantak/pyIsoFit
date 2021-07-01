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
# from utilityFunctions import _model_params, _model_param_lists
from modelProcedures import *
import math

_MODELS = [
    "Langmuir", "Quadratic", "BET", "Henry", "TemkinApprox", "DSLangmuir"
]

colours = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:olive']


def get_fit_tuples(model, guess):
    if model.lower() == "langmuir":
        return ('q', guess['q'], True, 0), \
               ('b', guess['b'], True, 0),

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
                 model=None,
                 guess=None,
                 meth='tnc',
                 x=None,
                 y=None,
                 params=None,
                 df_result=None):

        self.df = df  # Dataframe with uptake vs. pressure data
        self.temps = temps  # Temp in deg C
        self.meth = meth  # Optional to choose mathematical fitting method in lmfit (default is leastsq)
        self.compname = compname  # Name of component

        self.keyPressures = keyPressures  # names of the column headers for pressure and uptakes
        self.keyUptakes = keyUptakes  #
        self.model = model
        if type(self.df) is list and model.lower() != "dsl":
            raise Exception("\n\n\n\nEnter one dataframe, not a list of dataframes")

        if model is None:
            raise Exception("Enter a model as a parameter")

        self.x = x if x is not None else []
        self.y = x if y is not None else []
        self.params = params if y is not None else []
        self.df_result = df_result if df_result is not None else []

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

            if self.model.lower() == "langmuir linear 1":
                for i in range(len(self.keyPressures)):
                    self.x.append([1 / p for p in x2[i]])
                    self.y.append([1 / q for q in y2[i]])

            elif self.model.lower() == "langmuir linear 2":
                for i in range(len(self.keyPressures)):
                    self.y.append([p / q for (p, q) in zip(x2[i], y2[i])])
                    self.x.append(x2[i])

            elif self.model.lower() == "mdr" or self.model.lower() == "mdr td":
                for i in range(len(self.keyPressures)):
                    pressure = x2[i]
                    self.x.append([p / pressure[-1] for p in x2[i]])
                    self.y.append([i for i in y2])
                    del pressure
            else:
                self.x = x2
                self.y = y2
                # self.x = [[x for x in x2[i] if not math.isnan(x)] for i in range(len(x2))]
                # self.y = [[y for y in y2[i] if not math.isnan(y)] for i in range(len(y2))]

            base_params = self.model, self.x, self.y, self.guess, self.temps

            henry_constants = henry_approx(self.df, self.keyPressures, self.keyUptakes, show_hen, hen_tol,
                                           self.compname)

        # SINGLE LANGMUIR FITTING
        if "langmuir" in self.model.lower() and self.model.lower() != "langmuir td":
            final_result = langmuir_fit(*base_params, cond, henry_constants[0], self.meth)

        elif self.model.lower() == "langmuir td":
            final_result = langmuirTD_fit(*base_params, cond, self.meth)

        elif self.model.lower() == "gab":
            final_result = gab_fit(*base_params, meth)

        elif self.model.lower() == "dsl nc":
            final_result = dsl_fit_nc(*base_params, meth)

        elif self.model.lower() == "dsl":
            dsl_result = dsl_fit(self.df, self.keyPressures, self.keyUptakes,
                                 self.temps, self.compname, self.meth, self.guess, hen_tol)
            df_dict, results_dict, df_res_dict = dsl_result

        if self.model.lower() != "dsl":
            self.params = final_result[0]
            self.df_result.append(final_result[1])
            self.df_result.append(henry_constants[1])

        # else:
        #     isotherm = get_model(self.model)
        #     gmod = Model(isotherm, nan_policy="omit")
        #
        #     model_params = get_fit_tuples(self.model, self.guess)
        #
        #     params = []
        #     values_dict = {}
        #     for i in range(len(self.x)):
        #         pars = Parameters()
        #         pars.add_many(*model_params)
        #         results = gmod.fit(self.y[i], pars, x=self.x[i], method=meth)
        #         params.append(results)
        #         values_dict[i] = results.values
        #         del results, pars
        #
        #     values_dict_sorted = {}
        #     param_keys = _model_param_lists(self.model)
        #     params_list = [[] for i in range(len(param_keys))]
        #     for i in range(len(values_dict)):
        #         for j in range(len(param_keys)):
        # #             params_list[j].append(values_dict[i][param_keys[j]])

        if type(self.df) is list:
            self.params = results_dict
            for i in range(len(compname)):
                x_i, y_i = df_dict[compname[i]]
                self.x.append(x_i)
                self.y.append(y_i)
            self.df_result = df_res_dict


    def plot(self, logplot=False):
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

        else:
            plot_settings(logplot)

            for i in range(len(self.keyPressures)):
                plt.plot(self.x[i], self.params[i].best_fit, '-', color=colours[i],
                         label="{temps} °C Fit".format(temps=self.temps[i]))
                plt.plot(self.x[i], self.y[i], 'ko', color='0.75',
                         label="Data at {temps} °C".format(temps=self.temps[i]))
        plt.legend()
        plt.show()

    def save(self, filestring=None, filetype='.csv'):
        if filestring is None:
            filenames = ['fit_result', 'henry_result']

        if type(self.df_result) is dict:
            for comp in self.df_result:
                filenames = [filestring[0] + comp + filetype, filestring[1] + comp + filetype]
                if filetype == '.csv':
                    self.df_result[comp].to_csv(filenames[0])
                elif filetype == '.json':
                    self.df_result[comp].to_json(filenames[0])
                print('Conversion successful')
        else:
            filenames = [filestring[0]+filetype, filestring[1]+filetype]
            if filetype == '.csv':
                self.df_result[0].to_csv(filenames[0])
                self.df_result[1].to_csv(filenames[1])
            elif filetype == '.json':
                self.df_result[0].to_json(filenames[0])
                self.df_result[1].to_json(filenames[1])
            print('Conversion successful')




df2 = pd.read_csv('Computational Data (EPFL) N2.csv')
compname = 'N2'
temps = [10, 40, 100]
meth = 'tnc'
keyUptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
keyPressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']

# keyUptakes = ['Loading [mmol/g] 25', 'Loading [mmol/g] 50', 'Loading [mmol/g] 70']
# keyPressures = ['Pressure [bar] 25', 'Pressure [bar] 50', 'Pressure [bar] 70']

tolerance = 0.9999  # set minimum r squared value

langmuir = IsothermFit(df2, compname, temps, keyPressures, keyUptakes, "Langmuir")
langmuir.fit(False, True)
langmuir.plot(True)

