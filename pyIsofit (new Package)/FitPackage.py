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
from utilityFunctions import _model_param_lists, _model_df_titles, _temp_dep_models
from modelProcedures import *
import math

_MODELS = [
    "Langmuir", "Quadratic", "BET", "Henry", "TemkinApprox", "DSLangmuir"
]

colours = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:olive']

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
                 df_result=None,
                 guess_check=None,
                 henry_params=None,
                 rel_pres=False,
                 input_model=None,
                 emod_input=None):

        self.df = df  # Dataframe with uptake vs. pressure data
        self.temps = temps  # Temp in deg C
        self.meth = meth  # Optional to choose mathematical fitting method in lmfit (default is leastsq)
        self.compname = compname  # Name of component

        self.keyPressures = keyPressures  # names of the column headers for pressure and uptakes
        self.keyUptakes = keyUptakes  #
        self.model = model
        self.input_model = model

        if type(self.df) is list and model.lower() != "dsl":
            raise Exception("\n\n\n\nEnter one dataframe, not a list of dataframes")

        if model is None:
            raise Exception("Enter a model as a parameter")

        self.x = x if x is not None else []
        self.y = x if y is not None else []
        self.params = params if y is not None else []
        self.df_result = df_result if df_result is not None else []
        self.emod_input = emod_input if emod_input is not None else {}

        self.henry_params = henry_params
        self.rel_pres = rel_pres

        # ! Dictionary of parameters as a starting point for data fitting
        if type(self.df) is not list:
            self.guess = get_guess_params(model, df, keyUptakes, keyPressures)
        else:
            self.guess = None
        self.guess_check = self.guess

        # Override defaults if user provides param_guess dictionary
        if guess is not None:
            for param, guess_val in guess.items():
                if param not in list(self.guess.keys()):
                    raise Exception("%s is not a valid parameter"
                                    " in the %s model." % (param, model))
                self.guess[param] = guess_val

    def fit(self, cond=True, meth='leastsq', show_hen=False, hen_tol=0.999, rel_pres=False):

        if self.model.lower() == "dsl" and cond is True:
            self.x = []
            self.y = []

            if self.guess == self.guess_check:
                self.guess = None

            if type(self.df) is not list:
                self.df = [self.df]

            if type(self.compname) is not list:
                self.compname = [self.compname]

            dsl_result = dsl_fit(self.df, self.keyPressures, self.keyUptakes,
                                 self.temps, self.compname, self.meth, self.guess, hen_tol)

            df_dict, results_dict, df_res_dict, params_dict = dsl_result

            if self.model.lower() == 'dsl':
                self.params = results_dict
                for i in range(len(self.compname)):
                    x_i, y_i = df_dict[self.compname[i]]
                    self.x.append(x_i)
                    self.y.append(y_i)
                self.df_result = df_res_dict
                self.emod_input = params_dict

                return None

        henry_constants = henry_approx(self.df, self.keyPressures, self.keyUptakes, show_hen, hen_tol,
                                       self.compname)

        if self.model.lower() == "henry":
            self.henry_params = henry_constants
            return None

        if "mdr" in self.model.lower():
            self.rel_pres = True

        # Reading data from dataframe with respect to provided keys
        x2 = []
        y2 = []
        # Importing data and allocating to variables
        for i in range(len(self.keyPressures)):
            x2.append(self.df[self.keyPressures[i]].values)
            y2.append(self.df[self.keyUptakes[i]].values)

        if self.model.lower() == "langmuir linear 1":
            for i in range(len(self.keyPressures)):
                self.x.append([1 / p for p in x2[i]])
                self.y.append([1 / q for q in y2[i]])

        elif self.model.lower() == "langmuir linear 2":
            for i in range(len(self.keyPressures)):
                self.y.append([p / q for (p, q) in zip(x2[i], y2[i])])
                self.x.append(x2[i])

        elif rel_pres is True:
            self.rel_pres = True
            for i in range(len(self.keyPressures)):
                pressure = x2[i]
                self.x.append([p / pressure[-1] for p in x2[i]])
                self.y.append(y2[i])
                del pressure
        else:
            self.x = x2
            self.y = y2

        del x2, y2

        # nan filter for datasets - fixes bug where pandas reads empty cells as nan values
        x_filtr = []
        y_filtr = []
        for i in range(len(self.x)):
            if np.nan in self.x[i] or np.nan in self.y[i]:
                x_filtr.append(np.array(self.x[i][~np.isnan(self.x[i])]))
                y_filtr.append(np.array(self.y[i][~np.isnan(self.y[i])]))
            else:
                x_filtr.append(np.array(self.x[i]))
                y_filtr.append(np.array(self.y[i]))

        self.x = x_filtr
        self.y = y_filtr

        isotherm = get_model(self.model)
        gmod = Model(isotherm, nan_policy='omit')

        base_params = gmod, self.x, self.y, self.guess



        # SINGLE LANGMUIR FITTING

        if "langmuir" in self.model.lower() and self.model.lower() != "langmuir td":
            params, values_dict = langmuir_fit(*base_params, cond, henry_constants[0], self.meth)
            self.model = "langmuir"

        elif self.model.lower() == "langmuir td":
            params, values_dict = langmuirTD_fit(*base_params, self.temps, cond, self.meth)

        else:
            if self.model.lower() == "dsl" and cond is False:
                self.model = "dsl nc"
            elif self.model.lower() == "bddt 2n" or self.model.lower() == "bddt 2n-1" or self.model.lower() == "bddt":
                self.model = "bddt"

            params = []
            values_dict = {}
            for i in range(len(self.x)):
                pars = Parameters()
                model_params = get_fit_tuples(self.model, self.guess, self.temps, i, cond)
                pars.add_many(*model_params)
                results = gmod.fit(self.y[i], pars, x=self.x[i], method=meth)

                params.append(results)
                values_dict[i] = results.values
                del results, pars

        final_results_dict = {'T (C)': self.temps}

        param_keys = _model_param_lists[self.model.lower()]
        params_list = [[] for i in range(len(param_keys))]
        for i in range(len(values_dict)):
            for j in range(len(param_keys)):
                params_list[j].append(values_dict[i][param_keys[j]])

        df_keys = _model_df_titles[self.model.lower()]

        for i in range(len(df_keys)):
            final_results_dict[df_keys[i]] = params_list[i]


        c_list = []
        for i in range(len(self.x)):
            values = values_dict[i]
            c_innerlst = []
            if self.model.lower() in _temp_dep_models:
                c_innerlst.append(self.temps[i])
            for key in param_keys:
                c_innerlst.append(values[key])
            c_list.append(c_innerlst)

        # r_sq = [r2(self.x[i], self.y[i], isotherm, c_list[i]) for i in range(len(self.x))]
        r_sq = []
        for i in range(len(self.x)):
            rsqres = r2(self.x[i], self.y[i], isotherm, c_list[i])
            r_sq.append(rsqres)

        se = [mse(self.x[i], self.y[i], isotherm, c_list[i]) for i in range(len(self.x))]

        final_results_dict['R squared'] = r_sq
        final_results_dict['MSE'] = se

        df_result = pd.DataFrame.from_dict(final_results_dict)
        pd.set_option('display.max_columns', None)
        display(df_result)

        final_result = params, df_result

        if self.model.lower() != "dsl":
            self.params = final_result[0]
            self.df_result.append(final_result[1])
            self.df_result.append(henry_constants[1])

        if len(self.temps) >= 3:
            heat_calc(self.model, self.temps, final_results_dict, self.x)

    def plot(self, logplot=False):
        np.linspace(0, 10, 301)

        ##### Plotting results #####

        # plt.title()

        if type(self.df) is list:
            for i in range(len(self.df)):
                plot_settings(logplot)

                comp_x_params = self.params[self.compname[i]]
                plt.title(self.compname[i])
                for j in range(len(self.keyPressures)):
                    plt.plot(self.x[i][j], comp_x_params[j].best_fit, '-', color=colours[j],
                             label="{temps} °C Fit".format(temps=self.temps[j]))
                    plt.plot(self.x[i][j], self.y[i][j], 'ko', color='0.75',
                             label="Data at {temps} °C".format(temps=self.temps[j]))

        elif self.model.lower() == "henry":
            henry_const = self.henry_params[0]
            xy_dict = self.henry_params[2]
            x_hen = xy_dict['x']
            y_hen = xy_dict['y']
            plot_settings(logplot)
            for i in range(len(self.x)):
                y_henfit = henry(x_hen[i], henry_const[i])
                subplot_size = get_subplot_size(self.x, i)

                plt.subplot(subplot_size[0], subplot_size[1], subplot_size[2])
                plt.subplots_adjust(wspace=0.3, hspace=0.2)

                plt.title('Henry region at ' + str(self.temps[i]) + ' °C')
                plt.plot(x_hen[i], y_henfit, '-', color=colours[i],
                         label="{temps} °C Fit".format(temps=self.temps[i]))
                plt.plot(x_hen[i], y_hen[i], 'ko', color='0.75',
                         label="Data at {temps} °C".format(temps=self.temps[i]))

        else:
            plot_settings(logplot, self.input_model, self.rel_pres)

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
                filenames = [filenames[0] + comp + filetype, filenames[1] + comp + filetype]
                if filetype == '.csv':
                    self.df_result[comp].to_csv(filenames[0])
                elif filetype == '.json':
                    self.df_result[comp].to_json(filenames[0])
                print('File saved to directory')
        else:
            filenames = [filenames[0] + filetype, filenames[1] + filetype]
            if filetype == '.csv':
                self.df_result[0].to_csv(filenames[0])
                self.df_result[1].to_csv(filenames[1])
            elif filetype == '.json':
                self.df_result[0].to_json(filenames[0])
                self.df_result[1].to_json(filenames[1])
            print('File saved to directory')

    def plot_emod(self, yfracs, ext_model='extended dsl', logplot=False):
        if self.model.lower() != "dsl":
            print("""This isotherm model is not supported for extended models. Currently supported models are:
            - DSL """)
        if ext_model.lower() == 'extended dsl':
            q_dict = ext_dsl(self.emod_input, self.temps, self.x, self.compname, yfracs)

        for i in range(len(self.compname)):
            plot_settings(logplot)
            q = q_dict[self.compname[i]]
            plt.title(self.compname[i])
            for j in range(len(self.temps)):
                plt.plot(self.x[i][j], q[j], '--', color=colours[j],
                         label="{temps} °C Fit".format(temps=self.temps[j]))
            plt.legend()
            plt.show()

        return q_dict



#
# df1 = pd.read_csv('Computational Data (EPFL) N2.csv')
# df2 = pd.read_csv('Computational Data (EPFL) CO2.csv')
# df_list = [df1, df2]
# compname = ['N2', 'CO2']
# temps = [10, 40, 100]
#
# #keyUptakes = ['q']
# #keyPressures = ['p']
#
# keyUptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
# keyPressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']
#
# # keyUptakes = ['Loading [mmol/g] 25', 'Loading [mmol/g] 50', 'Loading [mmol/g] 70']
# # keyPressures = ['Pressure [bar] 25', 'Pressure [bar] 50', 'Pressure [bar] 70']
#
# tolerance = 0.9999  # set minimum r squared value
#
# langmuir = IsothermFit(df_list, compname, temps, keyPressures, keyUptakes, "dsl")
# langmuir.fit(cond=True, show_hen=True, meth='tnc')
# # langmuir.plot(logplot=False)
# langmuir.plot_emod(yfracs=[0.15, 0.85], logplot=True)

