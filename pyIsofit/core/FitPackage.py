
from pyIsofit.core.model_definitions import get_guess_params, get_model
from pyIsofit.ext_models.ext_dsl import ext_dsl
from pyIsofit.models.dsl import dsl_fit
from pyIsofit.models.langmuir import langmuir_fit, langmuirTD_fit
from pyIsofit.models.generic import generic_fit
from utilityFunctions import *


colours = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:olive']


class IsothermFit:
    """
    A class used to characterise pure-component isotherm data with an isotherm model and
    utilise the results to predict co-adsorption using extended models.

    The class is split up into four methods - fit, plot, save and plot_emod (plot extended model)
    The IsothermFit class is instantiated by passing isotherm data into it.

    Parameters
    ----------
    df : pd.DataFrame or list[pd.DataFrame]
        Pure-component isotherm data as a pandas dataframe. If datasets at different temperatures
        are required for fitting, the user must specify them in the same dataframe. A list of dataframes
        may be passed for the dual-site Langmuir isotherm model procedure where parameter results across different
        components are utilised. Must be inputted in the same order as compname (when passing as a list).

    temps: list[float]
        List of temperatures corresponding to each dataset within the dataframe for results formatting and for
        calculating heats of adsorption/ binding energies. Must be inputted in the same order as key_pressures and
        key_uptakes.

    key_pressures : list[str]
        List of unique column key(s) which correspond to each dataset's pressure values within the
        dataframe. Can input any number of keys corresponding to any number of datasets in the dataframe.
        If multiple dataframes are specified, make sure keys are identical across each dataframe for each temperature.
        Must be inputted in the same order as key_uptakes and temps.

    key_uptakes : list[str]
        List of unique column key(s) which correspond to each dataset's uptake values within the
        dataframe. Can input any number of keys corresponding to any number of datasets in the dataframe.
        If multiple dataframes are specified, make sure keys are identical across each dataframe for each temperature.
        Must be inputted in the same order as key_pressures and temps.

    model : str
        Model to be fit to dataset(s).

    compname: str or list[str], optional
        Name of pure component(s) for results formatting. If None is passed, self.compname is instantiated as an
        arbitrary letter or a list of arbitrary letters corresponding to each component. Must be inputted in the same
        order as compname (when passing as a list).

    temp_units : str, Optional
        Units of temperature input (temps). Default is degrees C. Can accept Kelvin, 'K'.
    """

    def __init__(self,
                 df=None,
                 temps=None,
                 key_pressures=None,
                 key_uptakes=None,
                 model=None,
                 compname=None,
                 temp_units='C'
                 ):

        self.df = df  # Dataframe with uptake vs. pressure data
        self.temps = temps  # Temp in deg C
        self.temp_units = temp_units

        if self.temp_units == 'K':
            self.temps = temps
        else:
            self.temps = [t + 273 for t in temps]

        self.compname = compname  # Name of component

        if compname is None and type(compname) is not list:
            self.compname = 'A'
        elif compname is None and type(compname) is list:
            letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
            self.compname = [letters[i] for i in range(len(compname))]


        self.keyPressures = key_pressures  # names of the column headers for pressure and uptakes
        self.keyUptakes = key_uptakes  #
        self.model = model
        self.input_model = model

        if type(self.df) is list and model.lower() != "dsl":
            raise Exception("\n\n\n\nEnter one dataframe, not a list of dataframes")

        if model is None:
            raise Exception("Enter a model as a parameter")

        self.x = []
        self.y = []
        self.params = []
        self.df_result = []
        self.emod_input = {}

        self.henry_params = []

    def fit(self, cond=True,
            meth='leastsq',
            show_hen=False,
            hen_tol=0.999,
            rel_pres=False,
            henry_off=False,
            guess=None,
            cust_bounds=None
            ):

        """ Fit model to data using """

        #
        if self.model.lower() == "dsl" and cond is True:
            self.x = []
            self.y = []

            if type(self.df) is not list:
                self.df = [self.df]

            if type(self.compname) is not list:
                self.compname = [self.compname]

            if type(guess) is not list and guess is not None:
                guess = [guess]

            dsl_result = dsl_fit(self.df, self.keyPressures, self.keyUptakes,
                                 self.temps, self.compname, self.meth, guess, hen_tol, show_hen, henry_off)

            df_dict, results_dict, df_res_dict, params_dict = dsl_result

            self.params = results_dict
            for i in range(len(self.compname)):
                x_i, y_i = df_dict[self.compname[i]]
                self.x.append(x_i)
                self.y.append(y_i)
            self.df_result = df_res_dict
            self.emod_input = params_dict

            return params_dict

        # ! Dictionary of parameters as a starting point for data fitting

        guess = get_guess_params(self.model, self.df, self.keyUptakes, self.keyPressures)

        # Override defaults if user provides param_guess dictionary
        if guess is not None:
            for param, guess_val in guess.items():
                if param not in list(guess.keys()):
                    raise Exception("%s is not a valid parameter"
                                    " in the %s model." % (param, self.model))
                guess[param] = guess_val

        if henry_off is False:
            henry_constants = henry_approx(self.df, self.keyPressures, self.keyUptakes, show_hen, hen_tol,
                                           self.compname)
        else:
            henry_constants = None, None

        if self.model.lower() == "henry":
            self.henry_params = henry_constants
            return None

        if "mdr" in self.model.lower():
            self.rel_pres = True

        self.x, self.y = get_xy(self.df, self.keyPressures, self.keyUptakes, self.model, rel_pres)

        base_params = self.model, self.x, self.y, guess, self.temps, cond, meth, cust_bounds

        if "langmuir" in self.model.lower() and self.model.lower() != "langmuir td":
            self.params, values_dict = langmuir_fit(*base_params, henry_constants[0], henry_off)
            self.model = "langmuir"

        elif self.model.lower() == "langmuir td":
            self.params, values_dict = langmuirTD_fit(*base_params)

        else:
            if self.model.lower() == "dsl" and cond is False:
                self.model = "dsl nc"
            elif self.model.lower() == "bddt 2n" or self.model.lower() == "bddt 2n-1" or self.model.lower() == "bddt":
                self.model = "bddt"

            self.params, values_dict = generic_fit(*base_params)

        final_results_dict, c_list = get_sorted_results(values_dict, self.model, self.temps)

        r_sq = [r2(self.x[i], self.y[i], get_model(self.model), c_list[i]) for i in range(len(self.x))]
        se = [mse(self.x[i], self.y[i], get_model(self.model), c_list[i]) for i in range(len(self.x))]

        final_results_dict['R squared'] = r_sq
        final_results_dict['MSE'] = se

        df_result = pd.DataFrame.from_dict(final_results_dict)
        pd.set_option('display.max_columns', None)
        print(df_result)

        if self.model.lower() != "dsl":
            self.df_result.append(df_result)
            self.df_result.append(henry_constants[1])

        if len(self.temps) >= 3:
            heat_calc(self.model, self.temps, final_results_dict, self.x)

    def plot(self, logplot=False):
        np.linspace(0, 10, 301)

        if type(self.df) is list:
            for i in range(len(self.df)):
                plot_settings(logplot)

                comp_x_params = self.params[self.compname[i]]
                plt.title(self.compname[i])
                for j in range(len(self.keyPressures)):
                    plt.plot(self.x[i][j], comp_x_params[j].best_fit, '-', color=colours[j],
                             label="{temps} K Fit".format(temps=self.temps[j]))
                    plt.plot(self.x[i][j], self.y[i][j], 'ko', color='0.75',
                             label="Data at {temps} K".format(temps=self.temps[j]))

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
                         label="{temps} K Fit".format(temps=self.temps[i]))
                plt.plot(x_hen[i], y_hen[i], 'ko', color='0.75',
                         label="Data at {temps} K".format(temps=self.temps[i]))

        else:
            plot_settings(logplot, self.input_model, self.rel_pres)

            for i in range(len(self.keyPressures)):
                plt.plot(self.x[i], self.params[i].best_fit, '-', color=colours[i],
                         label="{temps} K Fit".format(temps=self.temps[i]))
                plt.plot(self.x[i], self.y[i], 'ko', color='0.75',
                         label="Data at {temps} K".format(temps=self.temps[i]))
        plt.legend()
        plt.show()

    def save(self, filestring=None, filetype='.csv', save_henry=False):

        filenames = ['fit_result', 'henry_result'] if filestring is None else filestring

        if type(self.df_result) is dict:
            for comp in self.df_result:
                filenames = [filenames[0] + comp + filetype, filenames[1] + comp + filetype]
                if filetype == '.csv':
                    self.df_result[comp].to_csv('fitting_results/' + filenames[0])
                elif filetype == '.json':
                    self.df_result[comp].to_json('fitting_results/' + filenames[0])
                print('File saved to directory')
        else:
            filenames = [filenames[0] + filetype, filenames[1] + filetype]
            if filetype == '.csv':
                self.df_result[0].to_csv('fitting_results/' + filenames[0])
                if save_henry:
                    self.df_result[1].to_csv('fitting_results/' + filenames[1])
            elif filetype == '.json':
                self.df_result[0].to_json('fitting_results/' + filenames[0])
                if save_henry:
                    self.df_result[1].to_json('fitting_results/' + filenames[1])
            print('File saved to directory')

    def plot_emod(self, yfracs, ext_model='extended dsl', logplot=False):
        if self.model.lower() != "dsl":
            print("""This isotherm model is not supported for extended models. Currently supported models are:
            - DSL """)
            return None

        if ext_model.lower() == 'extended dsl':
            q_dict = ext_dsl(self.emod_input, self.temps, self.x, self.compname, yfracs)
        else:
            return None

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


# # df1 = pd.read_csv('SIFSIX-3-Cu CO2.csv')
# df1 = pd.read_csv('../Datasets for testing/Lewatit CO2.csv')
# # df2 = pd.read_csv('Computational Data (EPFL) CO2.csv')
# # df_list = [df1, df2]
# compname = 'CO2'
# temps = [25, 50, 75, 100]
# # temps = [0, 20]
# # temps = [25, 45, 55]
#
# keyUptakes = ['q1', 'q2', 'q3', 'q4']
# keyPressures = ['p1', 'p2', 'p3', 'p4']
#
# # keyUptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
# # keyPressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']
#
# # keyUptakes = ['q 298', 'q 318', 'q 328']
# # keyPressures = ['p 298', 'p 318', 'p 328']
#
# # keyUptakes = ['Loading [mmol/g] 25', 'Loading [mmol/g] 50', 'Loading [mmol/g] 70']
# # keyPressures = ['Pressure [bar] 25', 'Pressure [bar] 50', 'Pressure [bar] 70']
#
# # guess = {'q1': [1.4, 1.4, 1.4, 1.4],
# #          'q2': [3, 3, 3, 3],
# #          'b1': [10, 10, 10, 10],
# #          'b2': [1, 1, 1, 1]}
#
#
# langmuir = IsothermFit(df1, compname, temps, keyPressures, keyUptakes, "toth")
# langmuir.fit(cond=False, show_hen=True, meth='leastsq')
# langmuir.plot(logplot=True)
# langmuir.save()
# # langmuir.plot_emod(yfracs=[0.15, 0.85], logplot=True)
