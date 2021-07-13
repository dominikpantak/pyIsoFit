import pandas as pd

from pyIsofit.core.model_definitions import get_guess_params, _MODEL_FUNCTIONS
from pyIsofit.ext_models.ext_dsl import ext_dsl
from pyIsofit.models.dsl import dsl_fit
from pyIsofit.models.langmuir import langmuir_fit, langmuirTD_fit
from pyIsofit.models.generic import generic_fit
from utilityFunctions import *
from pyIsofit.core.exceptions import *
import logging

logger = logging.getLogger('pyIsofit')

colours = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:olive']

_MODELS = ['mdr', 'mdr td', 'langmuir', 'langmuir linear 1', 'langmuir linear 2', 'langmuir td',
           'dsl nc', 'dsl', 'gab', 'sips', 'toth', 'toth td', 'bddt', 'bddt 2n', 'bddt 2n-1',
           'dodo', 'bet']


class IsothermFit:
    """
    A class used to characterise pure-component isotherm data with an isotherm model and
    utilise the results to predict co-adsorption using extended models.

    The class is split up into four methods - fit, plot, save and plot_emod (plot extended model)
    The IsothermFit class is instantiated by passing isotherm data into it.

    """

    def __init__(self,
                 df=None,
                 temps=None,
                 key_pressures=None,
                 key_uptakes=None,
                 model=None,
                 compname=None,
                 temp_units='C',
                 ):
        """
        :param df: pd.DataFrame or list[pd.DataFrame]
                    Pure-component isotherm data as a pandas dataframe - must be uptake in mmol/g and pressure in bar or
                    equivalent. If datasets at different temperatures are required for fitting, the user must specify
                    them in the same dataframe. A list of dataframes may be passed for the dual-site Langmuir isotherm
                    model procedure where parameter results across different components are utilised. Must be inputted
                    in the same order as compname (when passing as a list).

        :param temps: list[float]
                    List of temperatures corresponding to each dataset within the dataframe for results formatting and
                    for calculating heats of adsorption/ binding energies. Must be inputted in the same order as
                    key_pressures and key_uptakes.

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

        :param compname: str or list[str], optional
                    Name of pure component(s) for results formatting. If None is passed, self.compname is instantiated
                    as anarbitrary letter or a list of arbitrary letters corresponding to each component. Must be
                    inputted in the same order as compname (when passing as a list).

        :param temp_units: str, Optional
                    Units of temperature input (temps). Default is degrees C. Can accept Kelvin, 'K'.

        """

        # Checks

        if df is None:
            raise ParameterError("Input Pandas Dataframe with pure-component isotherm data for fitting")

        if temps is None:
            raise ParameterError("Input temperature corresponding to each pure-component isotherm dataset within the "
                                 " dataframe for fitting")

        if key_pressures is None:
            raise ParameterError("Input list of unique column key(s) which correspond to each dataset's pressure "
                                 "values within the Dataframe")

        if key_uptakes is None:
            raise ParameterError("Input list of unique column key(s) which correspond to each dataset's uptake "
                                 "values within the Dataframe")

        if model.lower() is None:
            raise ParameterError("Enter a model as a parameter")

        if model.lower() not in _MODELS:
            raise ParameterError("Enter a valid model - List of supported models:\n " + str(_MODELS))

        len_check = [len(key_uptakes), len(key_pressures), len(temps)]

        if len(temps) != sum(len_check) / len(len_check):
            raise ParameterError("Lengths of key_uptakes, key_pressures or temps do not match. "
                                 "Check that the length of each list is the same, corresponding to each dataset")

        if type(df) is list and model.lower() != "dsl":
            raise ParameterError("Enter one dataframe, not a list of dataframes")

        if compname is None and type(compname) is not list:
            self.compname = 'A'
            logger.info('No component name passed - giving component an arbitrary name.')

        elif compname is None and type(compname) is list:
            letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
            self.compname = [letters[i] for i in range(len(compname))]
            logger.info('No component names passed - giving components arbitrary names.')
            del letters

        logger.info('Checks successfully passed')

        self.df = df
        self.temps = temps
        self.temp_units = temp_units

        # Calculate temps in K if C is provided
        if self.temp_units == 'K':
            self.temps = temps
        else:
            self.temps = [t + 273 for t in temps]

        self.compname = compname
        self.key_pressures = key_pressures
        self.key_uptakes = key_uptakes
        self.model = model.lower()
        self.input_model = model

        # Internal parameters that are passed between methods
        self.x = []
        self.y = []
        self.params = []
        self.df_result = []
        self.emod_input = {}
        self.henry_params = []
        self.rel_pres = False

    def fit(self, cond=False,
            meth='leastsq',
            show_hen=False,
            hen_tol=0.999,
            rel_pres=False,
            henry_off=False,
            guess=None,
            cust_bounds=None,
            fit_report=False
            ):

        """
        Fit model to data using Non-Linear Least-Squares Minimization.
        This method is a generic fitting method for all models included in this package.

        Parameters
        ----------
        :param cond : bool
                Input whether to add standardised fitting constraints to fitting procedure. These are different
                for each fitting. Currently only works for Langmuir, Langmuir td, DSL, BDDT


        Note:
        ---------
        Because the dsl constrained fitting procedure fits a list of dataframes, the generic fitting method is not
        used when 'dsl' is inputted with the fitting condition as true and the method returns the result from the
        dsl_fit function. This is because the dsl_fit function carries out its' own initial guess calculations and
        henry regime estimations. The user may interact with this model in the same way as with the rest, however guess
        must be inputted as a list of dictionaries (just as with the list of DataFrames and component names).
        Custom bounds cannot yet be inputted into this model as this is a WIP.
        """

        if self.model == "dsl" and cond is True:
            self.x = []
            self.y = []

            if type(self.df) is not list:
                self.df = [self.df]

            if type(self.compname) is not list:
                self.compname = [self.compname]

            if type(guess) is not list and guess is not None:
                guess = [guess]

            try:
                dsl_result = dsl_fit(self.df, self.key_pressures, self.key_uptakes,
                                     self.temps, self.compname, meth, guess, hen_tol, show_hen, henry_off)
            except ValueError:
                logger.critical('Fitting aborted! Try changing lmfit fitting method (default is "leastsq")'
                                ' by passing it as an argument in .fit() i.e. meth="tnc".\n'
                                'Recommended "tnc" or "nedler"\n')

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
        guess = get_guess_params(self.model, self.df, self.key_uptakes, self.key_pressures)

        # Override defaults if user provides param_guess dictionary
        if guess is not None:
            for param, guess_val in guess.items():
                if param not in list(guess.keys()):
                    raise Exception("%s is not a valid parameter"
                                    " in the %s model." % (param, self.model))
                guess[param] = guess_val

        if henry_off is False:
            henry_constants = henry_approx(self.df, self.key_pressures, self.key_uptakes, show_hen, hen_tol,
                                           self.compname)
        else:
            henry_constants = None, None

        if self.model == "henry":
            self.henry_params = henry_constants
            return None

        if "mdr" in self.model:
            self.rel_pres = True

        self.x, self.y = get_xy(self.df, self.key_pressures, self.key_uptakes, self.model, rel_pres)

        base_params = self.x, self.y, guess, self.temps, cond, meth, cust_bounds, fit_report

        if "langmuir" in self.model and self.model != "langmuir td":
            self.params, values_dict = langmuir_fit(self.model, *base_params, henry_constants[0], henry_off)
            self.model = "langmuir"

        elif self.model == "langmuir td":
            self.params, values_dict = langmuirTD_fit(self.model, *base_params)

        else:
            if self.model == "dsl" and cond is False:
                self.model = "dsl nc"
            if self.model == "bddt 2n" or self.model == "bddt 2n-1" or self.model == "bddt":
                self.model = "bddt"

            self.params, values_dict = generic_fit(self.model, *base_params)

        final_results_dict, c_list = get_sorted_results(values_dict, self.model, self.temps)

        r_sq = [r2(self.x[i], self.y[i], _MODEL_FUNCTIONS[self.model], c_list[i]) for i in range(len(self.x))]
        se = [mse(self.x[i], self.y[i], _MODEL_FUNCTIONS[self.model], c_list[i]) for i in range(len(self.x))]

        final_results_dict['R squared'] = r_sq
        final_results_dict['MSE'] = se

        df_result = pd.DataFrame.from_dict(final_results_dict)
        pd.set_option('display.max_columns', None)

        print(f'\n\n\n---- Component {self.compname} fitting results -----')
        print(df_result)

        if self.model != "dsl":
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
                for j in range(len(self.key_pressures)):
                    plt.plot(self.x[i][j], comp_x_params[j].best_fit, '-', color=colours[j],
                             label="{temps} K Fit".format(temps=self.temps[j]))
                    plt.plot(self.x[i][j], self.y[i][j], 'ko', color='0.75',
                             label="Data at {temps} K".format(temps=self.temps[j]))

        elif self.model == "henry":
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

            for i in range(len(self.key_pressures)):
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
        if self.model != "dsl":
            print("""This isotherm model is not supported for extended models. Currently supported models are:
            - DSL """)
            return None

        if ext_model == 'extended dsl':
            q_dict = ext_dsl(self.emod_input, self.temps, self.x, self.compname, yfracs)
        else:
            return None

        for i in range(len(self.compname)):
            plot_settings(logplot)
            q = q_dict[self.compname[i]]
            plt.title(f'Co-adsorption isotherm for component {self.compname[i]} at mol frac of {yfracs[i]}')
            for j in range(len(self.temps)):
                plt.plot(self.x[i][j], q[j], '--', color=colours[j],
                         label="{temps} K Fit".format(temps=self.temps[j]))
            plt.legend()
            plt.show()

        return q_dict


# df1 = pd.read_csv('SIFSIX-3-Cu CO2.csv')
# df1 = pd.read_csv('../Datasets for testing/Lewatit CO2.csv')
df1 = pd.read_csv('../Datasets for testing/Computational Data (EPFL) CO2.csv')
df2 = pd.read_csv('../Datasets for testing/Computational Data (EPFL) N2.csv')

df_list = [df1, df2]
compname = ['CO2', 'N2']
temps = [10, 40, 100]
# temps = [0, 20]
# temps = [25, 45, 55]

# keyUptakes = ['q1', 'q2', 'q3', 'q4']
# keyPressures = ['p1', 'p2', 'p3', 'p4']

keyUptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
keyPressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']

# keyUptakes = ['q 298', 'q 318', 'q 328']
# keyPressures = ['p 298', 'p 318', 'p 328']

# keyUptakes = ['Loading [mmol/g] 25', 'Loading [mmol/g] 50', 'Loading [mmol/g] 70']
# keyPressures = ['Pressure [bar] 25', 'Pressure [bar] 50', 'Pressure [bar] 70']


langmuir = IsothermFit(df_list, temps, keyPressures, keyUptakes, "dsl", compname)
langmuir.fit(cond=True, show_hen=True, meth='tnc')
langmuir.plot(logplot=True)
# langmuir.save()
langmuir.plot_emod(yfracs=[0.15, 0.5, 0.35], logplot=True)
