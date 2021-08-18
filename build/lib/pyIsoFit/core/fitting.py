"""
Main module for all of the features that pyIsoFit-master has.
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from src.pyIsoFit.core.model_fit_def import get_guess_params
from src.pyIsoFit.core.model_dicts import _MODEL_FUNCTIONS, _MODEL_PARAM_LISTS
from src.pyIsoFit.ext_models.ext_dsl import ext_dsl
from src.pyIsoFit.models.dsl import dsl_fit
from src.pyIsoFit.models.generic import generic_fit
from src.pyIsoFit.core.utility_functions import get_sorted_results, get_xy, heat_calc, plot_settings, \
    get_subplot_size, colours, save_func
from src.pyIsoFit.core.model_equations import mse, henry
from src.pyIsoFit.core.exceptions import ParameterError, SaveError
import logging
from IPython.display import display

from src.pyIsoFit.models.henry import henry_approx

logger = logging.getLogger('pyIsoFit-master')

_MODELS = ['mdr', 'mdr td', 'langmuir', 'langmuir linear 1', 'langmuir linear 2', 'langmuir td',
           'dsl nc', 'dsl', 'gab', 'sips', 'toth', 'toth td', 'bddt', 'bddt 2n', 'bddt 2n-1',
           'dodo', 'bet', 'henry']

_does_something = ['langmuir', 'linear langmuir 1', 'linear langmuir 2', 'langmuir td', 'bddt', 'bddt 2n-1', 'bddt 2n']


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

        elif compname is None and type(df) is list:
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
        self.df_result = None
        self.emod_input = {}
        self.henry_params = []
        self.rel_pres = False

    def info_params(self):
        """
        Prints information about the model to be fit
        (WIP)

        """
        print(f'Parameters for the {self.model} model:')
        print(_MODEL_PARAM_LISTS[self.model])

    def fit(self,
            cond=False,
            meth='leastsq',
            show_hen=False,
            hen_tol=0.999,
            rel_pres=False,
            henry_off=False,
            guess=None,
            cust_bounds=None,
            fit_report=False,
            weights=None,
            dsl_comp_a=None
            ):

        """
        Plotting method for the FitIsotherm class.
        Fits model to data using Non-Linear Least-Squares Minimization.
        This method is a generic fitting method for all models included in this package using the lmfit
        Parameters and Models class.

        Parameters
        ----------

        :param cond : bool
                Input whether to add standardised fitting constraints to fitting procedure. These are different
                for each fitting. Currently only works for Langmuir, Langmuir td, DSL, BDDT. Default is False

        :param meth : str
                Input the fitting algorithm which lmfit uses to fit curves. Default is 'leastsq' however lmfit includes
                many fitting algorithms which can be inputted (https://lmfit.github.io/lmfit-py/fitting.html).

        :param show_hen : bool
                Input whether to show the henry regime of the datasets approximated by the package. This is False by
                default.

        :param hen_tol : float or list[float]
                The henry region approximation function calculates the henry region by finding a line with the highest
                R squared value in the low pressure region of the dataset. This is done with a default R squared
                tolerance value (set to 0.999).

                For example, if a float is inputted (a different henry tolerance) this will be the henry tolerance value
                used by the function. i.e if 0.98 is inputted the henry regime will be across a large pressure range
                due to the low tolerance for the R squared value of the henry model fitting.

                This function also supports inputting the henry regimes manually. For this, input each henry regime for
                each dataset as a list i.e [1.2, 2.1, ... ]

        :param rel_pres : bool
                Input whether to fit the x axis data to relative pressure instead of absolute. Default is False

        :param henry_off : bool
                Input whether to turn off the henry regime fitting constraint when using the standardised fitting
                constraint to langmuir or dsl - this is usually done when fitting experimental data which has a messy
                low pressure region. Default is False.

        :param guess : dict
                Input custom guess values to override the default guess values. This must be inputted as a dictionary
                with the keys corresponding to the parameter string and the value corresponding to the list of guess
                values corresponding to each dataset.
                i.e for Langmuir: guess = {'q': [5, 5, 6], 'b':[100, 1000, 2000]}

        :param cust_bounds : dict
                Input custom bounds for the fitting. These are hard constraints and lmfit will fit only within these
                minimum and maximum values. Input these as a dictionary with the keys corresponding to the parameter
                string and the value corresponding to the list of tuples which include bounds for each dataset in the
                format (min, max).
                i.e for Langmuir: cust_bounds = {'q': [(4,6), (4, None), (5,10)], ... ect.}

        :param fit_report : bool
                Display a fitting report generated by lmfit for each dataset. Default is False

        :param weights : list[list[float]]
                Weights for fitting

        :param dsl_comp_a : str
                Manually input which component is the most adsorbed component (compA) for the dsl constrained
                fitting procedure.

        :return Returns a dictionary of fitting results


        Note:
        ---------
        Because the dsl constrained fitting procedure fits a list of dataframes, the generic fitting method is not
        used when 'dsl' is inputted with the fitting condition as true and the method returns the result from the
        dsl_fit function. This is because the dsl_fit function carries out its' own initial guess calculations and
        henry regime estimations. The user may interact with this model in the same way as with the rest, however guess
        must be inputted as a list of dictionaries (just as with the list of DataFrames and component names).
        Custom bounds cannot yet be inputted into this model as this is a WIP.
        """
        # When model is dsl and the constrained fitting condtion is true, the generic fitting method is ignored
        # and instead everything within this condition is run
        if self.model == "dsl" and cond is True:
            logger.info('DSL Fitting procedure commenced')

            # Input checks - dsl procedure requires lists of dataframes, components and guesses to function
            if type(self.df) is not list:
                self.df = [self.df]

            if type(self.compname) is not list:
                self.compname = [self.compname]

            if type(guess) is not list and guess is not None:
                guess = [guess]

            # Calling the DSL fitting function - see DSL function docstring for more info
            try:
                dsl_result = dsl_fit(self.df, self.key_pressures, self.key_uptakes,
                                     self.temps, self.compname, meth, guess, hen_tol, show_hen, henry_off, dsl_comp_a)
            except ValueError:
                # Often the fitting algorithm is the source of the error
                logger.critical('The model function generated NaN values and the fit aborted!'
                                'Please check your model function and/or set boundaries on parameters where applicable'
                                'In these cases try changing lmfit fitting method (default is "leastsq") to "tnc"')

            df_dict, results_dict, df_res_dict, params_dict = dsl_result

            # Allocating x and y axis co-ordinates to class variables
            # Produces a list of lists - outer corresponds to component, inner corresponds to temperatures
            self.params = results_dict
            for i in range(len(self.compname)):
                x_i, y_i = df_dict[self.compname[i]]
                self.x.append(x_i)
                self.y.append(y_i)

            # Allocating the resulting dataframes for use within the .save() method
            self.df_result = df_res_dict

            # Allocating the parameter result dictionary for use within the .plot_emod() method
            self.emod_input = params_dict

            return df_res_dict
        logger.info('Generic fitting procedure commenced')

        # cond=True only works for certain models - This notifies the user of this
        if self.model not in _does_something and cond is not False:
            logger.warning(f"WARNING You have set cond={cond} but cond for the model '{self.model}' does nothing.\n")

        # Calculating henry constants and displaying results (see function docstring)
        if self.model == "henry":
            show_hen = True

        henry_params = henry_approx(self.df, self.key_pressures, self.key_uptakes, show_hen, hen_tol,
                                    self.compname, henry_off)

        henry_constants = henry_params[0]

        # When model is henry, only the henry approximation function is run and the rest is ignored
        if self.model == "henry":
            logger.info('Henry model fitting only chosen')
            self.henry_params = henry_params
            self.df_result = henry_params[1]
            return None

        # ! Dictionary of parameters as a starting point for data fitting
        guess = get_guess_params(self.model, self.df, self.key_uptakes, self.key_pressures)
        logger.info('Guess values successfully obtained')

        # Override defaults if user provides param_guess dictionary
        if guess is not None:
            for param, guess_val in guess.items():
                if param not in list(guess.keys()):
                    raise ParameterError("%s is not a valid parameter"
                                         " in the %s model." % (param, self.model))
                guess[param] = guess_val
            logger.info('Guess values overridden with custom guess values')

        # MDR works best when dealing with relative pressures - This forces it on
        if "mdr" in self.model:
            logger.info('MDR chosen so relative pressure toggle force set')
            rel_pres = True
        self.rel_pres = rel_pres

        # Extracts the x and y parameter lists from the dataframe
        self.x, self.y = get_xy(self.df, self.key_pressures, self.key_uptakes, self.model, rel_pres)
        logger.info('x and y parameters successfully obtained')

        if weights is None:
            logger.info('No weights inputted - setting weights to x')
            weights = self.x

        # Everything except the model function is the same for the bddt fitting procedure
        # This reduces extra entries within the model_dicts.py file by standardising the input
        if self.model == "bddt 2n" or self.model == "bddt 2n-1" or self.model == "bddt":
            self.model = "bddt"

        # ----------Generic fitting procedure--------------

        # Allocate list of lmfit result objects to class and create values_dict variable
        self.params, values_dict = generic_fit(self.model, weights, self.y, guess, self.temps, cond, meth, cust_bounds,
                                               fit_report, henry_constants, henry_off)
        logger.info('Generic fit completed successfully')

        # Sorts the values_dict variable to a dictionary which can be used to create a dataframe result and creates a
        # list of result parameters to feed into the error checking functions
        final_results_dict, c_list = get_sorted_results(values_dict, self.model, self.temps)
        logger.info('Results sorted successfully')

        # Find the r squared and mean squared error (mse) values
        # r_sq = [r2(self.x[i], self.y[i], _MODEL_FUNCTIONS[self.model], c_list[i]) for i in range(len(self.x))]
        se = [mse(self.x[i], self.y[i], _MODEL_FUNCTIONS[self.model], c_list[i]) for i in range(len(self.x))]
        logger.info('Mean squared error calculated successfully')

        # Add r squared and mse results to final results dictionary
        # final_results_dict['R squared'] = r_sq
        final_results_dict['MSE'] = se

        # Create dataframe with fitting results
        df_result = pd.DataFrame.from_dict(final_results_dict)
        pd.set_option('display.max_columns', None)

        # Print fitting results
        print(f'\n\n\n---- Component {self.compname} fitting results -----')
        display(df_result)

        # Save dataframes for model results and henry regime to class for use within the .save() function
        self.df_result = df_result

        if len(self.temps) >= 3:
            heat_calc(self.model, self.temps, final_results_dict, self.x)

    def plot(self, logplot=(False, False)):
        """
        Plotting method for the FitIsotherm class.
        There are three plotting procedures:
         - For more than one component
         - For henry plots (This requires plotting on individual subplots)
         - The generic plotting procedure for any other model with one component (most models use this)

        :param logplot: tuple(bool)
            Whether to have an x and y log axis. Default is off for both x and y axis i.e (False, False) in the order
            (x, y)

        """
        np.linspace(0, 10, 301)

        fit_label = "{temps} K Fit"
        data_label = "Data at {temps} K"

        # Plotting for more than one component
        if type(self.df) is list:
            for i in range(len(self.df)):
                # Get plot settings
                plot_settings(logplot)

                comp_x_params = self.params[self.compname[i]]
                plt.title(self.compname[i])
                for j in range(len(self.key_pressures)):
                    plt.plot(self.x[i][j], comp_x_params[j].best_fit, '-', color=colours[j],
                             label=fit_label.format(temps=self.temps[j]))
                    plt.plot(self.x[i][j], self.y[i][j], 'ko', color='0.75',
                             label=data_label.format(temps=self.temps[j]))

        # Plotting for henry model
        elif self.model == "henry":
            henry_const = self.henry_params[0]
            xy_dict = self.henry_params[2]
            x_hen = xy_dict['x']
            y_hen = xy_dict['y']
            plot_settings(logplot)
            lenx = len(self.x)
            for i in range(len(x_hen)):
                y_henfit = henry(x_hen[i], henry_const[i])
                subplot_size = get_subplot_size(lenx, i)

                plt.subplot(subplot_size[0], subplot_size[1], subplot_size[2])
                plt.subplots_adjust(wspace=0.3, hspace=0.3)

                plt.title('Henry regime at ' + str(self.temps[i]) + ' K')
                plt.plot(x_hen[i], y_henfit, '-', color=colours[i],
                         label=fit_label.format(temps=self.temps[i]))
                plt.plot(x_hen[i], y_hen[i], 'ko', color='0.75',
                         label=data_label.format(temps=self.temps[i]))
                plt.legend()

        # Plotting for any other model
        else:
            plot_settings(logplot, self.input_model, self.rel_pres)

            for i in range(len(self.key_pressures)):
                plt.plot(self.x[i], self.params[i].best_fit, '-', color=colours[i],
                         label=fit_label.format(temps=self.temps[i]))
                plt.plot(self.x[i], self.y[i], 'ko', color='0.75',
                         label=data_label.format(temps=self.temps[i]))
        plt.legend()
        plt.show()

    def save(self, directory=None, filestring=None, filetype='.csv'):
        """
        Saves the model fitting result and henry region fitting result dataframes to directory as a .csv or .json
        file.

        :param directory:
            Full destination directory must be inputted for the user to save a file

        :param filestring: list[str] or str
            This is a list of strings corresponding to the file names, first position is fit result, second is
            henry result. Inputting this as a string for the fit result only will also work.

        :param filetype: str
            .csv or .json for saving

        """
        # Directory input check
        if directory is None:
            raise SaveError("\n\nPlease enter full directory for saving file separated by double dashes i.e "
                            "C:\\Users\\User\\pyIsoFit-master\\fittingresults\\")

        # File name input check
        if filestring is None:
            filestring = 'fit_result'

        # Creates file for each component when there are multiple components
        if type(self.df_result) is dict:
            for comp in self.df_result:
                save_func(directory, filestring, filetype, self.df_result[comp], comp)
        else:
            save_func(directory, filestring, filetype, self.df_result)

    def plot_emod(self, yfracs, ext_model='extended dsl', logplot=(False, False)):
        """
        Predicts co-adsorption isotherm data and plots it.

        :param yfracs: list[float]
            List of component mole fractions within the gas mixture

        :param ext_model: str
            Extended model for the method to predict co-adsorption with. Currently extended DSL is the only
            model included

        :param logplot: bool
            Whether to have an x and y log axis.

        :return:
            Returns a dictionary of co-adsorption uptakes for each component
        """

        # Checks
        if len(self.compname) < 2:
            raise ParameterError("Enter 2 components or more to use extended models")

        if self.model != "dsl":
            raise ParameterError("""This isotherm model is not supported for extended models. Currently supported
             models are:
            - DSL """)

        # Only extended dsl is currently supported so no need for if else statements
        q_dict = ext_dsl(self.emod_input, self.temps, self.x, self.compname, yfracs)

        # Plot results ext_dsl function
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
