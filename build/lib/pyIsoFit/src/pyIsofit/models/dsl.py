import pandas as pd
import numpy as np
from scipy import stats
from lmfit import Model, Parameters

from src.pyIsoFit.core.model_equations import langmuir1, dsl, dsltd, langmuirTD, r, bold, unbold, mse
from src.pyIsoFit.core.model_fit_def import get_guess_params
from src.pyIsoFit.models import henry_approx
from IPython.display import display


def dsl_fit(df_list: list,
            key_pressures: list,
            key_uptakes: list,
            temps: list,
            compnames: list,
            meth: str,
            guess: list,
            hentol: float,
            show_hen: bool,
            henry_off: bool,
            dsl_comp_a: str
            ):
    """
    Thermodynamically consistent fitting procedure written with accordance to thermodynamic consistency principles.
    These are similar procedures to ones described by Farmahini et al. (2018) (doi.org/10.1021/acs.iecr.8b03065).

    Comprised of three inner functions, corresponding to each step in the procedure, which are called sequentially
    for each component.

    The outer function extracts data from Pandas Dataframes, sorts them, finds initial guess values, finds henry
    constants and calls the inner functions to carry out fitting.


    :param df_list: list[DataFrame]
            List of Pandas Dataframes corresponding to each component.

    :param compnames: list[str]
            List of component names corresponding to each component.

    :param guess: list[dict]
            List of guess dictionaries corresponding to each component.

    :return: tuple of:
        [0] : Dictionary of x and y co-ordinates corresponding to each component for all temperatures in the form of:
            {'component n' : {'x': [list[x1], list[x2],..., list[xn]],
                            {'y': [list[y1], list[y2],..., list[yn]]},
            'component n+1' : {...}
            }

        [1] : Dictionary of lmfit ModelResult objects corresponding to each component for all temperatures in the
                form of:
            {'component n' : list[ModelResult], 'component n+1': ... }

        [2] : Dictionary of Pandas Dataframe objects corresponding to each component in the form of:
            {'component n' : Dataframe, 'component n+1' : ...}

        [3] : Dictionary of best fitting parameters corresponding to each component in the form of:
            {'component n' : {'q1' : list[pars],
                              'q2' : list[pars],
                              ...
                              },
            'component n+1' : {...}
            }


        xy_dict, results_dict, df_res_dict, params_dict
    """
    def step1(x, y, meth, guess, henry_constants):
        """
        The fitting procedure is started from the lowest temperature dataset as experimentally this leads to the
        determination to the closest value to the actual saturated adsorption capacity of the material.

        The model must also reduce to Henry’s law at a limit of zero
        pressure, and so the following constraint must be set in the fitting procedure:

        {KH = (q1 * b1) + (q2 * b2)}


        :return:
            Returns initial qsat values to be fixed for the remaining datasets.

        Note - This function is first used for finding the most adsorbed component from a list of dataframes
        """

        gmod = Model(dsl)
        pars1 = Parameters()

        pars1.add('q1', value=guess['q1'][0], min=0)
        pars1.add('q2', value=guess['q2'][0], min=0)
        pars1.add('b2', value=guess['b2'][0], min=0)
        if henry_off:
            pars1.add('b1', value=guess['b1'][0], min=0)
        else:
            pars1.add('delta', value=henry_constants[0], vary=False)
            pars1.add('b1', expr='(delta-q2*b2)/q1')  # KH = b1*q1 + b2*q2

        result1 = gmod.fit(y[0], pars1, x=x[0], method=meth)

        return result1.values['q1'], result1.values['q2']

        # This ends the function within the inner class and returns the qmax values to
        # be used in step 2

    def step2(x, y, meth, guess, henry_constants, temps, step1_pars, comp2=False):
        """
        The same procedure as in step 1 is followed for the remaining isotherm datasets with the same constraints as
        described in step 1. q1 and q2 are fixed at that of the first temperature dataset.
        Fitting results in q1, q2, b1 and b2 being found, after which Van't Hoff plots are made to find b01, b02, h1
        and h2.

        :return:
            Returns calculated b01, b02, h1 and h2 for use in step 3
        """
        q1fix, q2fix = step1_pars

        if comp2:
            # Depending on whether the package is fitting for the most adsorbed componennt or the
            # remaining components, the fitting procedure is different. In the methodology used,
            # The most adsorbed component is fitted to DSL but the remaining components are
            # fit to essentially the single langmuir isotherm
            gmod = Model(langmuir1)
            qtot = q1fix + q2fix

            c_list = []
            for i in range(len(x)):
                pars = Parameters()
                pars.add('q', value=qtot, min=qtot, max=qtot + 0.001)
                if henry_off:
                    pars.add('b2', value=guess['b2'][i])
                else:
                    pars.add('delta', value=henry_constants[i], vary=False)
                    pars.add('b', expr='delta/q')

                results = gmod.fit(y[i], pars, x=x[i], method=meth)
                c = [q1fix, q2fix, results.values['b'], results.values['b']]
                c_list.append(c)

                del results
                del pars

        else:
            gmod = Model(dsl)
            c_list = []
            for i in range(len(x)):
                pars = Parameters()
                pars.add('q1', value=q1fix, min=q1fix, max=q1fix + 0.001)
                pars.add('q2', value=q2fix, min=q2fix, max=q2fix + 0.001)
                pars.add('b2', value=guess['b2'][i], min=0)
                if henry_off:
                    pars.add('b1', value=guess['b1'][i])
                else:
                    pars.add('delta', value=henry_constants[i], vary=False)
                    pars.add('b1', expr='(delta-q2*b2)/q1', min=0)  # KH = b*q

                results = gmod.fit(y[i], pars, x=x[i], method=meth)
                c = [results.values['q1'], results.values['q2'],
                     results.values['b1'], results.values['b2']]
                c_list.append(c)

                del results
                del pars

        # allocating variables
        qsat1 = c_list[0][0]
        qsat2 = c_list[0][1]
        b1 = [param[2] for param in c_list]
        b2 = [param[3] for param in c_list]

        # Finding heat of adsorption for both sites
        T = np.array([1 / temp for temp in temps])
        ln_b1 = np.array([np.log(i) for i in b1])
        ln_b2 = np.array([np.log(i) for i in b2])
        mH1, bH1, rH1, pH1, sterrH1 = stats.linregress(T, ln_b1)
        mH2, bH2, rH2, pH2, sterrH2 = stats.linregress(T, ln_b2)

        h = [-mH1 * r, -mH2 * r]
        b0 = [np.exp(bH1), np.exp(bH2)]

        return qsat1, qsat2, h, b0
        # The package returns these sets of parameters to be used as initial guesses for the final step

    def step3(x, y, meth, temps, step2_pars, comp2=False):
        """
        The temperature dependent form of the DSL equation is now used for fitting, which is the DSL equation
        with the Van’t Hoff equation substituted for b1 and b2 giving a model with 6 fitting
        parameters; b01, b02, h1, h2, q1, q2. The parameters calculated at the end of step 2 are used as
        initial guess values for this fitting.

        :return: tuple of:
                A list of lmfit results objects [0], the results in the form of a dataframe [1],
                a dictionary of the fitting results
        """
        # unpacking tuples
        q1_in, q2_in, h_in, b0_in = step2_pars

        # We can now fit to the van't Hoff form of DSL with the initial guess values
        # obtained through step 2

        if comp2 == True:
            # Again, we fit to the single langmuir form when the least adsorbed species is used
            gmod = Model(langmuirTD)
            qtot = q1_in + q2_in
            c_list = []
            results_lst = []
            for i in range(len(x)):
                pars = Parameters()
                pars.add('t', value=temps[i], vary=False)
                pars.add('q', value=qtot, min=qtot, max=qtot + 0.001)
                pars.add('h', value=h_in[0])
                pars.add('b0', value=b0_in[0], min=0)

                results = gmod.fit(y[i], pars, x=x[i], method=meth)
                results_lst.append(results)

                c_list.append([results.values['t'], q1_in, q2_in, results.values['h'],
                               results.values['h'], results.values['b0'], results.values['b0']])

                del results
                del pars

        else:
            gmod = Model(dsltd)  # DSL
            c_list = []
            results_lst = []
            for i in range(len(x)):
                pars = Parameters()
                pars.add('t', value=temps[i], vary=False)
                pars.add('q1', value=q1_in, min=q1_in, max=q1_in + 0.001)
                pars.add('q2', value=q2_in, min=q2_in, max=q2_in + 0.001)
                pars.add('h1', value=h_in[0])
                pars.add('h2', value=h_in[1])
                pars.add('b01', value=b0_in[0], min=0)
                pars.add('b02', value=b0_in[1], min=0)

                results = gmod.fit(y[i], pars, x=x[i], method=meth)
                results_lst.append(results)
                c_list.append([results.values['t'], results.values['q1'], results.values['q2'], results.values['h1'],
                               results.values['h2'], results.values['b01'], results.values['b02']])

                del results
                del pars

        print(bold + "\nParameters found..." + unbold)

        t = [param[0] for param in c_list]

        # Allocating variables
        qmax1 = [param[1] for param in c_list]
        qmax2 = [param[2] for param in c_list]
        h1 = [param[3] for param in c_list]
        h2 = [param[4] for param in c_list]
        b01 = [param[5] for param in c_list]
        b02 = [param[6] for param in c_list]

        params_dict = {'q1': qmax1,
                       'q2': qmax2,
                       'h1': h1,
                       'h2': h2,
                       'b01': b01,
                       'b02': b02
                       }

        # Checking mse squared of fits
        se = [mse(x[i], y[i], dsltd, c_list[i]) for i in range(len(x))]

        # Displaying results
        pd.set_option('display.max_columns', None)
        df_result = pd.DataFrame(list(zip(t, qmax1, qmax2, h1, h2, b01, b02, se)),
                                 columns=['Temp(K)', 'qmax1 (mmol/g)',
                                          'qmax2 (mmol/g)', 'h1 (J/mol)', 'h2 (J/mol)', 'b01 (1/bar)', 'b02 (1/bar)',
                                          'mse'])

        display(pd.DataFrame(df_result))

        print(bold + "===============================================================================")
        print(bold + "===============================================================================")

        return results_lst, df_result, params_dict

    # Extracting x and y from datasets for each component
    xy_dict = {}
    df_res_dict = {}
    results_dict = {}
    params_dict = {}
    i = 0
    for df in df_list:
        x = [df[key_pressures[j]].values for j in range(len(key_pressures))]
        y = [df[key_uptakes[j]].values for j in range(len(key_pressures))]
        xy_dict[compnames[i]] = x, y
        del x, y
        i += 1

    # Creating a list of dictionaries of guess values for each component
    if guess is None:
        guess = [get_guess_params("dsl", df_list[i], key_uptakes, key_pressures) for i in range(len(df_list))]

    # Creating a list of henry constants for each component
    henry_const_lst = [henry_approx(df_list[i], key_pressures, key_uptakes, show_hen, hentol, compnames[i])[0] for i in
                       range(len(df_list))]

    # Checking the most adsorbed component - calling function step 1 for all components and checking which gives
    # highest q1 + q2.
    i_high = 0
    qhigh = 0
    if dsl_comp_a is None:
        for i in range(len(compnames)):
            qtest = step1(*xy_dict[compnames[i]], meth, guess[i], henry_const_lst[i])
            qcheck = qtest[0] + qtest[1]
            if qcheck > qhigh:
                # When the calculated qsat(total) is the highest of all components, that qsat is saved along with the
                # index of the component with the highest qsat
                qhigh = qcheck
                i_high = i
        comp_msg = f'{compnames[i_high]} was found to have the highest qsat(total) This will be used as ' \
                   f'component A. '
    else:
        # If the most adsorbed component name is inputted manually,
        i_high = compnames.index(dsl_comp_a)
        qtest = step1(*xy_dict[compnames[i_high]], meth, guess[i_high], henry_const_lst[i_high])
        qhigh = qtest[0] + qtest[1]
        comp_msg = f'You have manually inputted {compnames[i_high]} as the most adsorbed component'

    if len(df_list) > 1:
        print(comp_msg)
        print(f'{compnames[i_high]} shows a qsat(total) of {str(round(qhigh, 1))} mmol/g')
        # Component A refers to the fitting procedure for the most adsorbed component - i.e this component is fit to
        # the DSL isotherm model, while the rest of the components are fit to the single site Langmuir isotherm model


    # Allocates the most adsorbed dataframe to be used in the procedure
    print(bold + "_________________________" + compnames[i_high] + " RESULTS_____________________________")
    print(' ')

    # Calling the inner functions corresponding to the three step fitting procedure
    step1_compA = step1(*xy_dict[compnames[i_high]], meth, guess[i_high], henry_const_lst[i_high])

    step2_compA = step2(*xy_dict[compnames[i_high]], meth, guess[i_high], henry_const_lst[i_high], temps, step1_compA)

    step3_compA = step3(*xy_dict[compnames[i_high]], meth, temps, step2_compA)

    # Allocating the resulting tuple from step 3 to dictionaries to be returned by the outer function
    results_dict[compnames[i_high]] = step3_compA[0]
    df_res_dict[compnames[i_high]] = step3_compA[1]
    params_dict[compnames[i_high]] = step3_compA[2]

    # Iterating the procedure for 'component B' for all other dataframes (except the one with index i_high)
    for i in range(len(df_list)):
        if len(df_list) > 1 and i != i_high:
            print(bold + "_________________________" + compnames[i] + " RESULTS_____________________________")
            print(' ')
            step2_compB = step2(*xy_dict[compnames[i]], meth, guess[i], henry_const_lst[i], temps,
                                step1_compA, True)

            step3_compB = step3(*xy_dict[compnames[i]], meth, temps, step2_compB, True)
            results_dict[compnames[i]] = step3_compB[0]
            df_res_dict[compnames[i]] = step3_compB[1]
            params_dict[compnames[i]] = step3_compB[2]

    return xy_dict, results_dict, df_res_dict, params_dict
