from re import A
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from lmfit import Model, Parameters
from IPython.display import display
from modelFunctions import *
from utilityFunctions import *


def dsl_fit(df_lst, keyPressures, keyUptakes, temps=None, compnames=None,
              meth='tnc', guess=None, hentol=0.9999):
    # Here is the step procedure mentioned above
    # The outer class controls which step is being carried out
    # The first step is to find the initial q1, q2, b1, b2 values with the henry constraint se
    def step1(x, y, meth, guess, henry_constants):

        gmod = Model(dsl)
        pars1 = Parameters()

        pars1.add('q1', value=guess['q1'][0], min=0)
        pars1.add('q2', value=guess['q2'][0], min=0)
        pars1.add('b2', value=guess['b2'][0])
        pars1.add('delta', value=henry_constants[0], vary=False)
        pars1.add('b1', expr='(delta-q2*b2)/q1')  # KH = b1*q1 + b2*q2

        result1 = gmod.fit(y[0], pars1, x=x[0], method=meth)

        c1 = [result1.values['q1'], result1.values['q2'], result1.values['b1'], result1.values['b2']]
        return c1[0], c1[1]

        # This ends the function within the inner class and returns the qmax values to
        # be used in step 2

    def step2(x, y, meth, guess, henry_constants, temps, step1_pars, comp2=False):
        q1fix = step1_pars[0]
        q2fix = step1_pars[1]

        if comp2 == True:
            # Depending on whether the package is fitting for the most adsorbed componennt or the
            # remaining components, the fitting procedure is different. In the methodology used,
            # The most adsorbed component is fitted to DSL but the remaining components are
            # fitted to essentially the single langmuir isotherm
            gmod = Model(langmuir1)
            qtot = q1fix + q2fix

            c_list = []
            for i in range(len(x)):
                pars = Parameters()
                pars.add('q', value=qtot, min=qtot, max=qtot + 0.001)
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
                pars.add('b2', value=guess['b2'][0], min=0)
                pars.add('delta', value=henry_constants[i], vary=False)
                pars.add('b1', expr='(delta-q2*b2)/q1', min=0)  # KH = b*q

                results = gmod.fit(y[i], pars, x=x[i], method=meth)
                c = [results.values['q1'], results.values['q2'],
                       results.values['b1'], results.values['b2']]
                c_list.append(c)

                del results
                del pars

        # allocating variables
        qmax1 = [param[0] for param in c_list]
        qmax2 = [param[1] for param in c_list]
        b1 = [param[2] for param in c_list]
        b2 = [param[3] for param in c_list]

        # Finding heat of adsorption for both sites
        T = np.array([1 / (temp + 273) for temp in temps])
        ln_b1 = np.array([np.log(i) for i in b1])
        ln_b2 = np.array([np.log(i) for i in b2])
        mH1, bH1, rH1, pH1, sterrH1 = stats.linregress(T, ln_b1)
        mH2, bH2, rH2, pH2, sterrH2 = stats.linregress(T, ln_b2)

        h = [-0.001 * mH1 * r, -0.001 * mH2 * r]
        b0 = [np.exp(bH1), np.exp(bH2)]

        return qmax1[0], qmax2[0], h, b0
        # The package returns these sets of parameters to be used as initial guesses for the final step

    def step3(x, y, meth, guess, henry_constants, temps, step2_pars, comp2=False):
        tempsK = [t + 273 for t in temps]

        # unpacking tuples
        q1_in = step2_pars[0]
        q2_in = step2_pars[1]
        h_in = step2_pars[2]
        b0_in = step2_pars[3]

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
                pars.add('t', value=tempsK[i], vary=False)
                pars.add('q', value=qtot, min=qtot, max=qtot + 0.001)
                pars.add('h', value=h_in[0] * 1000)
                pars.add('b0', value=b0_in[0], min=0)

                results = gmod.fit(y[i], pars, x=x[i], method=meth)
                results_lst.append(results)

                c_list.append([results.values['t'], q1_in, q2_in, results.values['h'],
                          results.values['h'], results.values['b0'], results.values['b0']])

                del results
                del pars

        else:
            gmod = Model(dslTD, nan_policy='omit')  # DSL
            c_list = []
            results_lst = []
            for i in range(len(x)):
                pars = Parameters()
                pars.add('t', value=tempsK[i], vary=False)
                pars.add('q1', value=q1_in, min=q1_in, max=q1_in + 0.001)
                pars.add('q2', value=q2_in, min=q2_in, max=q2_in + 0.001)
                pars.add('h1', value=h_in[0] * 1000)
                pars.add('h2', value=h_in[1] * 1000)
                pars.add('b01', value=b0_in[0], min=0)
                pars.add('b02', value=b0_in[1], min=0)

                results = gmod.fit(y[i], pars, x=x[i], method=meth)
                results_lst.append(results)
                c_list.append([results.values['t'], results.values['q1'], results.values['q2'], results.values['h1'],
                          results.values['h2'], results.values['b01'], results.values['b02']])

                del results
                del pars

        print(bold + "\nParameters found...")

        # allocating variables, formatting and creating dataframe

        t = [param[0] for param in c_list]

        # UNFORMATTED VARIABLES
        q1 = [param[1] for param in c_list]
        q2 = [param[2] for param in c_list]
        h_1 = [param[3] for param in c_list]
        h_2 = [param[4] for param in c_list]
        b_01 = [param[5] for param in c_list]
        b_02 = [param[6] for param in c_list]

        # FORMATTED VARIABLES
        qmax1 = [np.round(param[1], 3) for param in c_list]
        qmax2 = [np.round(param[2], 3) for param in c_list]
        h1 = [np.round(param[3], 3) for param in c_list]
        h2 = [np.round(param[4], 3) for param in c_list]
        b01 = ["{:.3e}".format(param[5]) for param in c_list]
        b02 = ["{:.3e}".format(param[6]) for param in c_list]

        # Checking r squared of fits
        r_sq = [r2(x[i], y[i], dslTD, c_list[i]) for i in range(len(x))]
        se = [mse(x[i], y[i], dslTD, c_list[i]) for i in range(len(x))]

        # Displaying results
        pd.set_option('display.max_columns', None)
        df_result = pd.DataFrame(list(zip(t, qmax1, qmax2, h1, h2, b01, b02, r_sq, se)),
                                 columns=['Temp(K)', 'qmax1 (mmol/g)',
                                          'qmax2 (mmol/g)', 'h1 (J/mol)', 'h2 (J/mol)', 'b01 (1/bar)', 'b02 (1/bar)',
                                          'R sq', 'mse'])

        display(pd.DataFrame(df_result))

        print(bold + "===============================================================================")
        print(bold + "===============================================================================")

        return results_lst

    df_dict = {}
    results_dict = {}
    i = 0
    for df in df_lst:
        x = [df[keyPressures[j]].values for j in range(len(keyPressures))]
        y = [df[keyUptakes[j]].values for j in range(len(keyPressures))]
        df_dict[compnames[i]] = x, y
        del x, y
        i += 1

    i = 0
    i_high = 0
    qhigh = 0
    if guess is None:
        guess = [get_guess_params("DSL", df_list[i], keyUptakes, keyPressures) for i in range(len(df_list))]
    henry_const_lst = [henry_approx(df_list[i], keyPressures, keyUptakes, True, hentol, True) for i in range(len(df_list))]

    for i in range(len(compnames)):
        qtest = step1(*df_dict[compnames[i]], meth, guess[i], henry_const_lst[i])
        qcheck = qtest[0] + qtest[1]
        if qcheck > qhigh:
            qhigh += qcheck
            i_high += i

    print(compnames[i_high] + " shows the highest approx. qsat(total) of " + str(round(qhigh,1)) + " mmol/g")
    print("This will be used as component A")

    # Allocates the most adsorbed dataframe to be used in the procedure

    print(bold + "_________________________" + compnames[i_high] + " RESULTS_____________________________")
    print(' ')

    step1_compA = step1(*df_dict[compnames[i_high]], meth, guess[i_high], henry_const_lst[i_high])

    step2_compA = step2(*df_dict[compnames[i_high]], meth, guess[i_high], henry_const_lst[i_high], temps, step1_compA)

    step3_compA = step3(*df_dict[compnames[i_high]], meth, guess[i_high], henry_const_lst[i_high], temps, step2_compA)
    results_dict[compnames[i_high]] = step3_compA


    for i in range(len(df_list)):
        if len(df_list) > 1 and i != i_high:
            print(bold + "_________________________" + compnames[i] + " RESULTS_____________________________")
            print(' ')
            step2_compB = step2(*df_dict[compnames[i]], meth, guess[i], henry_const_lst[i], temps,
                                step1_compA, True)

            step3_compB = step3(*df_dict[compnames[i], meth, guess[i], henry_const_lst[i], temps, step2_compB, True)
            results_dict[compnames[i]] = step3_compB
    return df_dict, results_dict