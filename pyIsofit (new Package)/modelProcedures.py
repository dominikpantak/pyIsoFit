from re import A
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from lmfit import Model, Parameters
from IPython.display import display
from modelEquations import *
from utilityFunctions import *


def dsl_fit(df_list, keyPressures, keyUptakes, temps=None, compnames=None,
            meth='tnc', guess=None, hentol=0.9999, show_hen=False):
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
                pars.add('b2', value=guess['b2'][i], min=0)
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

        print(bold + "\nParameters found..." + unbold)

        # allocating variables, formatting and creating dataframe

        t = [param[0] for param in c_list]

        # UNFORMATTED VARIABLES
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

        return results_lst, df_result, params_dict

    df_dict = {}
    df_res_dict = {}
    results_dict = {}
    params_dict = {}
    i = 0
    for df in df_list:
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
    henry_const_lst = [henry_approx(df_list[i], keyPressures, keyUptakes, True, hentol, compnames[i])[0] for i in
                       range(len(df_list))]

    for i in range(len(compnames)):
        qtest = step1(*df_dict[compnames[i]], meth, guess[i], henry_const_lst[i])
        qcheck = qtest[0] + qtest[1]
        if qcheck > qhigh:
            qhigh += qcheck
            i_high += i

    print(compnames[i_high] + " shows the highest approx. qsat(total) of " + str(round(qhigh, 1)) + " mmol/g")
    print("This will be used as component A")

    # Allocates the most adsorbed dataframe to be used in the procedure

    print(bold + "_________________________" + compnames[i_high] + " RESULTS_____________________________")
    print(' ')

    step1_compA = step1(*df_dict[compnames[i_high]], meth, guess[i_high], henry_const_lst[i_high])

    step2_compA = step2(*df_dict[compnames[i_high]], meth, guess[i_high], henry_const_lst[i_high], temps, step1_compA)

    step3_compA = step3(*df_dict[compnames[i_high]], meth, guess[i_high], henry_const_lst[i_high], temps, step2_compA)
    results_dict[compnames[i_high]] = step3_compA[0]
    df_res_dict[compnames[i_high]] = step3_compA[1]
    params_dict[compnames[i_high]] = step3_compA[2]

    for i in range(len(df_list)):
        if len(df_list) > 1 and i != i_high:
            print(bold + "_________________________" + compnames[i] + " RESULTS_____________________________")
            print(' ')
            step2_compB = step2(*df_dict[compnames[i]], meth, guess[i], henry_const_lst[i], temps,
                                step1_compA, True)

            step3_compB = step3(*df_dict[compnames[i]], meth, guess[i], henry_const_lst[i], temps, step2_compB, True)
            results_dict[compnames[i]] = step3_compB[0]
            df_res_dict[compnames[i]] = step3_compB[1]
            params_dict[compnames[i]] = step3_compB[2]

    return df_dict, results_dict, df_res_dict, params_dict


def langmuir_fit(gmod, x, y, guess, cond, henry_constants, meth):
    if cond:
        print("Constraint 1: q sat = q_init for all temp")
        print("Constraint 2: qsat*b = Henry constant for all temp")

    q_guess = guess['q']
    b_guess = guess['b']
    params = []
    values_dict = {}

    for i in range(len(x)):
        pars = Parameters()

        # Creating intermediate parameter delta that will fix KH = b*q

        if cond:
            if i == 0:
                pars.add('q', value=q_guess[0], min=0)
            else:
                pars.add('q', value=q_fix, min=q_fix, max=q_fix + 0.001)

            pars.add('delta', value=henry_constants[i], vary=False)
            pars.add('b', expr='delta/q')  # KH = b*q
        else:
            pars.add('q', value=q_guess[i], min=0)
            pars.add('b', value=b_guess[i], min=0)

        results = gmod.fit(y[i], pars, x=x[i], method=meth)
        params.append(results)
        values_dict[i] = results.values
        if i == 0 and cond is True:
            q_fix = results.values['q']  # This only gets applied if cond=True

        del results, pars

    return params, values_dict


def langmuirTD_fit(gmod, x, y, guess, temps, cond, meth):
    q_guess = guess['q']
    h_guess = guess['h']
    b0_guess = guess['b0']

    values_dict = {}
    params = []

    tempsK = [t + 273 for t in temps]

    for i in range(len(x)):
        pars = Parameters()
        pars.add('t', value=tempsK[i], vary=False)
        if cond:
            if i == 0:
                pars.add('q', value=q_guess[0], min=0)
            else:
                pars.add('q', value=q_fix, min=q_fix, max=q_fix + 0.001)

        else:
            pars.add('q', value=q_guess[i], min=0)

        pars.add('h', value=h_guess[i] * 1000)
        pars.add('b0', value=b0_guess[i], min=0)

        results = gmod.fit(y[i], pars, x=x[i], method=meth)
        params.append(results)
        values_dict[i] = results.values
        if i == 0 and cond is True:
            q_fix = results.values['q']  # This only gets applied if cond=True

        del results, pars

    return params, values_dict


def heat_calc(model, temps, param_dict, x):
    site = ['A', 'B', 'C']
    t = np.array([1 / (temp + 273) for temp in temps])

    def isosteric_heat(t, yparams, j=0):
        ln_b = np.array([np.log(i) for i in yparams])
        mh, bh, rh, ph, sterrh = stats.linregress(t, ln_b)
        h = -0.001 * mh * r

        print("_______________________________________________________________________")
        print("Heat of adsorption for site " + site[j] + ":" + str(round(h, 2)) + " kJ/mol. \n" +
              "R sq of Van't Hoff: " + str(round(rh, 4)))

    if model.lower() == "gab":
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

    elif "langmuir" in model.lower() and model.lower() != "langmuir td":
        yparams = param_dict['b (1/bar)']
        isosteric_heat(t, yparams)

    elif model.lower() == "dsl nc":
        yparams = [param_dict['b1 (1/bar)'], param_dict['b2 (1/bar)']]
        for i in range(2):
            isosteric_heat(t, yparams[i], i)
    else:
        return None


def ext_dsl(param_dict, temps, x, comps, yfracs):
    if len(comps) < 2:
        print("Enter 2 components or more to use extended models")
        return None

    def vant_hoff(b0, h, t):
        return b0 * np.exp(-h / (r * (t + 273) ) )

    def calc_q(x, num_a, num_b, den_a, den_b):
        e1 = num_a * x / (1 + den_a * x )
        e2 = num_b * x / (1 + den_b * x )
        return e1 + e2

    params_sorted = {}
    """ First we sort the input data and calculate b1 and b2 from the heats of 
    adsorption and temperatures"""
    for i in range(len(comps)):
        params = param_dict[comps[i]]
        x_comp = x[i]
        b1_list_in = []
        b2_list_in = []

        for j in range(len(x_comp)):
            b1_list_in.append(vant_hoff(params['b01'][j], params['h1'][j], temps[j]))
            b2_list_in.append(vant_hoff(params['b02'][j], params['h2'][j], temps[j]))
        params_sorted[comps[i]] = {'q1': params['q1'],
                                   'q2': params['q2'],
                                   'b1': b1_list_in,
                                   'b2': b2_list_in,
                                   'y': yfracs[i]}

    """Input data has now been sorted and now the numerators of the two terms in the extended dsl equation 
    can be worked out. The aim here is to simplify the extended dsl equation into the following form:
    
    q* = num(siteA) * x / (1 + den(siteA) * x ) + num(siteB) * x / (1 + den(siteB) * x )
    
    """

    num_dict = {}
    for i in range(len(comps)):
        params = params_sorted[comps[i]]
        num_a_in = []
        num_b_in = []
        for j in range(len(temps)):
            num_a_in.append(params['q1'][j] * params['b1'][j] * params['y'])
            num_b_in.append(params['q2'][j] * params['b2'][j] * params['y'])

        num_dict[comps[i]] = {'a': num_a_in, 'b': num_b_in}

    """ The above sorts the numerator of site A and B into a dictionary for its corresponding
    component.
    
    The next part is calculating the denominators - this is a bit more tricky as we need to add
    more terms depending on the number of components inputted. The workaround for this was to
    create a list of the same length as there are the number of temperatures. Within this list
    there are inner lists of the same length as the number of components. This way, we are allocating
    parameters in the following way for example:
    
    b1 (at 20 C) = 4.2 for component A, 2.4 for component B ... x for component X 
    
    we also do this for b2
    """

    b1_list = [[] for _ in range(len(temps))]
    b2_list = [[] for _ in range(len(temps))]
    for i in range(len(temps)):
        for j in range(len(comps)):
            params = params_sorted[comps[j]]
            b1_list[i].append(params['b1'][i])
            b2_list[i].append(params['b2'][i])

    den_a_list = []
    den_b_list = []
    for i in range(len(temps)):
        b1_in = b1_list[i]
        b2_in = b2_list[i]
        den_a = 0
        den_b = 0
        for j in range(len(b1_in)):
            den_a += b1_in[j] * yfracs[j]
            den_b += b2_in[j] * yfracs[j]
        den_a_list.append(den_a)
        den_b_list.append(den_b)

    """Above we are iterating over the list of lists mentioned previously with the goal of getting:
    (b1A * yA) + (b1B * yB) ... (b1X * yX)
    
    relating to the denominators of the extended dsl equation:
    
    1 + (b1A * yA * P) + (b1B * yB * P) = 1 + (b1A * yA + b1B * yB) * P
    
    """

    q_dict = {}
    for i in range(len(comps)):
        x_comp = x[i]
        numer = num_dict[comps[i]]
        num_a = numer['a']
        num_b = numer['b']
        q_list = []
        for j in range(len(temps)):
            q = calc_q(np.array(x_comp[j]), num_a[j], num_b[j], den_a_list[j], den_b_list[j])
            q_list.append(q)

        q_dict[comps[i]] = q_list

    return q_dict