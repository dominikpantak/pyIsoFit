import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from lmfit import Model, Parameters
from IPython.display import display
from modelFunctions import *
from utilityFunctions import *


class DSL_fit:
    def __init__(self, temps, guess, keyPressures, keyUptakes, compnames, df_list, logplot=False, tol=0.9999,
                 meth='leastsq', fix=None, yA=None):

        # These are required for both the inner class and outer class
        self.temps = temps
        self.meth = meth

        # initial guess values for dsl fitting
        self.guess = guess

        self.keyPressures = keyPressures
        self.keyUptakes = keyUptakes
        self.compname = compnames

        # A list of dataframes is inputted here, meaning that the user can fit to any number of components simultaneously
        # Using the DSL procedure
        self.df_list = df_list
        # self.individ_sheet = individ_sheet
        self.fix = fix
        self.yA = yA
        self.tol = tol

    def fit_isotherm(self, temps, guess, keyPressures, keyUptakes, compnames, df_list, logplot=False, tol=0.9999,
                     meth='leastsq', fix=None, yA=None):

        class DSL:
            def __init__(self, df, temps, meth, q1_init, q2_init, b1_init, b2_init, keyPressures, keyUptakes, compname,
                         logplot=False, tol=0.9999, step=None, step1=None, step3=None, comp2=False):

                self.df = df  # Dataframe with uptake vs. pressure data
                self.temps = temps  # Temp in deg C
                self.meth = meth  # Optional to choose mathematical fitting method in lmfit (default is leastsq)

                # initial guess values for dsl fitting
                self.q1_init = q1_init
                self.q2_init = q2_init
                self.b1_init = b1_init
                self.b2_init = b2_init

                self.keyPressures = keyPressures  # names of the column headers for pressure and uptakes
                self.keyUptakes = keyUptakes
                self.compname = compname

                self.step = step

                self.step1 = step1
                self.step3 = step3

                self.comp2 = comp2

            def dslfit(self, df, temps, meth, q1_init, q2_init, b1_init, b2_init, keyPressures, keyUptakes, compname,
                       logplot=False, tol=0.9999, step=None, step1=None, step3=None, comp2=None):

                ##### setting up variables for fitting ####
                ###### allocating columns to lists ##### 

                # Importing data and allocating to variables
                y = []
                x = []
                for i in range(len(keyPressures)):
                    xi = np.insert(np.array(df[keyPressures[i]].values), 0, 0)
                    yi = np.insert(np.array(df[keyUptakes[i]].values), 0, 0)
                    x.append(xi)
                    y.append(yi)
                    del xi
                    del yi

                ########################################################

                # This section finds the henry region of the datasets
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
                    pres = x[i]
                    x_henry = [pres[0], pres[1], pres[2]]  # Starting with a minimum of two datapoints
                    counter = 3
                    rsq_ilst = []
                    hen_ilst = []
                    # This loop adds data points while the points correspond to a henry fit with an R^2 of above 0.9995
                    while rsq > 0 and counter < len(pres):
                        x_henry.append(pres[counter])
                        y_henry = dataset[:len(x_henry)]

                        hen = y_henry[-1] / x_henry[-1]
                        rsq = round(r2hen(x_henry, y_henry, henry, hen), 5)  # r squared calc.
                        rsq_ilst.append(rsq)
                        hen_ilst.append(hen)
                        counter += 1
                    rsq_lst.append(rsq_ilst)
                    hen_lst.append(hen_ilst)
                    # plt.figure()

                    abtol = []
                    itol = []
                    i2 = 0
                    for rsq in rsq_ilst:
                        if rsq > tol:
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

                    henry_len.append(henidx + 1)
                    # Saving Henry region parameters to later display
                    henry_constants.append(hen_ilst[rsqidx])
                    henry_limits.append(x_henry[henidx])
                    henry_rsq.append(rsq_ilst[rsqidx])
                    # sometimes data may not have a good henry region fit, which could abort the above while loop after the first
                    # iteration. This piece of code warns the user of this
                    if henidx + 1 < 4:
                        errHen.append(str(i + 1))
                    i += 1
                if step == 3 and errHen != []:
                    print(unbold + 'WARNING: Henry region for dataset(s) ' + ', '.join(
                        errHen) + ' were found to be made up of less than 4 points.')
                    print('         This may affect accuracy of results.')
                    print(
                        '         Henry region tolerance may be entered after log plot toggle parameter (default = 0.9999).')

                    # Here is the step procedure mentioned above
                # The outer class controls which step is being carried out
                # The first step is to find the initial q1, q2, b1, b2 values with the henry constraint set
                if step == 1:
                    gmod = Model(dsl)
                    pars1 = Parameters()
                    pars1.add('q1', value=q1_init, min=0)
                    pars1.add('q2', value=q2_init, min=0)
                    pars1.add('b2', value=b2_init)
                    pars1.add('delta', value=henry_constants[0], vary=False)
                    pars1.add('b1', expr='(delta-q2*b2)/q1')  # KH = b1*q1 + b2*q2

                    result1 = gmod.fit(y[0], pars1, x=x[0], method=meth)

                    c1 = [result1.values['q1'], result1.values['q2'], result1.values['b1'], result1.values['b2']]
                    return c1[0], c1[1]

                    # This ends the function within the inner class and returns the qmax values to
                    # be used in step 2
                if step == 2:
                    # This step calculates b1, b2 parameters for the remaining datasets with constraints:
                    # qmax fixed for all fittings from previous steps
                    # henry region constraint

                    q1fix = step1[0]
                    q2fix = step1[1]
                    ############################### FITTING ##########################################
                    if comp2 == True:
                        # Depending on whether the package is fitting for the most adsorbed componennt or the
                        # remaining components, the fitting procedure is different. In the methodology used,
                        # The most adsorbed component is fitted to DSL but the remaining components are
                        # fitted to essentially the single langmuir isotherm
                        gmod = Model(langmuir1)
                        qtot = q1fix + q2fix

                        c = []
                        for i in range(len(keyPressures)):
                            pars = Parameters()
                            pars.add('q', value=qtot, min=qtot, max=qtot + 0.001)
                            pars.add('delta', value=henry_constants[i], vary=False)
                            pars.add('b', expr='delta/q')

                            results = gmod.fit(y[i], pars, x=x[i], method=meth)
                            cee = [q1fix, q2fix, results.values['b'], results.values['b']]
                            c.append(cee)

                            del results
                            del pars

                    else:
                        gmod = Model(dsl)
                        c = []
                        for i in range(len(keyPressures)):
                            pars = Parameters()
                            pars.add('q1', value=q1fix, min=q1fix, max=q1fix + 0.001)
                            pars.add('q2', value=q2fix, min=q2fix, max=q2fix + 0.001)
                            pars.add('b2', value=b2_init, min=0)
                            pars.add('delta', value=henry_constants[i], vary=False)
                            pars.add('b1', expr='(delta-q2*b2)/q1', min=0)  # KH = b*q

                            results = gmod.fit(y[i], pars, x=x[i], method=meth)
                            cee = [results.values['q1'], results.values['q2'],
                                   results.values['b1'], results.values['b2']]
                            c.append(cee)

                            del results
                            del pars

                    # allocating variables
                    qmax1 = [param[0] for param in c]
                    qmax2 = [param[1] for param in c]
                    b1 = [param[2] for param in c]
                    b2 = [param[3] for param in c]
                    qtot = [param[0] + param[1] for param in c]

                    # Finding heat of adsorption for both sites
                    T = np.array([1 / (temp + 273) for temp in self.temps])
                    ln_b1 = np.array([np.log(i) for i in b1])
                    ln_b2 = np.array([np.log(i) for i in b2])
                    mH1, bH1, rH1, pH1, sterrH1 = stats.linregress(T, ln_b1)
                    mH2, bH2, rH2, pH2, sterrH2 = stats.linregress(T, ln_b2)

                    h = [-0.001 * mH1 * r, -0.001 * mH2 * r]
                    b0 = [np.exp(bH1), np.exp(bH2)]

                    # The package returns these sets of parameters to be used as initial guesses for the final step
                    return qmax1[0], qmax2[0], h, b0
                if step == 3:

                    # Creating dataframe for henry constants
                    # We can now display henry region results
                    print(bold + "\nHenry regions found...")

                    df_henry = pd.DataFrame(list(zip(self.temps, henry_constants, henry_limits, henry_len, henry_rsq)),
                                            columns=['Temperature (oC)', 'Henry constant (mmol/(bar.g))',
                                                     'Upper limit (bar)', 'Datapoints', 'R squared'])
                    display(pd.DataFrame(df_henry))

                    tempsK = [t + 273 for t in temps]

                    # unpacking tuples
                    q1_in = step3[0]
                    q2_in = step3[1]
                    h_in = step3[2]
                    b0_in = step3[3]

                    # We can now fit to the van't Hoff form of DSL with the initial guess values
                    # obtained through step 2

                    if comp2 == True:
                        # Again, we fit to the single langmuir form when the least adsorbed species is used   
                        gmod = Model(langmuirTD)
                        qtot = q1_in + q2_in
                        c = []
                        result = []
                        for i in range(len(keyPressures)):
                            pars = Parameters()
                            pars.add('t', value=tempsK[i], vary=False)
                            pars.add('q', value=qtot, min=qtot, max=qtot + 0.001)
                            pars.add('h', value=h_in[0] * 1000)
                            pars.add('b0', value=b0_in[0], min=0)

                            results = gmod.fit(y[i], pars, x=x[i], method=meth)
                            result.append(results)

                            c.append([results.values['t'], q1_in, q2_in, results.values['h'],
                                      results.values['h'], results.values['b0'], results.values['b0']])

                            del results
                            del pars


                    else:
                        gmod = Model(dslTD)  # DSL
                        c = []
                        result = []
                        for i in range(len(keyPressures)):
                            pars = Parameters()
                            pars.add('t', value=tempsK[i], vary=False)
                            pars.add('q1', value=q1_in, min=q1_in, max=q1_in + 0.001)
                            pars.add('q2', value=q2_in, min=q2_in, max=q2_in + 0.001)
                            pars.add('h1', value=h_in[0] * 1000)
                            pars.add('h2', value=h_in[1] * 1000)
                            pars.add('b01', value=b0_in[0], min=0)
                            pars.add('b02', value=b0_in[1], min=0)

                            results = gmod.fit(y[i], pars, x=x[i], method=meth)
                            result.append(results)
                            c.append(
                                [results.values['t'], results.values['q1'], results.values['q2'], results.values['h1'],
                                 results.values['h2'], results.values['b01'], results.values['b02']])

                            del results
                            del pars

                    print(bold + "\nParameters found...")

                    # allocating variables, formatting and creating dataframe

                    t = [param[0] for param in c]

                    # UNFORMATTED VARIABLES
                    q1 = [param[1] for param in c]
                    q2 = [param[2] for param in c]
                    h_1 = [param[3] for param in c]
                    h_2 = [param[4] for param in c]
                    b_01 = [param[5] for param in c]
                    b_02 = [param[6] for param in c]

                    # FORMATTED VARIABLES
                    qmax1 = [np.round(param[1], 3) for param in c]
                    qmax2 = [np.round(param[2], 3) for param in c]
                    h1 = [np.round(param[3], 3) for param in c]
                    h2 = [np.round(param[4], 3) for param in c]
                    b01 = ["{:.3e}".format(param[5]) for param in c]
                    b02 = ["{:.3e}".format(param[6]) for param in c]

                    # Checking r squared of fits
                    r_sq = [r2(x[i], y[i], dsl_h, c[i]) for i in range(len(keyPressures))]
                    se = [mse(x[i], y[i], dsl_h, c[i]) for i in range(len(keyPressures))]

                    # Displaying results
                    df_result = pd.DataFrame(list(zip(t, qmax1, qmax2, h1, h2, b01, b02, r_sq, se)),
                                             columns=['Temp(K)', 'qmax1 (mmol/g)',
                                                      'qmax2 (mmol/g)', 'h1 (J/mol)', 'h2 (J/mol)', 'b01 (1/bar)',
                                                      'b02 (1/bar)', 'R sq', 'mse'])

                    display(pd.DataFrame(df_result))

                    print(bold + "===============================================================================")
                    print(bold + "===============================================================================")

                    xaxis = 'pressure [bar]'
                    yaxis = 'uptake mmol/g'

                    ##### Plotting results #####
                    plt.figure(figsize=(8, 6))
                    plt.title(compname)
                    if logplot == True:
                        plt.xscale("log")
                        plt.yscale("log")
                    plt.xlabel(xaxis)
                    plt.ylabel(yaxis)
                    plt.tick_params(**tick_style)

                    for i in range(len(keyPressures)):
                        plt.plot(x[i], result[i].best_fit, '-', color=colours[i],
                                 label="{temps} °C Fit".format(temps=temps[i]))
                        plt.plot(x[i], y[i], 'ko', color='0.75',
                                 label="Data at {temps} °C".format(temps=temps[i]))

                    plt.grid()
                    plt.legend()
                    plt.show()

                    # These parameters will be fed into the procedure for the remaining components
                    if comp2 == True:
                        return [h_1[0], h_2[0], b_01[0], b_02[0]]
                    else:
                        return [q1[0], q2[0], h_1[0], h_2[0], b_01[0], b_02[0]]

        # ################ OUTER CLASS... USED FOR GOING THROUGH FARHAMINI PROCEDURE ###############
        # importing dataframes
        # if individ_sheet == True:
        # for df in df_list:

        keys = keyPressures, keyUptakes

        # The code below loops through the imported dataframes and checks which one is the most adsorbed component
        # When this is finished, it displays to the user which component is the most adsorbed
        try:
            i = 0
            i_high = 0
            qhigh = 0
            for df in df_list:
                test = DSL(df, temps, meth, *guess, *keys, compnames[i], logplot, tol, 1)
                qtest = test.dslfit(df, temps, meth, *guess, *keys, compnames[i], logplot, tol, 1)
                qcheck = qtest[0] + qtest[1]
                if qcheck > qhigh:
                    qhigh += qcheck
                    i_high += i
                i += 1
        except IndexError:  # This was added because its easy to forget to include the corresponding component names
            print("ERROR: Make sure all component names/ dataframes are listed")
            print(".\n.\n.\n.")

        print(compnames[i_high] + " shows the highest approx. qsat(total) of " + str(round(qhigh, 1)) + " mmol/g")
        print("This will be used as component A")

        # Allocates the most adsorbed dataframe to be used in the procedure
        compname_a = compnames[i_high]
        df_a = df_list[i_high]

        if fix == False:
            s4 = DSL(df_a, temps, meth, *guess, *keys, compname_a, logplot, tol, 4)
            s4.dslfit(df_a, temps, meth, *guess, *keys, compname_a, logplot, tol, 4)

        if not fix == False:
            # Going through steps 1 to 3 for the most adsorbed component. The results are
            # then used for the remaining components
            print(bold + "_________________________" + compname_a + " RESULTS_____________________________")
            print(' ')
            dsl_s1_co2 = DSL(df_a, temps, meth, *guess, *keys, compname_a, logplot, tol, 1)
            step1_co2 = dsl_s1_co2.dslfit(df_a, temps, meth, *guess, *keys, compname_a, logplot, tol, 1)

            dsl_s2_co2 = DSL(df_a, temps, meth, *guess, *keys, compname_a, logplot, tol, 2, step1_co2)
            step2_co2 = dsl_s2_co2.dslfit(df_a, temps, meth, *guess, *keys, compname_a, logplot, tol, 2, step1_co2)

            dsl_s3_co2 = DSL(df_a, temps, meth, *guess, *keys, compname_a, logplot, tol, 3, step1_co2, step2_co2)
            step3_co2 = dsl_s3_co2.dslfit(df_a, temps, meth, *guess, *keys, compname_a, logplot, tol, 3, step1_co2,
                                          step2_co2)
            i = 0
            # This only runs if there is more than one component in the system
            if len(df_list) > 1:
                i = 0
                for df in df_list:
                    # This condition makes sure that the procedure for the remaining components isn't repeated 
                    # for the most adsorbed component
                    if not i == i_high:
                        # Procedure for the remaining components
                        print(
                            bold + "_________________________" + compnames[i] + " RESULTS_____________________________")
                        print(' ')
                        dsl_step2_n2 = DSL(df, temps, meth, *guess, *keys, compnames[i], logplot, tol, 2, step1_co2,
                                           step2_co2, True)
                        step2_n2 = dsl_step2_n2.dslfit(df, temps, meth, *guess, *keys, compnames[i], logplot, tol, 2,
                                                       step1_co2, step2_co2, True)

                        dsl_step3_n2 = DSL(df, temps, meth, *guess, *keys, compnames[i], logplot, 3, step1_co2,
                                           step2_n2, True)
                        step3_n2 = dsl_step3_n2.dslfit(df, temps, meth, *guess, *keys, compnames[i], logplot, tol, 3,
                                                       step1_co2, step2_n2, True)
                        i += 1
                    else:
                        i += 1

        if len(df_list) == 2:
            i = 0
            for df in df_list:
                if not i == i_high:
                    ib = i
                i += 1

        if fix == 'extend':
            binary_components = step3_co2 + step3_n2
            print(binary_components)
            xA = []
            xB = []
            yB = 1 - yA
            df1 = df_a
            df2 = df_list[ib]
            for i in range(len(keyPressures)):
                xA.append(df1[keyPressures[i]].values)
                xB.append(df2[keyPressures[i]].values)

            tempsK = [t + 273 for t in temps]
            q_sA = []
            q_sB = []
            for i in range(len(keyPressures)):
                q_sA.append(ext_dslA(xA[i], tempsK[i], *binary_components, yA))
                q_sB.append(ext_dslB(xB[i], tempsK[i], *binary_components, yB))

            xaxis = 'pressure [bar]'
            yaxis = 'Absolute adsorption mmol/g'

            ##### Plotting results ####

            plt.figure(figsize=(8, 6))
            for i in range(len(keyPressures)):
                plt.plot(xA[i], q_sA[i], '--', color=colours[i],
                         label="{temps} °C Fit".format(temps=temps[i]))

            plt.title("Extended DSL model showing absolute adsorption of A at yA (frac): " + str(yA))
            plt.tick_params(**tick_style)
            if logplot == True:
                plt.xscale("log")
                # plt.yscale("log")
            plt.xlabel(xaxis)
            plt.ylabel(yaxis)
            plt.grid()
            plt.legend()

            plt.figure(figsize=(8, 6))
            for i in range(len(keyPressures)):
                plt.plot(xB[i], q_sB[i], '--', color=colours[i],
                         label="{temps} °C Fit".format(temps=temps[i]))

            plt.title("Extended DSL model showing absolute adsorption of B at yB (frac): " + str(np.round(yB, 2)))
            plt.tick_params(**tick_style)
            if logplot == True:
                plt.xscale("log")
                # plt.yscale("log")
            plt.xlabel(xaxis)
            plt.ylabel(yaxis)
            plt.grid()
            plt.legend()

            plt.show()

            return step3_co2 + step3_n2
