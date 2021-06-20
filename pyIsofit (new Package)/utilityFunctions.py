import numpy as np
from modelFunctions import *
import pandas as pd
from IPython.display import display

def henry_approx(df, keyPressures, keyUptakes, display_hen=False, tolerance=0.999, henry_only=False):
    #This section finds the henry region of the datasets
    x = []
    y = []
    for i in range(len(keyPressures)):
        xi = np.insert(np.array(df[keyPressures[i]].values), 0, 0)
        yi = np.insert(np.array(df[keyUptakes[i]].values), 0, 0)
        x.append(xi)
        y.append(yi)
        del xi
        del yi

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
        x_ = x[i]
        x_henry = [x_[0], x_[1], x_[2]] #Starting with a minimum of two datapoints
        counter = 3
        rsq_ilst = []
        hen_ilst = []
        #This loop adds data points while the points correspond to a henry fit with an R^2 of above 0.9995
        while rsq > 0 and counter < len(x_):
            x_henry.append(x_[counter])
            y_henry = dataset[:len(x_henry)]

            hen= y_henry[-1] / x_henry[-1]
            rsq = round(r2hen(x_henry, y_henry, henry, hen),5) #r squared calc.
            rsq_ilst.append(rsq)
            hen_ilst.append(hen)
            counter += 1
        rsq_lst.append(rsq_ilst)
        hen_lst.append(hen_ilst)
        #plt.figure()


        abtol = []
        itol = []
        i2 = 0
        for rsq in rsq_ilst:
            if rsq > tolerance:
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

        henry_len.append(henidx+1)
        #Saving Henry region parameters to later display
        henry_constants.append(hen_ilst[rsqidx])
        henry_limits.append(x_henry[henidx])
        henry_rsq.append(rsq_ilst[rsqidx])
        # sometimes data may not have a good henry region fit, which could abort the above while loop after the first
        # iteration. This piece of code warns the user of this

        if henidx+1 < 4:
            errHen.append(str(i+1))
        i += 1

    if henry_only == True:
        return henry_constants

    if errHen != []:
        print(unbold + 'WARNING: Henry region for dataset(s) ' + ', '.join(errHen) + ' were found to be made up of less than 4 points.')
        print('         This may affect accuracy of results.')
        print('         Henry region tolerance may be entered after log plot toggle parameter (default = 0.9999).')

    #Creating dataframe for henry constants
    df_henry = pd.DataFrame(list(zip(henry_constants, henry_limits, henry_len, henry_rsq)),
                            columns=['Henry constant (mmol/(bar.g))',
                                    'Upper limit (bar)','datapoints', 'R squared'])
    if display_hen == True:
        display(pd.DataFrame(df_henry))

    return henry_constants

def get_model(model):
    if model == "Langmuir":
        return langmuir1
    if model == "Langmuir linear 1":
        return langmuirlin1
    if model == "Langmuir linear 2":
        return langmuirlin2


def get_guess_params(model, df, key_uptakes, key_pressures):

    if model != "BDDT 2n" or model != "BDDT 2n-1" or model != "DoDo":
        henry_lim = henry_approx(df, key_pressures, key_uptakes, 0.999, True, True)
        saturation_loading = [1.1 * df[key_uptakes[i]].max() for i in range(len(key_pressures))]
        langmuir_b = [kh / qsat for (kh, qsat) in zip(henry_lim, saturation_loading)]

    if "Langmuir" in model and model != "Langmuir TD":
        return {
            "b": langmuir_b,
            "q" : saturation_loading
        }

    if model == "Langmuir TD":
        return {
            "b0": langmuir_b,
            "q" : saturation_loading,
            "h" : 5000
        }

    if model == "DSL":
        return {
            "q1": [0.5 * q for q in saturation_loading] ,
            "b1": [0.4 * b for b in langmuir_b],
            "q2": [0.5 * q for q in saturation_loading],
            "b2": [0.6 * b for b in langmuir_b]
        }

    if model == "GAB":
        return {
            "n": saturation_loading,
            "ka": langmuir_b,
            "ca": 0.01 * langmuir_b
        }
