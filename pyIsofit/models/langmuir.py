
from lmfit import Model, Parameters
from pyIsofit.core.utilityFunctions import get_model

def langmuir_fit(model, x, y, guess, temps, cond, meth, cust_bounds, henry_constants, henry_off):
    isotherm = get_model(model)
    gmod = Model(isotherm)

    if cond:
        print("Constraint 1: q sat = q_init for all temp")
        if not henry_off:
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

            if henry_off:
                pars.add('b', value=b_guess[i], min=0)
            else:
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

def langmuirTD_fit(model, x, y, guess, temps, cond, meth, cust_bounds):
    isotherm = get_model(model)
    gmod = Model(isotherm)

    q_guess = guess['q']
    h_guess = guess['h']
    b0_guess = guess['b0']

    values_dict = {}
    params = []

    for i in range(len(x)):
        pars = Parameters()
        pars.add('t', value=temps[i], vary=False)
        if cond:
            if i == 0:
                pars.add('q', value=q_guess[0], min=0)
            else:
                pars.add('q', value=q_fix, min=q_fix, max=q_fix + 0.001)

        else:
            pars.add('q', value=q_guess[i], min=0)

        pars.add('h', value=h_guess[i])
        pars.add('b0', value=b0_guess[i], min=0)

        results = gmod.fit(y[i], pars, x=x[i], method=meth)
        params.append(results)
        values_dict[i] = results.values
        if i == 0 and cond is True:
            q_fix = results.values['q']  # This only gets applied if cond=True

        del results, pars

    return params, values_dict