from lmfit import Model, Parameters
from pyIsofit.core.model_fit_def import bounds_check
from pyIsofit.core.model_dicts import _MODEL_FUNCTIONS


def langmuir_fit(model, x, y, guess, temps, cond, meth, cust_bounds, fit_report, henry_constants, henry_off):
    isotherm = _MODEL_FUNCTIONS[model]
    gmod = Model(isotherm)

    bounds = bounds_check('langmuir', cust_bounds, len(temps))

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
                pars.add('q', value=q_guess[0], min=bounds['q'][i][0], max=bounds['q'][i][1])
            else:
                pars.add('q', value=q_fix, min=q_fix, max=q_fix + 0.001)

            if henry_off:
                pars.add('b', value=b_guess[i], min=bounds['b'][i][0], max=bounds['b'][i][1])
            else:
                pars.add('delta', value=henry_constants[i], vary=False)
                pars.add('b', expr='delta/q')  # KH = b*q
        else:
            pars.add('q', value=q_guess[i], min=bounds['q'][i][0], max=bounds['q'][i][1])
            pars.add('b', value=b_guess[i], min=bounds['b'][i][0], max=bounds['b'][i][1])

        results = gmod.fit(y[i], pars, x=x[i], method=meth)
        params.append(results)
        values_dict[i] = results.values
        if fit_report:
            print(results.fit_report())
        if i == 0 and cond is True:
            q_fix = results.values['q']  # This only gets applied if cond=True

        del results, pars

    return params, values_dict


def langmuirTD_fit(model, x, y, guess, temps, cond, meth, cust_bounds, fit_report):
    isotherm = _MODEL_FUNCTIONS[model]
    gmod = Model(isotherm)

    bounds = bounds_check(cust_bounds, len(temps))

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
                pars.add('q', value=q_guess[0], min=bounds['q'][0], max=bounds['q'][1])
            else:
                pars.add('q', value=q_fix, min=q_fix, max=q_fix + 0.001)

        else:
            pars.add('q', value=q_guess[i], min=bounds['q'][0], max=bounds['q'][1])

        pars.add('h', value=h_guess[i], min=bounds['h'][0], max=bounds['h'][1])
        pars.add('b0', value=b0_guess[i], min=bounds['b0'][0], max=bounds['b0'][1])

        results = gmod.fit(y[i], pars, x=x[i], method=meth)
        params.append(results)
        values_dict[i] = results.values
        if fit_report:
            print(results.fit_report())
        if i == 0 and cond is True:
            q_fix = results.values['q']  # This only gets applied if cond=True

        del results, pars

    return params, values_dict
