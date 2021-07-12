from lmfit import Model, Parameters
from pyIsofit.core.utilityFunctions import get_model

def generic_fit(model, x, y, guess, temps, cond, meth, cust_bounds):
    isotherm = get_model(model)
    gmod = Model(isotherm)

    params = []
    values_dict = {}
    for i in range(len(x)):
        pars = Parameters()
        model_params = get_fit_tuples(model, guess, temps, i, cond, cust_bounds)
        pars.add_many(*model_params)
        results = gmod.fit(y[i], pars, x=x[i], method=meth)

        params.append(results)
        values_dict[i] = results.values
        del results, pars

    return params, values_dict
