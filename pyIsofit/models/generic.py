from lmfit import Model, Parameters
from pyIsofit.core.model_fit_def import get_fit_tuples
from pyIsofit.core.model_dicts import _MODEL_FUNCTIONS


def generic_fit(model, x, y, guess, temps, cond, meth, cust_bounds, fit_report):
    isotherm = _MODEL_FUNCTIONS[model]
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
        if fit_report:
            print("\n\n\n---- FIT REPORT FOR DATASET AT {temp} K -----".format(temp=temps[i]))
            print(results.fit_report())
        del results, pars

    return params, values_dict
