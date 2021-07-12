from utilityFunctions import *

def get_model(model):
    if model.lower() == "langmuir":
        return langmuir1
    if model.lower() == "langmuir linear 1":
        return langmuirlin1
    if model.lower() == "langmuir linear 2":
        return langmuirlin2
    if model.lower() == "langmuir td":
        return langmuirTD
    if model.lower() == "dsl":
        return dsl
    if model.lower() == "gab":
        return gab
    if model.lower() == "mdr":
        return mdr
    if model.lower() == "mdr td":
        return mdrtd
    if model.lower() == "sips":
        return sips
    if model.lower() == "toth":
        return toth
    if model.lower() == "bddt 2n" or model.lower() == "bddt":
        return bddt1
    if model.lower() == "bddt 2n-1":
        return bddt2
    if model.lower() == "dodo":
        return dodo
    if model.lower() == "bet":
        return bet
    if model.lower() == "toth td":
        return tothTD

def get_guess_params(model, df, key_uptakes, key_pressures):
    saturation_loading = [1.1 * df[key_uptakes[i]].max() for i in range(len(key_pressures))]
    henry_lim = henry_approx(df, key_pressures, key_uptakes, False)[0]
    langmuir_b = [kh / qsat for (kh, qsat) in zip(henry_lim, saturation_loading)]
    h_guess = [-5000 for i in range(len(key_pressures))]

    if "langmuir" in model.lower() and model.lower() != "langmuir td":
        return {
            "b": langmuir_b,
            "q": saturation_loading
        }

    if model.lower() == "langmuir td":
        return {
            "b0": langmuir_b,
            "q": saturation_loading,
            "h": h_guess
        }

    if model.lower() == "dsl" or model.lower() == "dsl nc":
        return {
            "q1": [0.5 * q for q in saturation_loading],
            "b1": [0.4 * b for b in langmuir_b],
            "q2": [0.5 * q for q in saturation_loading],
            "b2": [0.6 * b for b in langmuir_b]
        }

    if model.lower() == "gab":
        return {
            "n": [0.2 * q for q in saturation_loading],
            "ka": [1.1 * b for b in langmuir_b],
            "ca": [0.1 * b for b in langmuir_b]
        }

    if model.lower() == "mdr":
        return {
            "n0": saturation_loading,
            "n1": langmuir_b,
            "a": [0.1 * b for b in langmuir_b],
            "c": [10 * b for b in langmuir_b]
        }
    if model.lower() == "mdr td":
        return {
            "n0": saturation_loading,
            "n1": langmuir_b,
            "a": [0.1 * b for b in langmuir_b],
            "b": [10 * b for b in langmuir_b],
            "e": [-1000 for i in saturation_loading]
        }
    if model.lower() == "sips":
        return {
            "q": saturation_loading,
            "b": langmuir_b,
            "n": [1 for q in saturation_loading]
        }
    if model.lower() == "toth":
        return {
            "q": saturation_loading,
            "b": langmuir_b,
            "t": [0.5 for q in saturation_loading]
        }
    if "bddt" in model.lower():
        return {
            "c": langmuir_b,
            "n": [10 for i in saturation_loading],
            "g": [100 for i in saturation_loading],
            "q": [q * 0.5 for q in saturation_loading]
        }
    if model.lower() == "dodo":
        return {
            "ns": saturation_loading,
            "kf": langmuir_b,
            "nu": [b * 10 for b in saturation_loading],
            "ku": langmuir_b,
            "m": [5 for i in saturation_loading]
        }
    if model.lower() == "bet":
        return {
            "n": saturation_loading,
            "c": langmuir_b,
        }
    if model.lower() == "toth td":
        return {
            "q": saturation_loading,
            "b0": [b * 0.1 for b in langmuir_b],
            "t": [0.5 for q in saturation_loading],
            "h": h_guess
        }


def get_fit_tuples(model, guess, temps, i=0, cond=False, cust_bounds=None):
    if cust_bounds is None:
        bounds = _MODEL_BOUNDS[model.lower()]
    else:
        bounds = cust_bounds
    if model.lower() == "dsl nc":
        return ('q1', guess['q1'][i], True, *bounds['q1']), \
               ('q2', guess['q2'][i], True, *bounds['q2']), \
               ('b1', guess['b1'][i], True, *bounds['b1']), \
               ('b2', guess['b2'][i], True, *bounds['b2']),

    if model.lower() == "gab":
        return ('n', guess['n'][i], True, *bounds['n']), \
               ('ka', guess['ka'][i], True, *bounds['ka']), \
               ('ca', guess['ca'][i], True, *bounds['ca'])

    if model.lower() == "mdr":
        return ('n0', guess['n0'][i], True, *bounds['n0']), \
               ('n1', guess['n1'][i], True, *bounds['n1']), \
               ('a', guess['a'][i], True, *bounds['a']), \
               ('c', guess['c'][i], True, *bounds['c'])

    if model.lower() == "sips":
        return ('q', guess['q'][i], True, *bounds['q']), \
               ('b', guess['b'][i], True, *bounds['b']), \
               ('n', guess['n'][i], True, *bounds['n'])

    if model.lower() == "toth":
        return ('q', guess['q'][i], True, *bounds['q']), \
               ('b', guess['b'][i], True, *bounds['b']), \
               ('t', guess['t'][i], True, *bounds['t'])

    if model.lower() == "toth td":
        return ('temp', temps[i], False), \
               ('q', guess['q'][i], True, *bounds['q']), \
               ('b0', guess['b0'][i], True, *bounds['b0']), \
               ('t', guess['t'][i], True, *bounds['t']), \
               ('h', guess['h'][i], True, *bounds['h'])

    if model.lower() == "bddt":
        if cond is True:
            c_con = ('c', guess['c'][i], True, 0, 1)
        else:
            c_con = ('c', guess['c'][i], True, *bounds)
        return c_con, \
               ('n', guess['n'][i], True, *bounds['n']), \
               ('g', guess['g'][i], True, *bounds['g']), \
               ('q', guess['q'][i], True, *bounds['q'])

    if model.lower() == "dodo":
        return ('ns', guess['ns'][i], True, *bounds['ns']), \
               ('kf', guess['kf'][i], True, *bounds['kf']), \
               ('nu', guess['nu'][i], True, *bounds['ns']), \
               ('ku', guess['ku'][i], True, *bounds['ku']), \
               ('m', guess['m'][i], True, *bounds['m'])

    if model.lower() == "mdr td":
        return ('t', temps[i], False), \
               ('n0', guess['n0'][i], True, *bounds['n0']), \
               ('n1', guess['n1'][i], True, *bounds['n1']), \
               ('a', guess['a'][i], True, *bounds['a']), \
               ('b', guess['b'][i], True, *bounds['b']), \
               ('e', guess['e'][i], True, *bounds['e'])

    if model.lower() == "bet":
        return ('n', guess['n'][i], True, *bounds['n']), \
               ('c', guess['c'][i], True, *bounds['c']),


_MODEL_PARAM_LISTS = {
    'mdr': ['n0', 'n1', 'a', 'c'],
    'mdr td': ['n0', 'n1', 'a', 'b', 'e'],
    'langmuir': ['q', 'b'],
    'langmuir td': ['q', 'b0', 'h'],
    'dsl nc': ['q1', 'q2', 'b1', 'b2'],
    'gab': ['n', 'ka', 'ca'],
    'sips': ['q', 'b', 'n'],
    'toth': ['q', 'b', 't'],
    'toth td': ['q', 'b0', 't', 'h'],
    'bddt': ['c', 'n', 'g', 'q'],
    'dodo': ['ns', 'kf', 'nu', 'ku', 'm'],
    'bet': ['n', 'c']
}

_MODEL_DF_TITLES = {
    'mdr': ['n0', 'n1', 'a', 'c'],
    'mdr td': ['n0 (mmol/g)', 'n1 (mmol/g)', 'a', 'b', 'e (J/mol)'],
    'langmuir': ['q (mmol/g)', 'b (1/bar)'],
    'langmuir td': ['q (mmol/g)', 'b0 (1/bar)', 'h (kJ/mol)'],
    'dsl nc': ['q1 (mmol/g)', 'q2 (mmol/g)', 'b1 (1/bar)', 'b2 (1/bar)'],
    'gab': ['n (mmol/g)', 'ka (H2O activity coeff.)', 'ca (GAB const.)'],
    'sips': ['q (mmol/g)', 'b (1/bar)', 'n (heterogeneity parameter)'],
    'toth': ['q (mmol/g)', 'b (1/bar)', 't (heterogeneity parameter)'],
    'toth td': ['q (mmol/g)', 'b0 (1/bar)', 't (heterogenity parameter)', 'h (J/mol)'],
    'bddt': ['c (BET const.)', 'n (layers)', 'g', 'q (mmol/g)'],
    'dodo': ['ns (mmol/g)', 'kf', 'nμ (mmol/g)', 'kμ', 'm'],
    'bet': ['n (mmol/g)', 'c']
}

_MODEL_BOUNDS = {
    'dsl nc': {
        'q1': (0, None),
        'q2': (0, None),
        'b1': (0, None),
        'b2': (0, None)},
    'gab': {
        'n': (0, None),
        'ka': (0, None),
        'ca': (0, None)
    },
    'mdr': {
        'n0': (0, None),
        'n1': (0, None),
        'a': (0, None),
        'c': (0, None)
    },
    'sips': {
        'q': (0, None),
        'b': (0, None),
        'n': (0, None)
    },
    'toth': {
        'q': (0, None),
        'b': (0, None),
        't': (0, None)
    },
    'toth td': {
        'q': (0, None),
        'b0': (0, None),
        't': (0, None),
        'h': (None, None)
    },
    'bddt': {
        'c': (0, None),
        'n': (0, None),
        'g': (0, None),
        'q': (0, None)
    },
    'dodo': {
        'ns': (0, None),
        'kf': (0, None),
        'nu': (0, None),
        'ku': (0, None),
        'm': (0, None)
    },
    'mdr td': {
        'n0': (0, None),
        'n1': (0, None),
        'a': (0, None),
        'b': (0, None),
        'e': (None, None)
    },
    'bet': {
        'n': (0, None),
        'c': (0, None)
    }
}

_TEMP_DEP_MODELS = ['langmuir td', 'mdr td', 'toth td']