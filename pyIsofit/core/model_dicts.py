from pyIsofit.core.model_equations import *

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
        'b2': (0, None)
    },
    'langmuir': {
        'q': (0, None),
        'b': (0, None)
    },
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

_MODEL_FUNCTIONS = {
    'langmuir': langmuir1,
    'langmuir linear 1': langmuirlin1,
    'langmuir linear 2': langmuirlin2,
    'langmuir td': langmuirTD,
    'dsl': dsl,
    'dsl nc': dsl,
    'gab': gab,
    'mdr': mdr,
    'mdrtd': mdrtd,
    'sips': sips,
    'toth': toth,
    'bddt 2n': bddt1,
    'bddt 2n-1': bddt2,
    'bddt': bddt1,
    'dodo': dodo,
    'bet': bet,
    'toth td': tothTD
}

_TEMP_DEP_MODELS = ['langmuir td', 'mdr td', 'toth td']

# _MODEL_INFO = {
#     'langmuir': "This fits to the single site Langmuir isotherm model",
#     'langmuir linear 1': """This fits to the second linearized form of the
#                             single site Langmuir isotherm model. This creates a plot of 1/q vs. 1/P""",
#     'langmuir linear 2': """This fits to the second linearized form of the
#                             single site Langmuir isotherm model""",
#     'langmuir td': langmuirTD,
#     'dsl': dsl,
#     'dsl nc': dsl,
#     'gab': gab,
#     'mdr': mdr,
#     'mdrtd': mdrtd,
#     'sips': sips,
#     'toth': toth,
#     'bddt 2n': bddt1,
#     'bddt 2n-1': bddt2,
#     'bddt': bddt1,
#     'dodo': dodo,
#     'bet': bet,
#     'toth td': tothTD
# }
