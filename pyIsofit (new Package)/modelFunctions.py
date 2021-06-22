import numpy as np

tick_style = {'direction': 'in',
              'length': 4,
              'width': 0.7,
              'colors': 'black'}

r = 8.314
bold = '\033[1m'  # for making text bold
unbold = '\033[0m'  # for removing bold text


##############################################################################
########################### Isotherm models ##################################
##############################################################################

########## x is pressure ##############

# Single site langmuir forms
def langmuir1(x, q, b):
    return (q * b * x) / (1 + b * x)


def langmuirlin1(x, q, b):
    return (x / (b * q)) + (1 / (q))


def langmuirlin2(x, q, b):
    return (1 / (b * q)) + (x / q)


# Single site langmuir form with temperature dependent form
def langmuirTD(x, t, q, b0, h):
    b = b0 * np.exp(-h / (r * t))
    return q * ((b * x) / (1 + b * x))


###############################################
# Dual site langmuir forms -  normal and temperature dependent form
def dsl(x, q1, q2, b1, b2):
    site1 = q1 * ((b1 * x) / (1 + b1 * x))
    site2 = q2 * ((b2 * x) / (1 + b2 * x))
    return site1 + site2


def dslTD(x, t, q1, q2, h1, h2, b01, b02):
    b1 = b01 * np.exp(-h1 / (r * t))
    b2 = b02 * np.exp(-h2 / (r * t))
    site1 = q1 * ((b1 * x) / (1 + b1 * x))
    site2 = q2 * ((b2 * x) / (1 + b2 * x))
    return site1 + site2


#################################################


def mdrTD(x, t, n0, n1, a, b, e):
    k = (r * t) / (b * e)
    term1a = (1 - np.exp(- a * x)) * n0
    term1b = np.exp((-k * ((np.log(1 / x)) ** 2)))
    term2 = np.exp(- a * x) * n1 * x
    return term1a * term1b + term2


def mdr(x, n0, n1, a, c):
    term1a = (1 - np.exp(- a * x)) * n0
    term1b = np.exp((-c * ((np.log(1 / x)) ** 2)))
    term2 = np.exp(- a * x) * n1 * x
    return term1a * term1b + term2


################################################
# Calculation for b0 from obtained single langmuir parameters to be used as an initial guess for 'deltaH_langmuir'
def henry(x, kh):  # Henry's law function for the henry region calculation
    return [i * kh for i in x]


def b0_calc(b, h, t):
    return np.exp(np.log(b) + h / (r * t))


####################### TYPE III, IV, V isotherms #################################

# BET Extension for type IV and V isotherms

# For a maximum (2n-1) layers which can be fit into a capillary:
def bddt1(x, c, n, g, q):
    term1 = (c * x) / (1 - x)
    term2_num = 1 + (((n * g) / 2) - n) * x ** (n - 1) - (n * g - n + 1) * x ** n + (n * g / 2) * x ** (n + 1)
    term2_den = 1 + (c - 1) * x + ((c * g / 2) - c) * x ** n - (c * g / 2) * x ** (n + 1)
    return term1 * (term2_num / term2_den) * q


# For a maximum number of 2n layers:
def bddt2(x, c, n, g, q):
    term1 = (c * x) / (1 - x)
    term2_num = 1 + (((n * g) / 2) - n / 2) * x ** (n - 1) - (n * g + 1) * x ** n + (n * g / 2 + n / 2) * x ** (n + 1)
    term2_den = 1 + (c - 1) * x + ((c * g / 2) - c / 2) * x ** n - (c * g / 2 + c / 2) * x ** (n + 1)
    return term1 * (term2_num / term2_den) * q


# Do and Do isotherm
def dodo(x, ns, kf, nu, ku, m):
    term1 = (ns * kf * x) / ((1 - x) * (1 + (kf - 1) * x))
    term2 = (nu * ku * x ** m) / (1 + ku * x ** m)
    return term1 + term2


def gab(x, n, ka, ca):
    num = n * ka * ca * x
    den = (1 - ka * x) * (1 + (ca - 1) * ka * x)
    return num / den


###############################################################################
##################### EXTENDED MODELS ##################################

# Extended dual site langmuir model for the prediction of binary adsorption
# This calculates the absolute adsorption of A wrt pressure
def ext_dslA(x, t, q1A, q2A, h1A, h2A, b01A, b02A, h1B, h2B, b01B, b02B, yA):
    b1A = b01A * np.exp(-h1A / (r * t))
    b2A = b02A * np.exp(-h2A / (r * t))
    b1B = b01B * np.exp(-h1B / (r * t))
    b2B = b02B * np.exp(-h2B / (r * t))
    yB = 1 - yA
    e1 = q1A * (b1A * x * yA) / (1 + (b1A * x * yA) + (b1B * x * yB))
    e2 = q2A * (b2A * x * yA) / (1 + (b2A * x * yA) + (b2B * x * yB))
    return e1 + e2


def ext_dslB(x, t, q1B, q2B, h1A, h2A, b01A, b02A, h1B, h2B, b01B, b02B, yB):
    b1A = b01A * np.exp(-h1A / (r * t))
    b2A = b02A * np.exp(-h2A / (r * t))
    b1B = b01B * np.exp(-h1B / (r * t))
    b2B = b02B * np.exp(-h2B / (r * t))
    yA = 1 - yB
    e1 = q1B * (b1B * x * yB) / (1 + (b1B * x * yB) + (b1A * x * yA))
    e2 = q2B * (b2B * x * yB) / (1 + (b2B * x * yB) + (b2A * x * yA))
    return e1 + e2


def sel(qA, qB, yA):
    return (qA / qB) / (yA / (1 - yA))


############# EMPIRICAL MODELS ################
def sips(x, q, b, n):
    return q * (((b * x) ** (1 / n)) / (1 + (b * x) ** (1 / n)))


def toth(x, q, b, t):
    return q * ((b * x) / ((1 + (b * x) ** t) ** (1 / t)))


################################################################################

#############################################################################
######################      ERROR CALCS       ###############################
############################################################################


######### R squared calculation ###########
def r2(x, y, f, c):
    residuals = y - f(x, *c)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    result = 1 - (ss_res / ss_tot)
    if result <= 0:
        return 0.00000
    else:
        return result


def r2fix(x, y, result):
    residuals = y - result
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    rsq = 1 - (ss_res / ss_tot)
    return rsq


# This additional form of R squared calculation was required due to an error...
# ...when using henrys law in the above function
def r2hen(pressures, uptake, f, hen):
    residuals = uptake - f(pressures, hen)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((uptake - np.mean(uptake)) ** 2)
    result = 1 - (ss_res / ss_tot)
    if result <= 0:
        return 0.00000
    else:
        return result


def mse(x, y, f, c):
    y_predicted = f(x, *c)
    return np.square(np.subtract(y, y_predicted)).mean()
