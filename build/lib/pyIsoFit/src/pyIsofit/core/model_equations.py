"""
This includes model equation definitions, including error checking functions.

"""
import numpy as np

r = 8.314
bold = '\033[1m'  # for making text bold
unbold = '\033[0m'  # for removing bold text


# Single site langmuir forms
def langmuir1(x, q, b):
    return (q * b * x) / (1 + b * x)


def langmuirlin1(x, q, b):
    return (x / (b * q)) + (1 / (q))


def langmuirlin2(x, q, b):
    return (1 / (b * q)) + (x / q)


# Single site langmuir (temperature dependent form)
def langmuirTD(x, t, q, b0, h):
    b = b0 * np.exp(-h / (r * t))
    return q * ((b * x) / (1 + b * x))


###############################################
# Dual site langmuir forms -  normal and temperature dependent form
def dsl(x, q1, q2, b1, b2):
    site1 = q1 * ((b1 * x) / (1 + b1 * x))
    site2 = q2 * ((b2 * x) / (1 + b2 * x))
    return site1 + site2


def dsltd(x, t, q1, q2, h1, h2, b01, b02):
    b1 = b01 * np.exp(-h1 / (r * t))
    b2 = b02 * np.exp(-h2 / (r * t))
    site1 = q1 * ((b1 * x) / (1 + b1 * x))
    site2 = q2 * ((b2 * x) / (1 + b2 * x))
    return site1 + site2


#################################################
# Modified Dubinin–Radushkevich (MDR) - normal and temperature dependent form
def mdrtd(x, t, n0, n1, a, b, e):
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

# Henry's law function for the henry region calculation
def henry(x, kh):
    return [i * kh for i in x]


# Calculation for b0 from obtained single langmuir parameters to be used as an initial guess for 'deltaH_langmuir'
def b0_calc(b, h, t):
    return np.exp(np.log(b) + h / (r * t))


###############################################
# BET Extension for type IV and V isotherms (BDDT isotherm model)

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


# Do and Do isotherm model
def dodo(x, ns, kf, nu, ku, m):
    term1 = (ns * kf * x) / ((1 - x) * (1 + (kf - 1) * x))
    term2 = (nu * ku * x ** m) / (1 + ku * x ** m)
    return term1 + term2

# Guggenheim-Anderson-De Boer (GAB) isotherm model
def gab(x, n, ka, ca):
    num = n * ka * ca * x
    den = (1 - ka * x) * (1 + (ca - 1) * ka * x)
    return num / den

# Brunauer Emmett Teller (BET) isotherm model
def bet(x, n, c):
    return (c * x * n) / ((1 - x) * (1 - x + c * x))


#######################################################
# Sips isotherm model
def sips(x, q, b, n):
    return q * (((b * x) ** (1 / n)) / (1 + (b * x) ** (1 / n)))

# Toth isotherm model
def toth(x, q, b, t):
    return q * ((b * x) / ((1 + (b * x) ** t) ** (1 / t)))

# Toth temperature dependent model
def tothTD(x, temp, q, b0, h, t):
    b = b0 * np.exp(-h / (r * temp))
    return q * ((b * x) / ((1 + (b * x) ** t) ** (1 / t)))

########################################################
# R squared calculation
def r2(x, y, f, c):
    residuals = y - f(x, *c)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    result = 1 - (ss_res / ss_tot)
    if result <= 0:
        # Returns zero if the result is less than zero
        return 0.00000
    else:
        return result


# R squared calculation for henry model
def r2hen(pressures, uptake, f, hen):
    residuals = uptake - f(pressures, hen)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((uptake - np.mean(uptake)) ** 2)
    result = 1 - (ss_res / ss_tot)
    if result <= 0:
        return 0.00000
    else:
        return result

# Mean squared error calculation
def mse(x, y, f, c):
    y_predicted = f(x, *c)
    return np.square(np.subtract(y, y_predicted)).mean()
