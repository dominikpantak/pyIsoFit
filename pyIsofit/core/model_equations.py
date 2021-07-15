import numpy as np

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

def bet(x, n, c):
    return (c * x * n) / ((1 - x) * (1 - x + c * x))


def sel(qA, qB, yA):
    return (qA / qB) / (yA / (1 - yA))


############# EMPIRICAL MODELS ################
def sips(x, q, b, n):
    return q * (((b * x) ** (1 / n)) / (1 + (b * x) ** (1 / n)))


def toth(x, q, b, t):
    return q * ((b * x) / ((1 + (b * x) ** t) ** (1 / t)))

def tothTD(x, temp, q, b0, h, t):
    b = b0 * np.exp(-h / (r * temp))
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


