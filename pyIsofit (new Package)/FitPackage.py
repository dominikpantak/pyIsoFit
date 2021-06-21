#import FitPackage as mc
#import importlib

#importlib.reload(mc)

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from lmfit import Model, Parameters
#from IPython.display import display
from modelFunctions import *
from utilityFunctions import *

_MODELS = [
    "Langmuir", "Quadratic", "BET", "Henry", "TemkinApprox", "DSLangmuir"
]

colours = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:olive']

class IsothermFit:
    def __init__(self,
                df,
                compname,
                temps,
                keyPressures,
                keyUptakes,
                logplot=False,
                model=None,
                guess=None,
                meth='tnc',
                x=None,
                y=None,
                params=None):

        self.df = df                 #Dataframe with uptake vs. pressure data
        self.temps = temps           #Temp in deg C
        self.meth = meth             #Optional to choose mathematical fitting method in lmfit (default is leastsq)
        self.name = compname             #Name of component

        self.keyPressures = keyPressures #names of the column headers for pressure and uptakes
        self.keyUptakes = keyUptakes    #
        self.model = model
        if model == None:
            raise Exception("Enter a model as a parameter")
        self.logplot = logplot

        self.x = x if x is not None else []
        self.y = x if y is not None else []
        self.params = params if y is not None else []

        # ! Dictionary of parameters as a starting point for data fitting
        self.guess = get_guess_params(model, df, keyUptakes, keyPressures)
        # Override defaults if user provides param_guess dictionary
        if guess is not None:
            for param, guess_val in guess.items():
                if param not in list(self.guess.keys()):
                    raise Exception("%s is not a valid parameter"
                                    " in the %s model." % (param, model))
                self.guess[param] = guess_val

    def fit(self, cond=True, show_hen=False, hen_tol=0.999):
        # Reading data from dataframe with respect to provided keys
        x2 = []
        y2 = []
        #Importing data and allocating to variables
        for i in range(len(keyPressures)):
            x2.append(self.df[keyPressures[i]].values)
            y2.append(self.df[keyUptakes[i]].values)

        if self.model == "Langmuir linear 1":
            for i in range(len(self.keyPressures)):
                self.x.append([1 / p for p in x2[i]])
                self.y.append([1 / q for q in y2[i]])

        elif self.model == "Langmuir linear 2":
            for i in range(len(self.keyPressures)):
                self.y.append([p / q for (p, q) in zip(x2[i], y2[i])])
                self.x.append(x2[i])

        elif self.model == "MDR" or self.model == "MDR TD":
            for i in range(len(self.keyPressures)):
                pressure = x2[i]
                self.x.append(np.array([p / pressure[-1] for p in x2[i]]))
                self.y.append([i for i in y2])
                del pressure
        else:
            self.x = x2
            self.y = y2

        henry_constants = henry_approx(self.df, self.keyPressures, self.keyUptakes, show_hen, hen_tol)

        # SINGLE LANGMUIR FITTING
        if "Langmuir" in self.model and self.model != "Langmuir TD":
            isotherm = get_model(self.model)
            gmod = Model(isotherm, nan_policy="omit")
            if cond == True:
                print("Constraint 1: q sat = q_init for all temp")
                print("Constraint 2: qsat*b = Henry constant for all temp")
            c = []

            q_guess = self.guess['q']
            b_guess = self.guess['b']

            for i in range(len(self.keyPressures)):
                pars = Parameters()

                #Creating intermediate parameter delta that will fix KH = b*q

                if cond == True:
                    if i == 0:
                        pars.add('q', value=q_guess[0], min=0)
                    else:
                        pars.add('q', value=q_fix, min=q_fix, max=q_fix+0.001)

                    pars.add('delta', value=henry_constants[i], vary=False)
                    pars.add('b', expr='delta/q') #KH = b*q
                else:
                    pars.add('q', value=q_guess[i], min=0)
                    pars.add('b', value=b_guess[i], min=0)

                results = gmod.fit(self.y[i], pars, x=self.x[i], method=self.meth)
                cee = [results.values['q'], results.values['b']]
                self.params.append(results)
                c.append(cee)
                if i == 0 and cond == True:
                    q_fix = results.values['q']  #This only gets applied if cond=True

                del results, pars
            #allocating variables and creating dataframe
            #UNFORMATTED VARIABLES
            q_ = [param[0] for param in c]
            b_ = [param[1] for param in c]

            #FORMATTED VARIABLES
            qmax = [np.round(param[0], 3) for param in c]
            b = ["{:.3e}".format(param[1]) for param in c]

            # Checking r squared of fits
            r_sq = [r2(self.x[i], self.y[i], isotherm, c[i]) for i in range(len(self.keyPressures))]
            se = [mse(self.x[i], self.y[i], isotherm, c[i]) for i in range(len(self.keyPressures))]


            df_result = pd.DataFrame(list(zip(self.temps, qmax, b, r_sq, se)), columns=['Temperature (oC)','Qmax (mmol/g)',
                                                                                  'b (1/bar)', 'R squared', 'MSE'])
            #displaying dataframe
            display(pd.DataFrame(df_result))

            #calculating heat of adsorption
            T = np.array([1/(temp+273) for temp in self.temps])
            ln_b = np.array([np.log(i) for i in b_])
            mH, bH, rH, pH, sterrH = stats.linregress(T,ln_b)
            h = -0.001*mH*r

            print("_______________________________________________________________________")
            print("Heat of adsorption: " + str(round(h,2)) + " kJ/mol. \n" +
                  "R sq of Van't Hoff: " + str(round(rH, 4)))

            #b0 is also calculated and displayed to the user.
            #This can be then fed back into the class with delta H to fit to the van't Hoff form of langmuir
            b0 = np.exp(bH)
            print("List for intial guess values to feed into the temperature dependent Langmuir model:")
            print([q_[0], b_[0], h, b0])

        #if self.model == "DSL":


    def plot(self):
        p=np.linspace(0, 10, 301)

        ##### Plotting results #####
        plt.figure(figsize=(8, 6))
        #plt.title()
        if self.logplot == True:
            plt.xscale("log")
            plt.yscale("log")
        plt.xlabel('xaxis')
        plt.ylabel('yaxis')
        plt.tick_params(**tick_style)

        for i in range(len(self.keyPressures)):
            plt.plot(self.x[i], self.params[i].best_fit, '-', color = colours[i],
                     label="{temps} °C Fit".format(temps=self.temps[i]))
            plt.plot(self.x[i], self.y[i], 'ko', color = '0.75',
                     label="Data at {temps} °C".format(temps=self.temps[i]))

        plt.grid()
        plt.legend()
        plt.show()

df1 = pd.read_csv('Computational Data (EPFL) CO2.csv')
compname = 'CO2'
temps = [10, 40, 100]
meth = 'tnc'
keyUptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
keyPressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']

#keyPressures = ['Pressure1', 'Pressure2', 'Pressure3']
#keyUptakes = ['Uptake1', 'Uptake2', 'Uptake3']
tolerance = 0.9999 # set minimum r squared value

langmuir = IsothermFit(df1, compname, temps, keyPressures, keyUptakes, True, "Langmuir")
langmuir.fit(True, True)
langmuir.plot()

