from re import A
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from lmfit import Model, Parameters
from IPython.display import display
from modelFunctions import *
from utilityFunctions import *

def dsl_funct(df_lst, keyPressures, keyUptakes, temps=None, compnames=None, henry_constants=None, logplot=False, meth='leastsq', guess=None):
    # Here is the step procedure mentioned above
    # The outer class controls which step is being carried out
    # The first step is to find the initial q1, q2, b1, b2 values with the henry constraint set
    #q1_init = guess['q1']
    #q2_init = guess['q2']
    #b2_init = guess['b2']
    def step1(x, y, meth):

        gmod = Model(dsl)
        pars1 = Parameters()

        pars1.add('q1', value=q1_init, min=0)
        pars1.add('q2', value=q2_init, min=0)
        pars1.add('b2', value=b2_init)
        pars1.add('delta', value=henry_constants[0], vary=False)
        pars1.add('b1', expr='(delta-q2*b2)/q1') #KH = b1*q1 + b2*q2

        result1 = gmod.fit(y[0], pars1, x=x[0], method=meth)

        c1 = [result1.values['q1'], result1.values['q2'], result1.values['b1'], result1.values['b2']]
        return c1[0], c1[1]

        # This ends the function within the inner class and returns the qmax values to
        # be used in step 2
    
    def step2(x, y, meth, temps, compnames, step1_pars, comp2=False):
        q1fix = step1_pars[0]
        q2fix = step1_pars[1]

        if comp2 == True:
        #Depending on whether the package is fitting for the most adsorbed componennt or the
        #remaining components, the fitting procedure is different. In the methodology used,
        #The most adsorbed component is fitted to DSL but the remaining components are 
        #fitted to essentially the single langmuir isotherm
            gmod = Model(langmuir1)
            qtot = q1fix + q2fix
            
            
            c = []
            for i in range(len(x)):
                pars = Parameters()
                pars.add('q', value=qtot, min=qtot, max=qtot+0.001)
                pars.add('delta', value=henry_constants[i], vary=False)
                pars.add('b', expr='delta/q')
                
                results = gmod.fit(y[i], pars, x=x[i], method=meth)
                cee = [q1fix, q2fix, results.values['b'], results.values['b']]
                c.append(cee)
                
                del results
                del pars

        else:
            gmod = Model(dsl)
            c = []
            for i in range(len(x)):
                pars = Parameters()
                pars.add('q1', value=q1fix, min=q1fix, max=q1fix+0.001)
                pars.add('q2', value=q2fix, min=q2fix, max=q2fix+0.001)
                pars.add('b2', value=b2_init, min=0)
                pars.add('delta', value=henry_constants[i], vary=False)
                pars.add('b1', expr='(delta-q2*b2)/q1', min=0) #KH = b*q
                
                results = gmod.fit(y[i], pars, x=x[i], method=meth)
                cee = [results.values['q1'], results.values['q2'], 
                            results.values['b1'], results.values['b2']]
                c.append(cee)
                
                del results
                del pars
        
        #allocating variables
        qmax1 = [param[0] for param in c]
        qmax2 = [param[1] for param in c]
        b1 = [param[2] for param in c]
        b2 = [param[3] for param in c]
        qtot = [param[0] + param[1] for param in c]
        
        #Finding heat of adsorption for both sites
        T = np.array([1/(temp+273) for temp in temps])
        ln_b1 = np.array([np.log(i) for i in b1])
        ln_b2 = np.array([np.log(i) for i in b2])
        mH1, bH1, rH1, pH1, sterrH1 = stats.linregress(T,ln_b1)
        mH2, bH2, rH2, pH2, sterrH2 = stats.linregress(T,ln_b2)
        
        h = [-0.001*mH1*r, -0.001*mH2*r]
        b0 = [np.exp(bH1), np.exp(bH2)]

        return qmax1[0], qmax2[0], h, b0
        # The package returns these sets of parameters to be used as initial guesses for the final step

    def step3(x, y, meth, temps, compnames, step2_pars, comp2=False):
        tempsK = [t+273 for t in temps]

        #unpacking tuples
        q1_in = step2_pars[0]
        q2_in = step2_pars[1]
        h_in = step2_pars[2]
        b0_in = step2_pars[3]
        
        # We can now fit to the van't Hoff form of DSL with the initial guess values
        # obtained through step 2
        
        if comp2 == True:
            # Again, we fit to the single langmuir form when the least adsorbed species is used   
            gmod = Model(langmuirTD)
            qtot = q1_in + q2_in
            c = []
            result = []
            for i in range(len(x)):
                pars = Parameters()
                pars.add('t', value=tempsK[i], vary = False)
                pars.add('q', value=qtot, min=qtot, max=qtot+0.001)
                pars.add('h', value=h_in[0]*1000)
                pars.add('b0', value=b0_in[0], min=0)
                
                results = gmod.fit(y[i], pars, x=x[i], method=meth)
                result.append(results)
                
                c.append([results.values['t'], q1_in, q2_in, results.values['h'],
                            results.values['h'], results.values['b0'], results.values['b0']])
                
                del results
                del pars
                
        else:
            gmod = Model(dslTD) #DSL                            
            c = []
            result = []
            for i in range(len(x)):
                pars = Parameters()
                pars.add('t', value=tempsK[i], vary = False)
                pars.add('q1', value=q1_in, min=q1_in, max=q1_in+0.001)
                pars.add('q2', value=q2_in, min=q2_in, max=q2_in+0.001)
                pars.add('h1', value=h_in[0]*1000)
                pars.add('h2', value=h_in[1]*1000)
                pars.add('b01', value=b0_in[0], min=0)
                pars.add('b02', value=b0_in[1], min=0)                

                results = gmod.fit(y[i], pars, x=x[i], method=meth)
                result.append(results)
                c.append([results.values['t'], results.values['q1'], results.values['q2'], results.values['h1'], 
                            results.values['h2'], results.values['b01'], results.values['b02']])
                
                del results
                del pars
                
                
        print(bold + "\nParameters found...")

        #allocating variables, formatting and creating dataframe
        
        t = [param[0] for param in c]
        
        #UNFORMATTED VARIABLES
        q1 = [param[1] for param in c]
        q2 = [param[2] for param in c]
        h_1 = [param[3] for param in c]
        h_2 = [param[4] for param in c]
        b_01 = [param[5] for param in c]
        b_02 = [param[6] for param in c]
        
        #FORMATTED VARIABLES
        qmax1 = [np.round(param[1], 3) for param in c]
        qmax2 = [np.round(param[2], 3) for param in c]
        h1 = [np.round(param[3], 3) for param in c]
        h2 = [np.round(param[4], 3) for param in c]
        b01 = ["{:.3e}".format(param[5]) for param in c]
        b02 = ["{:.3e}".format(param[6]) for param in c]
        
        # Checking r squared of fits
        r_sq = [r2(x[i], y[i], dslTD, c[i]) for i in range(len(x))]
        se = [mse(x[i], y[i], dslTD, c[i]) for i in range(len(x))]
        
        #Displaying results
        df_result = pd.DataFrame(list(zip(t, qmax1, qmax2, h1, h2, b01, b02, r_sq, se)), 
                        columns=['Temp(K)','qmax1 (mmol/g)',
                                'qmax2 (mmol/g)','h1 (J/mol)', 'h2 (J/mol)', 'b01 (1/bar)', 'b02 (1/bar)','R sq', 'mse'])

        display(pd.DataFrame(df_result))
        
        print(bold + "===============================================================================")
        print(bold + "===============================================================================")
    
    df_dict = {}
    i = 0
    for df in df_lst:
        x = [df[keyPressures[j]].values for j in range(len(keyPressures))]
        y = [df[keyUptakes[j]].values for j in range(len(keyPressures))]
        df_dict[i] = x, y
        del x, y
        i += 1

    print(df_dict)


#df1 = pd.read_csv('zeo_CO2_experimental.csv')
#df2 = pd.read_csv('zeo_N2_experimental.csv')

df1 = pd.read_csv('Computational Data (EPFL) CO2.csv')
df2 = pd.read_csv('Computational Data (EPFL) N2.csv')

df_list = [df1, df2]



temps = [0, 25, 50]  
#temps = [20, 40, 60, 70]
# Need to input the dataframe as a list into the class
# Lists for pressure and uptake keys are also required

compnames = ['CO2', 'N2'] #Temp in C
meth = 'tnc' # Optional picking mathematical method for fitting (default is leastsq)
#initial guess values#
#       [q1    q2     b1      b2]
guess = [4.014630251577328, 2.5053142871054255, 7.663656259628368, 7.663655158791659] #Optional initial guess values for fitting

keyUptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
keyPressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']

#keyUptakes = ['Uptake1', 'Uptake2', 'Uptake3', 'Uptake4']
#keyPressures = ['Pressure1', 'Pressure2', 'Pressure3', 'Pressure4']

dsl_funct(df_list, keyPressures, keyUptakes)
    


    





