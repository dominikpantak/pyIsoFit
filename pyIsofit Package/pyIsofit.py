#!/usr/bin/env python
# coding: utf-8

# In[5]:


import Source as iso

class Models:
    def __init__(self, df, compname, temps, keyPressures, keyUptakes, guess=None, logplot=False,
                 cond=None, meth=None, cond2=None):
        self.df = df                 #Dataframe with uptake vs. pressure data
        self.temps = temps           #Temp in deg C
        self.meth = meth             #Optional to choose mathematical fitting method in lmfit (default is leastsq)
        self.name = compname             #Name of component

        self.keyPressures = keyPressures #names of the column headers for pressure and uptakes
        self.keyUptakes = keyUptakes    #

        self.guess = guess       
        self.cond = cond
        self.cond2 = cond2
        
    def sips(self, df, compname, temps, keyPressures, keyUptakes, guess, logplot=False, cond=None, meth=None):
        isotherm = iso.sips
        pars = df, compname, temps, isotherm, keyPressures, keyUptakes, guess, logplot
        empirical = iso.Empirical(*pars)
        empirical.isotherm_fit(*pars)
        if len(self.temps) > 3:
            print("WARNING: Less than 3 temperature datasets have been provided.")
            print("         pyIsofit cannot calculate binding energies/heats of adsorption.")
            print("         A temperature dependent variation of an isotherm model may be used, however")
            print("         this may give innacurate results.")
    def toth(self, df, compname, temps, keyPressures, keyUptakes, guess, logplot=False, cond=None, meth=None):
        isotherm = iso.toth
        pars = df, compname, temps, isotherm, keyPressures, keyUptakes, guess, logplot
        empirical = iso.Empirical(*pars)
        empirical.isotherm_fit(*pars)
        if len(self.temps) > 3:
            print("WARNING: Less than 3 temperature datasets have been provided.")
            print("         pyIsofit cannot calculate binding energies/heats of adsorption.")
            print("         A temperature dependent variation of an isotherm model may be used, however")
            print("         this may give innacurate results.")        
    def gab(self, df, compname, temps, keyPressures, keyUptakes, guess, logplot=False, cond=None, meth=None):
        pars = df, compname, temps, keyPressures, keyUptakes, guess, logplot #missing isotherms
        typeIII = iso.TypeIII_fit(*pars) # Calling Langmuir class to variable
        typeIII.isotherm_fit(*pars)
        if len(self.temps) > 3:
            print("WARNING: Less than 3 temperature datasets have been provided.")
            print("         pyIsofit cannot calculate binding energies/heats of adsorption.")
            print("         A temperature dependent variation of an isotherm model may be used, however")
            print("         this may give innacurate results.")      
        
    def bet_ext1(self, df, compname, temps, keyPressures, keyUptakes, guess, logplot=False, cond=None, meth=None):
        print("This form of BET extension is for a maximum of 2n-1 layers which can be fit into a capillary\n")
        isotherm = iso.BET_ext1
        if cond == True: 
            pars = df, compname, temps, isotherm, keyPressures, keyUptakes, guess, True, logplot
            print("Note: Fitting for Type V isotherm")
        else:
            pars = df, compname, temps, isotherm, keyPressures, keyUptakes, guess, False, logplot
            print("Parameter 'c' is constrained to < 1. Currently fitting to Type IV isotherm.")
            print("To remove constraint, set additional condition 'cond' to True. i.e isotherms.bet_ext1(...., guess, True)")
        
        typeIV_V = iso.TypeIV_V_fit(*pars) 
        typeIV_V.isotherm_fit(*pars)
    
    def bet_ext2(self, df, compname, temps, keyPressures, keyUptakes, guess, logplot=False, cond=None, meth=None):
        print("This form of BET extension is for a maximum of 2n layers which can be fit into a capillary.\n")
        isotherm = iso.BET_ext2
        if cond == True: 
            pars = df, compname, temps, isotherm, keyPressures, keyUptakes, guess, True, logplot
            print("Fitting for Type V isotherm")
        else:
            pars = df, compname, temps, isotherm, keyPressures, keyUptakes, guess, False, logplot
            print("Parameter 'c' is constrained to < 1. Currently fitting to Type IV isotherm.")
            print("To remove constraint, set additional condition 'cond' to True. i.e isotherms.bet_ext1(...., guess, True)")
        typeIV_V = iso.TypeIV_V_fit(*pars) 
        typeIV_V.isotherm_fit(*pars)
            
    def dodo(self, df, compname, temps, keyPressures, keyUptakes, guess, logplot=False, cond=None, meth=None):
        isotherm = iso.DoDo
        pars = df, compname, temps, isotherm, keyPressures, keyUptakes, guess, False, logplot
        
        typeIV_V = iso.TypeIV_V_fit(*pars) 
        typeIV_V.isotherm_fit(*pars)
        
    def langmuir1(self, df, compname, temps, keyPressures, keyUptakes, guess, logplot=False, cond=False, meth=None, cond2=0.9999):
        print("This fits to the single site Langmuir isotherm model")
        if meth == None:
            meth = 'leastsq'
        if cond == False:
            print("WARNING: 'qsat' is not fixed. Thermodynamic consistency criteria are not met.")
            print("Consider using qsat from the lowest temperature isotherm result, inputting it into the 'guess'")
            print("parameter, and setting condition to 'True' i.e isotherms.langmuir1(...., guess, True)")
        isotherm = iso.single_langmuir
        pars = df, temps, compname, isotherm, keyPressures, keyUptakes, guess, logplot, cond, meth, cond2
        typeI = iso.TypeI_fit(*pars) 
        typeI.lang(*pars)
        if len(self.temps) > 3:
            print("WARNING: Less than 3 temperature datasets have been provided.")
            print("         pyIsofit cannot calculate binding energies/heats of adsorption.")
            print("         A temperature dependent variation of an isotherm model may be used, however")
            print("         this may give innacurate results.")
        
    def langmuir2(self, df, compname, temps, keyPressures, keyUptakes, guess, logplot=False, cond=False, meth=None, cond2=0.9999):
        print("This fits to the first linearized form of the single site Langmuir isotherm model")
        print("This creates a plot of 1/q vs. 1/P")
        if meth == None:
            meth = 'leastsq'
        if cond == False:
            print("WARNING: 'qsat' is not fixed. Thermodynamic consistency criteria are not met.")
            print("Consider using qsat from the lowest temperature isotherm result, inputting it into the 'guess'")
            print("parameter, and setting condition to 'True' i.e isotherms.langmuir1(...., guess, True)")
        isotherm = iso.linear_langmuir1
        pars = df, temps, compname, isotherm, keyPressures, keyUptakes, guess, logplot, cond, meth, cond2
        typeI = iso.TypeI_fit(*pars) 
        typeI.lang(*pars)
        if len(self.temps) > 3:
            print("WARNING: Less than 3 temperature datasets have been provided.")
            print("         pyIsofit cannot calculate binding energies/heats of adsorption.")
            print("         A temperature dependent variation of an isotherm model may be used, however")
            print("         this may give innacurate results.")
    
    def langmuir3(self, df, compname, temps, keyPressures, keyUptakes, guess, logplot=False, cond=False, meth=None, cond2=0.9999):
        print("This fits to the second linearized form of the single site Langmuir isotherm model")
        print("This creates a plot of P/q vs. P")
        if meth == None:
            meth = 'leastsq'
        if cond == False:
            print("WARNING: 'qsat' is not fixed. Thermodynamic consistency criteria are not met.")
            print("Consider using qsat from the lowest temperature isotherm result, inputting it into the 'guess'")
            print("parameter, and setting condition to 'True' i.e isotherms.langmuir1(...., guess, True)")
        isotherm = iso.linear_langmuir2
        pars = df, temps, compname, isotherm, keyPressures, keyUptakes, guess, logplot, cond, meth, cond2
        typeI = iso.TypeI_fit(*pars) 
        typeI.lang(*pars)
        if len(self.temps) > 3:
            print("WARNING: Less than 3 temperature datasets have been provided.")
            print("         pyIsofit cannot calculate binding energies/heats of adsorption.")
            print("         A temperature dependent variation of an isotherm model may be used, however")
            print("         this may give innacurate results.")
    
    def langmuirTD(self, df, compname, temps, keyPressures, keyUptakes, guess, logplot=False, cond=False, meth=None, cond2=0.9999):
        print("This fits to the temperature dependent (TD) form of the single site Langmuir isotherm model")
        print("For greatest accuracy, run code on any of the non TD langmuir isotherm models, and paste the")
        print("given initial guess values into the guess parameter")
        if cond == False:
            print("WARNING: 'qsat' is not fixed. Thermodynamic consistency criteria are not met.")
            print("Consider using qsat from the lowest temperature isotherm result, inputting it into the 'guess'")
            print("parameter, and setting condition to 'True' i.e isotherms.langmuir1(...., guess, True)")
        isotherm = iso.deltaH_langmuir
        pars = df, temps, compname, isotherm, keyPressures, keyUptakes, guess, logplot, cond, meth, cond2
        typeI = iso.TypeI_fit(*pars) 
        typeI.lang(*pars)
    
    def mdr(self, df, compname, temps, keyPressures, keyUptakes, guess, logplot=False, cond=None, meth=None, cond2=0.9999):
        isotherm = iso.mdr
        pars = df, temps, compname, isotherm, keyPressures, keyUptakes, guess, logplot, cond, meth, cond2
        typeI = iso.TypeI_fit(*pars) 
        typeI.lang(*pars)
        if len(self.temps) > 3:
            print("WARNING: Less than 3 temperature datasets have been provided.")
            print("         pyIsofit cannot calculate binding energies/heats of adsorption.")
            print("         A temperature dependent variation of an isotherm model may be used, however")
            print("         this may give innacurate results.")
    
    def mdrTD(self, df, compname, temps, keyPressures, keyUptakes, guess, logplot=False, cond=None, meth=None, cond2=0.9999):
        isotherm = iso.mdr_temp
        pars = df, temps, compname, isotherm, keyPressures, keyUptakes, guess, logplot, cond, meth, cond2
        typeI = iso.TypeI_fit(*pars) 
        typeI.lang(*pars)
    
    def henry(self, df, compname, temps, keyPressures, keyUptakes, guess=None, logplot=False, cond=True, meth=None):
        isotherm = iso.henry
        print("This fits to Henry's law isotherm.")
        print("To fit to entire dataset, set condition to 0")
        if guess == None:
            print("By default, the minimum R squared value is 0.9999. To change it, set the condition to the desired value")
            print("i.e isotherms.henry(...., keyUptakes, 0.96)")
            tol = 0.9999
        tol = self.guess
        pars = df, compname, keyPressures, keyUptakes, temps, tol, logplot, cond
        henry = iso.Henry(*pars) 
        henry.isotherm_fit(*pars)
        
    def dsl(self, df, compname, temps, keyPressures, keyUptakes, guess, logplot=False, cond=None, meth=None, cond2=0.9999):
        print("This fits to the dual-site Langmuir isotherm model (DSL) based on thermodynamically\n consistent parameters.")
        print("The third fitting procedure described by Farmahini et. al. (2018) was used.\n")
        #print("Please refer to documentation for more information.\n")
        #print("This function can fit more than one dataframe. This is required for binary co-adsorption prediction.")
        #print("Please enter dataframes into a list parameter")
        if meth == None:
            meth = 'tnc'
        if not cond == None:
            params = temps, guess, keyPressures, keyUptakes, compname, df, logplot, cond2, meth, 'extend', cond
        else: 
            params = temps, guess, keyPressures, keyUptakes, compname, df, logplot, cond2, meth

        fitDSL = iso.DSL_fit(*params)
        fitDSL.fit_isotherm(*params)
 #params = temps, guess, keyPressures, keyUptakes, compnames, df_list, True, meth, 'extend', 0.85
#df2 = pd.read_csv('Computational Data (EPFL) N2.csv')
#df = [df1, df2]


# In[ ]:




