from pyIsofit import Models
import pandas as pd

df1 = pd.read_csv('Computational Data (EPFL) N2.csv')
df2 = pd.read_csv('Computational Data (EPFL) CO2.csv')
df_list = [df1, df2]
compname = ['N2', 'CO2']
temps = [10, 40, 100]

#keyUptakes = ['q']
#keyPressures = ['p']

keyUptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
keyPressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']

pars = df1, compname, temps, keyPressures, keyUptakes, guess, False

test1 = Models(*pars)
test1.dsl(*pars)