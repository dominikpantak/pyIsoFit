from FitPackage import IsothermFit
import pandas as pd

df2 = pd.read_csv('Computational Data (EPFL) N2.csv')
compname = 'N2'
temps = [10, 40, 100]
meth = 'tnc'
keyUptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
keyPressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']

# keyUptakes = ['Loading [mmol/g] 25', 'Loading [mmol/g] 50', 'Loading [mmol/g] 70']
# keyPressures = ['Pressure [bar] 25', 'Pressure [bar] 50', 'Pressure [bar] 70']

tolerance = 0.9999  # set minimum r squared value

langmuir = IsothermFit(df2, compname, temps, keyPressures, keyUptakes, "Langmuir")
langmuir.fit(False, True)
langmuir.plot(True)
