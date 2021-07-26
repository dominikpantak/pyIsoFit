"""
Testing file (not unit test)
---------------------------
"""


from fitting import IsothermFit
import pandas as pd




# df1 = pd.read_csv('SIFSIX-3-Cu CO2.csv')
# df1 = pd.read_csv('../Datasets for testing/Lewatit CO2.csv')
df1 = pd.read_csv('../../../tests/Datasets for testing/Computational Data (EPFL) CO2.csv')
df2 = pd.read_csv('../../../tests/Datasets for testing/Computational Data (EPFL) N2.csv')

# df1 = pd.read_csv('../Datasets for testing/data.csv')
# df_list = df1
compname = 'CO2'
# temps = [0, 25, 50]
temps = [10, 40, 100]
# temps = [0, 20]
# temps = [25, 45, 55]

# keyUptakes = ['y1', 'y2', 'y3']
# keyPressures = ['x1', 'x2', 'x3']


keyUptakes = ['Uptake (mmol/g)_13X_10 (°C)', 'Uptake (mmol/g)_13X_40 (°C)', 'Uptake (mmol/g)_13X_100 (°C)']
keyPressures = ['Pressure (bar)', 'Pressure (bar)', 'Pressure (bar)']

# keyUptakes = ['q 298', 'q 318', 'q 328']
# keyPressures = ['p 298', 'p 318', 'p 328']

# keyUptakes = ['Loading [mmol/g] 25', 'Loading [mmol/g] 50', 'Loading [mmol/g] 70']
# keyPressures = ['Pressure [bar] 25', 'Pressure [bar] 50', 'Pressure [bar] 70']

# cust_bounds = {
#     'q': [(4.97, 4.99), (4.81, 4.83), (0, 6)],
#     'b': [(39.5, 39.7), (0, 50), (0, 1000)]
# }

langmuir = IsothermFit(df1, temps, keyPressures, keyUptakes, "langmuir", compname)
langmuir.fit(cond=False, meth='leastsq', show_hen=True, hen_tol=0.999999)
# langmuir.plot(logplot=True)
langmuir.save(save_henry=True)
# langmuir.plot_emod(yfracs=[0.15, 0.85], logplot=False)