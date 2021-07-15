"""
Test file
---------------------------

FITTING ANY MODELS FOR A SINGLE DATAFRAME

To fit data first create a Pandas Dataframe variable using pd.read_csv() (or json). The file must contain all datasets
    to be fitted (any number of datasets), with unique keys for each pressure and uptake:
    i.e the column header for the pressure of the first dataset is 'p@0C', for the second dataset it's 'p@20C' ect.
        df = pd.read_csv('my_file.csv')

Next, define a list of integers (or floats) which include the temperatures of each dataset:
        temps = [0, 20, ...]

Next, define the keys for the package to extract the data. This requires a list of strings.
    i.e key_pressures = ['p@0C', 'p@20C', '...']
        key_uptakes = ['q@0C', 'q@20C', '...']

To fit this data to a model, create an instance of the IsothermFit class. This requires the user choosing a model from
the following list:

            ['mdr', 'mdr td', 'langmuir', 'langmuir linear 1', 'langmuir linear 2', 'langmuir td',
           'dsl nc', 'dsl', 'gab', 'sips', 'toth', 'toth td', 'bddt', 'bddt 2n', 'bddt 2n-1',
           'dodo', 'bet']

Notes:
        - 'td' means temperature dependent - i.e equilibrium constant is temperature dependent { b = b0 * exp(-H/RT) }
        - 'toth td' does not currently work (WIP)
        - 'langmuir linear 1' is using the linear form of langmuir with axis 1/q vs. 1/p
        - 'langmuir linear 2' is using the linear form of langmuir with axis p/q vs. p
        - There are two forms of the bddt isotherm, both with the same parameters. One is for an n (layers) of 2n,
         and the other is for a n (layers) of 2n-1. Choosing 'bddt' will fit to the 'bddt 2n' isotherm model.
        - High parameter models may require custom guess values to work further discussed below.
        - Inputting a list of dataframes is only supported for the 'dsl' fitting procedure - further discussed below

my_object = IsothermFit(df, temps, key_pressures, key_uptakes, "langmuir", compname)


"""


from fitting import IsothermFit
import pandas as pd




# df1 = pd.read_csv('SIFSIX-3-Cu CO2.csv')
# df1 = pd.read_csv('../Datasets for testing/Lewatit CO2.csv')
df1 = pd.read_csv('../Datasets for testing/Computational Data (EPFL) CO2.csv')
df2 = pd.read_csv('../Datasets for testing/Computational Data (EPFL) N2.csv')

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



langmuir = IsothermFit(df1, temps, keyPressures, keyUptakes, "dsl", compname)
langmuir.fit(cond=True, meth='tnc')
langmuir.plot(logplot=False)
# langmuir.save()
# langmuir.plot_emod(yfracs=[0.15, 0.85], logplot=False)
