""" Custom errors thrown by the package"""


class pgError(Exception):
    """Base error raised by pyIsoFit-master"""


class ParameterError(pgError):
    """Raised when one of the parameters is unsuitable"""


class SaveError(pgError):
    """Raised when the .save method is called incorrectly"""


class HenryError(pgError):
    """Raised when there is a problem with the henry regime approximation"""


_dsl_error_msg = """
******************************************************************************\n
The model function generated NaN values and the fit aborted!
Please check your model function and/or set boundaries on parameters where applicable
In these cases you can try:\n\n
* Changing lmfit fitting method (default is "leastsq") to "tnc". This is done by
  passing meth="tnc" as a kwarg within .fit()\n
* Inspecting henry regime estimation by passing show_hen=True as a kwarg to check the
  estimate is not off\n
* Because this is often an issue with the henry regime constraint, changing the henry
  regime tolerance by passing hen_tol as a kwarg in the following way may work:
     - A list of single floats representing the tolerance of the R^2 value which pyIsoFit uses to
       calculate the henry regime for each component. The values must be in the same order
       as the component names passed i.e hen_tol=[0.98, 0.99] corresponding to ["CO2", "N2"]
     - A list of lists of floats representing the upper henry regime for each component\'s
       datasets which you are passing i.e hen_tol=[[0.1, 0.2, 0.3], [0.1, 0.5, 0.3]] for ["CO2", "N2"] \n\n
Please restart the fitting after you have made these changes\n
If none of these solutions work, it is possible that this particular dataset cannot be 
fit to the DSL isotherm with the constraints you have chosen
"""