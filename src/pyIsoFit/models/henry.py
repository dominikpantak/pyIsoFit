import numpy as np
import pandas as pd
import logging as logger

from pyIsoFit.core.model_equations import r2hen, henry, bold, unbold
from IPython.display import display


def henry_approx(df, key_pressures, key_uptakes, display_hen=False, tol=0.999, compname="---", henry_off=False):
    """
        Henry approximation function used to estimate henry constants and the henry regime.

    This function works by first adding a (0,0) co-ordinate to each dataset as this is the behaviour exhibited in the
    henry regime. Next, for each temperature dataset, starting with a line made up of three of the first datapoints (
    including the 0,0 point). A gradient (Henry constant) and R squared value of this line is then found through
    linear regression and the package stores it in a variable. It is important to note that the equation that is used
    for linear regression is the Henry isotherm model: q = KH*p.

    This procedure is iterated, with each run adding an additional data point to the line, with the R squared value
    calculated after each fitting. This loop is repeated for the entire dataset. The function stores every calculated
    R2 value as a list, after which the lowest R squared value within this list that is above the minimum tolerance
    value is chosen.

    value. If there are no henry regime lines with an R2 greater than the tolerance value, the henry regime dataset
    corresponding to the maximum R2 value is chosen. The minimum number of datapoints that can be used for the Henry
    regime determination is 3.

    :param df: pd.DataFrame
                Pure-component isotherm data as a pandas dataframe - must be uptake in mmol/g and pressure in bar or
                equivalent. If datasets at different temperatures are required for fitting, the user must specify
                them in the same dataframe.

    :param key_pressures: list[str]
                List of unique column key(s) which correspond to each dataset's pressure values within the
                dataframe. Can input any number of keys corresponding to any number of datasets in the dataframe.
                If multiple dataframes are specified, make sure keys are identical across each dataframe for each
                temperature. Must be inputted in the same order as key_uptakes and temps.

    :param key_uptakes: list[str]
                List of unique column key(s) which correspond to each dataset's uptake values within the
                dataframe. Can input any number of keys corresponding to any number of datasets in the dataframe.
                If multiple dataframes are specified, make sure keys are identical across each dataframe for each
                temperature. Must be inputted in the same order as key_pressures and temps.

    :param display_hen: bool
                Input whether to show the henry regime of the datasets approximated by the package. This is False by
                default.

    :param tol: float or list[float]
                The henry region approximation function calculates the henry region by finding a line with the highest
                R squared value in the low pressure region of the dataset. This is done with a default R squared
                tolerance value (set to 0.999).

                For example, if a float is inputted (a different henry tolerance) this will be the henry tolerance value
                used by the function. i.e if 0.98 is inputted the henry regime will be across a large pressure range
                due to the low tolerance for the R squared value of the henry model fitting.

                This function also supports inputting the henry regimes manually. For this, input each henry regime for
                each dataset as a list i.e [1.2, 2.1, ... ]

    :param compname: str or list[str]
                Name of pure component(s) for results formatting.

    :param henry_off : bool
                Input whether to turn off the henry regime fitting constraint when using the standardised fitting
                constraint to langmuir or dsl - this is usually done when fitting experimental data which has a messy
                low pressure region. Default is False.

    :return: tuple
                Returns list of henry constant for each dataset [0], resulting DataFrame object [1], the x and y
                co-ordinates in the form of a dictionary for each dataset [2]
    """
    if henry_off:
        return None, None, None
    # This section finds the henry region of the datasets
    x = []
    y = []
    for i in range(len(key_pressures)):
        # Reads data from dataframe with respect to keys and adds a 0,0 point
        xi = np.insert(np.array(df[key_pressures[i]].values), 0, 0)
        yi = np.insert(np.array(df[key_uptakes[i]].values), 0, 0)
        x.append(xi)
        y.append(yi)
        del xi
        del yi

    henry_constants = []
    henry_limits = []
    henry_rsq = []
    henry_len = []
    err_lowpoints = []
    err_highrsq = []
    henidx_lst = []
    i = 0

    # Iterate for all datasets at each temperature
    for dataset in y:
        rsq = 1
        x_i = x[i]
        x_henry = [x_i[0], x_i[1]]  # Starting with a minimum of three datapoints
        j = 2
        # Create inner empty lists
        # rsq_ilst stores all calculated r squared values within the inner loops
        # hen_ilst stores all calculated henry constants within the inner loops
        rsq_ilst = []
        hen_ilst = []
        # This loop adds data points while the points correspond to a henry fit with an R^2 of above 0.9999
        if type(tol) == list:
            # Procedure when a list of henry regime limits are passed
            # Iterate over each datapoint within dataset for a given temperature
            while x_i[j] < tol[i] and j < len(x_i) - 3:
                # Add datapoint
                x_henry.append(x_i[j])
                # Create y axis - same length as x axis
                y_henry = dataset[:len(x_henry)]
                # Calculate henry constant
                hen = y_henry[-1] / x_henry[-1]
                # Find r squared value of fit
                rsq = round(r2hen(x_henry, y_henry, henry, hen), 5)
                # Append results to inner lists
                rsq_ilst.append(rsq)
                hen_ilst.append(hen)
                j += 1

            rsqidx = len(hen_ilst) - 1

        else:
            # Procedure when an r squared tolerance is passed
            while rsq > 0 and j < len(x_i):
                # Within the loop - same procedure as in the above condition
                x_henry.append(x_i[j])
                y_henry = dataset[:len(x_henry)]

                hen = y_henry[-1] / x_henry[-1]
                rsq = round(r2hen(x_henry, y_henry, henry, hen), 5)  # r squared calc.
                rsq_ilst.append(rsq)
                hen_ilst.append(hen)
                j += 1

            # Create empty lists
            # abtol stores list of r squared values above the tolerance value
            # itol stores a list of the index values within rsq_ilst
            abtol = []
            itol = []
            i2 = 0
            for rsq in rsq_ilst:
                if rsq > tol:
                    abtol.append(rsq)
                    itol.append(i2)
                i2 += 1

            # Check for an empty list of r squared values above the tolerance value - this can happen if theres a poor
            # Henry regime fitting or if an r squared tolerance is given which is too high
            if abtol == []:
                # In this case the line with the highest r squared value is chosen
                maxrsq = max(rsq_ilst)
                rsqidx = rsq_ilst.index(maxrsq)
                err_highrsq.append(str(i))

            else:
                # Choose the minimum value from the above tolerance r squared list
                rsqfin = min(abtol)
                minidx = abtol.index(rsqfin)
                rsqidx = itol[minidx]

        # +2 to compensate for the initial datapoints
        henidx = rsqidx + 2
        henidx_lst.append(henidx)

        # try:
        henry_len.append(henidx+1)
        # Saving Henry region parameters to later display
        henry_constants.append(hen_ilst[rsqidx])
        henry_limits.append(x_henry[henidx])
        henry_rsq.append(rsq_ilst[rsqidx])
            # sometimes data may not have a good henry region fit, which could abort the above while loop after the
            # first iteration. This piece of code warns the user of this
        # except IndexError:
        #     logger.error("ERROR - Increase henry region value of index " + str(i))

        # Record the index of the dataset where a henry regime is made up of less than 4 points
        if henidx < 3:
            err_lowpoints.append(str(i))

        i += 1

    # Create resulting henry regime datasets and make dictionary
    x_result = [x[i][:henidx_lst[i]+1] for i in range(len(x))]
    y_result = [y[i][:henidx_lst[i]+1] for i in range(len(y))]
    xy_dict = {"x": x_result, "y": y_result}

    # Creating dataframe for henry constants
    df_henry = pd.DataFrame(list(zip(henry_constants, henry_limits, henry_len, henry_rsq)),
                            columns=['Henry constant (mmol/(bar.g))',
                                     'Upper limit (bar)', 'datapoints', 'R squared'])

    if display_hen:
        print(bold + '\nHenry regime for component ' + compname + ':' + unbold)
        display(pd.DataFrame(df_henry))

    # Print a warning if a dataset contains less than 4 points within the henry regime
    if err_lowpoints and display_hen:
        logger.warning(
            unbold + f'WARNING for Dataset {", ".join(err_lowpoints)}: Datasets were found to be made up of less '
                     f'than 4 points.\n '
                     f'Henry region tolerance may be entered as a kwarg - either as a float (default = 0.999) '
                     f'or a list of floats specifying the upper limit for the henry regime for each dataset.\n')

    if err_highrsq and display_hen:
        logger.warning(
            f"WARNING for Dataset {', '.join(err_highrsq)}: Could not find a line with an r squared of above {tol}. "
            f"Creating a line with the highest r squared value. Consider adding henry regime manually by passing"
            f"hen_tol as a list of floats corresponding to the henry regimes of each dataset i.e hen_tol = [0.1, ...,]"
            f"\n")

    return henry_constants, df_henry, xy_dict