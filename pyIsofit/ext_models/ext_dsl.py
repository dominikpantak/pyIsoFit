import numpy as np
from pyIsofit.core.model_equations import r


def ext_dsl(param_dict, temps, x, comps, yfracs):
    """
    Extended DSL fitting procedure

    :param param_dict: dict
        Dictionary of parameters for co-adsorption calculations

    :param temps: list[float]
        List of temperatures corresponding to each dataset

    :param x: dict[list[list[float]]]
        Dictionary of component datasets' pressure values

    :param comps: list[str]
        List of strings corresponding to the component names

    :param yfracs: list[float]
        List of mole fractions corresponding to each component in the mixture (same order as components)

    :return:
        dictionary of uptake values for each dataset for each component

    Note:
    --------
    First input data is sorted and b1 and b2 are calculated from the heats of adsorption and temperatures.

    Once input data has been sorted numerators of the two terms in the extended dsl equation
    can be worked out. The aim here is to simplify the extended dsl equation into the following form:

    q* = num(siteA) * x / (1 + den(siteA) * x ) + num(siteB) * x / (1 + den(siteB) * x )

    The numerator of site A and B is sorted into a dictionary for its corresponding component.

    The denominators are then calculated - The workaround for this was to
    create a list of the same length as there are the number of temperatures. Within this list
    there are inner lists of the same length as the number of components. This way, we are allocating
    parameters in the following way for example:

    b1 (at 20 C) = 4.2 for component A, 2.4 for component B ... x for component X

    we also do this for b2

    Finally we iterate over the list of lists mentioned previously with the goal of getting:
    (b1A * yA) + (b1B * yB) ... (b1X * yX)

    relating to the denominators of the extended dsl equation:

    1 + (b1A * yA * P) + (b1B * yB * P) = 1 + (b1A * yA + b1B * yB) * P
    """

    def vant_hoff(b0, h, t):
        """
        Inner function for calculating temperature dependent parameter b
        """
        return b0 * np.exp(-h / (r * t))

    def calc_q(x, num_a, num_b, den_a, den_b):
        """

        :param x: list
            pressure values

        :param num_a: float
            numerator for site A

        :param num_b: float
            numerator for site B

        :param den_a: float
            denominator for site A

        :param den_b: float
            denominator for site B

        :return: list
            co-adsorption uptake
        """
        e1 = num_a * x / (1 + den_a * x)
        e2 = num_b * x / (1 + den_b * x)
        return e1 + e2

    params_sorted = {}

    # Step 1 - Calculating b1 and b2 and creating a dictionary of parameter lists for each component
    for i in range(len(comps)):
        params = param_dict[comps[i]]
        x_comp = x[i]
        b1_list_in = []
        b2_list_in = []

        for j in range(len(x_comp)):
            b1_list_in.append(vant_hoff(params['b01'][j], params['h1'][j], temps[j]))
            b2_list_in.append(vant_hoff(params['b02'][j], params['h2'][j], temps[j]))

        params_sorted[comps[i]] = {'q1': params['q1'],
                                   'q2': params['q2'],
                                   'b1': b1_list_in,
                                   'b2': b2_list_in,
                                   'y': yfracs[i]}

    # Step 2- Creating a dictionary of numerators (for site a and b) for each component
    num_dict = {}
    for i in range(len(comps)):
        params = params_sorted[comps[i]]
        num_a_in = []
        num_b_in = []
        for j in range(len(temps)):
            num_a_in.append(params['q1'][j] * params['b1'][j] * params['y'])
            num_b_in.append(params['q2'][j] * params['b2'][j] * params['y'])

        num_dict[comps[i]] = {'a': num_a_in, 'b': num_b_in}

    # Creating a list of b1 and b2 parameters in the form
    # b1_list = [[b1(comp1)(temp1), b1(comp2)(temp1)], [b1(comp1)(temp2), b1(comp2)(temp2)] ... ]
    # The inner list is the same length as the number of components and the outer list is the same length as the
    # number of temperatures.
    b1_list = [[] for _ in range(len(temps))]
    b2_list = [[] for _ in range(len(temps))]
    for i in range(len(temps)):
        for j in range(len(comps)):
            params = params_sorted[comps[j]]
            b1_list[i].append(params['b1'][i])
            b2_list[i].append(params['b2'][i])

    # Calculating denominators and creating a list of denominators
    den_a_list = []
    den_b_list = []
    for i in range(len(temps)):
        b1_in = b1_list[i]
        b2_in = b2_list[i]
        den_a = 0
        den_b = 0
        for j in range(len(b1_in)):
            den_a += b1_in[j] * yfracs[j]
            den_b += b2_in[j] * yfracs[j]
        den_a_list.append(den_a)
        den_b_list.append(den_b)

    # Step 3 - Calculating co-adsorption uptake for each component and creating a dictionary for each component and
    # dataset
    q_dict = {}
    for i in range(len(comps)):
        x_comp = x[i]
        numer = num_dict[comps[i]]
        num_a = numer['a']
        num_b = numer['b']
        q_list = []
        for j in range(len(temps)):
            q = calc_q(np.array(x_comp[j]), num_a[j], num_b[j], den_a_list[j], den_b_list[j])
            q_list.append(q)

        q_dict[comps[i]] = q_list

    return q_dict
