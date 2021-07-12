import numpy as np
from pyIsofit.core.modelEquations import r

def ext_dsl(param_dict, temps, x, comps, yfracs):
    if len(comps) < 2:
        print("Enter 2 components or more to use extended models")
        return None

    def vant_hoff(b0, h, t):
        return b0 * np.exp(-h / (r * t))

    def calc_q(x, num_a, num_b, den_a, den_b):
        e1 = num_a * x / (1 + den_a * x)
        e2 = num_b * x / (1 + den_b * x)
        return e1 + e2

    params_sorted = {}
    """ First we sort the input data and calculate b1 and b2 from the heats of 
    adsorption and temperatures"""
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

    """Input data has now been sorted and now the numerators of the two terms in the extended dsl equation 
    can be worked out. The aim here is to simplify the extended dsl equation into the following form:

    q* = num(siteA) * x / (1 + den(siteA) * x ) + num(siteB) * x / (1 + den(siteB) * x )

    """

    num_dict = {}
    for i in range(len(comps)):
        params = params_sorted[comps[i]]
        num_a_in = []
        num_b_in = []
        for j in range(len(temps)):
            num_a_in.append(params['q1'][j] * params['b1'][j] * params['y'])
            num_b_in.append(params['q2'][j] * params['b2'][j] * params['y'])

        num_dict[comps[i]] = {'a': num_a_in, 'b': num_b_in}

    """ The above sorts the numerator of site A and B into a dictionary for its corresponding
    component.

    The next part is calculating the denominators - this is a bit more tricky as we need to add
    more terms depending on the number of components inputted. The workaround for this was to
    create a list of the same length as there are the number of temperatures. Within this list
    there are inner lists of the same length as the number of components. This way, we are allocating
    parameters in the following way for example:

    b1 (at 20 C) = 4.2 for component A, 2.4 for component B ... x for component X 

    we also do this for b2
    """

    b1_list = [[] for _ in range(len(temps))]
    b2_list = [[] for _ in range(len(temps))]
    for i in range(len(temps)):
        for j in range(len(comps)):
            params = params_sorted[comps[j]]
            b1_list[i].append(params['b1'][i])
            b2_list[i].append(params['b2'][i])

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

    """Above we are iterating over the list of lists mentioned previously with the goal of getting:
    (b1A * yA) + (b1B * yB) ... (b1X * yX)

    relating to the denominators of the extended dsl equation:

    1 + (b1A * yA * P) + (b1B * yB * P) = 1 + (b1A * yA + b1B * yB) * P

    """

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