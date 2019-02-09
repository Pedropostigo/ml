from itertools import product


def expand_grid(params):
    """
    Function to compute all the combinations of a lists of parameters

    Parameters:
    params -- dict with the parameter names as keys and list of parameter values as values

    Return:
    grid -- a list of dicts containgin the combination of parameters
    """

    # list of parameters
    keys = list(params.keys())

    # get all combinations of parameters
    comb = []
    for param in params:
        comb.append(params[param])

    comb = list(product(*comb))

    # dict to return the combination of parameters
    grid = []

    # reorder the result of the combination of parameters in the grid dictionary
    for i in range(0, len(comb)):
        # create a dictionary to store the parameters
        par = {}
        
        # pupulate the dictionary with the parameters
        for j in range(0, len(keys)):
            par[keys[j]] = comb[i][j]

        # append the parameters to the grid
        grid.append(par)

    return grid