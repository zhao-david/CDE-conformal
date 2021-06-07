import numpy as np
import pandas as pd

def my_bimodal(n):
    """
    defines a bimodal distribution (with mean 1, overall sd = 1, and pi = .5)

    Arguments:
    ----------
    n : int number of samples to draw

    Returns:
    --------
    vector of random samples
    """
    mean_shift = .9
    inner_sd = np.sqrt(.2)

    group_idx = np.random.binomial(n = 1, p = 0.5, size = n)
    value = group_idx*np.random.normal(1-mean_shift, inner_sd, size = n) +\
        (1-group_idx)*np.random.normal(1+mean_shift, inner_sd, size = n)
    return value


def data_generation(n, sigma_num = 4):
    """
    This function generates x,y and group values

    xvals = noisy group numbers.

    Where the overall y distribution is distributed:
    (1) N(1,1),
    (2) Unif(1-sqrt(3), 1+sqrt(3))
    (3) Exp(1)
    or (4) 0.5N(0.1,var = 0.2) + 0.5N(1.9, var = 0.2),

    and we scale the sigmas and means by 4^(0:sigma_num)

    Arguments:
    ----------
    n : int, number of y values for each group
    sigma_num : int, number of sigma groups

    Returns:
    -------
    y_vec : numpy vector of length n*sigma_num*4, where
    n*(i*j-1):n*(i*j) values are associated with the ith overall distribution,
    with the jth sigma value scaling.
    """
    sigma_values = 4**np.arange(sigma_num)
    num_groups = sigma_num*4


    # x and group values:
    group_list = []
    for idx in np.arange(num_groups, dtype = int):
        group_list += [idx]*n

    group_info = np.array(group_list)
    add_noise = np.random.uniform(low = -.45, high = .45, size = num_groups*n)

    x = group_info + add_noise



    # y values

    y_generate_base = [lambda n : np.random.normal(1,1, size = n),
                   lambda n : np.random.uniform(1 - np.sqrt(3),
                                                1 + np.sqrt(3),
                                                size = n),
                   lambda n : np.random.exponential(1, size = n),
                   my_bimodal]

    y_generate_all = []
    for sigma_idx in np.arange(sigma_num , dtype = int):
        current_sigma = sigma_values[sigma_idx]
        for f_idx in np.arange(4, dtype = int):

            y_generate_all += [lambda n : current_sigma.copy() * y_generate_base.copy()[f_idx](n)]


    y_list = []
    for sigma_idx in np.arange(sigma_num , dtype = int):
        current_sigma = sigma_values[sigma_idx]
        for f_idx in np.arange(4, dtype = int):
            current_function = lambda n : current_sigma * y_generate_base[f_idx](n)
            y_list += list(current_function(n))
    y = np.array(y_list)

    data_all = pd.DataFrame(data = {"x" : x, "y" : y,
                                "group_info" : group_info})

    return(data_all)
