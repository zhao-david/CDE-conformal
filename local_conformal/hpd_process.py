import numpy as np


def _find_interval(grid, value):
    """
    find index that value falls on the 1d grid

    Arguments:
    ----------
    grid: an ordered numpy vector of values
    value: scalar which we are looking to find where it fits.

    Returns:
    --------
        index of grid which value is within [a,b) style.
    """
    if (grid[0] > value) | \
       (grid[grid.shape[0]-1] < value):
        return np.nan

    idx = np.sum(grid <= value) - 1

    return idx

def find_interval(grid, values):
    """
    vectorized version of _find_interval (find index that value falls
    on the 1d grid)

    Arguments:
    ----------
    grid: an ordered numpy vector of values
    values: numpy array which we are looking to find where they fits.

    Returns:
    --------
        index of grid which each value is within [a,b) style.
    """
    out = [_find_interval(grid, value) for value in values]
    return out

def inner_hpd_value_level(cdes, z_grid, z_test, z_delta, order = None):
    """
    inner function to caculate hpd values

    Arguments:
    ----------
    cdes: numpy vector (m,) grid of cde values (across potential z_grid)
    # for each z value
    z_grid: numpy vector (m,) on which the cdes is defined
    z_test: scalar z value we wish to estiamted it's HPD value.
    z_delta: distance between z_grid values (assumed constant)
    order:  tells us the ordering the cdes (if not provided) the order
    is calculated as the index ordering the z_grid this values of cdes smallest
    to largest

    Returns:
    --------
        HPD value for z_test
    """

    z_idx = _find_interval(z_grid, z_test)

    if np.isnan(z_idx):
        return 0
    v2 = cdes.copy()

    if order is None:
        order = v2.argsort()

    v2_index = (order == z_idx).argmax()

    v2 = v2[order]
    v2s = ((v2.cumsum())*z_delta)
    v2si = v2s[v2_index]
    return v2si

def hpd_coverage(cdes, z_grid, z_test, order = None):
    """
    Calculates 'coverage' based upon the HPD (the HPD value for any
    particular point on the grid).

    Arguments:
    ----------
    cdes: a numpy array of conditional density estimates;
        each row corresponds to an observation, each column corresponds to a grid
        point
    z_grid: a numpy array of the grid points at which cde_estimates is evaluated
    z_test: a numpy array of the true z values corresponding to the rows of cde_estimates

    Returns:
    --------
        A numpy array of HDP values for each point
    """
    nrow_cde, ncol_cde = cdes.shape
    n_samples = z_test.shape[0]
    n_grid_points = z_grid.shape[0]

    if nrow_cde != n_samples:
        raise ValueError("Number of samples in CDEs should be the same as in z_test."
                         "Currently %s and %s." % (nrow_cde, n_samples))
    if ncol_cde != n_grid_points:
        raise ValueError("Number of grid points in CDEs should be the same as in z_grid."
                         "Currently %s and %s." % (nrow_cde, n_grid_points))

    z_min = np.min(z_grid)
    z_max = np.max(z_grid)
    z_delta = (z_max - z_min)/(n_grid_points-1)

    vals = [inner_hpd_value_level(cdes[ii, ], z_grid, z_test[ii], z_delta,
                                 order)
            for ii in range(n_samples)]
    return np.array(vals)



def _profile_density(cdes, t_grid, z_delta):
    """
    create a vector with the amount of mass above a given threshold

    Arguments:
    ----------
    cdes : numpy vector (n,) of cde values associated with a set of z_grid
        values that are spaced z_delta apart.
    t_grid : numpy vector (T,) of thresholds we will be using for cumulative
        mass contained under (assumed to be ordered).
    z_delta : distance between z_grid values that created the cdes vector


    Returns:
    --------
        numpy vector (T, ) of cumulative mass of CDE based on t_grid thresholds

    Details:
    --------
    This function is a python version of the profile_density function from the
        R package predictionBands (https://github.com/rizbicki/predictionBands/)


    """

    v2 = cdes.copy()
    v2.sort()
    v2s = np.cumsum(v2)*z_delta

    indices = find_interval(v2, t_grid) # this is ok since v2 is already sorted
    above = np.logical_and(np.isnan(indices),t_grid > v2.max())

    v2s_out = np.zeros(len(indices))
    for i in np.arange(len(indices)):
        if not np.isnan(indices[i]):
            v2s_out[i] = v2s[indices[i]]
        elif above[i]:
            v2s_out[i] = v2s.max() # not 1
            #^this was done preserve structure if z_delta isn't a perfect input
        else: # below
            v2s_out[i] = 0

    return v2s_out

def profile_density(cde_mat, t_grid, z_delta):
    """
    For each cde vector (row of cde_mat), produce the amount of mass under
    thresholds in t_grid

    Arguments:
    ----------
    cdes : numpy array (n, m). Each row contains a set of cde values associated
        with a set of (m) z_grid values that are spaced z_delta apart.
    t_grid : numpy vector (T,) of thresholds we will be using for cumulative
        mass contained under (assumed to be ordered).
    z_delta : distance between z_grid values that created the cdes vector


    Returns:
    --------
        numpy array (n, T) of cumulative mass of CDE based on t_grid thresholds

    Details:
    --------
    This function is a python version of the profile_density function from the
        R package predictionBands (https://github.com/rizbicki/predictionBands/)

    """

    nrow = cde_mat.shape[0]
    nT = t_grid.shape[0]

    out_mat = np.zeros((nrow, nT))
    for i in np.arange(nrow):
        out_mat[i,:] = _profile_density(cde_mat[i,:], t_grid, z_delta)

    return out_mat
