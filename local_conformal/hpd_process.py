import numpy as np


def _find_interval(grid, value, le = True):
    """
    find index that value falls on the 1d grid

    Arguments:
    ----------
    grid: an ordered numpy vector of values
    value: scalar which we are looking to find where it fits.
    le : boolean. if true, then grid is of values [a,b) style, if false,
        is (a,b] style

    Returns:
    --------
        index of grid which value is within specified bins
    """

    if (grid[0] > value) | \
       (grid[grid.shape[0]-1] < value):
        return np.nan

    if le:
        idx = np.sum(grid <= value) - 1
    else:
        idx = np.sum(grid < value)

    return idx

def _find_interval_bins(grid, value):
    """
    find which interval a value is in relative to a grid, grid input is the
    center of each bins (with width the difference between points)

    Arguments:
    ----------
    grid: an ordered numpy vector of values (equally spaced points)
    values: numpy array which we are looking to find where they fits.

    Returns:
    --------
        index of grid which each value is within [a-delta,a+delta) style,
            where delta is the space between each grid point
    """

    grid_delta = grid[1] - grid[0]
    inner_grid = np.array(list(grid.ravel()) + \
                          [grid[-1] + grid_delta]) - grid_delta/2

    if (inner_grid[0] > value) | \
       (inner_grid[-1] < value):
        return np.nan

    idx = np.sum(inner_grid <= value) - 1

    return idx





def find_interval_bins(grid, values):
    """
    vectorized version of _find_interval_bins (find index that value falls
    on the 1d grid as centers of bins, for equally spaced grid)

    Arguments:
    ----------
    grid: an ordered numpy vector of values (equally spaced points)
    values: numpy array which we are looking to find where they fits.

    Returns:
    --------
        index of grid which each value is within [a-delta,a+delta) style,
            where delta is the space between each grid point
    """
    out = [_find_interval_bins(grid, value) for value in values]
    return out



def find_interval(grid, values, le = True):
    """
    vectorized version of _find_interval (find index that value falls
    on the 1d grid)

    Arguments:
    ----------
    grid: an ordered numpy vector of values
    values: numpy array which we are looking to find where they fits.
    le : boolean. if true, then grid is of values [a,b) style, if false,
        is (a,b] style

    Returns:
    --------
        index of grid which each value is within [a,b) style.
    """
    out = [_find_interval(grid, value, le = le) for value in values]
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

    z_idx = _find_interval_bins(z_grid, z_test)

    if np.isnan(z_idx):
        return 0
    v2 = cdes.copy()

    if order is None:
        order = v2.argsort()

    v2_index = (order == z_idx).argmax() # I don't think this is right!!!

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
    cdes: a numpy array (n, m) of conditional density estimates;
        each row corresponds to an observation, each column corresponds to a
        grid point
    z_grid: a numpy vector (m,) of the grid points at which cde_estimates is
        evaluated
    z_test: a numpy vector (n, ) of the true z values corresponding to the
        rows of cde_estimates

    Returns:
    --------
        A numpy array of HDP values for each point
    """
    nrow_cde, ncol_cde = cdes.shape
    n_samples = z_test.shape[0]
    n_grid_points = z_grid.shape[0]

    if nrow_cde != n_samples:
        raise ValueError("Number of samples in CDEs should be the same as in z_test. "
                         "Currently %s and %s." % (nrow_cde, n_samples))
    if ncol_cde != n_grid_points:
        raise ValueError("Number of grid points in CDEs should be the same as in z_grid. "
                         "Currently %s and %s." % (nrow_cde, n_grid_points))

    z_min = np.min(z_grid)
    z_max = np.max(z_grid)
    z_delta = (z_max - z_min)/(n_grid_points-1)

    vals = [inner_hpd_value_level(cdes[ii, ], z_grid, z_test[ii], z_delta,
                                 order)
            for ii in range(n_samples)]
    return np.array(vals)

def hpd_grid(cde_mat, z_delta):
    """
    calculate the hpd grid for a vector of cdes and a z_delta

    Arguments:
    ----------
    cde_mat: numpy vector (n,m) grid of cde values (across potential z_grid)
        for each z value
    z_delta: distance between z_grid values (assumed constant)

    Returns:
    --------
        numpy vector (n ,m) of HPD values
    """
    out = -1*np.ones(cde_mat.shape)
    for r_idx in range(out.shape[0]):
        out[r_idx,:]  = _hpd_grid(cdes = cde_mat[r_idx,:].ravel(),
                                  z_delta = z_delta)

    return out

def _hpd_grid(cdes, z_delta):
    """
    calculate the hpd grid for a vector of cdes and a z_delta

    Arguments:
    ----------
    cdes: numpy vector (m,) grid of cde values (across potential z_grid)
        for each z value
    z_delta: distance between z_grid values (assumed constant)

    Returns:
    --------
        numpy vector (m, ) of HPD values
    """
    v2 = cdes.copy()

    order = v2.argsort()

    v2 = v2[order]
    v2s = (v2.cumsum())*z_delta

    reorder = np.array([(order == i).argmax() for i in np.arange(cdes.shape[0])])

    return v2s[reorder]


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



def true_thresholds_out(true_cde, z_delta, expected_prop):
    """
    calculates thresholds for the cde to get desired HPD value

    Arguments:
    ----------
    true_cde: numpy array (n, d) array of cde values for a range of y values
        conditional on a x value (the row value)
    z_delta : float, space between y values (assumed equally spaced y values)
    expected_prop : numpy vector (p, ) of proportion of mass values desired
        to be contained

    Returns:
    --------
        threshold_mat : numpy array (n, p). For each row, we have the cde
            thresholds that would allow expected_prop[j] mass to be contained
            above this amount
    """

    n_row = true_cde.shape[0]

    threshold_mat = -1 * np.ones((n_row, expected_prop.shape[0]))

    for r_idx in np.arange(n_row):
        threshold_mat[r_idx,:] = _true_thresholds_out(cdes = true_cde[r_idx,:].ravel(),
                                                      z_delta = z_delta,
                                                      expected_prop = expected_prop)


    return threshold_mat

def _true_thresholds_out(cdes, z_delta, expected_prop):
    """
    calculates thresholds for the cde to get desired HPD value

    Arguments:
    ----------
    cdes: numpy vector (d, ) array of cde values for a range of y values
        conditional on a x value (the row value)
    z_delta : float, space between y values (assumed equally spaced y values)
    expected_prop : numpy vector (p, ) of proportion of mass values desired
        to be contained

    Returns:
    --------
        threshold_vec : numpy vector (p, ). For each row, we have the cde
            thresholds that would allow expected_prop[j] mass to be contained
            above this amount
    """
    v2 = cdes.copy()
    order = v2.argsort()[::-1]
    v2 = v2[order]
    v2s = ((v2.cumsum())*z_delta) #down the mass

    order_mass = find_interval(v2s, expected_prop, le = False)

    threshold_vec = np.zeros(len(order_mass))

    above = np.logical_and(np.isnan(order_mass), expected_prop > v2s.max())


    for i in np.arange(len(order_mass)):
        if not np.isnan(order_mass[i]):
            threshold_vec[i] = v2[order_mass[i]]
        elif above[i]:
            threshold_vec[i] = 0
        else: # below
            threshold_vec[i] = v2.max()
            #^this was done preserve structure if z_delta isn't a perfect input

    return threshold_vec
