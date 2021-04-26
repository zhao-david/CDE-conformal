import numpy as np


def find_interval(grid, value):
    """
    find index that value falls on the 1d grid

    @param grid: an ordered numpy vector of values
    @param value: scalar which we are looking to find where it fits.
    @return index of grid which value is within [a,b) style.
    """
    if (grid[0] > value) | \
       (grid[grid.shape[0]-1] < value):
        return np.nan

    idx = np.sum(grid <= value) - 1

    return idx

def inner_hpd_value_level(cdes, z_grid, z_test, z_delta, order = None):
    """
    inner function to caculate hpd values

    @param cdes numpy vector (m,) grid of cde values (across potential z_grid)
    # for each z value
    @param z_grid numpy vector (m,) on which the cdes is defined
    @param z_test scalar z value we wish to estiamted it's HPD value.
    @z_delta distance between z_grid values (assumed constant)
    @param order  tells us the ordering the cdes (if not provided) the order
    is calculated as the index ordering the z_grid this values of cdes smallest
    to largest
    @returns HPD value for z_test
    """

    z_idx = find_interval(z_grid, z_test)

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
    Calculates 'coverage' based upon the HPD
    @param cdes: a numpy array of conditional density estimates;
        each row corresponds to an observation, each column corresponds to a grid
        point
    @param z_grid: a numpy array of the grid points at which cde_estimates is evaluated
    @param z_test: a numpy array of the true z values corresponding to the rows of cde_estimates
    @returns A numpy array of values
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
