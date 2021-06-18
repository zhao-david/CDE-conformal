import numpy as np
import progressbar

from .grouping import average_within_groups

def difference_validity_and_efficiency(true_cde,
                                       predict_grid,
                                       true_grid,
                                       thresholds_predict,
                                       thresholds_true,
                                       df_cs_grouping,
                                       true_grouping = None,
                                       expected_prop = .5,
                                       z_delta = 1,
                                       verbose = True):
    """
    Calculates difference in validity and efficiency between predicted and true
    estimates at an individual level.

    Arguments:
    ----------
    true_cde : numpy array (n, p). CDE estimates across a range of z values
        that are separated by a common `z_delta`. Each row relates to a single
        observation of a CDE function, and the columns of the array should
        correspond to the CDE_i(y_j) evaluation.
    predict_grid : numpy array (n, p). Predicted scores (e.g. HDP, CDE values,
        etc.) across the same range of z values as `true_cde` with same
        row/column structure as well
    true_grid: numpy array (n, p). Same as `predict_grid` but the true scores
        under the oracle (aka knowing `true_cde`)
    thresholds_predict : numpy vector (n, m). Threshold cutoffs associated with the
        predict_grid scores for each row of the above arrays (`true_cde`,
        `predict_grid`, `true_grid`). The columns are associated with difference
        cutoffs.
    thresholds_true : numpy vector (n, m). Threshold cutoffs associated with the
        true_grid scores for each row of the above arrays (`true_cde`,
        `predict_grid`, `true_grid`). The columns are associated with difference
        cutoffs.
    df_cs_grouping : pandas data.frame (n, 2+) with "cs" and "grouping" columns
        **NOTE: this grouping should be relative to the model grouping, not
        true grouping.**
    true_grouping : numpy vector (n, ) vector of discrete grouping values
        relative to how to evaluate preformance per groups. **NOTE: this is
        not the same as the model estimated grouping.** Default of None means
        this part of the analysis is not done.
    expected_prop : numpy vector (m,). Amount of mass expected to be contained
        in each level set defined by columns of `thresholds`
    z_delta : difference between the range of z values used to create above
        arrays (`true_cde`, `predict_grid`, `true_grid`)
    verbose : boolean, logic if we should report progress of analysis

    Returns:
    --------
    Depending upon `true_grouping = None` you will recieve either:

    a tuple with the following info (this is observation based (n) ):
        validity_error : numpy array (n, m). For each row (i) of the above numpy
            matrices, we look at the absolute error between the expected_prop
            verse actual mass in the level set. The columns are ranging across
            the different thresholds and expected_prop values.
        validation_raw_bin_mat : numpy array (n, m). For each row (i) of the
            above numpy matrices, if the observation has a cs scores
            great than or equal to the quantile cutoff for the associated group
            (see `?lc.difference_actual_validity` for more detail)
        efficiency_error : numpy array (n, m). For each row (i) of the above
            numpy matrices, we look at the lebegue measure of the set difference
            between predicted level set and the true level set.

    -OR-

    a tuple with the following info (this is group based (n) ):
        validity_average : pandas DF (n_groups * m, 3), in the pivot_longer
            style of a data frame. For each row, we include the "grouping",
            a "quantile" value, and the "means" (mean) of the absolute error
            between the expected validity and true density contained in the
            estimated level set.
        validity_raw_test_average: pandas DF (n_groups * m, 3), in the
            pivot_longer style of a data frame. For each row, we include the
            "grouping",a "quantile" value, and the "means" (mean) of the
            absolute error between the expected validity and empirical
            containment for the test observations in the given group.
        efficiency_average : pandas DF (n_groups * m, 3), in the pivot_longer
            style of a data frame. For each row, we include the "grouping",
            a "quantile" value, and the "means" (mean) of the lebegue
            measure of the set difference between the optimal true level set
            and the estimated level set.

    Details:
    --------
    This function can do both HDP and CDE style cutoffs (can actually do any
    conformal score function evaluated on the same grid).

    For **validity**: we calculate the discrete probability mass of the
    predicted level set (at each given threshold) related to the expected_prop.

    For **validity_raw**: we calculate the empirical amount of observations
    that are contained in the given level set (binary if observation based,
    mean if group based).

    For **efficiency**: we calculate the lebegue set difference beween the
    predicted level set vs the true level set (as defined with `true_grid`)
    """

    # dimension checks -------

    assert true_cde.shape == predict_grid.shape and \
        true_cde.shape == true_grid.shape, \
        "dimensions of true_cde, predict_grid and true_grid should be the same"

    assert thresholds_predict.shape[1] == expected_prop.shape[0] and \
        thresholds_predict.shape == thresholds_true.shape, \
        "number of columns of thresholds_{predict, true} should be the same "+\
        "as the length of expected_prop"

    assert thresholds_predict.shape[0] == true_cde.shape[0], \
        "thresholds_{predict, true} should have the same number of rows as "+\
        "true_cde"

    if true_grouping is not None:
        assert true_grouping.shape[0] == true_cde.shape[0], \
            "true_grouping should have the same length as the "+\
            "true_cde's number of rows"

    assert df_cs_grouping.shape[0] == true_cde.shape[0] and \
        np.all([name in df_cs_grouping.columns for name in ["cs", "grouping"]]), \
        "df_cs_grouping should have correct numbeer of rows and desirable column names"

    # verbosity design (across thresholds) ---------
    if verbose:
        bar = progressbar.ProgressBar(widgets = [ progressbar.Bar(),
                                              ' (', progressbar.ETA(), ", ",
                                              progressbar.Timer(), ')'])
        threshold_iter = bar(np.arange(thresholds_predict.shape[1]))
    else:
        threshold_iter = range(thresholds_predict.shape[1])


    validity_error = -1 * np.ones((true_cde.shape[0],
                                   thresholds_predict.shape[1]))
    efficiency_error = -1 * np.ones((true_cde.shape[0],
                                     thresholds_predict.shape[1]))

    for t_idx in threshold_iter:
        threshold_predict_vec = thresholds_predict[:,t_idx]

        predict_indicator_ge_threshold = predict_grid >= \
            np.repeat(threshold_predict_vec.reshape((-1,1)),
                      predict_grid.shape[1], axis = 1)


        # oracle validity: with the underlying distribution ---------
        expected_p = expected_prop[t_idx]

        mask_cde_value = true_cde * predict_indicator_ge_threshold
        rowwise_mass = mask_cde_value.sum(axis = 1) * z_delta

        validity_error[:, t_idx] = np.abs(rowwise_mass - expected_p)


        # efficiency: compared to truth optimal set ---------
        threshold_true_vec = thresholds_true[:,t_idx]


        true_indicator_ge_threshold = true_grid >= \
            np.repeat(threshold_true_vec.reshape((-1,1)),
                      predict_grid.shape[1], axis = 1)

        level_set_diff = true_indicator_ge_threshold == predict_indicator_ge_threshold
        rowwise_diff = level_set_diff.sum(axis = 1)
        efficiency_error[:, t_idx] = rowwise_diff * z_delta

    # test validity --------------------

    validation_raw_bin_mat = difference_actual_validity(df_cs_grouping,
                                                        thresholds_predict)


    #
    # combining across true groups ---------
    #

    if true_grouping is  None:
        return validity_error, validation_raw_bin_mat, efficiency_error
    else:
        _, validity_average = average_within_groups(true_grouping,
                                                    validity_error)
        _, validity_raw_test_average = average_within_groups(true_grouping,
                                                             validation_raw_bin_mat)
        _, efficiency_average = average_within_groups(true_grouping,
                                                      efficiency_error)


        return validity_average, validity_raw_test_average, \
            efficiency_average



def difference_actual_validity(df_cs_test, thresholds_mat_test):
    """
    This function should return a binary matrix relative to the groupings
    and whether or not each true cs(y|x) is above the group conformal threshold

    Arguments:
    ----------
    df_cs_test : pandas df  (n, 3) with columns at least "cs"
    thresholds_mat_test : matrix (n, p) cutoffs per row of df_cs_test relative
        to previously defined quantile cutoffs

    Returns:
    --------
    binary_mat : boolean numpy array (n, p) if the observation has a cs scores
        great than or equal to the quantile cutoff for the associated group
    """
    n_quant = thresholds_mat_test.shape[1]
    cs_only = np.array(df_cs_test.cs).reshape((-1,1))

    binary_mat = np.repeat(cs_only,n_quant, axis = 1) >= thresholds_mat_test

    return binary_mat
