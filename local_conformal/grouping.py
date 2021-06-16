import numpy as np
import pandas as pd

def thresholds_per_group(g_df, desired_props, append = None):
    """
    Calculates thresholds of conformal scores per group

    Arguments:
    ----------
    g_df: pandas data frame (n, p). Each row is an observation. Expected to have
        columns "grouping" and "cs"
    desired_props : numpy vector (q, ). A vector of proportion values between
        0 and 1
    append : pandas data.frame (m, p2) with a "grouping" column that has similar
        structure to `g_df`. If None then output changes (see returns).
        Default is None.

    Returns:
    --------
    tuple of
        threshold_mat: numpy array (m, q). Each row corresponds to a row of
            `append` (if `append` is None, then so is this). The columns capture
            the quantile of the "cs" values with the same group (we use
            `interprolation = "higher"` as seen in `np.quatile`)
        cs_info : pandas data frame (n_groups, q+1) data fram with a "grouping"
            column, and the rest of the columns correspond to desired_props.
            Data contained is per "grouping" value, and captures the
            associated desired_props[j+1] quantiles of the "cs" scores - per
            group.
    """

    cs_breaks = g_df[["grouping", "cs"]].groupby("grouping").quantile(
                                        q = desired_props,
                                        interpolation ="higher").reset_index()

    cs_info = cs_breaks.pivot(index = "grouping",
           values = "cs", columns = "level_1").reset_index()

    if append is None:
        return (None, cs_info)

    ljoined = append[["grouping"]].join(cs_info.set_index("grouping"),
                                        on = "grouping", how = "left")

    threshold_mat = np.array(ljoined.drop(columns = "grouping"))

    return (threshold_mat, cs_info)


def average_within_groups(group_vec, info_mat, quantiles=None):
    """
    utilization function to take the average value for each column conditional
    on a group vector

    ********also remember we want to average across TRUE GROUPS! *******

    Arguments:
    ----------
    group_vec : numpy vector (n, ) probably integers / object based to indicate
        discrete classes
    info_mat : numpy array (n, p) rows correspond to group_vec's entries
    quantiles : numpy array (p, ) associated quantiles for each column in
        info_mat (if None converted, a integer vector is used)

    Returns:
    --------
    tuple of
        mean_info : pandas data frame (n_groups, p+1) of means per column and
            a groups column
        mean_info_pivot_longer : pandas data frame (n_groups*p, 3) each row
            tells us the group number, the quantile value and the mean value
    """

    if quantiles is None:
        quantiles = np.arange(info_mat.shape[1], dtype = int)

    col_names = np.array([str(q) for q in quantiles])

    g_df = pd.DataFrame(data = {"grouping": group_vec})
    info_df = pd.DataFrame(info_mat, columns = col_names)
    df_combined = pd.concat([g_df.reset_index(drop = True),
                             info_df.reset_index(drop = True)],
                           axis = 1)

    mean_info = df_combined.groupby("grouping").mean().reset_index()

    mean_info_pivot_longer = mean_info.melt(id_vars = "grouping",
                                            value_vars = col_names,
                                            var_name = "quantile",
                                            value_name = "means")

    return mean_info, mean_info_pivot_longer
