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
