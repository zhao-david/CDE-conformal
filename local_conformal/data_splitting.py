import numpy as np
import pandas as pd

def stratified_data_splitting(data, group_col = "group_info",
                              prop_vec = np.array([1/3,1/3,0,0,1/3])):
    """
    splits pd.DataFrame into any number of subsets

    Arguments:
    ----------
    data : pd.DataFrame to split
    group_col : string (or list) of column(s) to group by
    prop_vec : np vector of proportions for each new data split.

    Returns:
    --------
        a list of data frames with disjoint splits of the data based on
        desired proportions.
    """

    assert np.sum(prop_vec) == 1 and np.all(prop_vec >= 0), \
        "prop_vec should be a proper proportion vector (non-neg and sum to 1)"

    list_group_info = [group for _, group in data.groupby(group_col)]

    n_out = prop_vec.shape[0]
    n_groups = len(list_group_info)
    list_data = []
    for group_idx in np.arange(n_groups, dtype = np.int):
        inner_nrow = list_group_info[group_idx].shape[0]
        inner_row_vec = np.random.choice(inner_nrow, size = inner_nrow,
                                         replace = False)
        #ipdb.set_trace()
        inner_group_vec = np.random.choice(n_out, size = inner_nrow,
                                           p = prop_vec)

        for split_idx in np.arange(n_out, dtype = np.int):
            #ipdb.set_trace()
            current_data = list_group_info[group_idx].iloc[
                np.array(inner_row_vec[inner_group_vec == split_idx],
                         dtype = np.int)]
            if group_idx == 0:
                list_data.append(current_data)
            else:
                list_data[split_idx] = list_data[split_idx].append(current_data,
                                                                   ignore_index = True)


    return list_data
