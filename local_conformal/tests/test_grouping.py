import numpy as np
import pandas as pd
import local_conformal as lc

def test_thresholds_per_group():
    "test thresholds_per_group, basic"
    my_df = pd.DataFrame(data = {"grouping": [0,0,1,0,1,0,1,1],
                                 "cs":np.arange(8)})

    threshold_mat1, cs_info1 = lc.thresholds_per_group(my_df,
                                                    desired_props = np.arange(1,20)/20)

    assert threshold_mat1 is None, \
        "if no append is provided then first element returned should be none"



    my_df2 = my_df.copy()
    threshold_mat2, cs_info2 = lc.thresholds_per_group(my_df,
                                                    desired_props = np.arange(1,20)/20,
                                                    append = my_df2)

    assert np.all(cs_info1 == cs_info2), \
        "grouping information should not be effected by append data.frame"

    assert threshold_mat2.shape == (my_df2.shape[0], np.arange(1,20).shape[0]), \
        "threshold matrix (when append is not None) should be (n, n_quatiles)"

    # some static tests:
    ones_vec = np.quantile(np.array([2,4,6,7]), np.arange(1,20)/20,
                            interpolation = "higher")
    zeros_vec = np.quantile(np.array([0,1,3,5]), np.arange(1,20)/20,
                          interpolation = "higher")
    for z_idx in [0,1,3,5]:
        assert np.all(threshold_mat2[z_idx,:] == zeros_vec), \
            "row %i of static threshold_mat2 doesn't meet expectations" % z_idx

    for o_idx in [2,4,6,7]:
        assert np.all(threshold_mat2[o_idx,:] == ones_vec), \
            "row %i of static threshold_mat2 doesn't meet expectations" % o_idx


def test_average_within_groups():
    "test average_within_groups, basic - just structure"
    g_vec =  np.array([0,0,2,0,2,0,2,2])
    g_df = pd.DataFrame(data = {"grouping": g_vec})
    quantiles = np.arange(1,5)/5

    np.random.seed(1)
    info_mat = np.random.uniform(0,1, size = 8*4).reshape((8,-1))

    mi, mipl = lc.average_within_groups(g_vec, info_mat, quantiles)

    assert np.all([x in [0,2] for x in mi.grouping]) and \
        np.all([x in [0,2] for x in mipl.grouping]), \
        "expect only group indices that match the group vector"

    assert mi.shape == (2, 4+1) and \
        np.all(mi.columns == ["grouping"] + [str(q) for q in quantiles]), \
        "expect mean_info df to have shape and columns as in as in doc"

    assert mipl.shape == (2*4,3) and \
        np.all(mipl.columns == ["grouping", "quantile", "means"]), \
        "expected mean_info_pivot_longer df to have shape and columns as in doc"
