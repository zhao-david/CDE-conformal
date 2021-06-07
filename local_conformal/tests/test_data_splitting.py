import numpy as np
import pandas as pd
import local_conformal as lc

def test_stratified_data_splitting():
    """test stratified data splitting, basic"""
    data = pd.DataFrame(data = {"x": np.arange(120),
                               "group": [1]*40+[2]*40+[3]*40})
    for seed in [1,3]:
        np.random.seed(seed)
        out1 = lc.stratified_data_splitting(data, group_col = "group",
                                         prop_vec = np.array([.5,.5]))
        if seed == 1:
            assert out1[0].shape == (59,2) and out1[1].shape == (61,2), \
                "sizes should be fixed around .5/.5"
        if seed == 3:
            assert out1[0].shape == (60,2) and out1[1].shape == (60,2), \
                "sizes should be fixed around .5/.5"

        x_vals = np.array(out1[0].append(out1[1], ignore_index = True).x)
        x_vals.sort()
        assert np.all(x_vals == np.arange(120)), \
            "all data should be included 1 time between either of the " +\
            "sub data frames"
