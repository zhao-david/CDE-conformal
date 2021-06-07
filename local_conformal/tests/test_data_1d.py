import numpy as np
import pandas as pd
import local_conformal as lc
import matplotlib.pyplot as plt

def test_my_bimodal():
    """test my_bimodal basic"""

    np.random.seed(1)
    data = lc.my_bimodal(10000)

    assert np.abs(np.mean(data)-1) < .02 and \
        np.abs(np.median(data)-1) < .02, \
        "expected distribution to have mean and median 1"

    assert np.abs(np.std(data)-1) < .02, \
        "expected distribution to have sd 1"

    # weak approach to check that there are too modes
    counts, bins, patches = plt.hist(data, bins = 20)

    diffs = np.diff(counts)

    peak_switch = np.zeros(diffs.shape[0]-1)
    for idx in np.arange(diffs.shape[0]-1):
        peak_switch[idx] = diffs[idx] >= 0 and diffs[idx+1] < 0

    assert np.sum(peak_switch) == 2, \
        "should get exactly 2 peaks from the bimodal distribution"


def test_data_generation():
    """test data_generate, basic"""

    np.random.seed(2)
    # single sigma size
    data_1 = lc.data_generation(10000, sigma_num = 1)

    assert data_1.shape[0] == 10000*4, \
        "expected number of points should be n*4 when sigma_num == 1."

    means = data_1.groupby("group_info", as_index=False).mean()

    assert np.all(np.abs(means.y -1) < .02), \
        "each group's mean should be close to 1"

    assert np.all(np.abs(means.x -means.group_info) < .02), \
        "each group's x values mean should be close to to their group number"

    stds = data_1.groupby("group_info", as_index=False).std()

    assert np.all(np.abs(stds.y-1) < .02), \
        "each groups sd should be close to 1"

    # multiple sigma size
    data_4 = lc.data_generation(10000, sigma_num = 4)


    assert data_4.shape[0] == 10000*4*4, \
        "expected number of points should be n*4 when sigma_num == 4."

    means4 = data_4.groupby("group_info", as_index=False).mean()

    assert np.all(np.abs(means4.x -means4.group_info) < .02), \
        "each group's x values mean should be close to to their group number"

    means4["sigma"] = 4**np.repeat(np.arange(4),4)

    assert np.all(np.abs(means4.y -means4.sigma)/ means4.sigma < .02), \
        "each group's mean should be close to sigma value (4^sigma grouping)"

    stds4 = data_4.groupby("group_info", as_index=False).std()
    stds4["sigma"] = 4**np.repeat(np.arange(4),4)


    assert np.all(np.abs(stds4.y-stds4.sigma)/ stds4.sigma < .02), \
        "each groups sd should be close to sigma value (4^sigma grouping)"
