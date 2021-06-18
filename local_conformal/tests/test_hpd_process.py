import numpy as np

import local_conformal as lc

def test_find_interval():
    """test find_interval, basic"""

    grid = np.arange(5,10)
    value_na = 4
    value_mid = 7
    value_na2 = 11
    out1 = lc.hpd_process._find_interval(grid, value_na)
    out2 = lc.hpd_process._find_interval(grid, value_mid)
    out3 = lc.hpd_process._find_interval(grid, value_na)

    assert np.isnan(out1), \
        "expect value beyond grid to return NA (below)"
    assert np.isnan(out3), \
        "expect value beyond grid to return NA (above)"
    assert out2 == 2,\
        "7 should be in the bin [7,8)"


    values = np.array([value_na, value_mid, value_na2])

    out_all = lc.find_interval(grid, values)
    assert np.isnan(out_all[0]), \
        "expect value beyond grid to return NA (below)"
    assert np.isnan(out_all[2]), \
        "expect value beyond grid to return NA (above)"
    assert out_all[1] == 2,\
        "7 should be in the bin [7,8)"




def test_inner_hpd_value_level():
    """test inner_hpd_value_level, basic"""

    z_grid = np.array([-1.1,-.1,.9,1.9,2.9])
    cdes = np.array([5.1,1.1,2.1,4.1,3.1])
    z_delta = 1

    assert lc.hpd_process.inner_hpd_value_level(cdes, z_grid = z_grid,
                                                z_test = .1, z_delta = z_delta) == \
        1.1, \
        "error with inner_hpd_value_level (a)"

    assert lc.hpd_process.inner_hpd_value_level(cdes, z_grid = z_grid,
                                                z_test = .9, z_delta = z_delta) == \
        3.2, \
        "error with inner_hpd_value_level (b)"


    assert lc.hpd_process.inner_hpd_value_level(cdes, z_grid = z_grid,
                                                z_test = 1.8, z_delta = z_delta) == \
        10.4, \
        "error with inner_hpd_value_level (c)"

    assert lc.hpd_process.inner_hpd_value_level(cdes, z_grid = z_grid,
                                                z_test = -1, z_delta = z_delta) == \
        15.5, \
        "error with inner_hpd_value_level (d)"

    assert lc.hpd_process.inner_hpd_value_level(cdes, z_grid = z_grid,
                                                    z_test = 4, z_delta = z_delta) == \
            0, \
            "error with inner_hpd_value_level (outside above)"

    assert lc.hpd_process.inner_hpd_value_level(cdes, z_grid = z_grid,
                                                    z_test = -3, z_delta = z_delta) == \
            0, \
            "error with inner_hpd_value_level (outside below)"

    ### a more "cde styled one"
    cdes = np.array([.09,.24,.3,.26,.11])
    z_grid = np.arange(5.0) + .1
    z_delta = 1.0


    # Original
    out1 = lc.inner_hpd_value_level(cdes, z_grid, 3.1, z_delta)
    assert out1 == .7, \
        "error in inner_hpd_value getting desired amount above"
    out2 = lc.inner_hpd_value_level(cdes, z_grid, 3.2, z_delta)
    assert out2 == .7, \
        "error in inner_hpd_value getting desired amount above"

    out1_1 = lc.inner_hpd_value_level(cdes, z_grid, 2, z_delta)
    assert out1_1 == 1, \
        "error in inner_hpd_value with providing correct ordering"

    out3 = lc.inner_hpd_value_level(cdes, z_grid, -.7, z_delta)
    assert out3 == 0, \
        "expected z_values outside the grid to return 0 hpd values (lower)"
    out4 = lc.inner_hpd_value_level(cdes, z_grid, 6.7, z_delta)
    assert out4 == 0, \
        "expected z_values outside the grid to return 0 hpd values (upper)"

    # Original (explicit order)
    same_order = np.array([0,4,1,3,2], dtype = int)

    out1 = lc.inner_hpd_value_level(cdes, z_grid, 3.1, z_delta,
                                    order = same_order)
    assert out1 == .7, \
        "error in inner_hpd_value getting desired amount above"
    out2 = lc.inner_hpd_value_level(cdes, z_grid, 3.2, z_delta,
                                    order = same_order)
    assert out2 == .7, \
        "error in inner_hpd_value getting desired amount above"

    out1_1 = lc.inner_hpd_value_level(cdes, z_grid, 2, z_delta,
                                    order = same_order)
    assert out1_1 == 1, \
        "error in inner_hpd_value with providing correct ordering"

    out3 = lc.inner_hpd_value_level(cdes, z_grid, -.7, z_delta,
                                    order = same_order)
    assert out3 == 0, \
        "expected z_values outside the grid to return 0 hpd values (lower)"
    out4 = lc.inner_hpd_value_level(cdes, z_grid, 6.7, z_delta,
                                    order = same_order)
    assert out4 == 0, \
        "expected z_values outside the grid to return 0 hpd values (upper)"


    # new order
    new_order = np.arange(5, dtype = int)
    out1 = lc.inner_hpd_value_level(cdes, z_grid, 3.1, z_delta,
                          order= new_order)
    assert np.isclose(out1, .89), \
        "error in inner_hpd_value getting desired amount above"
    out2 = lc.inner_hpd_value_level(cdes, z_grid, 3.2, z_delta,
                          order= new_order)
    assert np.isclose(out1, .89), \
        "error in inner_hpd_value getting desired amount above"

    out3 = lc.inner_hpd_value_level(cdes, z_grid, -.7, z_delta,
                          order= new_order)
    assert out3 == 0, \
        "expected z_values outside the grid to return 0 hpd values (lower)"
    out4 = lc.inner_hpd_value_level(cdes, z_grid, 6.7, z_delta,
                          order= new_order)
    assert out4 == 0, \
        "expected z_values outside the grid to return 0 hpd values (upper)"



def test_hpd_coverage():
    """test hpd_coverage, basic"""
    cdes_mat = np.array([[.09,.24,.3,.26,.11],
                     [.3,.26,.25,.11,.08]])
    z_grid = np.arange(5.0) + .1
    z_test = np.array([3.1,3.1])
    out = lc.hpd_coverage(cdes = cdes_mat, z_grid = z_grid, z_test = z_test)
    assert np.all([np.isclose(out[ii], np.array([.7, .19])[ii]) for ii in [0,1]]), \
        "errors in hpd_coverage"

    order = np.arange(5)
    out2 = lc.hpd_coverage(cdes = cdes_mat, z_grid = z_grid, z_test = z_test,
                       order = order)
    assert np.all([np.isclose(out2[ii], np.array([.89, .92])[ii]) for ii in [0,1]]), \
        "errors in hpd_coverage, ordering approach"


def test_profile_density():
    """test profile density, basic"""

    cdes = np.array([.09,.24,.3,.26,.11])
    z_delta = 1
    t_grid = np.array([.08, .1, .26,.3, .4,.5])

    info_out = lc.hpd_process._profile_density(cdes, t_grid, z_delta)

    assert np.all(info_out == np.array([0, .09, .09+.11+.24+.26, 1,1,1])), \
        "expected _profile_density function incorrect on basic example"


    cdes_mat = np.array([[.09,.24,.3,.26,.11],
                    [.3,.26,.25,.11,.08]])
    t_grid = np.array([.08, .1, .26,.3, .4,.5])

    info_out_mat = lc.profile_density(cdes_mat, t_grid, z_delta)


    assert np.all(info_out_mat[0,:] == \
                  np.array([0, .09, .09+.11+.24+.26, 1,1,1])), \
        "expected profile_density function incorrect on basic example"

    assert np.all(info_out_mat[1,:] == \
                  np.array([.08, .08, .08+.11+.25+.26, 1,1,1])), \
        "expected profile_density function incorrect on basic example (ordered)"


def test___true_thresholds_out():
    """
    basic tests of _true_thresholds_out function
    """
    cdes = np.array([.09,.24,.3,.26,.11])
    z_delta = 1.0
    expected_prop = np.array([.08,.3, .55, .56,.57,1-.21, 1.1])

    thresholds_out = lc.hpd_process._true_thresholds_out(cdes, z_delta,
                                                         expected_prop)

    assert np.all(thresholds_out == np.array([.3, .3, .26,.26, .24,.24, 0])), \
        "static thresholds don't match"

    cdes2 = np.array([.09,.24,.3,.26,.11])*2
    z_delta2 = .5
    expected_prop2 = np.array([.08,.3, .55, .56,.57,1-.21, 1.1])

    thresholds_out2 = lc.hpd_process._true_thresholds_out(cdes2, z_delta2,
                                                          expected_prop2)

    assert np.all(thresholds_out2 == np.array([.3, .3, .26,.26, .24,.24, 0])*2), \
        "static thresholds don't match with non-one z_delta"


def test_true_thresholds_out():
    """
    test true_treshold_out, basic
    """

    cde_mat = np.array([[.09,.24,.3,.26,.11],
                       [.1,.11,.12,.13,.095]])
    z_delta = 1.0
    expected_prop = np.array([.08,.3, .55, .56,.57,1-.21, 1.1])

    t_out = lc.true_thresholds_out(cde_mat, z_delta, expected_prop)

    assert np.all(t_out == np.array([[.3, .3, .26,.26, .24,.24, 0],
                                     [.13, .11, .095, 0,0,0,0]])), \
        "basic threshold checks across multiple rows errored"


def test__hpd_grid():
    """
    test _hpd_grid, basic
    """
    cde_mat_1 = np.array([.11,.09,.24,.3,.26])

    out1_1 = lc.hpd_process._hpd_grid(cde_mat_1, z_delta = 1)

    assert np.all(out1_1 == np.array([.2, .09,  .44, 1.0, .7])), \
        "hpd values incorrect"

    cde_mat_2 = np.array([1,3,4,2,5,7,8,0])
    hpd_val_2 = np.array([1,6,10,3,15,22,30,0])/2

    out1_2 = lc.hpd_process._hpd_grid(cde_mat_2, z_delta = 1/2)

    assert np.all(out1_2 == hpd_val_2), \
        "hpd values incorrect when z_delta = .5"

def test_hpd_grid():
    """
    test hpd_grid, basic
    """
    cde_mat = np.array([[1,3,4,2,5,7,8,0],
                        [2,1,3,5,4,0,0,0]])

    hpd_mat = np.array([[1,6,10,3,15,22,30,0],
                        [3,1,6,15,10,0,0,0]])/3
    z_delta = 1/3

    hpd_out = lc.hpd_grid(cde_mat, z_delta = z_delta)

    assert np.all(np.abs(hpd_out - hpd_mat) < .00000001), \
        "hpd_grid multiple rows doesn't preform correctly with non-one z_delta"


def test__find_interval_bins():
    """
    test _find_interval_bins, basic structure
    """

    grid = np.arange(5)

    # first bin
    assert lc.hpd_process._find_interval_bins(grid, 0) == 0, \
        "bin containment in _find_intervals_bins is not working"

    assert lc.hpd_process._find_interval_bins(grid, -.5 + .000001) == 0, \
        "bin containment in _find_intervals_bins is not working "+\
        "(lower bound of bin)"

    assert lc.hpd_process._find_interval_bins(grid, 1-.500000001) == 0, \
        "bin containment in _find_intervals_bins is not working "+\
        "(upper bound of bin)"

    # below all bins
    assert lc.hpd_process._find_interval_bins(grid, -.500000001) is np.nan, \
        "bin containment in _find_intervals_bins is not working "+\
        "(lower bound of sequence + delta)"
    # above all bins
    assert lc.hpd_process._find_interval_bins(grid, 6) is np.nan, \
        "bin containment in _find_intervals_bins is not working "+\
        "(upper bound of sequence + delta)"

    # middle bin
    assert lc.hpd_process._find_interval_bins(grid, 2) == 2, \
        "bin containment in _find_intervals_bins is not working"

def test_find_interval_bins():
    """
    test find_interval_bins
    """
    grid = np.arange(5)/2
    vec_check = np.array([0, .25 - .001, -.25 + .001, -.251, 2.76, 1.5])

    assert np.all(lc.find_interval_bins(grid, vec_check) == \
                  [0,0,0, np.nan, np.nan, 3]), \
        "bin containment in find_intervals_bins is not working"

