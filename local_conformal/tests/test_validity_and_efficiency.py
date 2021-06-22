import numpy as np
import pandas as pd
import scipy.stats
import local_conformal as lc


def test_difference_validity_and_efficiency_basic():
    """
    basic test for difference_validity_and_efficiency
    """
    true_cde = np.ones((5,4))
    predict_grid = np.arange(20).reshape(5,4)
    thresholds_predict = (np.array([2,5,10,11,19]) + .01).reshape((-1,1))



    true_grid = np.repeat(np.array([1,1,0,0]).reshape((1,-1)),5,axis = 0)
    df_cs_grouping = pd.DataFrame(data = {"grouping": [1,1,0,1,0],
                                            "cs": [.5,.5,.5,.5,.5]})
    thresholds_true = np.ones((5,1))
    expected_prop = np.array([.5])
    z_delta = 1/4

    #thresholds_true.shape, thresholds_predict.shape, expected_prop.shape
    out_v, out_v2, out_e = lc.difference_validity_and_efficiency(true_cde = true_cde,
                                                     predict_grid = predict_grid,
                                                     true_grid = true_grid,
                                                     thresholds_predict = thresholds_predict,
                                                     thresholds_true = thresholds_true,
                                                     df_cs_grouping = df_cs_grouping,
                                                     true_grouping = None,
                                                     expected_prop = expected_prop,
                                                     z_delta = z_delta,
                                                     verbose = False)

    #proportion predict_grid >= thresholds_predict -> with goal of 50%
    assert np.all(out_v == np.abs(np.array([.25, .5, .25,1,0]) - .5).reshape((-1,1))), \
        "validation error in static test is incorrect (relatived to 50% coverage)"

    #true_grid vs 1*(predict_grid <= np.repeat(thresholds_predict, 4, axis =1))
    assert np.all(out_e == np.array([.25, 0,.25, .5, .5]).reshape((-1,1))), \
        "efficiency error in statistic test is incorrect (set diff incorrect)"


def test_difference_validity_and_efficiency_descriptive():
    """
    test for difference_validity_and_efficiency that is very descriptive
    for future debugging
    """


    true_cde = np.ones((5,4))
    predict_grid = np.arange(20).reshape(5,4)
    thresholds_predict = (np.array([2,5,10,11,19]) + .01).reshape((-1,1))

    # this means our regions relative to prediction will look like:
    # (value >= threshold)
    # False, False, False, True,
    # False, False, True, True,
    # False, False, False, True,
    # True, True, True, True,
    # False, False, False, False

    true_grid = np.repeat(np.array([0,0,1,1]).reshape((1,-1)),5,axis = 0)
    thresholds_true = .5*np.ones((5,1))
    # this means our regions relative to prediction will look like:
    # (value >= threshold)
    # False, False, True, True,
    # False, False, True, True,
    # False, False, True, True,
    # False, False, True, True,
    # False, False, True, True


    df_cs_grouping = pd.DataFrame(data = {"grouping": [1,1,0,1,0],
                                            "cs": [2.1,4,10.01, 0, 25]})

    # which are contained:
    # [True, False, Exact..., False, True]

    expected_prop = np.array([.5])
    z_delta = 1/4



    # efficiency:
    expected_efficiency_diff_mat = \
        np.array([[False, False, False, True],
                  [False, False, True, True],
                  [False, False, False, True],
                  [True, True, True, True],
                  [False, False, False, False]]) == \
        np.array([[False, False, True, True],
                 [False, False, True, True],
                 [False, False, True, True],
                 [False, False, True, True],
                 [False, False, True, True]])
    expected_out_e = expected_efficiency_diff_mat.sum(axis = 1).reshape((-1,1)) * z_delta


    # validity:
    # 1
    predict_level_set_prob_mask1 = true_cde * \
        np.array([[False, False, False, True],
                  [False, False, True, True],
                  [False, False, False, True],
                  [True, True, True, True],
                  [False, False, False, False]])
    validity_level1 = predict_level_set_prob_mask1.sum(axis = 1)*z_delta
    expected_out_v1 = np.abs(validity_level1-.5).reshape((-1,1))

    # 2
    # binary vector if contained...
    expected_contained2 = np.array([True, False, True, False, True]).reshape((-1,1))



    #thresholds_true.shape, thresholds_predict.shape, expected_prop.shape
    out_v, out_v2, out_e = lc.difference_validity_and_efficiency(true_cde = true_cde,
                                                     predict_grid = predict_grid,
                                                     true_grid = true_grid,
                                                     thresholds_predict = thresholds_predict,
                                                     thresholds_true = thresholds_true,
                                                     df_cs_grouping = df_cs_grouping,
                                                     true_grouping = None,
                                                     expected_prop = expected_prop,
                                                     z_delta = z_delta,
                                                     verbose = False)


    assert np.all(out_v == expected_out_v1), \
        "error in underlying discrete true validity calculation"

    assert np.all(out_v2 == expected_contained2), \
        "error in underlying discrete sample validity calculation"

    assert np.all(out_e == expected_out_e), \
        "error in underlying discrete true efficiency calculation"


def test_difference_validity_and_efficiency_uniform_hpd():
    """
    test of
    with continuous predicted HDP and discrete true HDP

    knows it will mess up with the efficiency for non-continuous conformal scores
    """

    # truth is uniform(0,1)
    yy = np.linspace(-2,2, 401)
    y_delta = yy[1]-yy[0]
    cde_unif = scipy.stats.uniform.pdf(yy)
    hpd_unif = np.zeros(yy.shape[0])
    hpd_unif[np.logical_and(yy <= np.sqrt(3), yy >= -np.sqrt(3))] = 1

    true_cde = np.repeat(cde_unif.reshape((1,-1)), 5, axis = 0)
    true_grid = np.repeat(hpd_unif.reshape((1,-1)), 5, axis = 0)

    # estimate is norm(0,1)
    cde_norm = scipy.stats.norm.pdf(yy)
    hpd_norm = lc.hpd_grid(cde_norm.reshape((1,-1)), y_delta).ravel()

    predict_grid = np.repeat(hpd_norm.reshape((1,-1)), 5, axis = 0)


    thresholds_predict = np.repeat(np.array([.25,.5,.75,.9]).reshape((1,-1)), 5, axis = 0)
    expected_prop = np.array([.25,.5,.75,.9])
    thresholds_true = np.repeat(np.array([.99]*4).reshape((1,-1)), 5, axis = 0)


    df_cs_grouping = pd.DataFrame(data = {"cs": np.array([.2,.3,.6,.76,.8]),
                                         "grouping": np.ones(5, dtype = int)})
    # just when the true obs are contained (aka cs >= threshold)
    expected_out_v2 = np.array([[False, False, False, False],
                               [True, False, False, False],
                               [True,True,False, False],
                               [True,True,True, False],
                               [True,True,True, False]])


    #true_cde.shape, predict_grid.shape, true_grid.shape
    #thresholds_true.shape, thresholds_predict.shape, expected_prop.shape
    out_v, out_v2, out_e = lc.difference_validity_and_efficiency(true_cde = true_cde,
                                                     predict_grid = predict_grid,
                                                     true_grid = true_grid,
                                                     thresholds_predict = thresholds_predict,
                                                     thresholds_true = thresholds_true,
                                                     df_cs_grouping = df_cs_grouping,
                                                     true_grouping = None,
                                                     expected_prop = expected_prop,
                                                     z_delta = y_delta,
                                                     verbose = False)


    assert np.all(out_v2 ==expected_out_v2), \
        "error in sample data breakouts (validity)"

    # checking masking of true_cde with prediction level sets
    # (taking advantance of the fact that the cutoffs are all the same...)
    expected_out_v1 = -1* np.ones((thresholds_predict.shape))
    for cut_idx, t_cut in enumerate(thresholds_predict[0,:]):
        mask_of_true_cde = true_cde * (predict_grid >= t_cut)
        true_mass_in_level_set = mask_of_true_cde.sum(axis = 1)* y_delta
        inner_diff = np.abs(true_mass_in_level_set - expected_prop[cut_idx])
        expected_out_v1[:,cut_idx] = inner_diff

    assert np.all(out_v == expected_out_v1), \
        "errors in true validity of level set estimation"

    # efficiency:
    # for uniform we'd actually want the prediction region to cover any set of the true uniform base
    # the lebegue measure can be seen as
    # desirable_share = prediction region *intersect* true uniform base
    # check if desirable_share is above or below minimum number of squares for get validity coverage
    # lebegue measure the difference above or below

    expected_out_e = -1* np.ones((thresholds_predict.shape))
    for cut_idx, t_cut in enumerate(thresholds_predict[0,:]):
        mask_of_true_cde = true_cde * (predict_grid >= t_cut)
        true_mass_in_level_set = np.mean(mask_of_true_cde)*y_delta

        non_zero_prob = true_cde[true_cde>0].max()

        inner_set_diff = np.abs(true_mass_in_level_set - expected_prop[cut_idx])/non_zero_prob
        expected_out_e[:,cut_idx] = inner_set_diff

    assert not np.all(out_e == expected_out_e), \
        "expected to error when dealing with uniform data/data with non-continuous conformal scores..."


def test_difference_actual_validity():
    """
    test difference_actual_validity, basic
    """

    df_test = pd.DataFrame(data = {"cs": np.array([.1,.3,.2,.4,-1,1.1])})
    t_mat = np.array([[0, .1, .2],
                      [-.1,0,.1],
                      [.4,1,.1],
                      [.4,1,1.2],
                      [.4,1,1.2],
                      [-.4,1,1.2]])
    expected = np.array([[True, True, False],
                  [True,True,True],
                  [False,False,True],
                  [True,False,False],
                  [False,False,False],
                  [True,True,False]])


    out = lc.difference_actual_validity(df_test, t_mat)

    assert np.all(out == expected), \
        "binary array for comparison with thresholds is incorrect"

def test_difference_efficiency_uniform():
    # truth is uniform(0,1)
    yy = np.linspace(-2,2, 401)
    y_delta = yy[1]-yy[0]
    cde_unif = scipy.stats.uniform.pdf(yy)
    hpd_unif = np.zeros(yy.shape[0])
    hpd_unif[np.logical_and(yy <= np.sqrt(3), yy >= -np.sqrt(3))] = 1

    true_cde = np.repeat(cde_unif.reshape((1,-1)), 5, axis = 0)
    true_grid = np.repeat(hpd_unif.reshape((1,-1)), 5, axis = 0)

    # estimate is norm(0,1)
    cde_norm = scipy.stats.norm.pdf(yy)
    hpd_norm = lc.hpd_grid(cde_norm.reshape((1,-1)), y_delta).ravel()

    predict_grid = np.repeat(hpd_norm.reshape((1,-1)), 5, axis = 0)


    thresholds_predict = np.repeat(np.array([.25,.5,.75,.9]).reshape((1,-1)), 5, axis = 0)
    expected_prop = np.array([.25,.5,.75,.9])
    thresholds_true = np.repeat(np.array([.99]*4).reshape((1,-1)), 5, axis = 0)

    out_e = lc.difference_efficiency_uniform(true_cde = true_cde,
                                  predict_grid = predict_grid,
                                  true_grid = true_grid,
                                  thresholds_predict = thresholds_predict,
                                  thresholds_true = thresholds_true,
                                  expected_prop = expected_prop,
                                  z_delta = y_delta,
                                  verbose = False)

    expected_out_e = -1* np.ones((thresholds_predict.shape))
    for cut_idx, t_cut in enumerate(thresholds_predict[0,:]):
        mask_of_true_cde = true_cde * (predict_grid >= t_cut)
        true_mass_in_level_set = mask_of_true_cde.sum(axis = 1)*y_delta

        non_zero_prob = true_cde[true_cde>0].max()

        inner_set_diff = np.abs(true_mass_in_level_set - expected_prop[cut_idx])/non_zero_prob
        expected_out_e[:,cut_idx] = inner_set_diff

    assert np.all(out_e == expected_out_e), \
        "uniform data doesn't correctly work with efficiency"


