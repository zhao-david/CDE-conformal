import numpy as np
import pandas as pd
import local_conformal as lc


def test_difference_validity_and_efficiency():
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
