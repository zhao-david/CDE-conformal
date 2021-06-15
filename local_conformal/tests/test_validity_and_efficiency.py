import numpy as np
import local_conformal as lc

def test_difference_validity_and_efficiency():
    true_cde = np.ones((5,4))
    predict_grid = np.arange(20).reshape(5,4)
    thresholds_predict = (np.array([2,5,10,11,19]) + .01).reshape((-1,1))



    true_grid = np.repeat(np.array([1,1,0,0]).reshape((1,-1)),5,axis = 0)
    thresholds_true = np.ones((5,1))
    expected_prop = np.array([.5])
    z_delta = 1/4

    #thresholds_true.shape, thresholds_predict.shape, expected_prop.shape
    out_v, out_e = lc.difference_validity_and_efficiency(true_cde = true_cde,
                                                     predict_grid = predict_grid,
                                                     true_grid = true_grid,
                                                     thresholds_predict = thresholds_predict,
                                                     thresholds_true = thresholds_true,
                                                     expected_prop = expected_prop,
                                                     z_delta = z_delta,
                                                     verbose = False)

    #proportion predict_grid >= thresholds_predict -> with goal of 50%
    assert np.all(out_v == np.abs(np.array([.25, .5, .25,1,0]) - .5).reshape((-1,1))), \
        "validation error in static test is incorrect (relatived to 50% coverage)"

    #true_grid vs 1*(predict_grid <= np.repeat(thresholds_predict, 4, axis =1))
    assert np.all(out_e == np.array([.25, 0,.25, .5, .5]).reshape((-1,1))), \
        "efficiency error in statistic test is incorrect (set diff incorrect)"
