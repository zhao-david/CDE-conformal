import numpy as np
import sklearn

import local_conformal as lc

def test_profile_grouping():
    """
    test profile_grouping, basic
    """

    # structure check
    np.random.seed(1)
    X = np.random.uniform(0,1,120).reshape((40,3))
    X2 = np.random.uniform(0,1,3*45).reshape((45,3))
    k = 3
    a,b,c = lc.profile_grouping(profile_train=X,
                             profile_test=X2, k=3, random_state=0)

    assert type(a) == sklearn.cluster._kmeans.KMeans and \
        a.cluster_centers_ is not None, \
        "expected first return object to be a fitted KMeans models"

    assert b is not None and b.shape[0] == X.shape[0] and \
        np.unique(b).shape[0] == 3, \
        "expected profile grouping to return vector with k classes for profile_train"

    assert c is not None and c.shape[0] == X2.shape[0] and \
        np.all(np.isin(c,np.unique(b))), \
        "expected profile grouping to return vector with at most k classes for profile_test"

    a2,b2,c2 = lc.profile_grouping(profile_train=X,
                                k=3, random_state=0)

    assert np.all(a.cluster_centers_ == a2.cluster_centers_), \
        "same Kmeans object should return with profile_train and K and random_state set to same value"


    a3,b3,c3 = lc.profile_grouping(Kmeans_model = a,
                             profile_test = X2, k=3, random_state=0)

    assert a3 is a, \
        "same Kmeans object should return with if Kmeans_model provided an no profile_train is provided"

    assert b3 is None, \
        "don't expect any clustering for profile_train if profile_train is None"

    assert np.all(c == c3), \
        "same clustering should result with using profile_train or the model generated from the same data, k and random_sample"


    X3 = np.random.uniform(0,1,3*60).reshape((60,3))
    a_diff, _,_ =  lc.profile_grouping(profile_train=X, k=2, random_state=1000)

    error_occurred = False
    try:
        a4,b3,c3 = lc.profile_grouping(profile_train=X,
                                    Kmeans_model = a_diff,
                                 profile_test = X2, k=3, random_state=0)
    except:
        error_occurred = True

    assert error_occurred, \
        "error should occur if profile_grouping has a non-None profile_train and Kmeans_model"



    # static point check
    X_static = np.array([[0,0],[0,1], [0,-1], [1,0],[-1,0],
                 [10,10], [10,11], [10,9], [11,10],[9,10]])
    X2_static = np.array([[3,3],[6,6]])
    k = 2
    model_static, g1_static, g2_static = \
        lc.profile_grouping(profile_train=X_static,
                         profile_test=X2_static,k=k, random_state=1000)
    assert np.all(g1_static == np.array([1]*5+[0]*5)) or \
           np.all(g1_static == np.array([0]*5+[1]*5)), \
        "static: 2 kmeans operates correctly"
    assert np.all(g2_static == np.array([1,0])) or \
           np.all(g2_static == np.array([0,1])), \
        "static: new points classified correctly"
    assert np.all(model_static.cluster_centers_ == np.array([[10,10],[0,0]])) or \
           np.all(model_static.cluster_centers_ == np.array([[0,0],[10,10]])),\
        "static clusters are correctly centered"
