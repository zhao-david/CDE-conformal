from sklearn.cluster import KMeans

def profile_grouping(profile_train=None,
                     Kmeans_model = None,
                     profile_test=None, k=3, random_state=0):
    """
    L2 clustering grouping using Kmeans

    @param profile_train numpy array (n, d)
    @param Kmeans_model fit sklearn Kmeans model
    @param profile_test numpy array (m,d)
    @param k number of clusters - if profile_train is non-None
    @param random_state integer random state for Kmeans model fit (if
    profile_train is non-None).

    @returns  Kmeans_model, grouping_train, grouping_test (later 2 can be
    None if profile_train or profile_test are None, respectively)

    Notes:
    ======
    izbicki used k = n_calibrate/100
    """
    assert not (profile_train is None and \
                Kmeans_model is None), \
        "Need 1 of profile_train and Kmeans_model parameters to not be None-see docs"

    assert not (profile_train is not None and \
                Kmeans_model is not None), \
        "either profile_train or Kmeans_model needs to be None-see docs"


    if profile_train is not None:
        Kmeans_model = KMeans(n_clusters=k,
                              random_state=random_state).fit(profile_train)
        grouping_train = Kmeans_model.predict(profile_train)
    else:
        grouping_train = None

    if profile_test is not None:
        grouping_test = Kmeans_model.predict(profile_test)
    else:
        grouping_test = None

    return Kmeans_model, grouping_train, grouping_test





