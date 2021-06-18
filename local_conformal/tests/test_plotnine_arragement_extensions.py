import numpy as np
import local_conformal as lc

def test__match_ids():
    nrow = 3
    ncol = 5
    out = lc.plotnine_arrangement_extensions._match_ids(nrow, ncol)
    
    assert np.all(out[0] == np.array([0]*5+[1]*5+[2]*5)), \
        "row ids don't match expected"
    
    assert np.all(out[1] == np.array([0,1,2]*5)), \
        "row ids don't match expected"
    
