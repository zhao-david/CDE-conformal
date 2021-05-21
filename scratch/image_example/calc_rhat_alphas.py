import argparse
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from collections import defaultdict



# calculate p-value at point x_i by resampling (x_i, U_i) where U_i is uniform
def main(pit_values_name, x_test_name, alphas=np.linspace(0.0, 1.0, 11), conf_level=0.05, GalSim=False): #,
         #points=[0,1,2,3,250,251,252,253,500,501,502,503,750,751,752,753]):
    
    alphas[-1] = 0.99
    #fine_alphas = np.linspace(0.01, 0.99, 99)
    
    with open(pit_values_name, 'rb') as handle:
        pit_values = pickle.load(handle)
    
    #x_test = np.load(x_test_name)
    with open(x_test_name, 'rb') as handle:
        x_test = pickle.load(handle)
    
    if GalSim:
        grid = x_test
    else:
        x_range = np.linspace(-2,2,41)
        x1, x2 = np.meshgrid(x_range, x_range)
        grid = np.hstack([x1.ravel().reshape(-1,1), x2.ravel().reshape(-1,1)])
    
    # calculate rhat alphas for each test point
    all_rhat_alphas = {}
    for alpha in alphas:
        ind_values = [1*(x<=alpha) for x in pit_values]
        rhat = MLPClassifier(alpha=0, max_iter=25000)
        rhat.fit(X=x_test, y=ind_values)
        
        # fit rhat at each point in prediction grid
        all_rhat_alphas[alpha] = rhat.predict_proba(grid)[:, 1]
    
    """
    # points for which we want local diagnostics
    points_of_interest = x_test[points, :]
    use_rhat_alphas = {}
    for alpha in fine_alphas:
        ind_values = [1*(x<=alpha) for x in pit_values]
        rhat = MLPClassifier(alpha=0, max_iter=25000)
        rhat.fit(X=x_test, y=ind_values)
        
        # fit rhat at each point of interest
        use_rhat_alphas[alpha] = rhat.predict_proba(points_of_interest)[:, 1]
    """
    
    # rhat_alphas for all alphas at all points
    all_rhat_alphas = pd.DataFrame(all_rhat_alphas)
    
    ## rhat_alphas for all fine alphas at points of interest
    #use_rhat_alphas = pd.DataFrame(use_rhat_alphas)
    
    date_str = datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    alphas_name = 'all_rhat_alphas_' + date_str + '.pkl'
    #alphas_name = 'use_rhat_alphas_' + date_str + '.pkl'
    if GalSim:
        alphas_name = 'GalSim_' + alphas_name
    with open(alphas_name, 'wb') as handle:
        pickle.dump(all_rhat_alphas, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #pickle.dump(use_rhat_alphas, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pit_values_name', action="store", type=str, default=None,
                        help='Name of pickled dictionary of PIT values')
    parser.add_argument('--x_test_name', action="store", type=str, default=None,
                        help='Name of saved numpy array of x_test values')
    parser.add_argument('--GalSim', action='store_true', default=False,
                        help='If true, we are running the GalSim example.')
    argument_parsed = parser.parse_args()
    
    main(
        pit_values_name=argument_parsed.pit_values_name,
        x_test_name=argument_parsed.x_test_name,
        GalSim=argument_parsed.GalSim
    )
