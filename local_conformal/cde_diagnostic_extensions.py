import numpy as np
import pandas as pd
from cde_diagnostics.classifiers import classifier_dict
import progressbar

from .hpd_process import hpd_coverage
import matplotlib.pyplot as plt

def local_pp_plot_data(x_train, pit_train, x_test,
                       alphas=np.linspace(0.0, 0.999, 101),
                       clf_name='MLP',
                       verbose=True):
    """
    calculate data to create a local P-P plot

    @param x_train numpy vector of x values (n, )
    @param pit_train pit values (can also be hpd values) of the y values
    associated with x_train point
    @param x_test numpy vector of new points to estimate the P-P values at
    @param alpha numpy vector (with values between 0 and 1 inclusive) to fit
    the P-P model to.
    @param clf_name string associated with cde_diagnostic classifier dictionary
    @param verbose boolean, if we should show progression of calculations
    across alpha values.

    @return pandas data frame with x_test, alpha values, and predicted actual
    mass
    """

    clf = classifier_dict[clf_name]


    ### calculate PIT values at point of interest x_test
    if verbose:
        bar = progressbar.ProgressBar()
        iter_alpha = bar(alphas)
    else:
        iter_alpha = alphas

    x_vec = []
    alpha_vec = []
    rhat_vec = []

    x_length = x_test.shape[0]
    for alpha in iter_alpha:

        ind_train = [1*(x<=alpha) for x in pit_train]
        rhat = clf

        rhat.fit(X=x_train, y=ind_train)
        rhat_val = rhat.predict_proba(x_test)[:, 1][0]
        x_vec += list(x_test)
        alpha_vec += [alpha]*x_length
        rhat_vec += [rhat_val]*x_length

    all_rhat_alpha_data = pd.DataFrame(data = {"x": x_vec,
                                               "alpha": alpha_vec,
                                               "rhat": rhat_vec},
                                      columns = ["x","alpha","rhat"])
    return all_rhat_alpha_data


def local_pp_plot_truth(x_test, model,
                        cde_truth,
                        y_n_grid=200,
                        return_data_only = False):
    """
    calculate data to create a local P-P plot and create them

    @param x_test numpy vector of new points to estimate the P-P values at
    @param model cde model (takes in x_test and a integer parameter n_grid,
    and it's .predict function returns the cdes and y_grid).
    @param cde_truth function a function that takes in x_test values and
    a grid of y_values and returns the true conditional density values.
    @param y_n_grid number of points to be in the y_grid
    @param return_data_only boolean, if we should only return data to make
    the plot or the plot as well. Assumes that if true, you are only examining
    a single value for x_test.

    @return P-P plot figure and data (or just data).
    """

    cdes, y_grid = model.predict(x_test, n_grid=y_n_grid)
    order_pred = cdes.argsort()
    #ipdb.set_trace()
    cdes_all = np.repeat(cdes, y_n_grid, 0)
    cdes_truth = cde_truth(x_test, y_grid)
    cdes_truth_all = np.repeat(cdes_truth, y_n_grid, 0)
    hpd_pred = hpd_coverage(cdes_all, y_grid, y_grid, order = order_pred) # order not required here
    hpd_truth = hpd_coverage(cdes_truth_all, y_grid, y_grid, order = order_pred)

    data_out = pd.DataFrame(data = {"hpd_pred": hpd_pred,
                                    "hpd_truth": hpd_truth},
                            columns = ["hpd_pred", "hpd_truth"])

    if return_data_only:
        return data_out
    else:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.scatter(x = data_out.hpd_pred,
                   y = data_out.hpd_truth, marker = ".")
        lims = [
            np.min([0,0]),
            np.max([1,1]),
        ]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)

        ### create confidence bands, by refitting the classifier using Unif[0,1] random values in place of true PIT values

        plt.title("True coverage of CDE model at %s" % str(x_test), fontsize=20)
        plt.xlabel(r'$\alpha$', fontsize=20)
        plt.ylabel("$r($" + r'$\alpha$' + "$)$", fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.close()

        return fig, data_out

