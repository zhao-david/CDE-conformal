import numpy as np
import torch
import torch.nn as nn
import progressbar

class MDNPerceptron(nn.Module):
    def __init__(self, n_hidden1, n_hidden2, n_gaussians):
        """
        create a multiple hidden layer MDN Perceptron for a 1D input (output is
        also in 1D space).

        Arguments:
        ----------
        n_hidden1: int, number of nodes in the first hidden layer
        n_hidden2: int, number of nodes in the second hidden layer
        n_guassians: int, number of gaussians in final output layer

        Returns:
        --------
            create a 2 layer MDN perceptron pytorch model

        """
        super().__init__()

        self.base_model = nn.Sequential(
            nn.Linear(1, n_hidden1),
            nn.Sigmoid(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.Sigmoid())

        self.z_pi = nn.Sequential(
            nn.Linear(n_hidden2, n_gaussians),
            nn.Softmax(dim = 1))
        self.z_mu = nn.Linear(n_hidden2, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden2, n_gaussians)

        self.y_range = None
        self.n_grid = None


    def forward(self, x):
        """
        Returns parameters of for a mixture of gaussians given x

        Arguments:
        ----------
        x: data x value (n, 1) torch.Tensor

        Returns:
        --------
        pi: probability distribution over the gaussians
        mu: vector of means of the gaussians
        sigma: vector representing the diagonals of the covariances of the
            gaussians
        """
        inner = self.base_model(x)

        pi = self.z_pi(inner)
        mu = self.z_mu(inner)
        sigma = torch.exp(self.z_sigma(inner)) # to get positive

        return pi, mu, sigma

    def sample(self, pi, mu, sigma):
        """
        Makes a random draw from a randomly selected mixture based on parameters

        Arguments:
        ----------
        pi: probability distribution over the gaussians
        mu: vector of means of the gaussians
        sigma: vector representing the diagonals of the covariances of the
            gaussians

        Returns:
        --------
            a random sample from a specified mixture of gaussians
        """
        mixture = torch.normal(mu, sigma)
        k = torch.multinomial(pi, 1, replacement=True).squeeze()
        result = mixture[range(k.size(0)), k]
        return result

    def loss_fn(self, y, pi, mu, sigma):
        """
        Computes the mean probability of the datapoint being drawn from all the
        gaussians parametized by the network.


        Arguments:
        ----------
        y: potential vector of y values (n, 1) torch.Tensor
        mu: vector of means of the gaussians (n, n_guassians)
        sigma: vector representing the diagonals of the covariances of the
            gaussians (n, n_guassians)
        pi: probability distribution over the gaussians (n, n_guassians)

        Returns:
        --------
            scalar loss

        Details / Notes:
        ----------------
        I'm slightly worried about the numerical stability given this is going
        back and forth between log() and exp().


        similar thoughs in tensorflow: https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca
        pytorch mixtures: https://pytorch.org/docs/stable/distributions.html
        """

        mixture = torch.distributions.normal.Normal(mu, sigma)
        log_prob_x = mixture.log_prob(y.reshape(-1,1))
        log_mix_prob = torch.log(pi)  # [B, k]
        return -torch.mean(torch.logsumexp(log_prob_x + log_mix_prob, dim=-1))

    def prep_for_cde(self, y_values, n_grid):
        """
        initialize paramters to allow for self.cde_predict_grid to analysis
        over a grid of y_values

        Arguments:
        ----------
        y_values: numpy or torch vector, either the min and max of the range or
            a list of y_values, which we will take the min and max of.
        n_grid: int, number of equally spaced point to be done on the grid

        Returns:
        --------
            Nothing. Will update some interal paramters of the model
        """
        self.y_range = (y_values.min(), y_values.max())
        self.n_grid = n_grid

    def cde_predict_grid(self, x_values):
        """
        calculate the estimated conditional density estimate across a grid
        of y_values (and defind by internal self.y_range and self.n_grid -- set
        using the self.prep_for_cde function)

        Arguments:
        ----------
        x_values:  data x value (n, 1) torch.Tensor

        Returns:
        --------
        tensor matrix (n, self.n_grid) where each value [i,j]
            is the CDE estimates conditonal on x[i] for yy[j] where
            yy is a sequence of equally spaced y values ranging from
            self.y_range[0] to self.y_range[1] (with n_grid points on this
            range)

        """


        if self.y_range is None or self.n_grid is None:
            raise LookupError("need to run prep_for_cde first to" +\
                              "define self.y_range and self.n_grid")

        y_grid = torch.from_numpy(np.linspace(self.y_range[0],
                                              self.y_range[1],
                                              self.n_grid)).float()
        pi, mu, sigma = self.forward(x_values)
        mixture = torch.distributions.normal.Normal(mu, sigma)
        log_mix_prob = torch.log(pi)

        log_prob_x_list = []

        for yy in y_grid:
            log_prob_x_list.append(mixture.log_prob(yy.repeat(pi.shape[0],1)).unsqueeze(2))

        log_prob_x_mat = torch.cat(log_prob_x_list, dim = 2)

        return torch.exp(torch.logsumexp(log_prob_x_mat + \
                                         log_mix_prob.unsqueeze(2).repeat(1,1,log_prob_x_mat.shape[2]),
                                         dim = 1))


    def cde_predict(self, x_values, y_values):
        """
        evaluate the estimated conditional density estimate on y_value[i] given
        x_values[i]

        Arguments:
        ----------
        x_values:  data x value (n, 1) torch.Tensor
        y_values: data y value (n,1) torch.Tensor

        Returns:
        --------
        tensor vector (n, 1) where each value [i,]
            is the CDE estimates conditonal on x_values[i] for y_values[j]
        """
        pi, mu, sigma = self.forward(x_values)
        mixture = torch.distributions.normal.Normal(mu, sigma)
        log_prob_x = mixture.log_prob(y_values.reshape(-1,1))
        log_mix_prob = torch.log(pi)  # [B, k]
        return torch.exp(torch.logsumexp(log_prob_x + log_mix_prob, dim=-1))



def tune_first_nn(x_train, y_train, x_val, y_val,
                  model_op_list = None, epochs=10000,
                  n_gaussians=3, n_hidden1=10, n_hidden2 = 10, lr=1e-3,
                  verbose = False):
    """
    create and evalulate a MDN Perceptron

    Arguments:
    ----------
    x_train: torch.Tensor (n, 1), training x
    y_train: torch.Tensor (n, 1), training y
    x_val: torch.Tensor (m, 1), validation x
    y_val: torch.Tensor (m, 1), validation y
    model_op_list : list with model and optimizer already created - else
        will create them ourselves (Default is: None)
    epochs : int number of epochs to train the model for
    n_guassians : int, number of guassians to be included in the MDN Perceptron
    n_hidden1 : int, number of nodes for first hidden layer
    n_hidden2 : int, number of nodes for second hidden layer
    lr : float, learning rate for the optimizer (default is Adam)
    verbose : boolean, if we should be verbose about the learning across epochs.

    Returns:
    --------
    tuned model, associated optimizer, and validation error
    """
    # model creation if necessary ------
    if model_op_list is None:
        model = MDNPerceptron(n_hidden1, n_hidden2, n_gaussians)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        model = model_op_list[0]
        optimizer = model_op_list[1]

    # verbosity for fit ------
    if verbose:
        bar = progressbar.ProgressBar(widgets = [ progressbar.Bar(),
                                              ' (', progressbar.ETA(), ", ",
                                              progressbar.Timer(), ')'])
        epoch_iter = bar(np.arange(epochs))
    else:
        epoch_iter = range(epochs)

    # actual fit ------
    for epoch in epoch_iter:
        optimizer.zero_grad()
        pi, mu, sigma = model(x_train)
        loss = model.loss_fn(y_train, pi, mu, sigma)
        loss.backward()
        optimizer.step()

    # get validation error -------
    pi_v, mu_v, sigma_v = model(x_val)
    error = model.loss_fn(y_val, pi_v, mu_v, sigma_v)

    return model, optimizer, error




class QuantilePerceptron(nn.Module):
    def __init__(self,
                 n_hidden1,
                 n_hidden2,
                 quantiles):
        """
        create a multiple hidden layer Quantile Perceptron for a 1D input
        (each quantile output is also in 1D space).

        Arguments:
        ----------
        n_hidden1: int, number of nodes in the first hidden layer
        n_hidden2: int, number of nodes in the second hidden layer
        quantiles: 1d Tensor of quantiles (between 0 and 1)

        Returns:
        --------
            creates a 2 layer quantile preceptron pytorch model
        """
        super().__init__()
        if type(quantiles) is np.ndarray:
            self.quantiles = torch.from_numpy(quantiles).float()
        else:
            self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2

        self.build_model()
        self.init_weights()

    def build_model(self):
        self.base_model = nn.Sequential(
            nn.Linear(1, self.n_hidden1),
            nn.ReLU(),
            nn.Linear(self.n_hidden1, self.n_hidden2),
            nn.ReLU(),
            nn.Linear(self.n_hidden2, self.num_quantiles)
        )

    def init_weights(self):
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.base_model(x)


    def loss_fn(self, preds, target):
        """
        Computes the quantile regression loss across predicts from model and
        true values.


         Args:
        -----
        preds: predicted quantiles from model (n, num_quantiles)
        target: potential vector of targe (y) values (n, 1) torch.Tensor
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)

        losses = torch.max(
                     torch.mul(self.quantiles, target - preds),
                     torch.mul(self.quantiles-1, target - preds)
                )
        loss = torch.mean(torch.sum(losses, dim = 1))
        return loss


def tune_second_nn(x_train, y_train, x_val, y_val,
                  model_op_list = None, epochs=10000,
                  n_hidden1=10,
                  n_hidden2 = 10, lr=1e-3,
                  quantiles=torch.from_numpy(np.arange(1,20)/20),
                  verbose = False):
    """
    create and evalulate a Quantile Perceptron

    Arguments:
    ----------
    x_train: torch.Tensor (n, 1), training x
    y_train: torch.Tensor (n, 1), training y
    x_val: torch.Tensor (m, 1), validation x
    y_val: torch.Tensor (m, 1), validation y
    model_op_list : list with model and optimizer already created - else
        will create them ourselves (Default is: None)
    epochs : int number of epochs to train the model for
    n_hidden1 : int, number of nodes for first hidden layer
    n_hidden2 : int, number of nodes for second hidden layer
    lr : float, learning rate for the optimizer (default is Adam)
    quantiles : numpy or torch.Tensor (n,), quantile values we wish to examine -
        each value must be between 0 and 1.
    verbose : boolean, if we should be verbose about the learning across epochs.

    Returns:
    --------
    tuned model, associated optimizer, and validation error
    """
    # model creation if necessary ------
    if model_op_list is None:
        model = QuantilePerceptron(n_hidden1, n_hidden2, quantiles)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        model = model_op_list[0]
        optimizer = model_op_list[1]

    # verbosity for fit ------
    if verbose:
        bar = progressbar.ProgressBar(widgets = [ progressbar.Bar(),
                                              ' (', progressbar.ETA(), ", ",
                                              progressbar.Timer(), ')'])
        epoch_iter = bar(np.arange(epochs))
    else:
        epoch_iter = range(epochs)

    # actual fit ------
    for epoch in epoch_iter:
        optimizer.zero_grad()
        preds = model(x_train)
        loss = model.loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

    # get validation error -------
    preds_v = model(x_val)
    error = model.loss_fn(preds_v, y_val)

    return model, optimizer, error



def torchify_data(*args):
    """
    make pandas columns to torch objects (n,1) shapes

    Arguments:
    ----------
    *args: a "list of parameters" that are pandas columns

    Returns:
    --------
    a real list of torch vector transformation of arguments.
    """
    out = list()
    for a in args:
        out.append(torch.from_numpy(np.array(a).reshape((-1,1))).float())

    return out
