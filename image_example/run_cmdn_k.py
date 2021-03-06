import argparse
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

import torch
from mdn_model import MDNPerceptron
from convolutional_mdn_model_1D import ConvMDNPerceptron
import torch.nn as nn


# run convolutional mixture density network for input image data; tune K the number of mixture components
def main(image_file_name, param_file_name, k=7, n_train=7000, n_val=3000, n_test=1000, n_channels=1, n_hidden=10, width=20, height=20,
         epochs=10000, lr=1e-3):
    
    # load data into PyTorch
    with open(image_file_name, 'rb') as handle:
        galaxies = pickle.load(handle)
    """
    with open(param_file_name, 'rb') as handle:
        prior_mat = pickle.load(handle)
    # parameter of interest (angle of image)
    alphas = prior_mat[:,0]
    """
    with open(param_file_name, 'rb') as handle:
        photoz = pickle.load(handle)
    
    x_train = galaxies[:n_train]
    x_val = galaxies[n_train:n_train+n_val]
    x_test = galaxies[n_train+n_val:n_train+n_val+n_test]
    y_train = photoz[:n_train]  #alphas[:n_train]
    y_val = photoz[n_train:n_train+n_val]  #alphas[n_train:n_train+n_val]
    y_test = photoz[n_train+n_val:n_train+n_val+n_test]  #alphas[n_train+n_val:n_train+n_val+n_test]
    
    assert(len(x_train) == n_train)
    assert(len(y_train) == n_train)
    assert(len(x_val) == n_val)
    assert(len(y_val) == n_val)
    assert(len(x_test) == n_test)
    assert(len(y_test) == n_test)
    
    x_data = torch.from_numpy(np.array(x_train).reshape(n_train,n_channels,width,height)).float()
    y_data = torch.Tensor(y_train).float()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_data = x_data.to(device)
    y_data = y_data.to(device)
    
    # fit the convolutional mixture density network for K=7 components
    
    # train model
    model = ConvMDNPerceptron(10, k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pi, mu, sigma = model(x_data)
        loss = model.loss_fn(y_data, pi, mu, sigma)
        loss.backward()
        optimizer.step()
        if epoch % (epochs / 10) == 0:
            print('Loss: ' + str(loss.item()))

    # evaluate model on test data
    pi_test, mu_test, sigma_test = model.forward(
        torch.from_numpy(np.array(x_test).reshape(n_test,n_channels,width,height)).float()
        )
    
    all_out_CMDN = [pi_test, mu_test, sigma_test]
    
    date_str = datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    with open('CMDN_test_k=%s_' % k + date_str + '.pkl', 'wb') as handle:
        pickle.dump(all_out_CMDN, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file_name', action="store", type=str,
                        default='data/galaxies_generated_20210210.pkl',
                        help='Name of pickled data file for galaxy images')
    parser.add_argument('--param_file_name', action="store", type=str,
                        default='data/prior_mat_20210209.pkl',
                        help='Name of pickled data file for params used to generate galaxy images')
    parser.add_argument('--n_train', action="store", type=int, default=7000,
                        help='Number of training samples')
    parser.add_argument('--n_val', action="store", type=int, default=3000,
                        help='Number of validation samples')
    parser.add_argument('--n_test', action="store", type=int, default=1000,
                        help='Number of test samples')
    parser.add_argument('--k', action="store", type=int, default=1,
                        help='Number of Gaussian components in mixture')
    parser.add_argument('--n_channels', action="store", type=int, default=1,
                        help='Number of channels (e.g. RGB) in images')
    parser.add_argument('--n_hidden', action="store", type=int, default=10,
                        help='Number of hidden units in penultimate hidden layer')
    parser.add_argument('--width', action="store", type=int, default=20,
                        help='Width of images in pixels')
    parser.add_argument('--height', action="store", type=int, default=20,
                        help='Height of images in pixels')
    parser.add_argument('--epochs', action="store", type=int, default=10000,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', action="store", type=float, default=1e-3,
                        help='Learning rate for Adam optimizer')
    argument_parsed = parser.parse_args()
    
    
    main(
        image_file_name=argument_parsed.image_file_name,
        param_file_name=argument_parsed.param_file_name,
        n_train=argument_parsed.n_train,
        n_val=argument_parsed.n_val,
        n_test=argument_parsed.n_test,
        k=argument_parsed.k,
        n_channels=argument_parsed.n_channels,
        n_hidden=argument_parsed.n_hidden,
        width=argument_parsed.width,
        height=argument_parsed.height,
        epochs=argument_parsed.epochs,
        lr=argument_parsed.lr
    )
