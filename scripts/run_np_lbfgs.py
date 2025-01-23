import sys, os
sys.path.insert(1, os.path.realpath(os.path.pardir))

import argparse
from math import sqrt

from utilities.utils import load_data
from utilities.models import LogisticModel, LinearModel
from utilities.train import train_lbfgs_NP, test

import torch
from opacus import PrivacyEngine

import numpy as np


def main(n_epochs, lr,
         data_path, dataset, verbose):
    """
    Load the train and test data.
    """
    train_loader, test_loader = load_data(dataset, data_path)
    """
    Train the model with L-BFGS.
    """
    model = LogisticModel(train_loader.dataset.dim(), 1)
    train_lbfgs_NP(model, train_loader, n_epochs, lr, verbose)
    res = test(model, test_loader, verbose)

    """
    Save the results.
    """
    results_out = os.path.join('results', dataset, 'np')
    if not os.path.exists(results_out):
        os.makedirs(results_out)
    params_dict = {'alg': 'lbfgs',
                           'dataset:': dataset,
                           'n_epochs': n_epochs,
                           'lr': lr,
                             }
    results_str = 'accuracy'
    params_str = 'params_lr={lr}_n_epochs={n_epochs}'.format(lr=str(lr), n_epochs=str(n_epochs))
    np.save(os.path.join(results_out, results_str), res)
    np.save(os.path.join(results_out, params_str), params_dict)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    main(**vars(args))
