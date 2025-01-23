import sys, os
sys.path.insert(1, os.path.realpath(os.path.pardir))

import argparse
from math import sqrt

from utilities.utils import load_data, get_num_classes
from utilities.noise_calibration import calibrate
from utilities.models import LogisticModel, LogisticMulti
from utilities.train import train, test

import torch
from opacus import PrivacyEngine

import numpy as np
from itertools import product


def main(n_epochs, batch_size_grid,
         lr_grid,
         data_path, optim,
         dataset, n_trials,
         verbose):
    results = []
    best_lrs = []
    candidate_grid = list(product(batch_size_grid, lr_grid))
    for n_trial in range(n_trials):
        best_res = 0
        best_lr = 0
        for batch_size, learning_rate in candidate_grid:
            """
            Load the train and test data.
            """
            train_loader, test_loader = load_data(dataset, data_path, batch_size)
            model = LogisticModel(train_loader.dataset.dim(), 1)
            if optim == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            elif optim == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            """
            Train for n_iter iterations.
            """
            for num_epoch in range(n_epochs):
                train(model, train_loader, optimizer, verbose)
            """
            Test .
            """
            curr_test_res = test(model, test_loader, verbose)
            if curr_test_res > best_res:
                best_res = curr_test_res
                best_lr = learning_rate                
        results.append(best_res)
        best_lrs.append(best_lr)
        print(f'Best result on test set on trial {n_trial}: {best_res:.2f}%, with learning rate {best_lr}')
    results_out = os.path.join('results', dataset, 'np')
    if not os.path.exists(results_out):
      os.makedirs(results_out)
    params_dict = {'alg': 'np_sgd',
                         'dataset:': dataset,
                             'n_epochs': n_epochs,
                             'batch_size': batch_size,
                             'learning_rate_grid': lr_grid,
                             'best_lr': best_lr,
                             'optim': optim
                             }
    results_str = 'results'
    params_str = 'params'
    np.save(os.path.join(results_out, results_str), results)
    np.save(os.path.join(results_out, params_str), params_dict)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """SGD parameters"""
    parser.add_argument('--n_epochs', type=int, default=60)
    parser.add_argument('--batch_size_grid', type=int, default=[256])
    parser.add_argument('--lr_grid', type=list, default=np.logspace(-5, -1, 5))
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    """Experimental arguments"""
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--verbose', type=bool, default=False)

    args = parser.parse_args()
    main(**vars(args))
