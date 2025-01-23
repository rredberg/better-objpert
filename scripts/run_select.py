import sys, os
sys.path.insert(1, os.path.realpath(os.path.pardir))

import argparse
from math import sqrt
import time

from utilities.utils import load_data
from utilities.noise_calibration import calibrate
from utilities.models import LogisticModel
from utilities.train import train, test

from itertools import product

import torch
from opacus import PrivacyEngine

import numpy as np
import random

def main(epsilon, delta,
         n_epochs, max_grad_norm, mu,
         lr_grid, batch_size_grid,
         n_trials, data_path, verbose,
         dataset):

    best_results = []
    best_params = []

    """ Sample K, the number of candidates, from the Poisson distribution with mean mu. We want n_trials samples. """
    K_vals = np.random.poisson(lam=mu, size=n_trials)
    for i, K in enumerate(K_vals):
        best_res = 0
        best_param = None
        """ Select K candidates by randomly selecting a learning rate and a batch size from their respective grids. """
        candidate_grid = list(product(batch_size_grid, lr_grid))
        select_indices = np.random.choice(range(len(candidate_grid)), size=min(K, len(candidate_grid)), replace=False)
        for ind in select_indices:
            expected_batch_size, learning_rate = candidate_grid[ind]
            train_loader, test_loader = load_data(dataset, data_path, expected_batch_size)
            prob = expected_batch_size / len(train_loader.dataset)
            n_iters = n_epochs * len(train_loader)
            sigma = calibrate({'alg': 'select',
                             'eps': epsilon,
                             'delta': delta,
                             'prob': prob,
                             'n_iters': n_iters,
                             'n_epochs': n_epochs,
                             'GS': max_grad_norm,
                             'mu': mu})
            model = LogisticModel(train_loader.dataset.dim(), 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            """
            Attach privacy engine.
            """
            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=sigma,
                max_grad_norm=max_grad_norm
                )
            """
            Train 'honest' DP-SGD for n_iter iterations.
            """
            for num_epoch in range(n_iters // len(train_loader)):
                train(model, train_loader, optimizer, verbose)
            """
            Test 'honest' DP-SGD.
            """
            curr_test_res = test(model, test_loader, verbose)
            if curr_test_res > best_res:
                best_res = curr_test_res
                best_lr = learning_rate
                best_bs = expected_batch_size
        best_results.append(best_res)
        best_params.append(best_param)
        print(f'Best accuracy on test set on trial {i}: {best_res:.2f}%, with learning rate {best_lr}')
    results_out = os.path.join('results', dataset, 'select')
    if not os.path.exists(results_out):
        os.makedirs(results_out)
    params_dict = {'alg': 'select',
                     'dataset:': dataset,
                         'eps': epsilon,
                         'delta': delta,
                         'prob': prob,
                         'n_iters': n_iters,
                         'GS': max_grad_norm,
                         'mu': mu,
                         'batch_size_grid': batch_size_grid,
                         'lr_grid': lr_grid,
                         'best_lr': best_lr,
                         'best_batch_size': best_bs
                         }
    print(f'Average accuracy on test set over {n_trials} trials: {np.mean(best_results):.2f}%.')
    j = len(os.listdir(results_out))
    results_str = 'accuracy_eps={epsilon}_delta={delta}_{j}'.format(epsilon=str(epsilon), delta=str(delta), j=j)
    params_str = 'params_eps={epsilon}_delta={delta}_{j}'.format(epsilon=str(epsilon), delta=str(delta), j=j)
    np.save(os.path.join(results_out, results_str), best_results)
    np.save(os.path.join(results_out, params_str), params_dict)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """Privacy budget"""
    parser.add_argument('--epsilon', type=float, default=.1)
    parser.add_argument('--delta', type=float, default=1e-5)
    """DP-SGD parameters"""
    parser.add_argument('--n_epochs', type=int, default=60)
    parser.add_argument('--max_grad_norm', type=float, default=sqrt(2))
    parser.add_argument('--batch_size_grid', type=list, default=[4096])
    parser.add_argument('--lr_grid', type=list, default=np.logspace(-5, -1, 10))
    """Selection parameters"""
    parser.add_argument('--mu', type=float, default=8.0)
    """Experimental arguments"""
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    main(**vars(args))
