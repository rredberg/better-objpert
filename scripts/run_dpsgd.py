import sys, os
sys.path.insert(1, os.path.realpath(os.path.pardir))

import argparse
from math import sqrt

from utilities.utils import load_data, get_num_classes
from utilities.noise_calibration import calibrate
from utilities.models import LogisticModel
from utilities.train import train, test

import torch
from opacus import PrivacyEngine

from itertools import product


import numpy as np


def main(epsilon,
         delta,
         n_epochs, batch_size_grid, max_grad_norm,
         lr_grid,
         data_path, noise_path,
         dataset, n_trials, optim, verbose):
    
    best_results = []
    best_lrs = []
    candidate_grid = list(product(batch_size_grid, lr_grid))
    for n_trial in range(n_trials):
        best_res = 0
        best_lr = 0
        for expected_batch_size, learning_rate in candidate_grid:
            """
            Load the train and test data.
            Note that train_loader and test_loader will have a deterministic batch size until we invoke the privacy engine.
            """
            train_loader, test_loader = load_data(dataset, data_path, expected_batch_size)
            prob = expected_batch_size / len(train_loader.dataset)
            """
            Calculate 'dishonest' noise scale.
            """
            n_iters = n_epochs * len(train_loader)
            sigma = calibrate({'alg': 'dpsgd',
                                     'eps': epsilon,
                                     'delta': delta,
                                     'prob': prob,
                                     'n_iters': n_iters,
                                     'GS': max_grad_norm,
                                     'noise_path': noise_path})
            """
            Train the model with DP-SGD for each candidate learning rate.
            """
            model = LogisticModel(train_loader.dataset.dim(), 1)
            if optim == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            elif optim == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
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
            Train 'cheating' DP-SGD for n_iter iterations.
            """
            for num_epoch in range(n_iters // len(train_loader)):
                train(model, train_loader, optimizer, verbose)
            """
            Test 'cheating' DP-SGD.
            """
            curr_test_res = test(model, test_loader, verbose)
            if curr_test_res > best_res:
                best_res = curr_test_res
                best_lr = learning_rate
        best_results.append(best_res)
        best_lrs.append(best_lr)
        print(f'Best accuracy on test set on trial {n_trial}: {best_res:.2f}%, with learning rate {best_lr}')
    results_out = os.path.join('results', dataset, 'dpsgd')
    if not os.path.exists(results_out):
        os.makedirs(results_out)
    params_dict = {'alg': 'dpsgd',
                         'dataset:': dataset,
                             'eps': epsilon,
                             'delta': delta,
                             'prob': prob,
                             'n_iters': n_iters,
                             'GS': max_grad_norm,
                             'batch_size': expected_batch_size,
                             'learning_rate_grid': lr_grid,
                             'best_lr': best_lr,
                             'batch_size': expected_batch_size
                             }
    print(f'Average best accuracy on test set over {n_trials} trials: {np.mean(best_results):.2f}%.')
    j = len(os.listdir(results_out)) / 2
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
    """Experimental arguments"""
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--noise_path', type=str, default='utilities/noise_cache')
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    main(**vars(args))
