import sys, os
sys.path.insert(1, os.path.realpath(os.path.pardir))

import argparse
from math import sqrt
import time
from copy import deepcopy

from utilities.utils import load_data, perturbed_model
from utilities.noise_calibration import calibrate, calibrate_amp_fix_lambda
from utilities.models import PerturbedLogisticModel
from utilities.train import lbfgs_step, train, test, grad_norm, add_output_noise

import torch
from opacus import PrivacyEngine

import numpy as np


def main(epsilon,
         delta,
         sigma_out,
         beta, L,
         tau, f,
         data_path,
         dataset, n_trials, learning_rate_grid,
         verbose
         ):
    """
    Load the train and test data.
    """
    train_loader, test_loader = load_data(dataset, data_path)
    """
    Calculate noise scale.
    """
    sigma, lambd = calibrate({'alg': 'amp',
                              'eps': epsilon,
                              'delta': delta,
                              'sigma_out': sigma_out,
                              'L': L,
                              'beta': beta,
                              'tau': tau,
                              'f': f
                               })
    """
    Train the model with L-BFGS to get an approximate minimizer whose gradient norm is at most tau.
    """
    results = []
    n_trial = 0
    while n_trial < n_trials:
      flag_count = 0 # Keeps track of how many learning rates successfully converged.
      while flag_count <= 0:
        omega_model = PerturbedLogisticModel(train_loader.dataset.dim(), 1, sigma, lambd)
        best_res = 0
        best_lr = 0
        for i, learning_rate in enumerate(learning_rate_grid):
          flag = True # Keeps track of whether the most recent learning rate converged.
          model = deepcopy(omega_model)
          optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
          grad_norm_prev = 10e6
          while grad_norm(model, optimizer, train_loader) > tau:
            g_norm = grad_norm(model, optimizer, train_loader)
            if grad_norm_prev <= g_norm: # Optimizer fails to converge: switch learning rate
              flag_count -= 1
              flag = False
              break
            grad_norm_prev = g_norm
            lbfgs_step(model, optimizer, train_loader, verbose)
          flag_count += 1
          if flag:
            curr_test_res = test(model, test_loader, verbose)
            if curr_test_res > best_res:
                best_res = curr_test_res
                best_lr = learning_rate
                best_model = model
      print(f'Best accuracy on test set on trial {n_trial}: {best_res:.2f}%, with learning rate {best_lr}')         
      n_trial += 1
      """
      Add output noise.
      """
      add_output_noise(best_model, sigma_out)
      """
      Test.
      """
      result = test(best_model, test_loader, 'logistic')
      results.append(result)
    results_out = os.path.join('results', dataset, 'amp')
    if not os.path.exists(results_out):
      os.makedirs(results_out)
    params_dict = {'alg': 'amp',
                         'dataset:': dataset,
                             'eps': epsilon,
                             'delta': delta,
                             'lambd': lambd,
                             'tau': tau,
                             'sigma_out': sigma_out,
                             'L': L,
                             'beta': beta,
                             'GS': L,
                             'learning_rate_grid': learning_rate_grid,
                             'best_lr': best_lr
                             }
    print(f'Average accuracy on test set over {n_trials} trials: {np.mean(results):.2f}%')
    results_str = 'accuracy_eps={epsilon}_delta={delta}'.format(epsilon=str(epsilon), delta=str(delta))
    params_str = 'params_eps={epsilon}_delta={delta}'.format(epsilon=str(epsilon), delta=str(delta))
    np.save(os.path.join(results_out, results_str), results)
    np.save(os.path.join(results_out, params_str), params_dict)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """Privacy budget"""
    parser.add_argument('--epsilon', type=float, default=.1)
    parser.add_argument('--delta', type=float, default=1e-5)
    """Parameters"""
    parser.add_argument('--sigma_out', type=float, default=0.15)
    parser.add_argument('--tau', type=float, default=0.0005)
    parser.add_argument('--L', type=float, default=sqrt(2))
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--learning_rate_grid', type=list, default=np.linspace(.1, 1, 5))
    parser.add_argument('--f', type=float, default=1.3)
    """Experimental arguments"""
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    main(**vars(args))
