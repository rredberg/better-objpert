# better-objpert
This directory contains code to reproduce the results of:

*"Improving the Privacy and Practicality of Objective Perturbation for Differentially Private Linear Learners."*\
**Rachel Redberg, Antti Koskela & Yu-Xiang Wang**\
https://arxiv.org/abs/2401.00583

# Installing
On top of fairly standard dependencies (`numpy`, `scipy`, `scikit-learn`, `matplotlib`), running the code in this repository will also require DP-specific libraries like `opacus` and `autodp`. When in doubt, all requirements can be installed with

`pip install -r requirements.txt`

# Getting the data

Pre-processed data can be found at: https://ucsb.box.com/s/hnlg7cgg3tyklf7kno9zydmc4fid5x46.

The file structure should look something like:

```
.
|-- utilities
|-- data
|   |-- adult
|   |-- gisette
|   |-- synthetic_L
|   |-- synthetic_H
|-- scripts
|-- requirements.txt
```


# Running the code

## Non-private baselines

To reproduce the results for the non-private baselines (using L-BFGS or SGD), run

`python -m scripts.run_np_lbfgs --dataset=[dataset]`   or `python -m scripts.run_np_sgd --dataset=[dataset]`

for datasets `adult`, `gisette`, `synthetic_L` and `synthetic_H`.

## Objective Perturbation

To reproduce the results for Approximate Minima Perturbation (i.e., "Algorithm 1" in the paper), run

`python -m scripts.run_amp --dataset=[dataset] --epsilon=[epsilon]`

for datasets `adult`, `gisette`, `synthetic_L` and `synthetic_H`; and for epsilon values `0.1`, `1`, and `8`.

Adding `--verbose=True` will give more detailed intermediary training output.

Default values for other parameters can also be tweaked, e.g.:

* `sigma_out`: the noise scale $\sigma_{out}$ for output perturbation,
* `tau`: the gradient norm (convergence) threshold $\tau$,
* `learning_rate_grid`: the candidate pool of learning rate values,
* `n_trials`: the number of trials over which to average the results.

## DP-SGD

To reproduce the results for "dishonest" DP-SGD (DP-SGD with non-private hyperparameter tuning), run

`python -m scripts.run_dpsgd --dataset=[dataset] --epsilon=[epsilon]`

Tunable parameters for DP-SGD include:

* `n_epochs`: number of epochs,
* `max_grad_norm`: bound on the gradient norm, i.e. clipping value,
* `batch_size_grid`: the candidate pool of batch sizes,
* `lr_grid`: the candidate pool of learning rate values,
* `optim`: the type of optimizer (`Adam` or `SGD`),
* `n_trials`: the number of trials over which to average the results.

## Private Selection


To reproduce the results for "honest" DP-SGD (DP-SGD with private hyperparameter tuning), run

`python -m scripts.run_select --dataset=[dataset] --epsilon=[epsilon]`

Tunable parameters for DP-SGD with private selection are mostly the same as for DP-SGD, but also include:

* `mu`: the expected number of sampled candidate hyperparameters.

# Accessing the results.

Each experiment produces a list of accuracies (of length `n_trials`) and the associated hyperparameters. All results are stored in the `results` folder (created on the fly) as `.npy` files, and can be opened via `np.load`.
