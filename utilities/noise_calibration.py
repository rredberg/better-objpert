from autodp.autodp_core import Mechanism
from autodp.mechanism_zoo import ExactGaussianMechanism
from utilities.utils import ObjectivePerturbationMechanism
from autodp.mechanism_zoo import NoisySGD_Mechanism
from autodp.calibrator_zoo import eps_delta_calibrator
from autodp.transformer_zoo import Composition
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

import numpy as np
import os

class AMP(Mechanism):
    """
    Approximate minima perturbation (AMP) is a composition of the objective perturbation mechanism and the Gaussian mechanism.
    params --
        sigma_obj: noise scale for objective perturbation
        sigma_out: noise scale for the gaussian mechanism
        lambd: regularization strength for objective perturbation
        beta: smoothness constant of the loss function
        L: Lipschitz constant of the loss function
        tau: gradient norm threshold for approximate minima
    """
    def __init__(self, sigma_obj, sigma_out, lambd, beta, L, tau, name = 'AMP'):

        Mechanism.__init__(self)
        self.name = name
        self.params = {'sigma': sigma_obj, 
                       'sigma_out': sigma_out,
                       'lambd': lambd,
                       'beta': beta,
                       'L': L,
                       'GS': L
                      }
        Delta_f = (2 * tau) / lambd
        objpert_mech = ObjectivePerturbationMechanism(self.params, name = 'ObjPert')
        gaussian_mech = ExactGaussianMechanism(sigma_out / Delta_f, name = 'Release_approx_theta') 
 
        compose = Composition() 
        mech = compose([objpert_mech, gaussian_mech], [1,1])
        
        self.set_all_representation(mech)

def calibrate_amp_fix_lambda(eps, delta, sigma_out, lambd, beta, L, tau):
	"""
	Calibrates the noise scale sigma_obj for the objective perturbation mechanism.
	params --
		(eps, delta): privacy budget for AMP
		sigma_out: noise scale for the gaussian mechanism
		lambd: regularization strength for objective perturbation
		beta: smoothness constant of the loss function
		L: Lipschitz constant of the loss function
		tau: gradient norm threshold for approximate minima
	"""
	amp_mech = lambda x: AMP(x, sigma_out, lambd, beta, L, tau)
	calibrate = eps_delta_calibrator()
	amp_mech_calibrated = calibrate(amp_mech, eps, delta, [0, 2000])
	sigma_obj = amp_mech_calibrated.params['sigma']
	return sigma_obj

def calibrate_amp(eps, delta, sigma_out, beta, L, tau, f):
	"""
	Calibrates the noise scale sigma and the regularization strength lambda for the
	Approximate Minima Perturbation mechanism.
	params --
		(eps, delta): privacy budget for AMP
		sigma_out: noise scale for the gaussian mechanism
		lambd: regularization strength for objective perturbation
		beta: smoothness constant of the loss function
		L: Lipschitz constant of the loss function
		tau: gradient norm threshold for approximate minima perturbation
		f: constant factor that regulates the trade-off between sigma and lambda.
		   We want to find sigma (for ObjPert) that is no more than f * sigma_gaussian, where
		   sigma_gaussian is the noise level for the Gaussian mechanism with the same privacy parameters.
	"""
	lambd = 2 * beta / eps
	sigma = None
	sigma_gaussian = calibrate_gaussian(eps, delta) * L
	while sigma == None or sigma > f * sigma_gaussian:
		try:
		    sigma = calibrate_amp_fix_lambda(eps, delta, sigma_out, lambd, beta, L, tau)
		except RuntimeError:
			pass
		lambd *= 1.05
	return sigma, lambd

def calibrate_dpsgd(eps, delta, prob, n_iters, GS):
    """
    Calibrates noise to add at each iteration of DP-SGD.
    params --
        (eps, delta): privacy budget,
        prob: expected_batch_size / size of training dataset,
        n_iters: number of iterations for which to run DP-SGD.
    """
    dpsgd_fix_params = lambda x: NoisySGD_Mechanism(prob, x, n_iters)
    calibrate = eps_delta_calibrator()
    mech = calibrate(dpsgd_fix_params, eps, delta, [0,500])
    sigma = mech.params['sigma']
    return sigma

def calibrate_select(eps, delta, prob, n_iters, GS, dist, dist_params):
	p_select = PrivateSelection()
	p_select_fix_params = lambda x: p_select(NoisySGD_Mechanism(prob, x, n_iters), dist, dist_params)
	calibrate = eps_delta_calibrator()
	mech = calibrate(p_select_fix_params, eps, delta, [0, 100])
	sigma = mech.params['sigma']
	return sigma

def calibrate_gaussian(eps, delta):
	gaussian_mech = lambda x: ExactGaussianMechanism(x)
	calibrate = eps_delta_calibrator()
	gaussian_mech_calibrated = calibrate(gaussian_mech, eps, delta, [0, 2000])
	return gaussian_mech_calibrated.params['sigma']

def calibrate_sigma_select(epsilon, delta, mu, sample_rate, n_epochs, GS, max_orders=256, sigma_min=0.1, sigma_max=1000, tol=1e-6):

	def get_eps_final(sigma):
		epsilons_dpsgd = compute_rdp(q=sample_rate, noise_multiplier=sigma, steps=steps, orders=alphas)
		epsilon_hats = np.log(1+1/(alphas-1))
		# This is the conversion formula by Canonne et al., 2020
		delta_hats=[]
		for eps_ in epsilon_hats:
		    delta_ = min((np.exp((alphas-1)*(epsilons_dpsgd-eps_))/alphas)*((1-1/alphas)**(alphas-1)))
		    delta_hats.append(delta_)
		# This is Thm. 6 by Papernot and Steinke, RDPs with the Poisson distributed K
		epsilons_tuning = epsilons_dpsgd + mu*np.array(delta_hats) + np.log(mu)/(alphas-1)
		eps_final, alpha = get_privacy_spent(orders=alphas, rdp=epsilons_tuning, delta=delta)
		return eps_final

	alphas = np.arange(2,max_orders+1)
	steps = int(1/sample_rate)*n_epochs
	eps_search = 10e6
	while eps_search > epsilon or abs(eps_search - epsilon) > tol:
		# do binary search
		eps_min = get_eps_final(sigma_min)
		eps_max = get_eps_final(sigma_max)
		sigma_search = (sigma_min + sigma_max) / 2
		eps_search = get_eps_final(sigma_search)
		if eps_search > epsilon:
			sigma_min = sigma_search
		else:
			sigma_max = sigma_search
	return GS * sigma_search

def calibrate(args_dict):
	method = args_dict['alg']
	if method == 'amp':
		return calibrate_amp(args_dict['eps'], args_dict['delta'], args_dict['sigma_out'],
			 		  args_dict['beta'], args_dict['L'], args_dict['tau'], args_dict['f'])
	elif method == 'dpsgd':
		return calibrate_dpsgd(args_dict['eps'], args_dict['delta'], args_dict['prob'], args_dict['n_iters'], args_dict['GS'])
	elif method == 'select':
		return calibrate_sigma_select(args_dict['eps'], args_dict['delta'], args_dict['mu'], args_dict['prob'], args_dict['n_epochs'], args_dict['GS'])
	else:
		raise Exception("{} is not a valid method. Key 'alg' needs to be one of: amp, dpsgd, select".format(method))
