import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utilities.models import PerturbedLogisticModel, PerturbedLinearModel
from autodp.autodp_core import Mechanism
from scipy.sparse import csr_matrix
from scipy.stats import rv_discrete, norm
from scipy.optimize import minimize_scalar
from scipy import special
from math import log
from copy import deepcopy

from torch.nn.functional import one_hot

""" Data and model utilities """

def load_data(dataset, data_path, batch_size=None):
	"""
	Loads
		train features: X_train,
		train labels  : Y_train,
		test  features: X_test,
		test  labels:   Y_test
	from the path data_path/dataset.
	"""
	path = os.path.join(data_path, dataset)
	x_train_path = os.path.join(path, 'x_train.npy')
	x_test_path = os.path.join(path, 'x_test.npy')
	y_train_path = os.path.join(path, 'y_train.npy')
	y_test_path = os.path.join(path, 'y_test.npy')
	train_dataset = DPMLDataset(x_train_path, y_train_path)
	test_dataset = DPMLDataset(x_test_path, y_test_path)
	if batch_size:
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # remove drop_last later
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
	else:
		train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
		test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
	return train_loader, test_loader


class DPMLDataset(Dataset):
	def __init__(self, data_path_x, data_path_y):
		self.X = torch.tensor(np.load(data_path_x, allow_pickle=True))
		# if len(self.X) == 3: # X is in sparse representation
		# 	data, row, col = self.X
		# 	coo = csr_matrix((data, (row, col)), shape=(data.shape[0], 20958)).tocoo()
		# 	self.X = torch.sparse.LongTensor(torch.LongTensor([coo.row.tolist(), coo.col.tolist()]),
  #                             torch.LongTensor(coo.data.astype(np.int32)))
		self.Y = torch.tensor(np.load(data_path_y, allow_pickle=True))

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.Y[idx]

	def dim(self):
		return self.X.shape[1]

def transform_y(y, k):
    y_k = deepcopy(y)
    class_ind = torch.where(y == k)
    non_class_ind = torch.where(y != k)
    y_k[class_ind] = 1
    y_k[non_class_ind] = 0
    return y_k

def get_num_classes(dataset, data_path):
    train_loader, test_loader = load_data(dataset, data_path)
    for _, y in train_loader:
        if 0 in y and len(torch.unique(y)) - 1 in y:
            return len(torch.unique(y))
        else:
            print("Danger! The y labels should range from 0...num_classes - 1.")

def perturbed_model(model_type, clip, L, sigma, lambd, train_loader):
	model_dict = {'linear': PerturbedLinearModel, 'logistic': PerturbedLogisticModel}
	if clip:
		model = model_dict[model_type](train_loader.dataset.dim(), 1, sigma, lambd, L, train_loader.dataset.X)
	else:
		model = model_dict[model_type](train_loader.dataset.dim(), 1, sigma, lambd)
	return model

""" Noise Caching utilities """		

def construct_key(args_dict):
	"""
	Constructs the dictionary key used for noise caching.
	"""
	return ''.join(['{key}={val}, '.format(key=str(key), val=str(val)) for key, val in args_dict.items()])[:-2]

def load_cache(path, name):
	try:
		print(os.path.join(path, name) + '.npy')
		cache = np.load(os.path.join(path, name) + '.npy', allow_pickle=True).item()
	except FileNotFoundError:
		cache = {}
	return cache

def save_cache(cache, path, name):
	np.save(os.path.join(path, name), cache)

class Cache(object):
	def __init__(self, func):
		self.func = func

	def __call__(self, args_dict):
		cache = load_cache(args_dict['noise_path'], args_dict['alg'])
		key = construct_key(args_dict)
		if key in cache:
			return cache[key]
		else:
			val = self.func(args_dict)
			cache[key] = val
			save_cache(cache, args_dict['noise_path'], args_dict['alg'])
			return val

""" Objective Perturbation utilities """

class ObjectivePerturbationMechanism(Mechanism):
    """
    param params:
        'sigma' --- is the noise level (linear perturbation of objective function)
        'lambd' --- is the regularization level (L2-norm squared)
        'L' -- is the smoothness constant (upper bound on the L2-norm of the gradient of the loss function)
        'beta' -- is the Lipschitz-smoothness constant (upper bound on max eigenvalue of the Hessian of the loss function)
        'd' -- is the dimension of the data
        'glm' -- is a Boolean indicating a generalized linear model
    """
    def __init__(self,params,name='ObjectivePerturbation'):
        
        Mechanism.__init__(self)
        
        self.name=name
        self.params={'sigma':params['sigma'],'lambd':params['lambd'], 'L':params['L'], 'beta':params['beta']}

        new_rdp = lambda x: RDP_objpert(self.params, x)
        self.propagate_updates(new_rdp, type_of_update='RDP')

def RDP_objpert(params, alpha):
    """
    :param params:
        'sigma' --- is the noise level (linear perturbation of objective function)
        'lambd' --- is the regularization level (L2-norm squared)
        'L' -- is the smoothness constant (upper bound on the L2-norm of the gradient of the loss function)
        'beta' -- is the Lipschitz-smoothness constant (upper bound on max eigenvalue of the Hessian of the loss function)
        'd' -- is the dimension of the data
        'glm' -- is a boolean indicating whether or not the model is a generalized linear model
    :param alpha: The order of the Renyi Divergence
    :return: Evaluation of the RDP's epsilon
    """
    sigma = params['sigma']
    lambd = params['lambd']
    L = params['L']
    beta = params['beta']
    if alpha > 1:
        return abs(-log(1 - beta/lambd)) + (L ** 2)/(2 * sigma ** 2) +  1/(alpha - 1) *  stable_mgf_foldnorm(alpha - 1, 0, L/sigma)
    elif alpha == 1:
        return abs(-log(1 - beta/lambd)) + (L ** 2)/(2 * sigma ** 2) + stable_mgf_foldnorm(1, 0, L/sigma)
    else:
        return 0

def stable_mgf_foldnorm(alpha, mu, sigma):
    """A numerically stable implementation of the log moment-generating function of the folded normal distribution.
    alpha -- moment is of order alpha
    mu -- mean of the original normal distribution
    sigma -- variance of the original normal distribution."""
    return (sigma ** 2) * (alpha ** 2)/2 + mu * alpha + log(norm.cdf(mu/sigma + sigma * alpha) + np.exp(-2 * mu * alpha) * norm.cdf(-mu/sigma + sigma * alpha))

""" Private Selection utilities """
def get_prod(k, gamma, eta):
    prod = 1
    for ell in range(int(k)):
        prod *= (ell + eta) / (ell + 1)
    return prod

class TNB(rv_discrete):

    def _pmf(self, k, gamma, eta):
        assert(0 < gamma < 1)
        assert(eta > -1)
        if eta == 0:
            return ((1 - gamma) ** k) / (k * log(1/gamma))
        else:
            prod_func = lambda x: get_prod(x, gamma, eta)
            prod_map = list(map(prod_func, k))
            return prod_map * ((1 - gamma) ** k) / (gamma ** (-eta) - 1)

def choose_gamma(q, grid_size, eta):
    fun = lambda gamma: abs(1 - TNB(a=1).cdf(grid_size - 1, gamma, eta) - q)
    return minimize_scalar(fun, bounds=[0, 1], method='bounded').x

def Phi(z):
    return 0.5*(1 + special.erf(z/np.sqrt(2)))

def compute_delta(Delta,eps,sigma,beta,lam):

    lim1 = abs(np.log(1-beta/lam)) + Delta**2/(2*sigma**2)

    if eps > lim1:
        eps_tilde = eps - abs(np.log(1-beta/lam))
        dd=  Phi(Delta/(2*sigma) - eps_tilde*sigma/Delta) - np.exp(eps_tilde)*Phi(-Delta/(2*sigma) - eps_tilde*sigma/Delta)
        delta=2*dd
    else:
        eps_hat = eps - abs(np.log(1-beta/lam)) -  Delta**2/(2*sigma**2)
        eps_tilde = Delta**2/(2*sigma**2)
        dd=  Phi(Delta/(2*sigma) - eps_tilde*sigma/Delta) - np.exp(eps_tilde)*Phi(-Delta/(2*sigma) - eps_tilde*sigma/Delta)
        delta = (1-np.exp(eps_hat)) + 2*np.exp(eps_hat)*dd

    return delta

def bisection_method(f, a, b, delta, sensitivity,beta,lam,epsilon,tol):
    if np.sign(f(sensitivity,epsilon,a,beta,lam)-delta) == np.sign(f(sensitivity,epsilon,b,beta,lam)-delta):
        raise Exception(
         "The scalars a and b do not bound a root")

    m = (a + b)/2

    if np.abs(f(sensitivity,epsilon,m,beta,lam)-delta) < tol:
        return m
    elif np.sign(f(sensitivity,epsilon,a,beta,lam)-delta) == np.sign(f(sensitivity,epsilon,m,beta,lam)-delta):
        return bisection_method(f, m, b, delta, sensitivity,beta,lam,epsilon,tol)
    elif np.sign(f(sensitivity,epsilon,b,beta,lam)-delta) == np.sign(f(sensitivity,epsilon,m,beta,lam)-delta):
        return bisection_method(f, a, m, delta, sensitivity,beta,lam,epsilon,tol)

