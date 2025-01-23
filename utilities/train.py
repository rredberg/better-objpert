import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
from utilities.models import LogisticModel, PerturbedLogisticModel
from utilities.utils import transform_y

from torch.nn.functional import one_hot


def train(model, train_loader, optimizer, verbose):
    """
    Trains a model for one epoch -- for now, this will always be a logistic regression model.
    """
    model.train()

    sigmoid = nn.Sigmoid()

    train_loss = 0
    correct = 0
    num_examples = 0


    for i, (inputs, target) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.float()

        target = target.float().reshape(-1, 1)

        output = model(inputs)
        loss = model.loss_func(output, target)

        loss.backward()
        optimizer.step()

        train_loss += model.loss_func(output, target).item()
        output_sig = sigmoid(output)

        correct += sum(output_sig.round() == target).item()
        num_examples += len(inputs)

    train_acc = 100. * correct / num_examples

    if verbose:

        print(f'Train set: Average loss: {train_loss:.4f}, '
        f'Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)')
        
        

def test(model, test_loader, verbose):
    """
    Evaluates the model on the test set.
    """
    model.eval()

    sigmoid = nn.Sigmoid()

    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for (inputs, target) in test_loader:

            inputs = inputs.float()
            target = target.float().reshape(-1, 1)
            output = model(inputs)
            test_loss = model.loss_func(output, target).item()
            output_sig = sigmoid(output)
            correct += sum(output_sig.round() == target).item()
            num_examples += len(inputs)

    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    if verbose:

        print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)')

    return test_acc


def train_lbfgs_NP(model, train_loader, n_epochs, lr, verbose):
    """
    Trains a model with the L-BFGS optimizer.
    params:
        model -- Pytorch model (linear or logistic),
        train_loader -- loads training data. For L-BFGS, we want batch_size = len(train_data).
    """
    model.train()
    optimizer = torch.optim.LBFGS(model.parameters(), lr)
        
    for epoch in range(n_epochs):
    
        for i, (inputs, target) in enumerate(train_loader):
            
            inputs = inputs.float()
            target = target.float().flatten()
            
            def loss_closure():
                optimizer.zero_grad()
                output = model(inputs).flatten()
                loss = model.loss_func(output, target)
                loss.backward()

                sigmoid = torch.nn.Sigmoid()
                train_loss = model.loss_func(output, target).item()
                output_sig = sigmoid(output)
                correct = sum(output_sig.round() == target).item()
                train_loss /= len(inputs)
                train_acc = 100. * correct / len(inputs)

                if verbose:
                    
                    print(f'Train set: Average loss: {train_loss:.4f}, '
                      f'Accuracy: {correct}/{len(inputs)} ({train_acc:.2f}%)')
                
                return loss
            
            optimizer.step(loss_closure)

def train_lbfgs_NP_linear(model, train_loader, n_epochs):
    """
    Trains a model with a perturbed loss function using LBFGS with a stopping condition that the gradient norm be less than
    tolerance_grad.
    params:
        model -- Pytorch model with a perturbed loss,
        train_loader -- loads training data. For LBFGS, we want batch_size = len(train_data),
        n_epochs -- number of epochs to train for.
    """
    model.train()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)
        
    for epoch in range(n_epochs):
    
        for i, (inputs, target) in enumerate(train_loader):
            
            inputs = inputs.float()
            target = target.float().flatten()
            
            def loss_closure():
                optimizer.zero_grad()
                output = model(inputs).flatten()
                loss = model.loss_func(output, target)
                loss.backward()

                return loss
            
            optimizer.step(loss_closure)

def lbfgs_step(model, optimizer, train_loader, verbose):
    """
    Trains a model with a perturbed loss function using LBFGS with a stopping condition that the gradient norm be less than
    tolerance_grad.
    params:
        model -- Pytorch model with a perturbed loss,
        optimizer -- LBFGS optimizer
        train_loader -- loads training data. For LBFGS, we want batch_size = len(train_data),
        verbose (bool) -- decides what intermediary output to print.
    """
    model.train()
    
    for i, (inputs, target) in enumerate(train_loader):
        
        inputs = inputs.float()
        target = target.float().flatten()
        
        def closure():
            optimizer.zero_grad()
            output = model(inputs).flatten()
            loss = model.loss_func(output, target)
            loss.backward()

            sigmoid = torch.nn.Sigmoid()
            train_loss = model.loss_func(output, target).item()
            output_sig = sigmoid(output)
            correct = sum(output_sig.round() == target).item()
            train_loss /= len(inputs)
            train_acc = 100. * correct / len(inputs)

            if verbose:
                
                print(f'Train set: Average loss: {train_loss:.4f}, '
                  f'Accuracy: {correct}/{len(inputs)} ({train_acc:.2f}%)')

            return loss

        optimizer.step(closure=closure)

def add_output_noise(model, sigma_out):
    for p in model.parameters():
        noise = torch.normal(mean=torch.zeros(p.shape), std=sigma_out*torch.ones(p.shape))
        p.data.copy_(p.data + noise)

def grad_norm(model, optimizer, data_loader, k=None):
    optimizer.zero_grad()
    inputs, target  = list(data_loader)[0]
    inputs = inputs.float()
    if k:
        target = transform_y(target, k)
    target = target.float()
    output = model(inputs).flatten()
    loss = model.loss_func(output, target)
    loss.backward()
    grad_norm = torch.norm(gather_flat_grad(model))
    return grad_norm

def grad_norm_multi(model, optimizer, data_loader, k):
    optimizer.zero_grad()
    inputs, target  = list(data_loader)[0]
    inputs = inputs.float()
    target = target.float()
    output = model(inputs).flatten()
    loss = model.loss_func(output, target[:, k])
    loss.backward()
    grad_norm =  torch.norm(gather_flat_grad(model))
    return grad_norm

def gather_flat_grad(model):
    views = []
    for p in model.parameters():
        if p.grad is None:
            view = p.new(p.numel()).zero_()
        elif p.grad.is_sparse:
            view = p.grad.to_dense().view(-1)
        else:
            view = p.grad.view(-1)
        views.append(view)
    return torch.cat(views, 0)

