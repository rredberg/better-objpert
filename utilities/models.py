import torch

class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

class LogisticModel(Model):
    def __init__(self, input_dim, output_dim):
        super(LogisticModel, self).__init__(input_dim, output_dim)
        self.loss_func = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

class LogisticMulti(Model):
    def __init__(self, input_dim, output_dim):
        super(LogisticMulti, self).__init__(input_dim, output_dim)
        self.loss_func = torch.nn.CrossEntropyLoss()

class LinearModel(Model):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__(input_dim, output_dim)
        self.loss_func = torch.nn.MSELoss()

class PerturbedLogisticModel(Model):
    def __init__(self, input_dim, output_dim, sigma, lambd, clipping_factor=None, inputs=None):
        super(PerturbedLogisticModel, self).__init__(input_dim, output_dim)
        if clipping_factor:
            self.np_loss_func = self.construct_clipped_loss(clipping_factor, inputs)
        else:
            self.np_loss_func = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.lambd = lambd
        self.layer_noise, self.bias_noise = [torch.normal(mean=torch.zeros(size=p.size()), std=sigma*torch.ones(size=p.size())) for p in self.parameters()]

    def regularization(self):       
        return 1/2 * (torch.norm(torch.concat([self.linear.weight.flatten(), self.linear.bias]), p=2)) ** 2
    
    def perturbation(self):
        return torch.dot(self.linear.weight.flatten(), self.layer_noise.flatten()) + self.bias_noise * self.linear.bias
    
    def loss_func(self, output, target):
        loss = self.np_loss_func(output, target) + self.lambd * self.regularization() + self.perturbation()
        return loss

    def sigmoid_inv(self, v):
        return torch.log(v) - torch.log(1 - v)

    def clipped_loss(self, y, B, x_norm):
        logistic_loss = self.construct_logistic_loss(y)
        def clip(x_T_theta):
            z_1 = self.sigmoid_inv(y - B / x_norm)
            z_2 = self.sigmoid_inv(y + B / x_norm)
            if x_T_theta < z_1:
                return (- B / x_norm) * (x_T_theta - z_1) + logistic_loss(z_1)
            elif x_T_theta > z_2:
                return (B / x_norm) * (x_T_theta - z_2) + logistic_loss(z_2)
            else:
                return logistic_loss(x_T_theta)
        return clip            

    def construct_logistic_loss(self, y):
        loss = torch.nn.BCEWithLogitsLoss()
        return lambda z: loss(torch.tensor(z).float(), torch.tensor(y).float())

    def construct_clipped_loss(self, B, inputs): # this could maybe be put into Model later
        def sum_clipped_loss(output, target):
            loss = 0
            for input_i, output_i, target_i in zip(inputs, output, target):
                x_norm = torch.norm(input_i)
                little_clip_loss = self.clipped_loss(target_i, B, x_norm)
                loss += little_clip_loss(output_i)
            return loss
        return sum_clipped_loss

class PerturbedLinearModel(Model):
    def __init__(self, input_dim, output_dim, sigma, lambd, clipping_factor=None, inputs=None):
        super(PerturbedLinearModel, self).__init__(input_dim, output_dim)
        if clipping_factor:
            self.np_loss_func = self.construct_clipped_loss(clipping_factor, inputs)
        else:
            self.np_loss_func = torch.nn.MSELoss(reduction='sum')
        self.lambd = lambd
        self.layer_noise, self.bias_noise = [torch.normal(mean=torch.zeros(size=p.size()), std=sigma*torch.ones(size=p.size())) for p in self.parameters()]

    def regularization(self):       
        return 1/2 * (torch.norm(torch.concat([self.linear.weight.flatten(), self.linear.bias]), p=2)) ** 2
    
    def perturbation(self):
        return torch.dot(self.linear.weight.flatten(), self.layer_noise.flatten()) + self.bias_noise * self.linear.bias
    
    def loss_func(self, output, target):
        loss = self.np_loss_func(output, target) + self.lambd * self.regularization() + self.perturbation()
        return loss

    def clipped_loss(self, y, B, x_norm):
        linear_loss = self.construct_linear_loss(y)
        def clip(x_T_theta):
            z_1 = y - B / x_norm
            z_2 = y + B / x_norm
            if x_T_theta < z_1:
                return (- B / x_norm) * (x_T_theta - z_1) + linear_loss(z_1)
            elif x_T_theta > z_2:
                return (B / x_norm) * (x_T_theta - z_2) + linear_loss(z_2)
            else:
                return linear_loss(x_T_theta)
        return clip            

    def construct_linear_loss(self, y):
        loss = torch.nn.MSELoss()
        return lambda z: loss(torch.tensor(z).float(), torch.tensor(y).float())


    def construct_clipped_loss(self, B, inputs):
        def sum_clipped_loss(output, target):
            loss = 0
            for input_i, output_i, target_i in zip(inputs, output, target):
                x_norm = torch.norm(input_i)
                little_clip_loss = self.clipped_loss(target_i, B, x_norm)
                loss += little_clip_loss(output_i)
            return loss
        return sum_clipped_loss

