import torch
from torch import nn
import numpy as np

class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()
        
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        
    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)
        out = layer(self._concrete_dropout(x, p))
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))        
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        
        input_dimensionality = x[0].numel() # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        
        regularization = weights_regularizer + dropout_regularizer
        return out, regularization
        
    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1
        unif_noise = torch.rand_like(x)
        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        return x
    
class ConcreteDense(nn.Module):
    def __init__(self, X_dim, Y_dim, nb_features, weight_regularizer, dropout_regularizer, verbose=True):
        super(ConcreteDense, self).__init__()
        self.verbose = verbose
        self.linear1 = nn.Linear(X_dim, nb_features)
        self.linear2 = nn.Linear(nb_features, nb_features)
        self.linear3 = nn.Linear(nb_features, nb_features)

        self.linear4_mu = nn.Linear(nb_features, Y_dim)
        self.linear4_logvar = nn.Linear(nb_features, Y_dim)

        self.conc_drop1 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop2 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop3 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop_mu = ConcreteDropout(weight_regularizer=weight_regularizer,
                                             dropout_regularizer=dropout_regularizer)
        self.conc_drop_logvar = ConcreteDropout(weight_regularizer=weight_regularizer,
                                                 dropout_regularizer=dropout_regularizer)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        regularization = torch.empty(5, device=x.device)
        
        x1, regularization[0] = self.conc_drop1(x, nn.Sequential(self.linear1, self.relu))
        x2, regularization[1] = self.conc_drop2(x1, nn.Sequential(self.linear2, self.relu))
        x3, regularization[2] = self.conc_drop3(x2, nn.Sequential(self.linear3, self.relu))

        mean, regularization[3] = self.conc_drop_mu(x3, self.linear4_mu) # ~ [batch, Y_dim]
        log_var, regularization[4] = self.conc_drop_logvar(x3, self.linear4_logvar) # ~ [batch, Y_dim] 
        
        return mean, log_var, regularization.sum()