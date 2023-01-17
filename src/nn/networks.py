import torch
from .layers import *
from .enums import LRPRule, ExplainingMethod
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExplainableNet(nn.Module):
    def __init__(self, model=None, data_mean=0, data_std=1, lrp_rule_first_layer=LRPRule.z_b,
                 lrp_rule_next_layers=LRPRule.alpha_beta, beta=None, gamma=None):
        super(ExplainableNet, self).__init__()

        # replace relus by differentiable counterpart for beta growth
        self.beta = beta
        self.gamma = gamma
        self.activation_fn = F.relu if beta is None else torch.nn.Softplus(beta=self.beta)

        self.layers = nn.ModuleList([])
        self.lrp_rule_first_layer = lrp_rule_first_layer
        self.lrp_rule_next_layers = lrp_rule_next_layers

        self.data_mean = data_mean
        self.data_std = data_std


        if model is not None:
            self.fill_layers(model)
            alpha = 1-self.beta if self.beta is not None else None # note: I'm not sure where to get alpha from.
            self.change_lrp_rules(lrp_rule_fl=lrp_rule_first_layer, lrp_rule_nl=lrp_rule_next_layers, 
                                    alpha=alpha, beta=self.beta, gamma=self.gamma)

        # remove activation function in last layer
        self.layers[-1].activation_fn = None

        self.R = 0

    def fill_layers(self, model):
        if hasattr(model, 'seq'):
            layers = list(model.seq) 
        elif hasattr(model, 'features') and hasattr(model, 'classifier'):
            layers = list(model.features) + list(model.classifier)
        else:
            raise Exception("Unsupported Network Architecture.")

        lrp_rule = self.lrp_rule_first_layer

        for layer in layers:
            new_layer = self.create_layer(layer, lrp_rule)
            if new_layer == 0:
                continue
            self.layers.append(new_layer)
            lrp_rule = self.lrp_rule_next_layers

    def create_layer(self, layer, lrp_rule):
        if type(layer) == torch.nn.Conv2d:
            new_layer = Convolutional(in_channels=layer.in_channels,
                                      out_channels=layer.out_channels,
                                      kernel_size=layer.kernel_size,
                                      stride=layer.stride,
                                      padding=layer.padding,
                                      activation_fn=self.activation_fn,
                                      lrp_rule=lrp_rule,
                                      data_mean=self.data_mean,
                                      data_std=self.data_std)
            new_layer.conv.weight.data = layer.weight.data
            new_layer.conv.bias.data = layer.bias.data

        elif type(layer) == nn.MaxPool2d:
            new_layer = MaxPool(kernel_size=layer.kernel_size,
                                stride=layer.stride,
                                padding=layer.padding)
        elif type(layer) == nn.Linear:
            new_layer = Dense(in_dim=layer.in_features, out_dim=layer.out_features, activation_fn=self.activation_fn,
                              lrp_rule=lrp_rule)
            new_layer.linear.weight.data = layer.weight.data
            new_layer.linear.bias.data = layer.bias.data

        elif isinstance(layer, nn.Flatten):
            new_layer = Flatten(start_dim=layer.start_dim, end_dim=layer.end_dim)

        elif type(layer) == (nn.Dropout or nn.Dropout2d):
            new_layer = layer

        elif type(layer) == nn.ReLU:
            return 0

        else:
            print('ERROR: unknown layer', type(layer))
            return None

        return new_layer

    def change_beta(self, beta):
        self.beta_activation = beta
        for layer in self.layers:
            if hasattr(layer, "activation_fn") and layer.activation_fn is not None:
                layer.activation_fn = torch.nn.ReLU() if self.beta_activation is None else torch.nn.Softplus(beta=self.beta_activation)


    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        self.R = x

        return x

    def change_lrp_rules(self, lrp_rule_fl=LRPRule.z_b, lrp_rule_nl=LRPRule.alpha_beta, alpha=None, beta=None, gamma=None, start_l=0, end_l=-1):
        for layer in self.layers[start_l:end_l]:
            if type(layer) == nn.Dropout or type(layer) == nn.Dropout2d or type(
                    layer) == MaxPool or type(layer) == Flatten:  # ignore dropout and pooling layer layer
                continue
            layer.set_lrp_rule(lrp_rule_nl, alpha, beta, gamma)

        if lrp_rule_fl==LRPRule.z_b:
            self.layers[0].set_lrp_rule(LRPRule.z_b) # z_b rule does not require parameters
        elif lrp_rule_fl is not None:
            self.layers[0].set_lrp_rule(lrp_rule_fl, alpha, beta, gamma)

        

    def classify(self, x):
        outputs = self.forward(x)
        return F.softmax(outputs, dim=1), torch.max(outputs, 1)[1]

    def analyze(self, method=ExplainingMethod.lrp, R=None, index=None):
        if R is None:
            R = self.R.clone()
        if index is not None:
            # mask out everything expect the desired index
            mask = torch.zeros(R.shape, dtype=int)
            mask[np.arange(len(mask)), index] = 1
            R = mask * R

        for layer in reversed(self.layers):
            if type(layer) == nn.Dropout or type(layer) == nn.Dropout2d:  # ignore Dropout layer
                continue
            R = layer.analyze(method, R)

        return R

    def learn_pattern(self, x):
        with torch.no_grad():
            for layer in self.layers:
                # no patterns to be learned for Dropout layers
                if type(layer) == nn.Dropout or type(layer) == nn.Dropout2d:
                    x = layer(x)
                else:
                    x = layer.learn_pattern(x)
        return x

    def compute_pattern(self):
        with torch.no_grad():
            for layer in self.layers:
                # no patterns to be computed for Dropout layers
                if type(layer) == nn.Dropout or type(layer) == nn.Dropout2d:
                    pass
                else:
                    layer.compute_pattern()

    def print_lrp_info(self):
        for i, l in enumerate(self.layers):
            try:    print(i, l.lrp_info())
            except: print(i, str(l).split('(')[0])
