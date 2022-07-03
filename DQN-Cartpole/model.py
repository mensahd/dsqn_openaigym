import math

import torch
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from sklearn.preprocessing import MinMaxScaler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, architecture, seed):
        """Initialize parameters and build model.
        Params
        ======
            architecture:
        """
        super(QNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.seed = torch.manual_seed(seed)
        for i in range(len(architecture) - 1):
            self.layers.append(nn.Linear(architecture[i], architecture[i + 1]))

    def forward(self, x):
        """Build a network that maps state -> action values."""
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # no ReLu activation in the output layer
        return self.layers[-1](x)

    def forward_return_all(self, x):
        """
        Same as the forward function but returns all intermediate results that are necessary for
        conversion to a spiking network
        """
        all_neurons_output = []
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
            all_neurons_output.append(x)

        # no ReLu activation in the output layer
        x = self.layers[-1](x)
        all_neurons_output.append(x)
        return all_neurons_output


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


class DSNN(nn.Module):
    def __init__(self, architecture, seed, alpha, beta, batch_size, threshold, simulation_time, scaler = MinMaxScaler,
                 two_neuron = False, population_coding = False, population_size=1, add_bias = False, encoding="potential", decoding="potential" ):
        """

        """

        self.two_neuron = two_neuron
        self.add_bias = add_bias
        self.population_coding = population_coding

        self.encoding = encoding
        self.decoding = decoding

        self.architecture = architecture[:]

        self.population_size = population_size

        if self.two_neuron:
            self.architecture[0] = self.architecture[0]*2
        if self.population_coding:
            self.architecture[0] = self.architecture[0] * self.population_size
        if self.add_bias:
            self.architecture[0] = self.architecture[0] + 1

        self.seed = random.seed(seed)

        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.simulation_time = simulation_time
        self.batch_size = batch_size
        self.spike_fn = SurrGradSpike.apply

        # Initialize the network weights
        self.weights = []
        for i in range(len(architecture) - 1):
            self.weights.append(torch.empty((self.architecture[i], self.architecture[i + 1]),
                                            device=device, dtype=torch.float, requires_grad=True))
            torch.nn.init.normal_(self.weights[i], mean=0.0, std=1)
        self.scaler = scaler

        # if self.scaler == MinMaxScaler and self.two_neuron:
        #    self.scaler.get_feature_names_out()


    def forward(self, inputs, loihi=False):
        syn = []
        mem = []
        for l in range(0, len(self.weights)):
            syn.append(torch.zeros((self.batch_size, self.weights[l].shape[1]), device=device,
                                   dtype=torch.float))
            mem.append(torch.zeros((self.batch_size, self.weights[l].shape[1]), device=device,
                                   dtype=torch.float))

        # Here we define two lists which we use to record the membrane potentials and output spikes
        mem_rec = []
        spk_rec = []

        # Here we loop over time
        for t in range(self.simulation_time):
            # append the new timestep to mem_rec and spk_rec
            mem_rec.append([])
            spk_rec.append([])

            if t == 0:
                for l in range(len(self.weights)):
                    mem_rec[-1].append(mem[l]*self.batch_size)
                    spk_rec[-1].append(mem[l]*self.batch_size)

            # We take the input as it is, multiply is by the weights, and we inject the outcome
            # as current in the neurons of the first hidden layer
            input = inputs.detach().clone()

            # loop over layers
            for l in range(len(self.weights)):
                if loihi:
                    # define impulse
                    if l == 0:
                        h = torch.einsum("ab,bc->ac", [input, self.weights[0]])
                    else:
                        h = torch.einsum("ab,bc->ac", [spk_rec[-1][l - 1], self.weights[l]*64])
                else:
                    if l == 0:
                        h = torch.einsum("ab,bc->ac", [input, self.weights[0]])
                    else:
                        h = torch.einsum("ab,bc->ac", [spk_rec[-1][l - 1], self.weights[l]])

                # calculate the new synapse potential
                if l == 0:
                    new_syn = 0*syn[l] + h
                else:
                    new_syn = self.alpha*syn[l] + h

                new_mem = self.beta*mem[l] + new_syn

                # calculate the spikes for all layers but the last layer (decoding='potential')
                if l < (len(self.weights) - 1):
                    mthr = new_mem - self.threshold
                    out = self.spike_fn(mthr)
                    c = (mthr > 0)
                    new_mem[c] = 0
                else:
                    # else reset is 0 (= no reset)
                    mthr = new_mem - self.threshold
                    out = self.spike_fn(mthr)
                    c = torch.zeros_like(new_mem, dtype=torch.bool, device=device)

                mem[l] = new_mem
                syn[l] = new_syn

                mem_rec[-1].append(mem[l])
                spk_rec[-1].append(out)

        # return the final recorded membrane potential (len(mem_rec)-1) in the output layer (-1)
        return mem_rec[-1][-1], mem_rec, spk_rec

    def plot_voltage_traces(self, mem, spk=None, dim=(1, 1), spike_height=5):
        gs = GridSpec(*dim)
        if spk is not None:
            dat = (mem + spike_height * spk).detach().cpu().numpy()
        else:
            dat = mem.detach().cpu().numpy()
        for i in range(np.prod(dim)):
            if i == 0:
                a0 = ax = plt.subplot(gs[i])
            else:
                ax = plt.subplot(gs[i], sharey=a0)
            ax.plot(dat[i])
            ax.axis("off")

    def load_state_dict(self, layers):
        """Method to load weights and biases into the network"""
        weights = layers[0]
        #biases = layers[1]
        for l in range(0,len(weights)):
            self.weights[l] = weights[l].detach().clone().requires_grad_(True)

    def state_dict(self):
        """Method to copy the layers of the SQN. Makes explicit copies, no references."""
        weights_copy = []
        bias_copy = []
        for l in range(0, len(self.weights)):
            weights_copy.append(self.weights[l].detach().clone())
        return weights_copy, bias_copy

    def parameters(self):
        parameters = []
        for l in range(0, len(self.weights)):
            parameters.append(self.weights[l])

        return parameters
