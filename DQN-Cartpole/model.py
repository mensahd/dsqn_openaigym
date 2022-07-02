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
                 two_neuron = False, population_coding = False, population_size=1, add_bias = True, encoding="potential", decoding="potential" ):
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

        time_step = 1e-3 ### in accordance to the set SNN hyperparameter
        tau_eff = 20e-3/time_step

        # We prep the input for two-neuron encoding
        input = inputs.detach().clone()

        spike_train = self.encode_input_layer(input)

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
            

            # loop over layers
            for l in range(len(self.weights)):
                if loihi:
                    # define impulse
                    if l == 0:
                        input = spike_train[t]
                        h = torch.einsum("ab,bc->ac", [input, self.weights[0]])
                    else:
                        h = torch.einsum("ab,bc->ac", [spk_rec[-1][l - 1], self.weights[l]*64])
                else:
                    if l == 0:
                        input = spike_train[t]
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
        final_layer_output = self.decode_output_layer(mem_rec, spk_rec)

        return mem_rec[-1][-1], mem_rec, spk_rec



    def current2firing_time(self, inputs, tau=20, tmax=1.0, epsilon=1e-7):
        """ Computes the firing times of a spike train based on the poisson distribution

        Args:
        x -- The "current" values

        Keyword args:
        tau -- The membrane time constant of the LIF neuron to be charged
        tmax -- The maximum time returned 
        epsilon -- A generic (small) epsilon > 0

        Returns:
        Time to first spike for each "current" x
        """
        idx = inputs < self.threshold
        clipped_inputs = np.clip(inputs.cpu(),self.threshold+epsilon,1e9)
        T = tau*np.log(clipped_inputs/(clipped_inputs-self.threshold))
        T = np.clip(T, 0, tmax)
        T[idx] = tmax
        return T

    def normalize_inputs(self, inputs, limits):
        differences = [element[1]- element[0] for element in limits]

        shifted_array = inputs - [ element[0] for element in limits]
        normalized_array = shifted_array / differences
        
        return normalized_array

    def transform_two_inputs(self, inputs):
        bigger_zero = inputs.where(inputs > 0, torch.tensor(0.).to(device)).bool()
        smaller_zero = inputs.where(inputs < 0, torch.tensor(0.).to(device)).bool()
        bigger_input = inputs.where(bigger_zero, torch.tensor(0.).to(device))
        smaller_input = inputs * -1
        smaller_input = smaller_input.where(smaller_zero, torch.tensor(0.).to(device))
        split_input = torch.cat((bigger_input, smaller_input),dim=1)
        return split_input


    def encode_input_layer(self, input):

        if self.scaler:
            input = torch.from_numpy(self.scaler.transform(input.cpu())).to(device)
        ## Two neuron encoding with split positive and negative inputs
        if self.two_neuron:
            input = self.transform_two_inputs(input)
        if self.population_coding:
            #
            bracket_size = 1 / self.population_size
            deviation = bracket_size
            shift = bracket_size / 2.0
            value_centers = np.array([i * bracket_size for i in range(self.population_size)]) + shift
            input_pops = np.tile(value_centers, input.size(dim=1))
            input = torch.repeat_interleave(input, self.population_size, dim=-1)
            denom_vals = np.array(input - input_pops)
            input = torch.tensor(np.exp((-0.5) * np.power((denom_vals) / deviation, 2)))
        if self.add_bias:
            bias_factor = 0.5
            input = torch.hstack(
                [input, (torch.ones((input.size(dim=0), 1)) * bias_factor).to(device)])

        if (self.encoding == "ttfs"):

            spike_train = self.encode_ttfs(input)
        elif( self.encoding=="poisson"):
            spike_train = self.encode_poisson(input)
        elif ( self.encoding == "fre"):
            spike_train = self.encode_fre(input)
        else:
            transformed_input = input.unsqueeze(dim=1).float()
            spike_train = transformed_input.repeat( self.simulation_time, 1, 1)

        return spike_train

    def encode_poisson(self, transformed_input):
        # We take the input and create a poisson distro for the input on
        # every timestep, that will be fed into the first level
        dims = (self.simulation_time, transformed_input.shape[0], transformed_input.shape[1])
        random_distribution = torch.tensor(np.random.uniform(low=0, high=1,
                                                             size=dims), device=device)
        spike_train = (transformed_input > random_distribution).float().to(device)

        #reconstruct_input = np.sum(spike_train.cpu().numpy(), axis=0) / self.simulation_time
        return spike_train

    def encode_fre(self, input):
        # stepsize = (self.simulation_time / ((torch.ones(transformed_input.size()) * (transformed_input) * self.simulation_time)+1))
        # bracketsizes = (torch.ones(transformed_input.size()) * (transformed_input) * self.simulation_time)+2
        # preliminary = np.array(np.tile(np.arange(self.simulation_time)[np.newaxis],(transformed_input.shape[0], transformed_input.shape[1],1)))
        # divider_array = np.asarray((transformed_input*self.simulation_time)+2).reshape(transformed_input.shape[0], transformed_input.shape[1],1).astype(int)
        # my_array = np.tile(np.arange(self.simulation_time)[np.newaxis],(2,1))
        twenties = np.arange(self.simulation_time)
        spike_train = []
        for state_array in input:
            values_train = np.ones((0, self.simulation_time))
            for state_value in state_array:
                # Zero values should also have a single spike, the time steps should thus be split into two equal parts
                # Adjustement of the value in regards to the max time step needs to be done
                zero_adjusted_state_value = min(np.round(
                    state_value * self.simulation_time * ((self.simulation_time - 2) / self.simulation_time) + 2),
                    self.simulation_time)
                divided_array = np.array_split(twenties, zero_adjusted_state_value)

                # take the "index" of the end of each subarray to determine the spike indices
                indices = np.array([subarray[-1] for subarray in divided_array[:-1]])

                # create spike trains
                single_train = np.zeros(self.simulation_time)
                np.put_along_axis(single_train, indices, 1, axis=0)
                values_train = np.vstack((values_train, single_train))
            spike_train.append(values_train.T)
        spike_train = torch.tensor(np.transpose(np.array(spike_train), (1, 0, 2))).float()
        return spike_train

    def encode_ttfs(self, input):
        time_step = 1e-3  ### in accordance to the set SNN yperparameter
        tau_eff = 20e-3 / time_step
        spike_train = self.current2firing_time(input, tau_eff, tmax=self.simulation_time - 1)
        spike_train = np.ceil(spike_train).to(device)

        zeros = torch.zeros((input.shape[0], self.simulation_time))

        #TODO: encode spiking correctly
        #spike_values = torch.ones(spike_train.shape, dtype=torch.long)

        return spike_train

    def decode_output_layer(self, mem_rec, spk_rec):
        def first_spike(train, axis, invalid_val=self.simulation_time):
            spikes_times = np.where(train.any(axis=axis), train.argmax(axis=axis), invalid_val)
            np.argmin(spikes_times)

        if self.decoding == "potential":
            return mem_rec[-1][-1]
        elif self.decoding == "ttfs":
            return first_spike(spk_rec, axis=1)
        elif self.decoding == "rate":
            last_layer = [rec[-1] for rec in spk_rec]
            detached_last_layer = torch.stack(last_layer).cpu().detach().numpy()
            return np.sum(detached_last_layer, axis=0) / self.simulation_time
        elif self.decoding == "population":
            return mem_rec[-1][-1]  ##TODO
        else:
            return mem_rec[-1][-1]

    def init_scaler(self):
        #[position of cart, velocity of cart, angle of pole, rotation rate of pole]
        limits = np.asarray([[-5, -3, -0.21 , -3],
                              [5, 3 , 0.21, -3]])
        self.scaler = MinMaxScaler()
        # transform data
        self.scaler.fit(limits)

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
