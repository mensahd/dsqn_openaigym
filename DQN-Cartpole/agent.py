import os
import sys
import gym
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import sunblaze_envs

import numpy as np
import torch.nn.functional as F

from model import DSNN
from collections import namedtuple, deque

sys.path.append('../')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def quantize_tensor(x, min_val, max_val, qmin=-128, qmax=127):
    scale = (max_val - min_val)/(qmax - qmin)

    zero_point = 0
    q_x = zero_point + (x/scale)
    q_x.clamp(qmin, qmax).round_()
    q_x = q_x.round().int()
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward",
                                                                "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).\
            float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).\
            long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).\
            float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).
                                 astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Agent:
    def __init__(self, env, policy_net, target_net, architecture, batch_size, memory_size, gamma,
                 eps_start, eps_end, eps_decay, update_every, target_update_frequency, optimizer,
                 learning_rate, num_episodes, max_steps, i_run, result_dir, seed, tau, SQN=False,
                 random=False, quantization=False):
        if random:
            #self.env = sunblaze_envs.make('SunblazeCartPoleRandomExtreme-v0')
            self.env = gym.make("CartPole-v0")
        else:
            self.env = gym.make(env)
        self.env.seed(seed)
        self.env._max_episode_steps = max_steps
        self.policy_net = policy_net
        self.target_net = target_net

        if quantization:
            self.loihi_policy_net = DSNN(architecture, seed, self.policy_net.alpha,
                                         self.policy_net.beta, self.policy_net.weight_scale,
                                         batch_size, 64, self.policy_net.simulation_time)

        self.architecture = architecture
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.update_every = update_every
        self.target_update_frequency = target_update_frequency
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.i_run = i_run
        self.result_dir = result_dir
        self.seed = seed
        self.tau = tau
        self.SQN = SQN
        self.random = random
        self.quantization = quantization

        # Initialize Replay Memory
        self.memory = ReplayBuffer(self.memory_size, self.batch_size, self.seed)

        # Initialize time step
        self.t_step = 0
        self.t_step_total = 0

        # Init actionspace visualization if necessary
        self.visualize_actionspace = False
        if self.visualize_actionspace:
            self.state_list = np.zeros((1,4))
            self.as_dir = "/content/drive/My Drive/Uni/MasterArbeit/dsqn_examples/" + \
                          "results/action_space_{}".format(self.env.unwrapped.spec.id)
            if not os.path.exists(self.as_dir):
                os.mkdir(self.as_dir)

    def dequantize_tensor(self, q_x):
        return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

    def quantize_weights(self, weights):
        combined_weights = torch.cat([torch.flatten(w) for w in weights])
        min_val = torch.min(combined_weights)
        max_val = torch.max(combined_weights)
        quantized_weights = []
        for w in weights:
            w = quantize_tensor(w, min_val, max_val, qmin=-256, qmax=254).tensor
            odd_mask = (w%2 == 1)
            w[odd_mask] -= 1
            quantized_weights.append(w)

        return quantized_weights

    def select_action(self, state, eps=0.):
        state = torch.from_numpy(state)
        state = state.unsqueeze(0).to(device)
        if random.random() > eps:
            with torch.no_grad():
                if self.SQN:
                    if self.quantization:
                        state = quantize_tensor(state, -4.5, 4.5).tensor.float()
                        weights = self.policy_net.weights
                        q_weights = [q_w.float() for q_w in self.quantize_weights(weights)]
                        self.loihi_policy_net.weights = q_weights
                        final_layer_values = self.loihi_policy_net.forward(state.float(), loihi=True)[0].cpu().data.numpy()
                    else:
                        final_layer_values = self.policy_net.forward(state.float())[0].cpu().data.numpy()

                    return np.argmax(final_layer_values[0])
                else:
                    return np.argmax(self.policy_net.forward(state.float())[0].cpu().data.numpy())
        else:
            return random.choice(np.arange(self.architecture[-1]))

    def plot_hist(self, hist_data,i):
      # plot the space values for the respective dimension for assessment
        fig, ax = plt.subplots(1,1)
        plt.grid(True)
        _ = plt.hist(hist_data, bins=200) 
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))      
        plt.title("Histogram of dimension {}".format(i))  
        plt.savefig(self.as_dir + '/env_hist_dim{}.png'.format(i), dpi=1000)
        plt.close(fig)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if self.visualize_actionspace:
            self.state_list = np.vstack((self.state_list,state))

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.optimize_model(experiences)
        if self.visualize_actionspace and self.t_step_total  % 1000 == 0:
            # If enough samples are available in memory, get random subset and learn
            for i in range(4):
                self.plot_hist(self.state_list[:,i],i)

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        if self.SQN:
            Q_targets_next = self.target_net.forward(next_states)[0].\
                detach().max(1)[0].unsqueeze(1)
                #gather(1, actions)
        else:
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next*(1 - dones))

        # Get expected Q values from local model
        if self.SQN:
            Q_expected = self.policy_net.forward(states)[0].gather(1, actions)
        else:
            Q_expected = self.policy_net.forward(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        if self.t_step_total % self.target_update_frequency == 0:
            self.soft_update()

    def soft_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_agent(self):
        best_average = -np.inf
        best_average_after = np.inf
        scores = []
        smoothed_scores = []
        scores_window = deque(maxlen=100)
        eps = self.eps_start

        for i_episode in range(1, self.num_episodes + 1):
            state = self.env.reset()

            np.round(state,1)

            score = 0
            for t in range(self.max_steps):
                self.t_step_total += 1
                action = self.select_action(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                eps = max(self.eps_end, self.eps_decay * eps)
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            smoothed_scores.append(np.mean(scores_window))

            if smoothed_scores[-1] > best_average:
                best_average = smoothed_scores[-1]
                best_average_after = i_episode
                if self.SQN:
                    if self.quantization:
                        torch.save(self.loihi_policy_net.state_dict(),
                               self.result_dir + '/checkpoint_DSQN_Loihi_{}.pt'.format(self.i_run))
                    else:
                        torch.save(self.policy_net.state_dict(),
                               self.result_dir + '/checkpoint_DSQN_{}.pt'.format(self.i_run))
                else:
                    torch.save(self.policy_net.state_dict(),
                               self.result_dir + '/checkpoint_DQN_{}.pt'.format(self.i_run))

            print("Episode {}\tAverage Score: {:.2f}\t Epsilon: {:.2f}".
                  format(i_episode, np.mean(scores_window), eps), end='\r')

            if i_episode % 100 == 0:
                print("\rEpisode {}\tAverage Score: {:.2f}".
                      format(i_episode, np.mean(scores_window)))

        print('Best 100 episode average: ', best_average, ' reached at episode ',
              best_average_after, '. Model saved in folder best.')
        
        return smoothed_scores, scores, best_average_after


def evaluate_agent(policy_net, env, num_episodes, max_steps, gym_seeds, epsilon=0,
                   SQN=False, quantization=True):
    """

    """
    rewards = []

    for i_episode in range(num_episodes):
        env.seed(int(gym_seeds[i_episode]))
        env._max_episode_steps = max_steps
        state = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #if quantization:
        #    state = quantize_tensor(state, -4.5, 4.5).tensor.float()
        total_reward = 0
        for t in range(max_steps):
            if quantization:
                state = quantize_tensor(state, -4.5, 4.5).tensor.float()
            if random.random() >= epsilon:
                if quantization:
                    final_layer_values = policy_net.forward(state.float(), loihi=True)[0].cpu().data.numpy()
                    action = np.argmax(final_layer_values[0])
                else: 
                    final_layer_values = policy_net.forward(state.float())[0].cpu().data.numpy()
                    action = np.argmax(final_layer_values)
            else:
                action = random.randint(0, env.action_space.n - 1)

            observation, reward, done, _ = env.step(action)
            state = torch.from_numpy(observation).float().unsqueeze(0).to(device)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        print("Episode: {}".format(i_episode), end='\r')

    return rewards
