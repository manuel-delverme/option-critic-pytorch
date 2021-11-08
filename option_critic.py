from math import exp
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli


class OptionCriticFeatures(nn.Module):
    def __init__(self, in_features, num_actions, num_options, temperature=1.0, eps_start=1.0, eps_min=0.1, eps_decay=int(1e6), eps_test=0.05, device='cpu', testing=False):
        super(OptionCriticFeatures, self).__init__()

        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0

        self.features = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.Q = nn.Linear(64, num_options)  # Policy-Over-Options
        self.terminations = nn.Sequential(
            nn.Linear(64, num_options),
            nn.Tanh(),
        )
        self.Q_options = nn.Linear(64, num_actions * num_options)

        self.to(device)
        self.train(not testing)

    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        # Q = self.Q(state)
        # next_option = Q.argmax(dim=-1)
        return bool(option_termination.item())  # , next_option.item()

    def option_pi(self, option, state):
        logits_Q = self.Q_options(state).reshape(-1, self.num_options, self.num_actions)
        logits_Qo = logits_Q[torch.arange(logits_Q.shape[0]), option, :].squeeze(1)
        # logits_Qo = logits_Q.gather(1, option)
        action_dist = (logits_Qo / self.temperature).softmax(dim=-1)
        return Categorical(action_dist)

    def get_action(self, state, option):
        action_dist = self.option_pi(option, state)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy

    def greedy_option(self, state):
        return self.Q(state).argmax(dim=-1).item()

    def epsgreedy_option(self, state):
        if self.testing and np.random.rand() < self.epsilon:
            current_option = np.random.choice(self.num_options)
        else:
            current_option = self.Q(state).argmax(dim=-1).item()
        return current_option

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps


class OptionCriticConv(OptionCriticFeatures):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = self.in_features
        del self.in_features
        self.magic_number = 7 * 7 * 64
        self.num_steps = 0

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.modules.Flatten(),
            nn.Linear(self.magic_number, 512),
            nn.ReLU()
        )

        self.Q = nn.Linear(512, self.num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(512, self.num_options)  # Option-Termination
        self.Q_options = nn.Linear(512, self.num_actions * self.num_options)

        self.to(self.device)
        self.train(not self.testing)


class OptionCriticTabular(OptionCriticFeatures):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = self.Q.weight.device

        self.features = nn.Identity()
        self.Q = nn.Linear(self.in_features, self.num_options)
        self.terminations = nn.Sequential(
            nn.Linear(self.in_features, self.num_options),
            nn.Tanh(),
        )
        self.Q_options = nn.Linear(self.in_features, self.num_actions * self.num_options)
        self.to(device=device)


def critic_loss(model, model_prime, obs, options, rewards, next_obs, dones, discount):
    assert len(obs.shape) == 2
    assert len(options.shape) == 2
    assert len(dones.shape) == 2

    # batch_idx = torch.arange(len(options)).long()
    # options = torch.LongTensor(options).to(model.device)
    # rewards = torch.FloatTensor(rewards).to(model.device)
    masks = ~dones

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    states = model.features(obs)
    Q = model.Q(states)

    # the update target contains Q_next, but for stable learning we use prime network for this
    next_states_prime = model_prime.features(next_obs)
    next_Q_prime = model_prime.Q(next_states_prime)  # detach?

    # Additionally, we need the beta probabilities of the next state
    next_states = model.features(next_obs)
    next_termination_probs = model.terminations(next_states).detach()
    next_options_term_prob = next_termination_probs.gather(-1, options)

    next_continuation_prob = 1 - next_options_term_prob
    # masks = ~dones
    # next_Qo_prime = next_Q_prime.gather(-1, options)
    Vnext = next_Q_prime.max(dim=-1, keepdim=True).values

    # Now we can calculate the update target gt
    next_Qo_prime = next_Q_prime.gather(-1, options)
    gt = rewards + masks * discount * (next_continuation_prob * next_Qo_prime + next_options_term_prob * Vnext)

    # to update Q we want to use the actual network, not the prime

    return F.mse_loss(Q.gather(-1, options), gt.detach(), reduction="mean")


def actor_loss(obs, options, logps, entropies, rewards, dones, next_obs, model: OptionCriticFeatures, model_prime: OptionCriticFeatures, discount: float, termination_reg: float,
               entropy_reg: float):
    assert len(obs.shape) == 2
    assert len(options.shape) == 2
    assert len(logps.shape) == 2
    assert len(entropies.shape) == 2
    assert len(dones.shape) == 2

    state = model.features(obs)
    next_state = model.features(next_obs)
    next_state_prime = model_prime.features(next_obs)

    option_term_prob = model.terminations(state).gather(-1, options)
    next_options_term_prob = model.terminations(next_state).gather(-1, options).detach()

    Q = model.Q(state).detach().squeeze()
    next_Q_prime = model_prime.Q(next_state_prime).detach().squeeze()

    # Target update gt
    next_continuation_prob = 1 - next_options_term_prob
    masks = ~dones
    next_Qo_prime = next_Q_prime.gather(-1, options)
    Vnext = next_Q_prime.max(dim=-1, keepdims=True).values

    gt = rewards + masks * discount * (next_continuation_prob * next_Qo_prime + next_options_term_prob * Vnext)

    # The termination loss
    Qo = Q.gather(-1, options)
    V = Q.max(dim=-1, keepdims=True).values.detach()
    termination_loss = option_term_prob * (Qo.detach() - V + termination_reg) * masks

    # actor-critic policy gradient with entropy regularization
    policy_loss = -logps * (gt.detach() - Qo) - entropy_reg * entropies
    return (termination_loss + policy_loss).sum()
