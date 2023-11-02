from IPython import embed
from collections import deque
import torch.nn.functional as F
import torch.nn as nn
import torch
import gym
import os
import random
import numpy as np
from tqdm import tqdm
import copy
import time

class Context(nn.Module):
    """
      This layer just does non-linear transformation(s)
    """

    def __init__(self,
                 hidden_sizes=[50],
                 input_dim=None,
                 hidden_activation=F.relu,
                 history_length=None,
                 n_recurrent_layer=3,
                 action_dim=None,
                 obsr_dim=None,
                 device='cpu',
                 model_type='gru'
                 ):

        super(Context, self).__init__()
        self.hid_act = hidden_activation
        self.fcs = []  # list of linear layer
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        self.hist_length = history_length
        self.device = device
        self.action_dim = action_dim
        self.obsr_dim = obsr_dim
        self.n_recurrent_layer = n_recurrent_layer
        self.model_type = model_type
        # build LSTM or multi-layers FF
        if model_type == 'lstm':
            self.recurrent = nn.LSTM(self.input_dim,
                                     self.hidden_sizes[0],
                                     bidirectional=False,
                                     batch_first=True,
                                     num_layers=self.n_recurrent_layer)
        elif model_type == 'gru':
            self.recurrent = nn.GRU(self.input_dim,
                                    self.hidden_sizes[0],
                                    bidirectional=False,
                                    batch_first=True,
                                    num_layers=self.n_recurrent_layer)
        elif model_type == 'ff':
            self.fcs.append(nn.Linear(self.input_dim, self.hidden_sizes[0]))
            for i in range(len(self.hidden_sizes) - 1):
                self.fcs.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
            self.fcs = nn.ModuleList(self.fcs)
        else:
            raise NotImplementedError

    def init_recurrent(self, bsize=None):
        '''
            init hidden states
            Batch size can't be none
        '''
        # The order is (num_layers, minibatch_size, hidden_dim)
        # LSTM ==> return (torch.zeros(1, bsize, self.hidden_sizes[0]),
        #        torch.zeros(1, bsize, self.hidden_sizes[0]))
        return torch.zeros(1, bsize, self.hidden_sizes[0]).to(self.device)

    def forward(self, data):
        '''
            pre_x : B * D where B is batch size and D is input_dim
            pre_a : B * A where B is batch size and A is input_dim
            previous_reward: B * 1 where B is batch size and 1 is input_dim
        '''
        bsize, _, _ = data.shape
        # init lstm/gru
        if self.model_type == 'lstm' or self.model_type == 'gru':
            self.recurrent.flatten_parameters()
            _, hidden = self.recurrent(data)  # hidden is (1, B, hidden_size)
            if self.n_recurrent_layer == 1:
                out = hidden.squeeze(0)  # (1, B, hidden_size) ==> (B, hidden_size)
            else:
                #             print('hidden shape', hidden.shape)
                out = hidden[0].squeeze(0)  # (1, B, hidden_size) ==> (B, hidden_size)
            return out
        elif self.model_type == 'ff':
            out = data
            for fc in self.fcs:
                out = self.hid_act(fc(out))
            return out

# generate a world model


class WorldModel_Hybrid(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hist_len=None,
                 hidden_dim=80,
                 hidden_recurrent=20,
                 n_recurrent_layer=3,
                 representation_dim=10,
                 device='cpu',
                 model_type='ff'):
        super(WorldModel_Hybrid, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_recurrent = hidden_recurrent
        self.representation_dim = representation_dim
        self.n_recurrent_layer = n_recurrent_layer
        self.device = device
        self.context = Context(hidden_sizes=[self.hidden_recurrent],
                               input_dim=state_dim + action_dim + 1,
                               history_length=hist_len,
                               n_recurrent_layer=n_recurrent_layer,
                               action_dim=action_dim,
                               obsr_dim=state_dim,
                               device=self.device,
                               model_type=model_type)
        self.l1 = nn.Linear(self.state_dim + self.action_dim + self.hidden_recurrent, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.l3 = nn.Linear(self.hidden_dim, self.representation_dim)

        self.reward_model = nn.Sequential(
            nn.Linear(representation_dim, 1),
        )

    def forward(self, state, action, preinfo):
        # preinfo is a list of [previous_action, previous_reward, pre_x]
        combined = self.context(preinfo)
        x = torch.cat([state, action, combined], 1)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        reward_next = self.reward_model(h)
        return reward_next

    def get_representation(self, state, action, preinfo):
        # preinfo is a list of [previous_action, previous_reward, pre_x]
        combined = self.context(preinfo)
        x = torch.cat([state, action, combined], 1)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h

    def loss(self, state, action, preinfo, reward_next):
        reward_next_pred = self.forward(state, action, preinfo)
        loss = F.mse_loss(reward_next_pred, reward_next)
        return loss

class BC_NN(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hist_len=None,
                 hidden_dim=80,
                 hidden_recurrent=20,
                 n_recurrent_layer=3,
                 representation_dim=10,
                 device='cpu',
                 model_type='ff'):
        super(BC_NN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_recurrent = hidden_recurrent
        self.representation_dim = representation_dim
        self.n_recurrent_layer = n_recurrent_layer
        self.device = device
        self.context = Context(hidden_sizes=[self.hidden_recurrent],
                               input_dim=state_dim + action_dim + 1,
                               history_length=hist_len,
                               n_recurrent_layer=n_recurrent_layer,
                               action_dim=action_dim,
                               obsr_dim=state_dim,
                               device=self.device,
                               model_type=model_type)
        self.l1 = nn.Linear(self.state_dim + self.hidden_recurrent, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.l3 = nn.Linear(self.hidden_dim, self.representation_dim)

        self.linear_out = nn.Sequential(
            nn.Linear(representation_dim, self.action_dim),
        )

    def forward(self, state, preinfo):
        # preinfo is a list of [previous_action, previous_reward, pre_x]
        combined = self.context(preinfo)
        x = torch.cat([state, combined], 1)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        act_pred = self.linear_out(h)
#         print('act pred shaoe (batched)', act_pred.shape)
        return act_pred

    def forward_single(self, state, preinfo):
        # preinfo is a list of [previous_action, previous_reward, pre_x]
        combined = self.context(preinfo).reshape(1, -1)
        x = torch.cat([state, combined], 1)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        act_pred = self.linear_out(h)
#         print('act pred shaoe (single)', act_pred.shape)
        return act_pred

    def loss(self, state, preinfo, action):
        action_pred = self.forward(state, preinfo)
        loss = F.mse_loss(action_pred, action)
        return loss

def eval_policy(policy=None,
                hist_len=None,
                env_name=None):
    env = gym.make(env_name)
    state_eval = env.reset(options={"x_init": 0, "y_init": 0})[0]
    eval_r_list = []
    previous_info = deque(maxlen=hist_len)
    for i in range(hist_len):
        previous_info.append(
            np.hstack((np.zeros_like(state_eval),
                       np.zeros(env.action_space.shape),
                       np.zeros(1))))
    for step_i in range(ENV_MAX_STEP):
        a = policy.forward_single(torch.as_tensor(state_eval).float().cuda().unsqueeze(0), torch.as_tensor(previous_info).float().cuda().unsqueeze(0)).cpu().detach().numpy()[0]
        next_state_eval, r, terminated, truncated, _ = env.step(a)
        previous_info.append(
            np.hstack((np.asarray(state_eval),
                       np.asarray(a),
                       np.asarray([r]))))
        state_eval = next_state_eval
        eval_r_list.append(r)
        print('decision step', step_i, 'reward', r)
        if terminated or truncated:
            break
    print('total reward', np.sum(eval_r_list))
    return np.sum(eval_r_list)

def accountable_batched_controller(buffer_sa,
                                     preinfo_set,
                                     buffer_r_cumavg,
                                     n_rollout=100,
                                     n_keep=5,
                                     hist_len=None,
                                     n_epoch=2000,
                                     env_name=None,
                                     use_latent=True,
                                     use_hist=True,
                                     quantile=0.99):
    env = gym.make(env_name)
    state_eval = env.reset(options={"x_init": 0, "y_init": 0})[0]
    eval_r_list = []
    if use_latent:

        state = torch.tensor(buffer_sa[:, :ENV_STATE_DIM], dtype=torch.float32)
        action = torch.tensor(buffer_sa[:, ENV_STATE_DIM:], dtype=torch.float32)
        preinfo = torch.tensor(preinfo_set, dtype=torch.float32)
        # do the following in a batch:
        bs_ = 1000
        buffer_latent_rep = []
        for i in range(0, len(state), bs_):
            buffer_latent_rep.append(world_model.get_representation(state[i:i+bs_].cuda(), action[i:i+bs_].cuda(), preinfo[i:i+bs_].cuda()).cpu().detach())
        buffer_latent_rep = torch.cat(buffer_latent_rep, 0)
        # buffer_latent_rep = world_model.get_representation(state, action, preinfo).cpu().detach()
    else:
        if use_hist:
            buffer_sa = np.concatenate((buffer_sa, np.asarray(preinfo_set).reshape(len(preinfo_set), -1)), 1)
    previous_info = deque(maxlen=hist_len)
    for i in range(hist_len):
        previous_info.append(
            np.hstack((np.zeros_like(state_eval),
                       np.zeros(env.action_space.shape),
                       np.zeros(1))))
    cum_rew = 0
    for step_i in range(ENV_MAX_STEP):
        start_time = time.time()
        sampled_action = np.random.uniform(ENV_ACTION_LOW, ENV_ACTION_HIGH, (n_rollout, ENV_ACTION_DIM))
        forw_state_eval = state_eval[np.newaxis, :].repeat(n_rollout, 0)
        rep_pre_info = np.array(previous_info)[np.newaxis, :, :].repeat(n_rollout, 0)
        if not use_latent and use_hist:
            explor_xa = np.concatenate((forw_state_eval, sampled_action), 1)
            explor_xa = np.concatenate((explor_xa, rep_pre_info.reshape(n_rollout, -1)), 1)
        else:
            explor_xa = np.concatenate((forw_state_eval, sampled_action), 1)
#             print('explor xa', explor_xa.shape)
        if use_latent:
            test_state = torch.tensor(explor_xa[:, :ENV_STATE_DIM], dtype=torch.float32).cuda()
            test_action = torch.tensor(explor_xa[:, ENV_STATE_DIM:], dtype=torch.float32).cuda()
            test_preinfo = torch.tensor(np.array(previous_info)[np.newaxis, :, :].repeat(n_rollout, 0), dtype=torch.float32).cuda()
#             print(test_state.shape, test_action.shape, test_preinfo.shape)
            test_latent_rep = world_model.get_representation(test_state, test_action, test_preinfo).cpu().detach()
        print('time for sampling', time.time() - start_time)
        mid_time = time.time()
        if not use_latent:
            if not use_hist:
                weights, idx_list, out_err = LP_solver_KNN(test_latent_reps=torch.as_tensor(explor_xa).cuda(),
                                                           train_latent_reps=torch.as_tensor(buffer_sa).cuda(),
                                                           n_epoch=n_epoch,
                                                           n_keep=n_keep)
            else:
                weights, idx_list, out_err = LP_solver_KNN(test_latent_reps=torch.as_tensor(explor_xa).cuda(),
                                                           train_latent_reps=torch.as_tensor(buffer_sa).cuda(),
                                                           n_epoch=n_epoch,
                                                           n_keep=n_keep)
        else:
            weights, idx_list, out_err = LP_solver_KNN(test_latent_reps=torch.as_tensor(test_latent_rep).cuda(),
                                                       train_latent_reps=torch.as_tensor(buffer_latent_rep).cuda(),
                                                       n_epoch=n_epoch,
                                                       n_keep=n_keep)
        print('time for LP', time.time() - mid_time)
        mid_time2 = time.time()
        err_reg = (out_err > np.quantile(out_err, quantile)) * -99
        a = sampled_action[np.argmax(err_reg + (buffer_r_cumavg.reshape(-1,)[idx_list] * weights.cpu().numpy()).sum(1))]

        next_state_eval, r, terminated, truncated, _ = env.step(a)
        if use_latent:
            previous_info.append(
                np.hstack((np.asarray(state_eval),
                           np.asarray(a),
                           np.asarray([r]))))
        state_eval = next_state_eval
        eval_r_list.append(r)
        cum_rew += r
        print('decision step', step_i, 'reward', r, 'cum_rew', cum_rew)
        if terminated or truncated:
            break
        print('time for forward step', time.time() - mid_time2)
    print('total reward', np.sum(eval_r_list))
    return np.sum(eval_r_list)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action,
                hist_len=2,
                 hidden_dim=128,
                 hidden_recurrent=30,
                 n_recurrent_layer=1,
                 representation_dim=20,
                 device='cuda',
                 model_type='gru'):
        super(Actor, self).__init__()
        self.hidden_recurrent = hidden_recurrent
        self.device = device
        self.l1 = nn.Linear(state_dim + hidden_recurrent, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.context = Context(hidden_sizes=[self.hidden_recurrent],
                               input_dim=state_dim + action_dim + 1,
                               history_length=hist_len,
                               n_recurrent_layer=n_recurrent_layer,
                               action_dim=action_dim,
                               obsr_dim=state_dim,
                               device=self.device,
                               model_type=model_type)


        self.max_action = max_action


    def forward(self, state, pre_info):
        state = state * PRE_SET_STATE_MULTIPLIER
        combined = self.context(pre_info)
        x = torch.cat([state, combined], 1)

        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,hist_len=4,
                 hidden_dim=80,
                 hidden_recurrent=20,
                 n_recurrent_layer=1,
                 representation_dim=20,
                 device='cuda',
                 model_type='gru'):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim + hidden_recurrent, 256)
        self.l2 = nn.Linear(256, representation_dim)
        self.l3 = nn.Linear(representation_dim, 1)
        self.context = Context(hidden_sizes=[hidden_recurrent],
                               input_dim=state_dim + action_dim + 1,
                               history_length=hist_len,
                               n_recurrent_layer=n_recurrent_layer,
                               action_dim=action_dim,
                               obsr_dim=state_dim,
                               device=device,
                               model_type=model_type)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim + hidden_recurrent, 256)
        self.l5 = nn.Linear(256, representation_dim)
        self.l6 = nn.Linear(representation_dim, 1)


    def forward(self, state, action, pre_info):
        state = state * PRE_SET_STATE_MULTIPLIER
        combined = self.context(pre_info)

        sa = torch.cat([state, action, combined], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action, pre_info):
        state = state * PRE_SET_STATE_MULTIPLIER
        combined = self.context(pre_info)
        sa = torch.cat([state, action, combined], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def get_representation(self, state, action, pre_info):
        state = state * PRE_SET_STATE_MULTIPLIER
        combined = self.context(pre_info)
        sa = torch.cat([state, action, combined], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.002,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state, pre_info):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        pre_info = torch.FloatTensor(pre_info).unsqueeze(0).to(device)
        return self.actor(state, pre_info).cpu().data.numpy().flatten()


    def train(self, buffer_sa, buffer_r, preinfo_set, bs=100, update_policy=True):
        self.total_it += 1

        # Sample replay buffer
        idx = np.random.choice(buffer_sa.shape[0]-1, bs)
        state = torch.tensor(buffer_sa[idx, :ENV_STATE_DIM], dtype=torch.float32).cuda()
        next_state = torch.tensor(buffer_sa[idx+1, :ENV_STATE_DIM], dtype=torch.float32).cuda()
        action = torch.tensor(buffer_sa[idx, ENV_STATE_DIM:], dtype=torch.float32).cuda()
        next_action = torch.tensor(buffer_sa[idx+1, ENV_STATE_DIM:], dtype=torch.float32).cuda()
        reward = torch.tensor(buffer_r[idx], dtype=torch.float32).cuda()
        pre_info = torch.tensor(preinfo_set[idx], dtype=torch.float32).cuda()
        pre_info_next= torch.tensor(preinfo_set[idx+1], dtype=torch.float32).cuda()
#         state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state, pre_info_next) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, pre_info_next)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, pre_info)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            if update_policy:
                # Compute actor losse
                actor_loss = -self.critic.Q1(state, self.actor(state, pre_info), pre_info).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if update_policy:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def MPPI_planner(init_state, prev_inf, world_model, reward_model, horizon=10, num_rollouts=20, n_refine=10, gamma=0.1):
    """
    MPPI planner: Model Predictive Path Integral (Williams et al. 2016)
    """
    action_dim = ENV_ACTION_DIM
    sigma = 0.3  # this is a hyperparameter
    mu = np.random.uniform(-1, 1, (horizon, action_dim))
    orig_prev_inf = copy.deepcopy(prev_inf)
    for n in range(n_refine):
        prev_inf = copy.deepcopy(orig_prev_inf)
        # rollout to update mu:
        state = init_state[np.newaxis, :].repeat(num_rollouts, 0) # n_roll, 3
        reward_list = []
        action_list = []

        for t in range(horizon):
            # select action
            action = mu[t] + np.random.randn(num_rollouts, action_dim) * sigma  # shape: (num_rollouts, action_dim)
            state = torch.as_tensor(state).float().cuda()
            action = torch.as_tensor(action).float().cuda().clip(-1,1)

            pre_info = torch.as_tensor(np.asarray(prev_inf)).float().cuda()

            # get reward
            reward = reward_model(state, action, pre_info).cpu().detach().squeeze().numpy()
            # update state
            pred_ds = world_model(state, action, pre_info).cpu().detach().numpy()

            reward_list.append(reward)
            action_list.append(action.cpu().detach().numpy())
            for j in range(num_rollouts):
                prev_inf[j].append(np.hstack((state[j].cpu().detach().numpy(),
                               action[j].cpu().detach().numpy(),
                               reward[j])))
            state = pred_ds + state.cpu().detach().numpy()

        # perform Eq.(7) in DADS (ref: https://arxiv.org/pdf/1907.01657.pdf)
        # mu_i = \sum_{k=1}^K  (exp(gamma * r_k)) / (\sum_{p=1}^K exp(gamma * r_p)) * action_p
        reward_list = np.asarray(reward_list).sum(0)  # shape: (num_rollouts, action_dim)
        action_list = np.asarray(action_list)  # shape: (horizon, num_rollouts, action_dim)

        exp_reward = np.exp(gamma * reward_list)  # shape: (num_rollouts, )
        exp_reward_sum = np.sum(exp_reward)  # shape: ()
        mu = np.sum(exp_reward[np.newaxis, :, np.newaxis].repeat(horizon, 0).repeat(action_dim, -1) * action_list, 1) / \
            exp_reward_sum  # shape: (horizon, num_rollouts)

    optimal_action = mu[0]  # shape: (horizon, action_dim)
    return optimal_action

def eval_policy_td3(policy):
    env = gym.make('Pendulum-v1')
    state_eval = env.reset(options={"x_init": 0, "y_init": 0})[0]
    eval_r_list = []
    previous_info = deque(maxlen=args.hist_len)
    for i in range(args.hist_len):
        previous_info.append(
            np.hstack((np.zeros_like(state_eval),
                       np.zeros(env.action_space.shape),
                       np.zeros(1))))
    for step_i in range(ENV_MAX_STEP):
        a = policy.select_action(state_eval, previous_info)
        next_state_eval, r, terminated, truncated, _ = env.step(a)
        previous_info.append(
            np.hstack((np.asarray(state_eval),
                       np.asarray(a),
                       np.asarray([r]))))
        state_eval = next_state_eval
        eval_r_list.append(r)
        if terminated or truncated:
            break
        #print('decision step', step_i, 'reward', r)
    #print('total reward', np.sum(eval_r_list))
    return np.sum(eval_r_list)

class WorldModel_Dynamics(nn.Module):
        def __init__(self, state_dim, action_dim,
                     hist_len=None,
                     hidden_dim=80,
                     hidden_recurrent=20,
                     n_recurrent_layer=3,
                     representation_dim=10,
                     device='cpu',
                     model_type='ff'):
            super(WorldModel_Dynamics, self).__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            self.hidden_recurrent = hidden_recurrent
            self.representation_dim = representation_dim
            self.n_recurrent_layer = n_recurrent_layer
            self.device = device
            self.context = Context(hidden_sizes=[self.hidden_recurrent],
                                   input_dim=state_dim + action_dim + 1,
                                   history_length=hist_len,
                                   n_recurrent_layer=n_recurrent_layer,
                                   action_dim=action_dim,
                                   obsr_dim=state_dim,
                                   device=self.device,
                                   model_type=model_type)
            self.l1 = nn.Linear(self.state_dim + self.action_dim + self.hidden_recurrent, self.hidden_dim)
            self.l2 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.l3 = nn.Linear(self.hidden_dim, self.representation_dim)

            self.dynamics_model = nn.Linear(representation_dim, self.state_dim)

        def forward(self, state, action, preinfo):
            # preinfo is a list of [previous_action, previous_reward, pre_x]
            combined = self.context(preinfo)
            x = torch.cat([state, action, combined], 1)
            h = F.relu(self.l1(x))
            h = F.relu(self.l2(h))
            h = self.l3(h)
            pred_next_state = self.dynamics_model(h)
            return pred_next_state

        def get_representation(self, state, action, preinfo):
            # preinfo is a list of [previous_action, previous_reward, pre_x]
            combined = self.context(preinfo)
            x = torch.cat([state, action, combined], 1)
            h = F.relu(self.l1(x))
            h = F.relu(self.l2(h))
            h = self.l3(h)
            return h

        def loss(self, state, action, preinfo, next_state):
            pred_next_state = self.forward(state, action, preinfo)
            loss = F.mse_loss(pred_next_state, next_state)
            return loss

def LP_solver_KNN(
    test_latent_reps: torch.Tensor,
    train_latent_reps: torch.Tensor,
    n_epoch: int = 100,
    reg_factor: float = 0.1,
    n_keep: int = 20,
) -> None:
    corpus_size = train_latent_reps.shape[0]
    n_test = test_latent_reps.shape[0]
    preweights = torch.zeros(
        (n_test, n_keep),
        device=test_latent_reps.device,
        requires_grad=True,
    ).cuda()
    optimizer = torch.optim.Adam([preweights])
    idx_list = torch.zeros((n_test, n_keep), dtype=torch.long)
    # generate kNN masks
    # find the k-nearest neighbors of each test point
    for i in range(n_test):
        dist = ((train_latent_reps - test_latent_reps[i]) ** 2).sum(1)
        _, idx = torch.sort(dist)
        idx_list[i] = idx[:n_keep]

    # select index of k-nearest neighbors
    train_latent_reps = train_latent_reps[idx_list]

    for epoch in range(n_epoch):
        optimizer.zero_grad()
        weights = F.softmax(preweights, dim=-1)
        # weights shape is (n_test, n_keep)
        # train_latent_reps shape is (n_test, n_keep, representation_dim)
        # time those two together:
        corpus_latent_reps = (weights.unsqueeze(-1) * train_latent_reps).sum(1)
        error = ((corpus_latent_reps - test_latent_reps) ** 2).sum()
        #weights_sorted = torch.sort(weights)[0]
        #regulator = (weights_sorted[:, : (corpus_size - n_keep)]).sum()
        loss = error
        loss.backward()
        optimizer.step()
        if (epoch + 1) % n_epoch == 0:
            print(
                f"Weight Fitting Epoch: {epoch+1}/{n_epoch} ; Error: {error.item():.3g} ;"
            )
    final_error = ((corpus_latent_reps - test_latent_reps) ** 2).sum(1).cpu().detach()
    weights = torch.softmax(preweights, dim=-1).cpu().detach()
    return weights, idx_list, final_error


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Pendulum-v1')
    parser.add_argument('--n_rollout', type=int, default=1000)
    parser.add_argument('--hist_len', type=int, default=2)
    parser.add_argument('--n_epoch', type=int, default=2000)
    parser.add_argument('--rand_seed', type=int, default=0)
    parser.add_argument('--dsize', type=int, default=500000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--quantile', type=float, default=0.5)
    args = parser.parse_args()
    # set random seed
    # torch.manual_seed(args.rand_seed)
    # np.random.seed(args.rand_seed)
    # random.seed(args.rand_seed)

    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alias = f"env_{args.env_name}_rollout_{args.n_rollout}_keep_autofill_gamma_{args.gamma}_epoch_{args.n_epoch}_seed_{args.rand_seed}_dsize_{args.dsize}_quantile_{args.quantile}"
    buffer = np.load(f'Dataset/pendulum_forward.npy', allow_pickle=True).item()
    buffer_sa = np.concatenate((buffer['state'], buffer['action']), 1)
    buffer_r = buffer['reward']
    buffer_r_cumavg = np.zeros_like(buffer_r)

    get_args_env = gym.make(args.env_name)
    ENV_MAX_STEP = get_args_env._max_episode_steps
    ENV_STATE_DIM = get_args_env.observation_space.shape[0]
    ENV_ACTION_DIM = get_args_env.action_space.shape[0]
    ENV_ACTION_HIGH = get_args_env.action_space.high[0]
    ENV_ACTION_LOW = get_args_env.action_space.low[0]
    ENV_SA_DIM = ENV_STATE_DIM + ENV_ACTION_DIM

    for mod in range(ENV_MAX_STEP):
        idx = [i*ENV_MAX_STEP + mod for i in range(int(1e6/ENV_MAX_STEP))]
        for j in range(int(1e6/ENV_MAX_STEP)):
            buffer_r_cumavg[idx[j]] = buffer_r[idx[j]: (j+1)*ENV_MAX_STEP].mean()

    LEN = args.dsize
    buffer_sa = buffer_sa[:LEN]
    buffer_r = buffer_r[:LEN]
    buffer_r_cumavg = buffer_r_cumavg[:LEN]
    buffer2 = np.load(f'Dataset/pendulum_inverse.npy', allow_pickle=True).item()
    buffer_sa2 = np.concatenate((buffer2['state'], buffer2['action']), 1)
    buffer_r2 = buffer2['reward']
    buffer_r_cumavg2 = np.zeros_like(buffer_r2)
    for mod in range(ENV_MAX_STEP):
        idx = [i*ENV_MAX_STEP + mod for i in range(int(1e6/ENV_MAX_STEP))]
        for j in range(int(1e6/ENV_MAX_STEP)):
            buffer_r_cumavg2[idx[j]] = buffer_r2[idx[j]: (j+1)*ENV_MAX_STEP].mean()

    buffer_sa2 = buffer_sa2[:LEN]
    buffer_r2 = buffer_r2[:LEN]
    buffer_r_cumavg2 = buffer_r_cumavg2[:LEN]

    buffer_sa = np.concatenate((buffer_sa, buffer_sa2), 0)
    buffer_r = np.concatenate((buffer_r, buffer_r2), 0)
    buffer_r_cumavg = np.concatenate((buffer_r_cumavg, buffer_r_cumavg2), 0)
    buffer_ns = np.concatenate((buffer['next_state'][:LEN], buffer2['next_state'][:LEN]), 0)
    LEN = args.dsize * 2

    # build preinfo set based on buffer_sa
    preinfo_set = []
    padded_buffer_sa = np.concatenate((np.zeros((args.hist_len, ENV_SA_DIM)), buffer_sa), 0)
    padded_buffer_r = np.concatenate((np.zeros((args.hist_len, 1)), buffer_r.reshape(-1, 1)), 0)
    for i in range(LEN):
        preinfo_set.append(np.concatenate((padded_buffer_sa[i:i+args.hist_len],
                                           padded_buffer_r[i:i+args.hist_len]), 1))

    preinfo_set = np.array(preinfo_set)

    world_model = WorldModel_Hybrid(state_dim=ENV_STATE_DIM,
                                    action_dim=ENV_ACTION_DIM,
                                    hidden_dim=128,
                                    hidden_recurrent=128,
                                    n_recurrent_layer=1,
                                    hist_len=args.hist_len,
                                    representation_dim=20,
                                    device=device,
                                    model_type='gru').cuda()
    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)

    bc_policy = BC_NN(state_dim=ENV_STATE_DIM,
                      action_dim=ENV_ACTION_DIM,
                      hidden_dim=128,
                      hidden_recurrent=128,
                      n_recurrent_layer=1,
                      hist_len=args.hist_len,
                      representation_dim=64,
                      device=device,
                      model_type='gru').cuda()
    optimizer = torch.optim.Adam(bc_policy.parameters(), lr=1e-3)
    for epoch in range(4000):
        idx = np.random.choice(buffer_sa.shape[0]-1, 5000)
        state = torch.tensor(buffer_sa[idx, :ENV_STATE_DIM], dtype=torch.float32).cuda()
        pre_info = torch.tensor(preinfo_set[idx], dtype=torch.float32).cuda()
        action = torch.tensor(buffer_sa[idx, ENV_STATE_DIM:], dtype=torch.float32).cuda()
        loss = bc_policy.loss(state, pre_info, action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('epoch', epoch, 'loss', loss.item())
    bc_results = []
    for rep_bc_i in range(1):
        bc_results.append(eval_policy(policy=bc_policy, hist_len=args.hist_len, env_name = args.env_name))


    # Q-Learning
    PRE_SET_STATE_MULTIPLIER = 1
    policy = TD3(state_dim = ENV_STATE_DIM, action_dim = ENV_ACTION_DIM, max_action = ENV_ACTION_HIGH)
    eval_hist = []
    for epoch in range(100000):
        batch_size = np.random.choice([10240])
        policy.train(buffer_sa, buffer_r, preinfo_set, bs = batch_size)
        if (epoch+1)% 100 ==0:
            avg_return = eval_policy_td3(policy)
            eval_hist.append(avg_return)
            print('epoch', epoch+1, 'avg_return', avg_return)
            if epoch > 50000 and np.std(eval_hist[-5:]) < 1 and np.mean(eval_hist[-5:]) > -1:
                break # early stopping...


    for epoch in range(4000):
        idx = np.random.choice(buffer_sa.shape[0], 500)
        state = torch.tensor(buffer_sa[idx, :ENV_STATE_DIM], dtype=torch.float32).cuda()
        action = torch.tensor(buffer_sa[idx, ENV_STATE_DIM:], dtype=torch.float32).cuda()
        reward_next = torch.tensor(buffer_r[idx], dtype=torch.float32).cuda()

        loss = world_model.loss(state, action, torch.tensor(preinfo_set[idx]).float().cuda(), reward_next)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('epoch', epoch, 'loss', loss.item())
    esc_results_latent_100_mb = accountable_batched_controller(buffer_sa,
                                                            preinfo_set,
                                                            buffer_r_cumavg,
                                                            n_rollout=args.n_rollout,
                                                            n_epoch = 1000,
                                                            n_keep=100,
                                                            hist_len=args.hist_len,
                                                            env_name = args.env_name,
                                                            quantile=args.quantile)

    world_model = policy.critic_target
    esc_results_latent_100 = accountable_batched_controller(buffer_sa,
                                                            preinfo_set,
                                                            buffer_r_cumavg,
                                                            n_rollout=args.n_rollout,
                                                            n_epoch = 1000,
                                                            n_keep=100,
                                                            hist_len=args.hist_len,
                                                            env_name = args.env_name,
                                                            quantile=args.quantile)
    knn_latent_100 = accountable_batched_controller(buffer_sa,
                                                      preinfo_set,
                                                      buffer_r_cumavg,
                                                      n_rollout=args.n_rollout,
                                                      n_epoch = 1000,
                                                      n_keep=100,
                                                      use_latent=True,
                                                      hist_len=args.hist_len,
                                                      env_name = args.env_name,
                                                      quantile=1.0)
    knn_result_100 = accountable_batched_controller(buffer_sa,
                                                      preinfo_set,
                                                      buffer_r_cumavg,
                                                      n_rollout=args.n_rollout,
                                                      n_epoch = 1000,
                                                      n_keep=100,
                                                      use_latent=False,
                                                      hist_len=args.hist_len,
                                                      env_name = args.env_name,
                                                      quantile=1.0)
    knn_result_1 = accountable_batched_controller(buffer_sa,
                                                      preinfo_set,
                                                      buffer_r_cumavg,
                                                      n_rollout=args.n_rollout,
                                                      n_epoch = 1000,
                                                      n_keep=1,
                                                      hist_len=args.hist_len,
                                                      env_name = args.env_name,
                                                      quantile=1.0)

    results = {'esc_results_latent_100': esc_results_latent_100,
                'esc_results_latent_100_mb': esc_results_latent_100_mb,
                'knn_latent_100': knn_latent_100,
               'knn_result_100': knn_result_100,
               'knn_result_1': knn_result_1,
               'bc_results': bc_results}



    results['QL_eval'] = eval_hist[-10:]


    # Model-Based + MPC
    reward_model = WorldModel_Hybrid(state_dim=ENV_STATE_DIM,
                                    action_dim=ENV_ACTION_DIM,
                                    hidden_dim=128,
                                    hidden_recurrent=128,
                                    n_recurrent_layer=1,
                                    hist_len=args.hist_len,
                                    representation_dim=64,
                                    device=device,
                                    model_type='gru').cuda()
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
    for epoch in range(10000):
        idx = np.random.choice(buffer_sa.shape[0], 5000)
        state = torch.tensor(buffer_sa[idx, :ENV_STATE_DIM], dtype=torch.float32).cuda()
        action = torch.tensor(buffer_sa[idx, ENV_STATE_DIM:], dtype=torch.float32).cuda()
        reward_next = torch.tensor(buffer_r[idx], dtype=torch.float32).cuda()
        loss = reward_model.loss(state, action, torch.tensor(preinfo_set[idx]).float().cuda(), reward_next)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('epoch', epoch, 'loss', loss.item())



    world_model = WorldModel_Dynamics(state_dim=ENV_STATE_DIM,
                                    action_dim=ENV_ACTION_DIM,
                                    hidden_dim=128,
                                    hidden_recurrent=128,
                                    n_recurrent_layer=1,
                                    hist_len=args.hist_len,
                                    representation_dim=64,
                                    device=device,
                                    model_type='gru').cuda()

    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    for epoch in range(10000):
        idx = np.random.choice(buffer_sa.shape[0], 5000)
        state = torch.tensor(buffer_sa[idx, :ENV_STATE_DIM], dtype=torch.float32).cuda()
        action = torch.tensor(buffer_sa[idx, ENV_STATE_DIM:], dtype=torch.float32).cuda()

        state_next = torch.tensor(buffer_ns[idx], dtype=torch.float32).cuda()

        delta_state = state_next - state

        loss = world_model.loss(state, action, torch.tensor(preinfo_set[idx]).float().cuda(), delta_state)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('epoch', epoch, 'loss', loss.item())




    env_name = args.env_name
    env = gym.make(env_name)
    init_state = env.reset()[0]

    state = init_state
    total_reward = 0
    N_ROLL = 30


    previous_info = []
    for i in range(N_ROLL):
        previous_info.append(deque(maxlen=args.hist_len))
    for j in range(N_ROLL):
        for i in range(args.hist_len):
            previous_info[j].append(
                np.hstack((np.zeros_like(init_state),
                           np.zeros(ENV_ACTION_DIM),
                           np.zeros(1))))

    cum_rew_mbrl = 0
    for t in tqdm(range(ENV_MAX_STEP)):
        action = MPPI_planner(state,
                              copy.deepcopy(previous_info),
                              world_model,
                              reward_model,
                              horizon=20,
                              num_rollouts=N_ROLL,
                              n_refine=30,
                              gamma=0.1)
        action = np.clip(action, -1, 1)
        next_state, reward, done1, done2, _ = env.step(action)
        cum_rew_mbrl += reward
    #     print('action', action)
        print(reward, 'cum_rew', cum_rew_mbrl)
        estimated_next_state = state + world_model(torch.as_tensor(state).cuda().unsqueeze(0), torch.as_tensor(action).cuda().unsqueeze(0), torch.as_tensor(np.asarray(previous_info)).float().cuda()[0].unsqueeze(0)).cpu().detach().numpy()
        for j in range(N_ROLL):
            previous_info[j].append(np.hstack((state,
                                   action,
                                   reward)))
        state = next_state
        total_reward += reward
        if done1 or done2:
            break


    print('episodic reward', total_reward)

    results['mpc'] = total_reward
    os.makedirs('results', exist_ok=True)
    np.save(f'results/results_{alias}.npy', results)
