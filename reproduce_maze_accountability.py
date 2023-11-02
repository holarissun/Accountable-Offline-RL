import gym
import numpy as np
from gym.spaces import Box
from IPython import embed
import torch

ENV_MAX_STEP = 60
ENV_STATE_DIM = 2
ENV_ACTION_DIM = 2
ENV_ACTION_HIGH = 1
ENV_ACTION_LOW = -1
ENV_SA_DIM = ENV_STATE_DIM + ENV_ACTION_DIM


torch.cuda.set_device(0)

default_config = dict(
    early_done=False,
    int_initialize=False,
    use_walls=True,
    init_loc=[0,0],  # None for random location
    _show=True,
    max_ts = None
)

class Wall:
    def __init__(self, p1, p2):
        self.start = tuple(p1)
        self.end = tuple(p2)

    def intersect(self, p3, p4):
        """Return True if line (self.start, self.end) intersect with line
        (p3, p4)"""
        p1 = self.start
        p2 = self.end
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]
        v1x = p3[0] - p1[0]
        v1y = p3[1] - p1[1]
        v2x = p4[0] - p1[0]
        v2y = p4[1] - p1[1]
        if (vx * v1y - vy * v1x) * (vx * v2y - vy * v2x) > 1e-6:
            return False
        vx = p4[0] - p3[0]
        vy = p4[1] - p3[1]
        v1x = p1[0] - p3[0]
        v1y = p1[1] - p3[1]
        v2x = p2[0] - p3[0]
        v2y = p2[1] - p3[1]
        if (vx * v1y - vy * v1x) * (vx * v2y - vy * v2x) > 1e-6:
            return False
        return True

class FourWayGridWorld(gym.Env):
    def __init__(self, env_config=None):
        self.N = 16
        self.observation_space = Box(0, self.N, shape=(2,))
        self.action_space = Box(-1, 1, shape=(2,))
        self.config = default_config
        if isinstance(env_config, dict):
            self.config.update(env_config)

        self.map = np.ones((self.N + 1, self.N + 1), dtype=np.float32)
        self.map.fill(-0.1)
        self._fill_map()
        if isinstance(env_config['max_ts'], int):
            self.max_tx = env_config['max_ts']
        else:
            self.max_tx = self.N * 2
        self.walls = []
        self._fill_walls()
        self.reset()

    def _fill_walls(self):
        """Let suppose you have three walls, two vertical, one horizontal."""
        if self.config['use_walls']:
            print('Building Walls!!!')
            for wall_i in self.config['wall_list']:
                self.walls.append(wall_i)
            # self.walls.append(Wall([8,0], [8, 7.5]))
            # self.walls.append(Wall([8, 8.5], [8, 15.5]))

    def _fill_map(self):
        self.map[self.config['rewarding_loc'][0], self.config['rewarding_loc'][1]] = self.config['reward']
        self.traj = []
        self.traj_hist = []

    @property
    def done(self):
        if self.map[int(round(self.loc[0])), int(round(self.loc[1]))] > 0:
            return True
        else:
            return self.step_num >= self.max_tx

    def step(self, action):
        action = np.clip(action, -1, 1).astype(np.float32)
        new_loc = np.clip(self.loc + action, 0, self.N)
        if any(w.intersect(self.loc, new_loc) for w in self.walls):
            '''this is for MDP-Rejection'''
            pass
            '''this is for MDP-ET'''
            # reward = - 10.
            # self.step_num +=1 
            # self.traj.append(self.loc)
            # return self.loc, reward, True, {}
        else:
            self.loc = new_loc
        reward = self.map[int(round(self.loc[0])), int(round(self.loc[1]))]
        self.step_num += 1
        self.traj.append(self.loc)
        return self.loc, reward, self.done, {}

    def render(self, mode=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        img = ax.imshow(
            np.transpose(self.map)[::-1, :], aspect=1,
            extent=[-0.5, self.N + 0.5, -0.5, self.N + 0.5], cmap=plt.cm.hot_r
        )
        fig.colorbar(img)
        ax.set_aspect(1)
        for w in self.walls:
            x = [w.start[0], w.end[0]]
            y = [w.start[1], w.end[1]]
            ax.plot(x, y, c='orange')
        if len(self.traj) > 0:
            traj = np.asarray(self.traj)
            ax.plot(traj[:, 0], traj[:, 1], c='blue', alpha=0.75)
        ax.set_xlabel('X-coordinate', fontsize= 13)
        ax.set_ylabel('Y-coordinate', fontsize= 13)
        ax.set_xlim(0, self.N)
        ax.set_ylim(0, self.N)
        if self.config["_show"]:
            plt.show()
        return fig, ax

    def render_hist(self, mode=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        img = ax.imshow(
            np.transpose(self.map)[::-1, :], aspect=1,
            extent=[-0.5, self.N + 0.5, -0.5, self.N + 0.5], cmap=plt.cm.hot_r
        )
        fig.colorbar(img)
        ax.set_aspect(1)
        for w in self.walls:
            x = [w.start[0], w.end[0]]
            y = [w.start[1], w.end[1]]
            ax.plot(x, y, c='orange')
        if len(self.traj_hist) > 0:
            for i, traj_i in enumerate(self.traj_hist):
                if i < len(self.traj_hist)/2:
                    if len(traj_i) >0:
                        traj_i = np.asarray(traj_i)
                        ax.plot(traj_i[:, 0], traj_i[:, 1], c='blue', alpha=0.25)
                elif i > len(self.traj_hist)/2:
                    if len(traj_i) >0:
                        traj_i = np.asarray(traj_i)
                        ax.plot(traj_i[:, 0], traj_i[:, 1], c='green', alpha=0.25)
        ax.set_xlabel('X-coordinate', fontsize= 13)
        ax.set_ylabel('Y-coordinate', fontsize= 13)
        ax.set_xlim(0, self.N)
        ax.set_ylim(0, self.N)
        if self.config["_show"]:
            plt.show()
        return fig, ax
    
    def render_hist_onecolor(self, mode=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        img = ax.imshow(
            np.transpose(self.map)[::-1, :], aspect=1,
            extent=[-0.5, self.N + 0.5, -0.5, self.N + 0.5], cmap=plt.cm.hot_r
        )
        fig.colorbar(img)
        ax.set_aspect(1)
        for w in self.walls:
            x = [w.start[0], w.end[0]]
            y = [w.start[1], w.end[1]]
            ax.plot(x, y, c='orange')
        if len(self.traj_hist) > 0:
            for i, traj_i in enumerate(self.traj_hist):
                if i < len(self.traj_hist)/2:
                    if len(traj_i) >0:
                        traj_i = np.asarray(traj_i)
                        ax.plot(traj_i[:, 0], traj_i[:, 1], c='blue', alpha=0.25)
                elif i > len(self.traj_hist)/2:
                    if len(traj_i) >0:
                        traj_i = np.asarray(traj_i)
                        ax.plot(traj_i[:, 0], traj_i[:, 1], c='blue', alpha=0.25)
        ax.set_xlabel('X-coordinate', fontsize= 13)
        ax.set_ylabel('Y-coordinate', fontsize= 13)
        ax.set_xlim(0, self.N)
        ax.set_ylim(0, self.N)
        if self.config["_show"]:
            plt.show()
        return fig, ax
    
    def reset(self):
        if self.config['init_loc'] is not None:
            self.loc = np.asarray(self.config['init_loc'])
        else:
            if self.config['int_initialize']:
                self.loc = np.random.randint(
                    0, self.N + 1, size=(2,)
                ).astype(np.float32)
            else:
                self.loc = np.random.uniform(
                    0, self.N, size=(2,)
                ).astype(np.float32)
        self.step_num = 0

        self.traj_hist.append(self.traj)
        self.traj = []
        return self.loc

    def seed(self, s=None):
        if s is not None:
            np.random.seed(s)

def draw(compute_action, env_config):
    """compute_action is a function that take current obs (array with shape
    (2,)) as input and return the action: array with shape (2,)."""
    import matplotlib.pyplot as plt
    env_config['_show'] = False
    env = FourWayGridWorld(env_config)
    fig, ax = env.render()
    for i in range(17):
        for j in range(17):
            action = compute_action(np.asarray([i, j]))
            ax.arrow(i, j, action[0], action[1], head_width=0.2, shape='left')
    plt.show()






def eval_policy(policy, eval_episodes=10):
    eval_env = FourWayGridWorld(test_env_config)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        eval_env.loc = [0,8]
        while not done:
            action = policy(np.array(state))
            state, reward, done, _= eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward



task1_config = dict(rewarding_loc=[8, 16],
                    use_walls=True,
                    init_loc=[0, 0],
                    reward = 10,
                    wall_list = [Wall([8,0], [8, 7.5]),
                                 Wall([8, 8.5], [8, 15.5])],
                    max_ts = None
                    )

task2_config = dict(rewarding_loc=[8, 8],
                    use_walls=True,
                    init_loc=[0, 0],
                    reward = 10,
                    wall_list = [Wall([8,0], [8, 7.5]),
                                 Wall([8, 8.5], [8, 15.5])],
                    max_ts = None
                    )


task3_config = dict(rewarding_loc=[16, 0],
                    use_walls=True,
                    init_loc=[8, 16],
                    reward = 10,
                    wall_list = [Wall([8,0], [8, 7.5]),
                                 Wall([8, 8.5], [8, 15.5])],
                    max_ts = None
                    )


task4_config = dict(rewarding_loc=[16, 0],
                    use_walls=True,
                    init_loc=[8, 8],
                    reward = 10,
                    wall_list = [Wall([8,0], [8, 7.5]),
                                 Wall([8, 8.5], [8, 15.5])],
                    max_ts = None
                    )

task_conservation_config = dict(rewarding_loc=[16, 8],
                                use_walls=True,
                                init_loc=[0, 8],
                                reward = 10,
                                wall_list = [Wall([8,0], [8, 3]),
                                            Wall([8, 4], [8, 12],),
                                            Wall([8, 13], [8, 16],)],
                                max_ts = None
                                )

def task1234_policy(state, eps = 0.15, decay = 0.5,
                   task_config = None):
    target_position = task_config['rewarding_loc']
    direc = (np.array(target_position) - np.array(state)) / np.linalg.norm((np.array(target_position) - np.array(state)))
    return direc * np.random.uniform(decay, 1) + np.random.randn(2) * eps


total_state_hist = []
total_action_hist = []
total_next_state_hist = []
total_reward_hist = []



for i, test_env_config in enumerate([task1_config, task2_config, task3_config, task4_config]):
    env = FourWayGridWorld(test_env_config)
    print(f'environent: Task {i}')
#     env.render()


    
    EVAL_SIZE = 5000
    state = env.reset()
    for i in range(EVAL_SIZE):
        action = task1234_policy(state, task_config = test_env_config)
        next_state, reward, done, _ = env.step(action)
        total_state_hist.append(state)
        total_action_hist.append(action)
        total_next_state_hist.append(next_state)
        total_reward_hist.append(reward)
        state = next_state
        if done:
            state, done = env.reset(), False
            if i > EVAL_SIZE-30:
                break

    env.render_hist_onecolor()


shuffled_idx = np.arange(len(total_state_hist))
np.random.shuffle(shuffled_idx)
total_state_hist = np.array(total_state_hist)[shuffled_idx]
total_action_hist = np.array(total_action_hist)[shuffled_idx]
total_next_state_hist = np.array(total_next_state_hist)[shuffled_idx]
total_reward_hist = np.array(total_reward_hist)[shuffled_idx]

print(len(total_state_hist))
print(len(total_action_hist))
print(len(total_next_state_hist))
print(len(total_reward_hist))
from IPython import embed
from collections import deque
import torch.nn.functional as F
import torch.nn as nn
import torch
import gym
import os
import random
import numpy as np


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


task_test_config = dict(rewarding_loc=[16, 0],
                    use_walls=True,
                    init_loc=[0, 0],
                    reward = 10,
                    wall_list = [Wall([8,0], [8, 7.5]),
                                 Wall([8, 8.5], [8, 15.5])],
                    max_ts = 100
                    )

test_env_config = task_test_config

def accountable_batched_controller(buffer_sa,
                                         buffer_r,
                                         n_rollout=100,
                                         n_keep=5,
                                         hist_len=None,
                                         n_epoch=2000,
                                         use_latent=False,
                                         use_hist=False,
                                         quantile=0.99):
    env = FourWayGridWorld(test_env_config)
    state_eval = env.reset()
    eval_r_list = []
    if use_latent:
        state = torch.tensor(buffer_sa[:, :ENV_STATE_DIM], dtype=torch.float32).cuda()
        action = torch.tensor(buffer_sa[:, ENV_STATE_DIM:], dtype=torch.float32).cuda()
        preinfo = torch.tensor(preinfo_set, dtype=torch.float32).cuda()
        buffer_latent_rep = world_model.get_representation(state, action, preinfo).cpu().detach()
    else:
        if use_hist:
            buffer_sa = np.concatenate((buffer_sa, np.asarray(preinfo_set).reshape(len(preinfo_set), -1)), 1)
    previous_info = deque(maxlen=hist_len)
    for i in range(hist_len):
        previous_info.append(
            np.hstack((np.zeros_like(state_eval),
                       np.zeros(env.action_space.shape),
                       np.zeros(1))))
    traj_eval = [state_eval]
    a_idx_eval = []
    for step_i in range(ENV_MAX_STEP):
        sampled_action = np.random.uniform(ENV_ACTION_LOW, ENV_ACTION_HIGH, (n_rollout, ENV_ACTION_DIM))
        forw_state_eval = state_eval[np.newaxis, :].repeat(n_rollout, 0)
        rep_pre_info = np.array(previous_info)[np.newaxis, :, :].repeat(n_rollout, 0)
        if not use_latent and use_hist:
            explor_xa = np.concatenate((forw_state_eval, sampled_action), 1)
            explor_xa = np.concatenate((explor_xa, rep_pre_info.reshape(n_rollout, -1)), 1)
        else:
            explor_xa = np.concatenate((forw_state_eval, sampled_action), 1)
        if use_latent:
            test_state = torch.tensor(explor_xa[:, :ENV_STATE_DIM], dtype=torch.float32).cuda()
            test_action = torch.tensor(explor_xa[:, ENV_STATE_DIM:], dtype=torch.float32).cuda()
            test_preinfo = torch.tensor(np.array(previous_info)[np.newaxis, :, :].repeat(n_rollout, 0), dtype=torch.float32).cuda()
            test_latent_rep = world_model.get_representation(test_state, test_action, test_preinfo).cpu().detach()
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

        err_reg = (out_err > np.quantile(out_err, quantile)) * -99
        selected_a_idx = np.argmax(err_reg + (buffer_r_cumavg.reshape(-1,)[idx_list] * weights.cpu().numpy()).sum(1))
        a = sampled_action[selected_a_idx]
        action_reference_idx = idx_list[selected_a_idx]
        next_state_eval, r, done, _ = env.step(a)
        traj_eval.append(next_state_eval)
        a_idx_eval.append(action_reference_idx)
        if use_latent:
            previous_info.append(
                np.hstack((np.asarray(state_eval),
                           np.asarray(a),
                           np.asarray([r]))))
        state_eval = next_state_eval
        eval_r_list.append(r)
        print('idx list is ', np.shape(action_reference_idx))
        print('decision step', step_i, 'reward', r)
        
        if done:
            break
    
    print('total reward', np.sum(eval_r_list))
    env.render()
    return np.sum(eval_r_list), traj_eval, a_idx_eval


buffer_sa = np.concatenate((total_state_hist, total_action_hist), 1)
buffer_r_cumavg = np.asarray(total_reward_hist)

eval_traj_dict = {}
reward_dict = {}
a_idx_dict = {}
for qt in [0.1, 0.3, 0.5]:
    eval_traj_dict[str(qt)] = []
    reward_dict[str(qt)] = []
    a_idx_dict[str(qt)] = []
    for repeat in range(100):
        rwd, traj, a_idx = accountable_batched_controller(buffer_sa,
                                              buffer_r_cumavg,
                                              n_rollout=5000,
                                              n_keep=100,
                                              hist_len=2,
                                              use_latent=False,
                                              use_hist=False,
                                              quantile=qt)
        eval_traj_dict[str(qt)].append(traj)
        reward_dict[str(qt)].append(rwd)
        a_idx_dict[str(qt)].append(a_idx)
        
for eps in ['0.1', '0.3', '0.5']:
    print('eps:', eps, 'performance:', np.mean(reward_dict[eps]).round(3), np.std(reward_dict[eps]).round(3) )
    
inverse_idx_list = [0] * len(shuffled_idx)
for i, idx in enumerate(shuffled_idx):
    inverse_idx_list[idx] = i
    
sorted_state_list = total_state_hist[inverse_idx_list]


splitted_trajs = []
subtraj = []
for i in range(len(sorted_state_list)):
    subtraj.append(sorted_state_list[i])
    if i < len(sorted_state_list)-1:
        if np.linalg.norm(sorted_state_list[i+1] - np.array([0., 0.]))==0:
            splitted_trajs.append(subtraj)
            subtraj = []
        if np.linalg.norm(sorted_state_list[i+1] - np.array([8., 8.]))==0:
            splitted_trajs.append(subtraj)
            subtraj = []
        if np.linalg.norm(sorted_state_list[i+1] - np.array([8., 16.]))==0:
            splitted_trajs.append(subtraj)
            subtraj = []

for traj_j in range(2):
    step_list = []
    traj = a_idx_dict['0.1'][traj_j]
    for step_i in range(len(traj)):
        ref_list_now = np.asarray(shuffled_idx)[traj[step_i].numpy()]
        n_policy1 = (ref_list_now<len(sorted_state_list)/4).sum()
        n_policy2 = ((len(sorted_state_list)/4 < ref_list_now) & (ref_list_now<len(sorted_state_list)/4*2)).sum()
        n_policy3 = ((len(sorted_state_list)/4*2 < ref_list_now) & (ref_list_now<len(sorted_state_list)/4*3)).sum()
        n_policy4 = ((len(sorted_state_list)/4*3 < ref_list_now) & (ref_list_now<len(sorted_state_list)/4*4)).sum()
        step_list.append([n_policy1, n_policy2, n_policy3, n_policy4])
        
        
    import numpy as np
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(nrows=2,ncols=1,sharex='col', sharey=False, height_ratios = [1,5], figsize=(5,6.5))
    ax1 = ax[0]
    data = np.asarray(step_list).T

    x = np.asarray(eval_traj_dict['0.1'][traj_j])[1:,0]
    width = 1.0
    
    bottom_y = np.zeros(len(data[0]))
    data = np.array(data)
    sums = np.sum(data, axis=0)
    for i in data:
        y = i / sums
        ax1.bar(x, y, width, bottom=bottom_y)
        bottom_y = y + bottom_y
    ax1.set_title('Percent Decision Corpus', fontsize=25)
    ax2 = ax[1]
    img = ax2.imshow(
        np.transpose(env.map)[::-1, :], aspect=1,
        extent=[-0.5, env.N + 0.5, -0.5, env.N + 0.5], cmap=plt.cm.hot_r
    )
    # fig.colorbar(img)
    ax2.set_aspect(1)
    for w in env.walls:
        x = [w.start[0], w.end[0]]
        y = [w.start[1], w.end[1]]
        ax2.plot(x, y, c='gray')
        
    traj_now = np.asarray(eval_traj_dict[eps][traj_j])
    ax2.plot(traj_now[:,0], traj_now[:,1], c='#9467bd', alpha=1.0, linewidth = 2)
#     ax2.set_xlim([0,16])
    plt.scatter([0], [0], color='r', marker='o', s=200)
    plt.scatter([0], [0], color='r', marker='o', s=30, label = 'Start')
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.show()

    
import matplotlib.pyplot as plt

for eps in ['0.1']:
    
    fig, ax = plt.subplots(figsize=(4,4))
    img = ax.imshow(
        np.transpose(env.map)[::-1, :], aspect=1,
        extent=[-0.5, env.N + 0.5, -0.5, env.N + 0.5], cmap=plt.cm.hot_r
    )
    # fig.colorbar(img)
    ax.set_aspect(1)
    for w in env.walls:
        x = [w.start[0], w.end[0]]
        y = [w.start[1], w.end[1]]
        ax.plot(x, y, c='gray')

    # show dataset trajectories
    for i, traj_i in enumerate(splitted_trajs):
        if i < len(splitted_trajs)/4-50:
            if len(traj_i) >0:
                traj_i = np.asarray(traj_i)
                ax.plot(traj_i[:, 0], traj_i[:, 1], c='#1f77b4', alpha=0.8, linewidth = 0.25)
        elif i < len(splitted_trajs)/2:
            if len(traj_i) >0:
                traj_i = np.asarray(traj_i)
                ax.plot(traj_i[:, 0], traj_i[:, 1], c='#ff7f0e', alpha=0.8, linewidth = 0.25)
        elif i < len(splitted_trajs)/4*3:
            if len(traj_i) >0:
                traj_i = np.asarray(traj_i)
                ax.plot(traj_i[:, 0], traj_i[:, 1], c='#2ca02c', alpha=0.8, linewidth = 0.25)
        elif i < len(splitted_trajs):
            if len(traj_i) >0:
                traj_i = np.asarray(traj_i)
                ax.plot(traj_i[:, 0], traj_i[:, 1], c='#d62728', alpha=0.8, linewidth = 0.25)
                
    plt.title(f'Decision Corpus Trajectories', fontsize=15)
    ax.set_xlim(0, env.N)
    ax.set_ylim(0, env.N)

    plt.scatter([8], [16], color='gray', marker='D', s=200)
    plt.scatter([8], [16], color='gray', marker='D', s=30, label = 'Goal $\pi_1$')
    
    plt.scatter([8], [8], color='gray', marker='s', s=200)
    plt.scatter([8], [8], color='gray', marker='s', s=30, label = 'Goal $\pi_2$')
    
    plt.scatter([16], [0], color='black', marker='s', label = 'Goal $\pi_3$, $\pi_4$')
    
    plt.scatter([0], [0], color='r', marker='o', s=120)
    plt.scatter([0], [0], color='r', marker='o', s=30, label = 'Start $\pi_1$, $\pi_2$')
    plt.scatter([8], [16], color='orange', marker='o', s=120)
    plt.scatter([8], [16], color='orange', marker='o', s=30, label = 'Start $\pi_3$')
    plt.scatter([8], [8], color='pink', marker='o', s=120)
    plt.scatter([8], [8], color='pink', marker='o', s=30, label = 'Start $\pi_4$')
    plt.show()
