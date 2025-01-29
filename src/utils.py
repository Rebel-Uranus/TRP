import torch
import numpy as np
import os
import glob
import json
import random
import time
import augmentations
import subprocess
from datetime import datetime
import copy


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def soft_update_params_quadra(net1, net2, target_net1, target_net2, tau):
    for param1, param2, target_param1, target_param2 in zip(net1.parameters(), net2.parameters(), target_net1.parameters(), target_net2.parameters()):
        target_param1.data.copy_(
            tau * param1.data/2 + tau * param2.data/2 + (1 - tau) * target_param1.data + (1 - tau) * target_param2.data
        )
        target_param2.data.copy_(
            tau * param1.data/2 + tau * param2.data/2 + (1 - tau) * target_param1.data + (1 - tau) * target_param2.data
        )


def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        'timestamp': str(datetime.now()),
        # 'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
        'args': vars(args)
    }
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


def load_config(key=None):
    path = os.path.join('setup', 'config.cfg')
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def listdir(dir_path, filetype='jpg', sort=True):
    fpath = os.path.join(dir_path, f'*.{filetype}')
    fpaths = glob.glob(fpath, recursive=True)
    if sort:
        return sorted(fpaths)
    return fpaths


def prefill_memory(obses, capacity, obs_shape):
    """Reserves memory for replay buffer"""
    c, h, w = obs_shape
    for _ in range(capacity):
        frame = np.ones((3, h, w), dtype=np.uint8)
        obses.append(frame)
    return obses


class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, prefill=True, jump=5):
        self.capacity = capacity
        self.batch_size = batch_size
        self.jump = jump

        # self._obses = []
        # if prefill:
        #     self._obses = prefill_memory(self._obses, capacity, obs_shape)
        # print(obs_shape)
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self._obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self._next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.real_dones = np.empty((capacity, 1), dtype=np.float32) # for auxiliary task
        self.priority = np.zeros(capacity, dtype=np.float32)
        self.sample_idxs = None
        # self.reward_num = np.zeros(10, dtype=np.int)
        # self.num_idx = 0
        # self.interval = 1000/10
        # self.priority_idx = 0

        self.idx = 0
        self.full = False

    # def update_priority(self, total_reward):
    #     if self.priority_idx <= self.idx:
    #         self.priority[self.priority_idx : self.idx] = total_reward
    #     else:
    #         self.priority[self.priority_idx : ] = total_reward
    #         self.priority[ : self.idx] = total_reward
    #     self.priority_idx = self.idx
    #     idx = int(total_reward / self.interval)
    #     self.reward_num[idx] += 1
    #     # if idx> self.num_idx:
    #     #     self.num_idx = idx

    def update_priority(self, loss):
        self.priority[self.sample_idxs] = loss.cpu().numpy()
        #     self.num_idx = idx
        
    def add(self, obs, action, reward, next_obs, done):
        # obses = (obs, next_obs)
        # print(np.array(obs).shape, np.array(next_obs).shape)
        # if self.idx >= len(self._obses):
        #     self._obses.append(obses)
        # else:
        #     self._obses[self.idx] = (obses)
        np.copyto(self._obses[self.idx], obs)
        np.copyto(self._next_obses[self.idx], next_obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.real_dones[self.idx], isinstance(done, int))   # "not done" is always True
        # print(not done, isinstance(done, int))
        self.priority[self.idx] = self.priority.max()

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _get_idxs(self, n=None):
        if n is None:
            n = self.batch_size
        return np.random.randint(
            0, self.capacity if self.full else self.idx, size=n
        )

    def _encode_obses(self, idxs):
        # obses, next_obses = [], []
        # for i in idxs:
        #     obs, next_obs = self._obses[i]
        #     obses.append(np.array(obs, copy=False))
        #     next_obses.append(np.array(next_obs, copy=False))
        # return np.array(obses), np.array(next_obses)
        return self._obses[idxs], self._next_obses[idxs]

    def sample_soda(self, n=None):
        idxs = self._get_idxs(n)
        obs, _ = self._encode_obses(idxs)
        return torch.as_tensor(obs).cuda().float()

    def __sample__(self, n=None):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        return obs, actions, rewards, next_obs, not_dones

    def sample_curl(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        pos = augmentations.random_crop(obs.clone())
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones, pos

    def sample_drq(self, n=None, pad=4):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.random_shift(obs, pad)
        next_obs = augmentations.random_shift(next_obs, pad)

        return obs, actions, rewards, next_obs, not_dones

    def sample_svea(self, n=None, pad=4):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.random_shift(obs, pad)

        return obs, actions, rewards, next_obs, not_dones

    def sample(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        # print(obs.shape, actions.shape, rewards.shape, next_obs.shape, not_dones.shape)
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones
    
    def sample_btsac(self, n=None):
        sampled_size = 0
        # res_ids = []
        idxs = None
        # i = 0
        # print('*********')
        # start = time.time()
        if idxs is None:
            idxs = self._get_idxs_btsac(n)
            self.sample_idxs = copy.deepcopy(idxs)
            idxs = idxs.reshape(-1, 1)
            step = np.arange(self.jump).reshape(1, -1) # this is a range
            idxs = idxs + step 
        # else:
        #     tmp = self._get_idxs_btsac(n)
        #     tmp = idxs.reshape(-1, 1)
        #     step = np.arange(self.jump + 1).reshape(1, -1) # this is a range
        #     tmp = tmp + step 
        #     idxs = np.concatenate((idxs,new))
        # while sampled_size < n:
            # i += 1
            
        ###################### old one    
        real_dones = torch.as_tensor(self.real_dones[idxs]).cuda()
        # valid_idxs = torch.where((real_dones.mean(1)==0).squeeze(-1))[0].cpu().numpy()
        other_idxs = torch.where((real_dones.mean(1)>0).squeeze(-1))[0].cpu().numpy()
        # new_idxs = idxs[valid_idxs] # (B, jump+1)
        # new_idxs = new_idxs[:n] if new_idxs.shape[0] >= n else new_idxs
        # sampled_size = new_idxs.shape[0]
        # if sampled_size<n:
        idxs[other_idxs] = idxs[other_idxs] - self.jump - 1
        real_dones = torch.as_tensor(self.real_dones[idxs]).cuda()
        valid_idxs = torch.where((real_dones.mean(1)==0).squeeze(-1))[0].cpu().numpy()
        # other_idxs = torch.where((real_dones.mean(1)>0).squeeze(-1))[0].cpu().numpy()
        idxs = idxs[valid_idxs] # (B, jump+1)
        # if idxs.shape[0]!=n*2:
        #     print(idxs.shape[0])
        idxs = idxs[:n] if idxs.shape[0] >= n else idxs
        # sampled_size = new_idxs.shape[0]
        
        #################### new one
        # real_dones = torch.as_tensor(self.real_dones[idxs])
        # row, col = torch.where((real_dones>0).squeeze(-1))

        
        # print(time.time()-start)
        # start = time.time()
        # if i>1:
        #     print(i)
        # print('sample:', idxs, idxs.shape, sampled_size, n, self.jump)
        obs = self._btsac_encode_obses(idxs[:,:2])
        # obs = self._btsac_encode_obses(idxs[:,[0,-1]])
        next_obs = self._btsac_encode_next_obses(idxs[:,-1])
        # print(time.time()-start)
        # start = time.time()
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        # obs = obs.flatten(0,1)
        # next_obs = next_obs.flatten(0,1)
        # print(time.time()-start)
        # start = time.time()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        # mask = torch.zeros(n*2, self.jump + 2).cuda()
        # for i in range(len(row)):
        #     actions[row[i],col[i]:,:] = 0
        #     rewards[row[i],col[i]:,:] = 0
        #     mask[row[i],2+col[i]:] = 1
        #     mask[row[i]+n,2+col[i]:] = 1
        # row = np.array([])
        # col = np.array([])
        not_dones = torch.as_tensor(self.not_dones[idxs[:,-1]]).cuda()

        # pad = 4
        # b,t,c,w,h = obs.size()
        # obs = augmentations.random_shift(obs.reshape(-1, c, w, h), pad)
        # obs = obs.reshape(b, t, c, w, h)

        

        # return obs, actions, rewards, not_dones, row, col
        return obs, actions, rewards, next_obs, not_dones
    
    def _btsac_encode_obses(self, idxs):
        # obses, next_obses = [], []
        # for i in idxs:
        #     for j in i:
        #         obs, next_obs = self._obses[j]
        #         obses.append(np.array(obs, copy=False))
        #         next_obses.append(np.array(next_obs, copy=False))
        # return np.array(obses), np.array(next_obses)
        return self._obses[idxs]
    
    def _btsac_encode_next_obses(self, idxs):
        return self._next_obses[idxs]
    
    def _get_idxs_btsac(self, n=None):
        if n is None:
            sample_size = self.batch_size*2
        else:
            sample_size = n*2
        return np.random.randint(
            0, self.capacity - self.jump - 1 if self.full else self.idx - self.jump - 1, size=sample_size
        )

        # prob = (self.priority.max()+10) - self.priority
        # prob = prob[ : self.capacity - self.jump - 1] if self.full else prob[ : self.idx - self.jump - 1]
        # prob = prob / prob.sum()

        # idx = (self.priority[ : self.capacity - self.jump - 1] / self.interval if self.full else self.priority[ : self.idx - self.jump - 1] / self.interval).astype(np.int)
        # w = np.array([1/((self.reward_num>0).sum() * x) if x>0 else 0 for x in self.reward_num])
        # prob = w[idx]
        # if prob.sum() == 0:
        #     prob = np.ones_like(prob)
        # prob = prob / prob.sum()
        # # print(prob.shape, self.idx)

        # prob = self.priority[ : self.capacity - self.jump - 1] if self.full else self.priority[ : self.idx - self.jump - 1]
        # if prob.sum() == 0:
        #     prob = np.ones_like(prob)
        # prob = prob / prob.sum()
        

        # res = random.choices(
        #     np.arange(0, self.capacity - self.jump - 1 if self.full else self.idx - self.jump - 1),
        #     weights=prob,
        #     k=sample_size
        # )  
        # return np.array(res)


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0]//3

    def frame(self, i):
        return self._force()[i*3:(i+1)*3]


def count_parameters(net, as_int=False):
    """Returns total number of params in a network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f'{count:,}'
