import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 添加数据
    def append(self, exp):
        self.buffer.append(exp)

    # 获取一批数据
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, isOver_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, isOver = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            isOver_batch.append(isOver)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(isOver_batch).astype('float32')

    # 获取当前数据记录的大小
    def __len__(self):
        return len(self.buffer)
