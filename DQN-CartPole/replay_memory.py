import collections
import random

import paddle


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 添加数据
    def append(self, exp):
        self.buffer.append(exp)

    # 获取一批数据
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        batch_state, batch_action, batch_reword, batch_next_state, batch_done = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, isOver = experience
            batch_state.append(s)
            batch_action.append(a)
            batch_reword.append(r)
            batch_next_state.append(s_p)
            batch_done.append(isOver)
        # 转换为张量数据
        batch_state = paddle.to_tensor(batch_state, dtype='float32')
        batch_action = paddle.to_tensor(batch_action, dtype='int64')
        batch_reword = paddle.to_tensor(batch_reword, dtype='float32')
        batch_next_state = paddle.to_tensor(batch_next_state, dtype='float32')
        batch_done = paddle.to_tensor(batch_done, dtype='int64')

        return batch_state, batch_action, batch_reword, batch_next_state, batch_done

    # 获取当前数据记录的大小
    def __len__(self):
        return len(self.buffer)
