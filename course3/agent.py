import numpy as np
import parl
from parl import layers
from paddle import fluid


class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, tuple)
        assert isinstance(act_dim, int)
        # 预测图像的shape
        self.obs_dim = obs_dim
        # 动作组合的数量
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        # 刚开始时就要完全同步目标模型
        self.alg.sync_target(decay=0)

    # 创建PaddlePaddle程序
    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(name='obs', shape=self.obs_dim, dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(name='obs', shape=self.obs_dim, dtype='float32')
            action = layers.data(name='action', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(name='next_obs', shape=self.obs_dim, dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    # 预测动作
    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        logits = self.fluid_executor.run(program=self.pred_program,
                                         feed={'obs': obs.astype('float32')},
                                         fetch_list=[self.pred_act])
        logits = np.squeeze(logits)
        policy = self.softmax(logits)
        return policy

    # Fluid版本不支持动态计算，只能自定义一个softmax
    def softmax(self, x):
        x_row_max = x.max(axis=-1)
        x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
        x = x - x_row_max
        x_exp = np.exp(x)
        x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
        softmax = x_exp / x_exp_row_sum
        return softmax

    # 执行模型学习
    def learn(self, obs, action, reward, next_obs, terminal):
        feed = {
            'obs': obs.astype('float32'),
            'action': action,
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(program=self.learn_program,
                                              feed=feed,
                                              fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost
