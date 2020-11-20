import numpy as np
import parl
from parl import layers
from paddle import fluid


class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, tuple)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        # 刚开始时就要完全同步目标模型
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.sample_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(name='obs', shape=self.obs_dim, dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.sample_program):
            obs = layers.data(name='obs', shape=self.obs_dim, dtype='float32')
            self.sample_act, _ = self.alg.sample(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(name='obs', shape=self.obs_dim, dtype='float32')
            act = layers.data(name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(name='next_obs', shape=self.obs_dim, dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.critic_cost, self.actor_cost = self.alg.learn(obs, act, reward, next_obs, terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(program=self.pred_program,
                                      feed={'obs': obs.astype('float32')},
                                      fetch_list=[self.pred_act])[0]
        return act

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(program=self.sample_program,
                                      feed={'obs': obs.astype('float32')},
                                      fetch_list=[self.sample_act])[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        [critic_cost, actor_cost] = self.fluid_executor.run(program=self.learn_program,
                                                            feed=feed,
                                                            fetch_list=[self.critic_cost, self.actor_cost])
        self.alg.sync_target()
        return critic_cost[0], actor_cost[0]
