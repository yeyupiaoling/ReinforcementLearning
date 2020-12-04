from collections import defaultdict

import cv2
import numpy
import numpy as np
import parl
import retro
from parl.env.vector_env import VectorEnv
from parl.utils.rl_utils import calc_gae

import retrowrapper
from agent import Agent
from model import Model


@parl.remote_class
class Actor(object):
    def __init__(self, config):
        self.config = config

        self.envs = []
        for _ in range(config['env_num']):
            env = retrowrapper.RetroWrapper(game=config['env_name'],
                                            state=retro.State.DEFAULT,
                                            use_restricted_actions=retro.Actions.DISCRETE,
                                            players=1,
                                            obs_type=retro.Observations.IMAGE)
            self.envs.append(env)
        self.vector_env = VectorEnv(self.envs)

        self.obs_batch = self.vector_env.reset()
        temp = []
        for o in self.obs_batch:
            obs = self.preprocess(o)
            temp.append(obs)
        self.obs_batch = temp

        model = Model(self.config['act_dim'])
        algorithm = parl.algorithms.A3C(model, vf_loss_coeff=config['vf_loss_coeff'])
        self.agent = Agent(algorithm, config)

    # 改变游戏的布局环境，减低输入图像的复杂度
    def change_obs_color(self, obs, src, target):
        for i in range(len(src)):
            index = (obs == src[i])
            obs[index] = target[i]
        return obs

    # 图像预处理
    def preprocess(self, observation):
        assert self.config['obs_shape'][0] == 1 or self.config['obs_shape'][0] == 3
        w, h, c = observation.shape
        observation = observation[25:h, 15:w]
        if self.config['obs_shape'][0] == 1:
            # 把图像转成灰度图
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            # 把其他的亮度调成一种，减低图像的复杂度
            observation = self.change_obs_color(observation, [66, 88, 114, 186, 189, 250], [255, 255, 255, 255, 255, 0])
            observation = cv2.resize(observation, (self.config['obs_shape'][2], self.config['obs_shape'][1]))
            observation = numpy.expand_dims(observation, axis=0)
        else:
            observation = cv2.resize(observation, (self.config['obs_shape'][2], self.config['obs_shape'][1]))
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            observation = observation.transpose((2, 0, 1))
        observation = observation / 255.0
        return observation

    def sample(self):
        sample_data = defaultdict(list)

        env_sample_data = {}
        for env_id in range(self.config['env_num']):
            env_sample_data[env_id] = defaultdict(list)

        total_reward = 0
        for i in range(self.config['sample_batch_steps']):
            actions_batch, values_batch = self.agent.sample(np.stack(self.obs_batch))
            next_obs_batch, reward_batch, done_batch, info_batch = self.vector_env.step(actions_batch)
            total_reward = total_reward + sum(reward_batch)

            temp = []
            for o in next_obs_batch:
                obs = self.preprocess(o)
                temp.append(obs)
            next_obs_batch = temp

            for env_id in range(self.config['env_num']):
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['actions'].append(actions_batch[env_id])
                env_sample_data[env_id]['rewards'].append(reward_batch[env_id])
                env_sample_data[env_id]['dones'].append(done_batch[env_id])
                env_sample_data[env_id]['values'].append(values_batch[env_id])

                # Calculate advantages when the episode is done or reach max sample steps.
                if done_batch[env_id] or i == self.config['sample_batch_steps'] - 1:
                    next_value = 0
                    if not done_batch[env_id]:
                        next_obs = np.expand_dims(next_obs_batch[env_id], 0)
                        next_value = self.agent.value(next_obs)

                    values = env_sample_data[env_id]['values']
                    rewards = env_sample_data[env_id]['rewards']
                    advantages = calc_gae(rewards, values, next_value, self.config['gamma'], self.config['lambda'])
                    target_values = advantages + values

                    sample_data['obs'].extend(env_sample_data[env_id]['obs'])
                    sample_data['actions'].extend(env_sample_data[env_id]['actions'])
                    sample_data['advantages'].extend(advantages)
                    sample_data['target_values'].extend(target_values)

                    env_sample_data[env_id] = defaultdict(list)

            self.obs_batch = next_obs_batch
        print("当前得分：", total_reward)

        # size of sample_data: env_num * sample_batch_steps
        for key in sample_data:
            sample_data[key] = np.stack(sample_data[key])

        return sample_data

    def set_weights(self, params):
        self.agent.set_weights(params)
