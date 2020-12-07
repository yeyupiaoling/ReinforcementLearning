from collections import defaultdict

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
        # 生成指定数量的游戏环境
        self.envs = []
        for _ in range(config['env_num']):
            env = retrowrapper.RetroWrapper(game=config['env_name'],
                                            use_restricted_actions=retro.Actions.DISCRETE,
                                            skill_frame=4,
                                            resize_shape=(1, 112, 112),
                                            render_preprocess=False)
            self.envs.append(env)
        # 把全部的游戏环境打包，通过这个工具可以方便对跟个游戏操作
        self.vector_env = VectorEnv(self.envs)
        # 获取全部环境的初始界面
        self.obs_batch = self.vector_env.reset()
        # 获取每个Actor的模型
        model = Model(self.config['action_dim'])
        algorithm = parl.algorithms.A3C(model, vf_loss_coeff=config['vf_loss_coeff'])
        self.agent = Agent(algorithm, config)

    def sample(self):
        # 全部数据都存放在这里返回
        sample_data = defaultdict(list)

        env_sample_data = {}
        for env_id in range(self.config['env_num']):
            env_sample_data[env_id] = defaultdict(list)

        total_reward = 0
        # 根据设定，获取指定步数的数据
        for i in range(self.config['sample_batch_steps']):
            # 执行预测
            actions_batch, values_batch = self.agent.sample(np.stack(self.obs_batch))
            # 执行游戏
            next_obs_batch, reward_batch, done_batch, info_batch = self.vector_env.step(actions_batch)
            # 记录分数
            total_reward = total_reward + sum(reward_batch)

            for env_id in range(self.config['env_num']):
                # 记录游戏的数据
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['actions'].append(actions_batch[env_id])
                env_sample_data[env_id]['rewards'].append(reward_batch[env_id])
                env_sample_data[env_id]['dones'].append(done_batch[env_id])
                env_sample_data[env_id]['values'].append(values_batch[env_id])

                # 当游戏结束之后或者到达了指定的步数
                if done_batch[env_id] or i == self.config['sample_batch_steps'] - 1:
                    # 如果游戏结束了，值就为0
                    next_value = 0
                    # 如果不是结束游戏的，就计算值
                    if not done_batch[env_id]:
                        next_obs = np.expand_dims(next_obs_batch[env_id], 0)
                        next_value = self.agent.value(next_obs)

                    # 计算奖励
                    values = env_sample_data[env_id]['values']
                    rewards = env_sample_data[env_id]['rewards']
                    advantages = calc_gae(rewards, values, next_value, self.config['gamma'], self.config['lambda'])
                    target_values = advantages + values

                    # 记录训练数据
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

    # 把train中的参数复制到Actor的模型中
    def set_weights(self, params):
        self.agent.set_weights(params)
