import numpy as np
import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils.scheduler import PiecewiseScheduler, LinearDecayScheduler


class Agent(parl.Agent):
    def __init__(self, algorithm, config):
        """
        Args:
            algorithm (`parl.Algorithm`): 强学习算法
            config (dict): 配置文件参数
        """

        self.obs_shape = config['obs_shape']
        super(Agent, self).__init__(algorithm)
        # 学习率衰减
        self.lr_scheduler = LinearDecayScheduler(config['start_lr'], config['max_sample_steps'])

        self.entropy_coeff_scheduler = PiecewiseScheduler(config['entropy_coeff_scheduler'])

    # 创建PaddlePaddle程序
    def build_program(self):
        self.sample_program = fluid.Program()
        self.predict_program = fluid.Program()
        self.value_program = fluid.Program()
        self.learn_program = fluid.Program()

        # 给Actor生成数据的程序
        with fluid.program_guard(self.sample_program):
            obs = layers.data(name='obs', shape=self.obs_shape, dtype='float32')
            sample_actions, values = self.alg.sample(obs)
            self.sample_outputs = [sample_actions, values]

        # 用于预测的程序
        with fluid.program_guard(self.predict_program):
            obs = layers.data(name='obs', shape=self.obs_shape, dtype='float32')
            self.predict_actions = self.alg.predict(obs)

        with fluid.program_guard(self.value_program):
            obs = layers.data(name='obs', shape=self.obs_shape, dtype='float32')
            self.values = self.alg.value(obs)

        # 用于训练的程序
        with fluid.program_guard(self.learn_program):
            obs = layers.data(name='obs', shape=self.obs_shape, dtype='float32')
            actions = layers.data(name='actions', shape=[], dtype='int64')
            advantages = layers.data(name='advantages', shape=[], dtype='float32')
            target_values = layers.data(name='target_values', shape=[], dtype='float32')
            lr = layers.data(name='lr', shape=[1], dtype='float32', append_batch_size=False)
            entropy_coeff = layers.data(name='entropy_coeff',
                                        shape=[1],
                                        dtype='float32',
                                        append_batch_size=False)

            total_loss, pi_loss, vf_loss, entropy = self.alg.learn(
                obs, actions, advantages, target_values, lr, entropy_coeff)
            self.learn_outputs = [total_loss, pi_loss, vf_loss, entropy]
        # 获取并行计算程序
        self.learn_program = parl.compile(self.learn_program, total_loss)

    def sample(self, obs_np):
        """
        Args:
            obs_np: 游戏的图像，shape为([N] + observation_space).
                    游戏图像通道顺序为NCHW.

        Returns:
            sample_ids: 游戏动作，类型为int64，shape为[N]
            values: 模型输出的值，类型为float32，shape为[N]
        """
        obs_np = obs_np.astype('float32')

        sample_actions, values = self.fluid_executor.run(program=self.sample_program,
                                                         feed={'obs': obs_np},
                                                         fetch_list=self.sample_outputs)
        return sample_actions, values

    def predict(self, obs_np):
        """
        Args:
            obs_np: 游戏的图像，shape为([N] + observation_space).
                    游戏图像通道顺序为NCHW.
        Returns:
            sample_ids: 游戏动作，类型为int64，shape为[N]
        """
        obs_np = obs_np.astype('float32')

        predict_actions = self.fluid_executor.run(program=self.predict_program,
                                                  feed={'obs': obs_np},
                                                  fetch_list=[self.predict_actions])[0]
        return predict_actions

    def value(self, obs_np):
        """
        Args:
            obs_np: 游戏的图像，shape为([N] + observation_space).
                    游戏图像通道顺序为NCHW.

        Returns:
            values: 模型输出的值，类型为float32，shape为[N]
        """
        obs_np = obs_np.astype('float32')

        values = self.fluid_executor.run(program=self.value_program,
                                         feed={'obs': obs_np},
                                         fetch_list=[self.values])[0]
        return values

    # 执行模型学习
    def learn(self, obs_np, actions_np, advantages_np, target_values_np):
        """
        Args:
            obs_np: 游戏的图像，shape为([N] + observation_space).
                    游戏图像通道顺序为NCHW.
            actions_np: 游戏动作，类型为int64，shape为[N]
            advantages_np: 奖励值，类型为float32，shape为[N]
            target_values_np: 目标模型值，类型为float32，shape为[N]
        """

        obs_np = obs_np.astype('float32')
        actions_np = actions_np.astype('int64')
        advantages_np = advantages_np.astype('float32')
        target_values_np = target_values_np.astype('float32')

        lr = self.lr_scheduler.step(step_num=obs_np.shape[0])
        entropy_coeff = self.entropy_coeff_scheduler.step()

        total_loss, pi_loss, vf_loss, entropy = self.fluid_executor.run(
            self.learn_program,
            feed={
                'obs': obs_np,
                'actions': actions_np,
                'advantages': advantages_np,
                'target_values': target_values_np,
                'lr': np.array([lr], dtype='float32'),
                'entropy_coeff': np.array([entropy_coeff], dtype='float32')
            },
            fetch_list=self.learn_outputs)
        return total_loss, pi_loss, vf_loss, entropy, lr, entropy_coeff
