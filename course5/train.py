import os
import queue
import threading
import time
from collections import defaultdict

import numpy as np
import parl
import retro
from parl.utils import logger, summary, machine_info, get_gpu_count
from parl.utils.time_stat import TimeStat
from parl.utils.window_stat import WindowStat

import retro_util
from actor import Actor
from agent import Agent
from config import config
from model import Model


class Learner(object):
    def __init__(self, config):
        self.config = config

        # 这里创建游戏单纯是为了获取游戏动作的维度
        env = retro_util.RetroEnv(game=config['env_name'],
                                  use_restricted_actions=retro.Actions.DISCRETE,
                                  resize_shape=config['obs_shape'],
                                  render_preprocess=False)
        action_dim = env.action_space.n
        self.config['action_dim'] = action_dim

        # 这里创建的模型是真正学习使用的
        model = Model(action_dim)
        algorithm = parl.algorithms.A3C(model, vf_loss_coeff=config['vf_loss_coeff'])
        self.agent = Agent(algorithm, config)

        # 只支持单个GPU
        if machine_info.is_gpu_available():
            assert get_gpu_count() == 1, 'Only support training in single GPU,\
                    Please set environment variable: `export CUDA_VISIBLE_DEVICES=[GPU_ID_TO_USE]` .'

        # 加载预训练模型
        if self.config['restore_model']:
            logger.info("加载预训练模型...")
            self.agent.restore(self.config['model_path'])

        # 记录训练的日志
        self.total_loss_stat = WindowStat(100)
        self.pi_loss_stat = WindowStat(100)
        self.vf_loss_stat = WindowStat(100)
        self.entropy_stat = WindowStat(100)
        self.lr = None
        self.entropy_coeff = None

        self.best_loss = None

        self.learn_time_stat = TimeStat(100)
        self.start_time = None

        # ========== Remote Actor ===========
        self.remote_count = 0
        self.sample_data_queue = queue.Queue()

        self.remote_metrics_queue = queue.Queue()
        self.sample_total_steps = 0

        self.params_queues = []
        self.create_actors()

    # 开始创建指定数量的Actor，并发放的集群中
    def create_actors(self):
        # 连接到集群
        parl.connect(self.config['master_address'])
        logger.info('Waiting for {} remote actors to connect.'.format(self.config['actor_num']))

        # 循环生成多个Actor线程
        for i in range(self.config['actor_num']):
            # 更新参数的队列
            params_queue = queue.Queue()
            self.params_queues.append(params_queue)

            self.remote_count += 1
            logger.info('Remote actor count: {}'.format(self.remote_count))
            # 创建Actor的线程
            remote_thread = threading.Thread(target=self.run_remote_sample, args=(params_queue,))
            remote_thread.setDaemon(True)
            remote_thread.start()

        logger.info('All remote actors are ready, begin to learn.')
        self.start_time = time.time()

    # 创建Actor，并使用无限循环更新Actor的模型参数和获取游戏数据
    def run_remote_sample(self, params_queue):
        # 创建Actor
        remote_actor = Actor(self.config)

        while True:
            # 获取train的模型参数
            latest_params = params_queue.get()
            # 设置Actor中的模型参数
            remote_actor.set_weights(latest_params)
            # 获取一小批的游戏数据
            batch = remote_actor.sample()
            # 将游戏数据添加的数据队列中
            self.sample_data_queue.put(batch)

    # 开始模型训练
    def step(self):
        """
        1. 启动所有Actor，同步参数和样本数据;
        2. 收集所有Actor生成的数据;
        3. 更新参数.
        """

        # 获取train中模型最新的参数
        latest_params = self.agent.get_weights()
        # 将参数同步给没有Actor线程的参数队列
        for params_queue in self.params_queues:
            params_queue.put(latest_params)

        train_batch = defaultdict(list)
        # 获取每个Actor生成的数据
        for i in range(self.config['actor_num']):
            sample_data = self.sample_data_queue.get()
            for key, value in sample_data.items():
                train_batch[key].append(value)

            # 记录训练步数
            self.sample_total_steps += sample_data['obs'].shape[0]

        # 将各个Actor的数据打包的训练数据
        for key, value in train_batch.items():
            train_batch[key] = np.concatenate(value)

        # 执行一次训练
        with self.learn_time_stat:
            total_loss, pi_loss, vf_loss, entropy, lr, entropy_coeff = self.agent.learn(
                obs_np=train_batch['obs'],
                actions_np=train_batch['actions'],
                advantages_np=train_batch['advantages'],
                target_values_np=train_batch['target_values'])

        # 记录训练数据
        self.total_loss_stat.add(total_loss)
        self.pi_loss_stat.add(pi_loss)
        self.vf_loss_stat.add(vf_loss)
        self.entropy_stat.add(entropy)
        self.lr = lr
        self.entropy_coeff = entropy_coeff

    # 保存训练日志
    def log_metrics(self):
        # 避免训练还未开始的情况
        if self.start_time is None:
            return
        # 获取最好的模型
        if self.best_loss is None:
            self.best_loss = self.total_loss_stat.mean
        else:
            if self.best_loss > self.total_loss_stat.mean:
                self.best_loss = self.total_loss_stat.mean
                self.save_model("model_%d" % int(self.best_loss))
        # 训练数据写入到日志中
        summary.add_scalar('total_loss', self.total_loss_stat.mean, self.sample_total_steps)
        summary.add_scalar('pi_loss', self.pi_loss_stat.mean, self.sample_total_steps)
        summary.add_scalar('vf_loss', self.vf_loss_stat.mean, self.sample_total_steps)
        summary.add_scalar('entropy', self.entropy_stat.mean, self.sample_total_steps)
        summary.add_scalar('lr', self.lr, self.sample_total_steps)
        summary.add_scalar('entropy_coeff', self.entropy_coeff, self.sample_total_steps)
        logger.info('total_loss: {}'.format(self.total_loss_stat.mean))

    # 保存模型
    def save_model(self, model_name="model"):
        # 避免训练还未开始的情况
        if self.start_time is None:
            return
        save_path = os.path.join(self.config['model_path'], model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.agent.save(save_path)

    # 检测训练步数是否达到最大步数
    def should_stop(self):
        return self.sample_total_steps >= self.config['max_sample_steps']


if __name__ == '__main__':
    learner = Learner(config)
    assert config['log_metrics_interval_s'] > 0
    assert config['save_model_interval_s'] > 0

    start1 = time.time()
    while not learner.should_stop():
        start = time.time()
        while time.time() - start < config['log_metrics_interval_s']:
            learner.step()
        # 保存日柱子
        learner.log_metrics()
        # 保存模型
        if time.time() - start1 > config['save_model_interval_s']:
            start1 = time.time()
            learner.save_model()
    print("================ 训练结束！================")
    # 最后结束之前保存模型
    learner.save_model("final_model")
