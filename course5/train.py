import os
import queue
import threading
import time
from collections import defaultdict

import numpy as np
import parl
import retro
from parl.utils import logger, summary
from parl.utils.time_stat import TimeStat
from parl.utils.window_stat import WindowStat

from actor import Actor
from agent import Agent
from config import config
from model import Model


class Learner(object):
    def __init__(self, config):
        self.config = config

        # =========== Create Agent ==========
        env = retro.RetroEnv(game=config['env_name'],
                             state=retro.State.DEFAULT,
                             use_restricted_actions=retro.Actions.DISCRETE,
                             players=1,
                             obs_type=retro.Observations.IMAGE)
        act_dim = env.action_space.n
        self.config['act_dim'] = act_dim

        model = Model(act_dim)
        algorithm = parl.algorithms.A3C(model, vf_loss_coeff=config['vf_loss_coeff'])
        self.agent = Agent(algorithm, config)

        # ========== Learner ==========
        self.total_loss_stat = WindowStat(100)
        self.pi_loss_stat = WindowStat(100)
        self.vf_loss_stat = WindowStat(100)
        self.entropy_stat = WindowStat(100)
        self.lr = None
        self.entropy_coeff = None

        self.learn_time_stat = TimeStat(100)
        self.start_time = None

        # ========== Remote Actor ===========
        self.remote_count = 0
        self.sample_data_queue = queue.Queue()

        self.remote_metrics_queue = queue.Queue()
        self.sample_total_steps = 0

        self.params_queues = []
        self.create_actors()

    def create_actors(self):
        """ Connect to the cluster and start sampling of the remote actor.
        """
        parl.connect(self.config['master_address'])

        logger.info('Waiting for {} remote actors to connect.'.format(self.config['actor_num']))

        for i in range(self.config['actor_num']):
            params_queue = queue.Queue()
            self.params_queues.append(params_queue)

            self.remote_count += 1
            logger.info('Remote actor count: {}'.format(self.remote_count))

            remote_thread = threading.Thread(target=self.run_remote_sample, args=(params_queue,))
            remote_thread.setDaemon(True)
            remote_thread.start()

        logger.info('All remote actors are ready, begin to learn.')
        self.start_time = time.time()

    def run_remote_sample(self, params_queue):
        """ Sample data from remote actor and update parameters of remote actor.
        """
        remote_actor = Actor(self.config)

        while True:
            latest_params = params_queue.get()
            remote_actor.set_weights(latest_params)
            batch = remote_actor.sample()
            self.sample_data_queue.put(batch)

    def step(self):
        """
        1. kick off all actors to synchronize parameters and sample data;
        2. collect sample data of all actors;
        3. update parameters.
        """

        latest_params = self.agent.get_weights()
        for params_queue in self.params_queues:
            params_queue.put(latest_params)

        train_batch = defaultdict(list)
        for i in range(self.config['actor_num']):
            sample_data = self.sample_data_queue.get()
            for key, value in sample_data.items():
                train_batch[key].append(value)

            self.sample_total_steps += sample_data['obs'].shape[0]

        for key, value in train_batch.items():
            train_batch[key] = np.concatenate(value)

        with self.learn_time_stat:
            total_loss, pi_loss, vf_loss, entropy, lr, entropy_coeff = self.agent.learn(
                obs_np=train_batch['obs'],
                actions_np=train_batch['actions'],
                advantages_np=train_batch['advantages'],
                target_values_np=train_batch['target_values'])

        self.total_loss_stat.add(total_loss)
        self.pi_loss_stat.add(pi_loss)
        self.vf_loss_stat.add(vf_loss)
        self.entropy_stat.add(entropy)
        self.lr = lr
        self.entropy_coeff = entropy_coeff

    def log_metrics(self):
        """ Log metrics of learner and actors
        """
        if self.start_time is None:
            return

        summary.add_scalar('total_loss', self.total_loss_stat.mean, self.sample_total_steps)
        summary.add_scalar('pi_loss', self.pi_loss_stat.mean, self.sample_total_steps)
        summary.add_scalar('vf_loss', self.vf_loss_stat.mean, self.sample_total_steps)
        summary.add_scalar('entropy', self.entropy_stat.mean, self.sample_total_steps)
        summary.add_scalar('lr', self.lr, self.sample_total_steps)
        summary.add_scalar('entropy_coeff', self.entropy_coeff, self.sample_total_steps)
        logger.info('total_loss: {}'.format(self.total_loss_stat.mean))

    # 保存模型
    def save_model(self):
        if self.start_time is None:
            return

        if not os.path.exists(os.path.dirname(self.config['model_path'])):
            os.makedirs(os.path.dirname(self.config['model_path']))
        self.agent.save(self.config['model_path'])

    def should_stop(self):
        return self.sample_total_steps >= self.config['max_sample_steps']


if __name__ == '__main__':
    learner = Learner(config)
    assert config['log_metrics_interval_s'] > 0

    start1 = time.time()
    while not learner.should_stop():
        start = time.time()
        while time.time() - start < config['log_metrics_interval_s']:
            learner.step()
        learner.log_metrics()
        # 保存模型
        if time.time() - start1 > config['save_model_interval_s']:
            start1 = time.time()
            learner.save_model()
