import argparse
import retro_util
import numpy as np
import parl
import retro
from agent import Agent
from model import Model
from parl.utils import logger
from parl.utils.rl_utils import calc_gae, calc_discount_sum_rewards
from scaler import Scaler


def run_train_episode(env, agent, scaler: Scaler):
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    step = np.zeros((1, 1, 112), dtype=np.float32)
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while True:
        # env.render()
        # 添加时间特征
        obs = np.concatenate((obs, step), axis=1)
        # 时间步长增量特征
        step += 1e-3
        obs = np.expand_dims(obs, axis=0)
        unscaled_obs.append(obs)
        # 中心和比例尺观测
        obs = (obs - offset) * scale
        obs = obs.astype('float32')
        observes.append(obs)

        # 获取动作
        action = agent.policy_sample(obs)
        action = np.squeeze(action)
        actions.append(action)

        # 执行游戏
        obs, reward, isOver, info = env.step(action)
        print(info)

        rewards.append(reward)

        if isOver:
            break

    return (np.concatenate(observes), np.array(actions, dtype='float32'),
            np.array(rewards, dtype='float32'), np.concatenate(unscaled_obs))


def run_evaluate_episode(env, agent, scaler):
    obs = env.reset()
    rewards = []
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while True:
        env.render()
        obs = obs.reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        obs = (obs - offset) * scale  # center and scale observations
        obs = obs.astype('float32')

        action = agent.policy_predict(obs)
        obs, reward, done, _ = env.step(np.squeeze(action))
        rewards.append(reward)

        step += 1e-3  # increment time step feature

        if done:
            break
    return np.sum(rewards)


def collect_trajectories(env, agent, scaler, episodes):
    trajectories, all_unscaled_obs = [], []
    for e in range(episodes):
        obs, actions, rewards, unscaled_obs = run_train_episode(env, agent, scaler)
        trajectories.append({'obs': obs,
                             'actions': actions,
                             'rewards': rewards,
                             })
        all_unscaled_obs.append(unscaled_obs)
    # 更新伸缩观察的运行统计信息
    scaler.update(np.concatenate(all_unscaled_obs))
    return trajectories


def build_train_data(trajectories, agent):
    train_obs, train_actions, train_advantages, train_discount_sum_rewards = [], [], [], []
    for trajectory in trajectories:
        pred_values = agent.value_predict(trajectory['obs'])

        # scale rewards
        scale_rewards = trajectory['rewards'] * (1 - args.gamma)

        discount_sum_rewards = calc_discount_sum_rewards(scale_rewards, args.gamma).astype('float32')

        advantages = calc_gae(scale_rewards, pred_values, 0, args.gamma, args.lam)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        advantages = advantages.astype('float32')

        train_obs.append(trajectory['obs'])
        train_actions.append(trajectory['actions'])
        train_advantages.append(advantages)
        train_discount_sum_rewards.append(discount_sum_rewards)

    train_obs = np.concatenate(train_obs)
    train_actions = np.concatenate(train_actions)
    train_advantages = np.concatenate(train_advantages)
    train_discount_sum_rewards = np.concatenate(train_discount_sum_rewards)

    return train_obs, train_actions, train_advantages, train_discount_sum_rewards


def main():
    env = retro_util.RetroEnv(game='SuperMarioBros-Nes',
                              use_restricted_actions=retro.Actions.DISCRETE,
                              skill_frame=3,
                              resize_shape=(1, 111, 112),
                              render_preprocess=False,
                              is_train=True)

    obs_dim = env.observation_space.shape
    action_dim = env.action_space.n
    obs_dim = (1, 112, 112)  # add 1 to obs dim for time step feature

    scaler = Scaler(obs_dim)

    model = Model(obs_dim, action_dim)
    alg = parl.algorithms.PPO(model=model,
                              act_dim=action_dim,
                              policy_lr=model.policy_lr,
                              value_lr=model.value_lr)
    agent = Agent(alg, obs_dim, action_dim, args.kl_targ, loss_type=args.loss_type)

    # 预热并初始化scaler
    collect_trajectories(env, agent, scaler, episodes=5)

    total_steps = 0
    print("开始训练...")
    while total_steps < args.train_total_steps:
        trajectories = collect_trajectories(env, agent, scaler, episodes=args.episodes_per_batch)
        total_steps += sum([t['obs'].shape[0] for t in trajectories])
        total_train_rewards = sum([np.sum(t['rewards']) for t in trajectories])

        train_obs, train_actions, train_advantages, train_discount_sum_rewards = build_train_data(trajectories, agent)

        policy_loss, kl = agent.policy_learn(train_obs, train_actions, train_advantages)
        value_loss = agent.value_learn(train_obs, train_discount_sum_rewards)

        logger.info('Steps {}, Train reward: {}, Policy loss: {}, KL: {}, Value loss: {}'
                    .format(total_steps, total_train_rewards / args.episodes_per_batch, policy_loss, kl, value_loss))
        # if total_steps % 500 == 0:
        #     eval_reward = run_evaluate_episode(env, agent, scaler)
        #     logger.info('Steps {}, Evaluate reward: {}'.format(total_steps, eval_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        help='Mujoco environment name',
                        default='HalfCheetah-v2')
    parser.add_argument('--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('--lam',
                        type=float,
                        help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('--kl_targ', type=float, help='D_KL target value', default=0.003)
    parser.add_argument('--episodes_per_batch',
                        type=int,
                        help='Number of episodes per training batch',
                        default=5)
    parser.add_argument('--loss_type',
                        type=str,
                        help="Choose loss type of PPO algorithm, 'CLIP' or 'KLPEN'",
                        default='CLIP')
    parser.add_argument('--train_total_steps',
                        type=int,
                        default=int(1e7),
                        help='maximum training steps')
    parser.add_argument('--test_every_steps',
                        type=int,
                        default=int(1e4),
                        help='the step interval between two consecutive evaluations')

    args = parser.parse_args()

    main()
