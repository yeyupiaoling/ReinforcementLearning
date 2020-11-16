import argparse
import cv2
import retro
import numpy as np
import parl
from agent import Agent
from model import Model
from parl.utils import logger
from replay_memory import ReplayMemory

ACTOR_LR = 1e-4  # actor模型的学习率
CRITIC_LR = 1e-3  # critic模型的学习速率
GAMMA = 0.99  # 奖励系数
TAU = 0.001  # 衰减参数
MEMORY_SIZE = int(1e4)  # 内存记忆
MEMORY_WARMUP_SIZE = 1e3  # 热身大小
BATCH_SIZE = 128  # batch大小
REWARD_SCALE = 0.1  # 奖励比例
ENV_SEED = 1  # 固定随机情况
RESIZE_SHAPE = (1, 224, 224)  # 训练缩放的大小，减少模型计算，原大小（224,240）


# 图像预处理
def preprocess(observation):
    assert RESIZE_SHAPE[0] == 1 or RESIZE_SHAPE[0] == 3
    observation = cv2.resize(observation, (RESIZE_SHAPE[1], RESIZE_SHAPE[2]))
    if RESIZE_SHAPE[0] == 1:
        # 把图像转成灰度图
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # 显示处理过的图像
        cv2.imshow("preprocess", observation)
        observation = np.expand_dims(observation, axis=0)
    else:
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        # 显示处理过的图像
        cv2.imshow("preprocess", observation)
        observation = observation.transpose((2, 0, 1))
    observation = observation / 255.0
    return observation


def run_train_episode(env, agent, rpm, render=False):
    obs = env.reset()
    obs = preprocess(obs)
    total_reward = 0
    lives = 2
    while True:
        if render:
            # 显示视频图像
            env.render()
        # 预测动作
        action = agent.predict(obs.astype('float32'))

        # 利用高斯分布添加噪声
        action = np.random.normal(action, 1.0)
        # 将动作固定在0和1
        action = [1 if a > 0 else 0 for a in action]

        next_obs, reward, terminal, info = env.step(action)
        next_obs = preprocess(next_obs)

        # 死一次就惩罚
        if info['lives'] < lives:
            reward = -10
            lives = info['lives']

        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, terminal)

        # 训练模型
        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if terminal:
            break
    return total_reward


# 评估模型
def run_evaluate_episode(env, agent, render=False):
    obs = env.reset()
    total_reward = 0
    while True:
        if render:
            # 显示视频图像
            env.render()
        obs = preprocess(obs)
        action = agent.predict(obs.astype('float32'))
        # 将动作固定在0和1
        action = [1 if a > 0 else 0 for a in action]

        next_obs, reward, terminal, info = env.step(action)

        obs = next_obs
        total_reward += reward

        if terminal:
            break
    return total_reward


def main():
    # 初始化游戏
    env = retro.make(game='SnowBrothers-Nes')
    env.seed(ENV_SEED)

    # 游戏的图像形状和动作形状
    obs_dim = RESIZE_SHAPE
    act_dim = env.action_space.shape[0]

    # 创建模型
    model = Model(act_dim)
    algorithm = parl.algorithms.DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm, obs_dim, act_dim)

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, act_dim)

    print("开始预热...")
    while rpm.size() < MEMORY_WARMUP_SIZE:
        run_train_episode(env, agent, rpm, render=args.show_play)

    print("开始正式训练...")
    episode = 0
    while episode < args.train_total_episode:
        # 训练
        for i in range(50):
            train_reward = run_train_episode(env, agent, rpm, render=args.show_play)
            episode += 1
            logger.info('Episode: {} Reward: {}'.format(episode, train_reward))

        # 评估
        evaluate_reward = run_evaluate_episode(env, agent, render=args.show_play)
        logger.info('Episode {}, Evaluate reward: {}'.format(episode, evaluate_reward))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_total_episode', type=int, default=int(1e4), help='maximum training episodes')
    parser.add_argument('--show_play', type=bool, default=True, help='if show game play')

    args = parser.parse_args()

    main()
