import argparse
import os
import cv2
import numpy as np
import parl
import retro
from agent import Agent
from model import ActorModel, CriticModel
from parl.utils import logger, summary
from replay_memory import ReplayMemory

ACTOR_LR = 1e-4  # actor模型的学习率
CRITIC_LR = 1e-4  # critic模型的学习速率
GAMMA = 0.99  # 奖励系数
TAU = 0.005  # 衰减参数
MEMORY_SIZE = int(1e5)  # 内存记忆大小
WARMUP_SIZE = 1e4  # 热身大小
BATCH_SIZE = 32  # batch大小
ENV_SEED = 1  # 固定随机情况
RESIZE_SHAPE = (1, 112, 112)  # 训练缩放的大小，减少模型计算，原大小（224,240）


# 改变游戏的布局环境，减低输入图像的复杂度
def change_obs_color(obs, src, target):
    for i in range(len(src)):
        index = (obs == src[i])
        obs[index] = target[i]
    return obs


# 图像预处理
def preprocess(observation, render=False):
    assert RESIZE_SHAPE[0] == 1 or RESIZE_SHAPE[0] == 3
    w, h, c = observation.shape
    observation = observation[25:h, 15:w]
    if RESIZE_SHAPE[0] == 1:
        # 把图像转成灰度图
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # 把其他的亮度调成一种，减低图像的复杂度
        observation = change_obs_color(observation, [66, 88, 114, 186, 189, 250], [255, 255, 255, 255, 255, 0])
        observation = cv2.resize(observation, (RESIZE_SHAPE[2], RESIZE_SHAPE[1]))
        if render:
            # 显示处理过的图像
            cv2.imshow("preprocess", observation)
            cv2.waitKey(1)
        observation = np.expand_dims(observation, axis=0)
    else:
        observation = cv2.resize(observation, (RESIZE_SHAPE[2], RESIZE_SHAPE[1]))
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        if render:
            # 显示处理过的图像
            cv2.imshow("preprocess", observation)
            cv2.waitKey(1)
        observation = observation.transpose((2, 0, 1))
    observation = observation / 255.0
    return observation


# 训练模型
def run_train_episode(env, agent, rpm, render=False):
    obs = env.reset()
    obs = preprocess(obs, render)
    total_reward = 0
    steps = 0
    lives = 2
    while True:
        steps += 1
        if render:
            # 显示视频图像
            env.render()
        if rpm.size() < WARMUP_SIZE:
            # 获取随机动作
            action = env.action_space.sample()
        else:
            # 预测动作
            action = agent.sample(obs)
            # 获取动作
            action = [0 if a < 0 else 1 for a in action]

        # 执行游戏
        next_obs, reward, isOver, info = env.step(action)
        next_obs = preprocess(next_obs, render)

        # 死一次就直接结束
        if info['lives'] < lives:
            isOver = True

        # 记录数据
        rpm.append(obs, action, reward, next_obs, isOver)

        # 训练模型
        if rpm.size() > WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if isOver:
            break
    return total_reward, steps


# 评估模型
def run_evaluate_episode(env, agent, render=False):
    obs = env.reset()
    total_reward = 0
    while True:
        if render:
            # 显示视频图像
            env.render()
        obs = preprocess(obs, render)
        # 预测动作
        action = agent.predict(obs)
        action = np.squeeze(action)
        # 获取动作
        action = [0 if a < 0 else 1 for a in action]
        # 执行游戏
        next_obs, reward, isOver, info = env.step(action)

        obs = next_obs
        total_reward += reward

        if isOver:
            break
    return total_reward


def main():
    # 初始化游戏
    env = retro.make(game=args.env)
    env.seed(ENV_SEED)

    # 游戏的图像形状
    obs_dim = RESIZE_SHAPE
    # 动作维度
    action_dim = env.action_space.n
    # 动作正负的最大绝对值
    max_action = 1

    # 创建模型
    actor = ActorModel(action_dim)
    critic = CriticModel()
    algorithm = parl.algorithms.SAC(actor=actor,
                                    critic=critic,
                                    max_action=max_action,
                                    gamma=GAMMA,
                                    tau=TAU,
                                    actor_lr=ACTOR_LR,
                                    critic_lr=CRITIC_LR)
    agent = Agent(algorithm, obs_dim, action_dim)

    # 创建记录数据存储器
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, action_dim)

    total_steps = 0 
    while total_steps < args.train_total_steps:
        # 训练
        train_reward, steps = run_train_episode(env, agent, rpm, render=args.show_play)
        logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward))
        summary.add_scalar('train/episode_reward', train_reward, total_steps)
        total_steps += steps

        # 评估
        if total_steps % 1000 == 0:
            evaluate_reward = run_evaluate_episode(env, agent, render=args.show_play)
            logger.info('Steps {}, Evaluate reward: {}'.format(total_steps, evaluate_reward))
            summary.add_scalar('eval/episode_reward', evaluate_reward, total_steps)

        # 保存模型
        if not os.path.exists(os.path.dirname(args.model_path)):
            os.makedirs(os.path.dirname(args.model_path))
        agent.save(args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        default='SnowBrothers-Nes',
                        help='Nes environment name')
    parser.add_argument('--train_total_steps',
                        type=int,
                        default=int(1e7),
                        help='maximum training steps')
    parser.add_argument('--show_play',
                        type=bool,
                        default=True,
                        help='if show game play')
    parser.add_argument('--model_path',
                        type=str,
                        default='models/model.ckpt',
                        help='save model path')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.2,
                        help='Temperature parameter α determines the relative importance of the \
        entropy term against the reward (default: 0.2)')
    args = parser.parse_args()
    main()
