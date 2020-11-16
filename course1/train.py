import gym
import numpy as np
from parl.utils import logger
import parl
from model import Model
from agent import Agent
from replay_memory import ReplayMemory

LEARN_FREQ = 5  # 更新参数步数
MEMORY_SIZE = 20000  # 内存记忆
MEMORY_WARMUP_SIZE = 200  # 热身大小
BATCH_SIZE = 64  # batch大小
LEARNING_RATE = 0.0005  # 学习率大小
GAMMA = 0.99  # 奖励系数
E_GREED = 0.1  # 探索初始概率
E_GREED_DECREMENT = 1e-6  # 在训练过程中，降低探索的概率
MAX_EPISODE = 10000  # 训练次数


def run_train(agent, env, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        # 获取随机动作和执行游戏
        action = agent.sample(obs)
        next_obs, reward, isOver, _ = env.step(action)

        # 记录数据
        rpm.append((obs, [action], reward, next_obs, isOver))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_isOver) = rpm.sample(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_isOver)

        total_reward += reward
        obs = next_obs

        # 结束游戏
        if isOver:
            break
    return total_reward


# 评估模型
def evaluate(agent, env):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        isOver = False
        while not isOver:
            action = agent.predict(obs)
            env.render()
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
    # 实例化一个游戏环境，参数为游戏名称
    env = gym.make('CartPole-v0')

    # 图像输入形状和动作维度
    action_dim = env.action_space.n
    obs_shape = env.observation_space.shape

    # 创建存储执行游戏的内存
    rpm = ReplayMemory(MEMORY_SIZE)

    # 创建模型
    model = Model(act_dim=action_dim)
    algorithm = parl.algorithms.DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(algorithm=algorithm,
                  obs_dim=obs_shape[0],
                  act_dim=action_dim,
                  e_greed=E_GREED,
                  e_greed_decrement=E_GREED_DECREMENT)

    # 预热
    print("开始预热...")
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train(agent, env, rpm)

    # 开始训练
    print("开始正式训练...")
    episode = 0
    while episode < MAX_EPISODE:
        # 训练
        for i in range(0, 50):
            train_reward = run_train(agent, env, rpm)
            episode += 1
            logger.info('Episode: {}, Reward: {:.2f}, e_greed: {:.2f}'.format(episode, train_reward, agent.e_greed))

        # 评估
        eval_reward = evaluate(agent, env)
        logger.info('episode:{}    test_reward:{}'.format(episode, eval_reward))


if __name__ == '__main__':
    main()
