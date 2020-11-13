import parl
import retro
from parl.utils import logger
from model import Model
from agent import Agent
from replay_memory import ReplayMemory


LEARN_FREQ = 5  # 更新参数步数
MEMORY_SIZE = 20000  # 内存记忆
MEMORY_WARMUP_SIZE = 200  # 热身大小
BATCH_SIZE = 32  # batch大小
LEARNING_RATE = 0.0005  # 学习率大小
GAMMA = 0.99  # 奖励系数
ENV_SEED = 1  # 固定随机情况
E_GREED = 0.9  # 探索初始概率
E_GREED_DECREMENT = 1e-6  # 在训练过程中，降低探索的概率
MAX_EPISODE = 200000  # 训练次数
SHOW_PLAY = False  # 显示游戏界面


def run_episode(agent, env, rpm, render=False):
    total_reward = 0
    obs = env.reset()
    obs = obs.transpose((2, 0, 1))
    obs = obs / 255.0
    step = 0
    lives = 2
    while True:
        step += 1
        if render:
            # 显示视频图像
            env.render()
        # 获取随机动作和执行游戏
        action = agent.sample(obs, env)
        next_obs, reward, isOver, info = env.step(action)
        next_obs = next_obs.transpose((2, 0, 1))
        next_obs = next_obs / 255.0

        # 死一次就惩罚
        if info['lives'] < lives:
            reward = -10
            lives = info['lives']

        # 动作转成十进制类别标签
        aa = ''.join([str(a) for a in action])
        action = [int(aa, 2)]
        # 记录数据
        rpm.append((obs, action, reward, next_obs, isOver))

        # 训练模型
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_isOver) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_isOver)

        total_reward += reward
        obs = next_obs
        # 结束游戏
        if isOver:
            break
    return total_reward


# 评估模型
def evaluate(agent, env, render=False):
    obs = env.reset()
    episode_reward = 0
    isOver = False
    while not isOver:
        obs = obs.transpose((2, 0, 1))
        obs = obs / 255.0
        action = agent.predict(obs)
        if render:
            # 显示视频图像
            env.render()
        obs, reward, isOver, _ = env.step(action)
        episode_reward += reward
    return episode_reward


def main():
    # 初始化游戏
    env = retro.make(game='SnowBrothers-Nes')
    env.seed(ENV_SEED)

    # 游戏的图像形状和动作形状
    obs_dim = (env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1])
    action_dim = env.action_space.shape[0]

    # 创建存储执行游戏的内存
    rpm = ReplayMemory(MEMORY_SIZE)

    # 创建模型，因为有9个动作，每个动作有0和1状态，所以是2的9次方
    model = Model(act_dim=2**action_dim)
    algorithm = parl.algorithms.DQN(model, act_dim=2**action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(algorithm=algorithm,
                  obs_dim=obs_dim,
                  act_dim=action_dim,
                  e_greed=E_GREED,
                  e_greed_decrement=E_GREED_DECREMENT)

    # 预热
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(agent, env, rpm, render=SHOW_PLAY)

    # 开始训练
    episode = 0
    while episode < MAX_EPISODE:
        # 训练
        for i in range(50):
            train_reward = run_episode(agent, env, rpm, render=SHOW_PLAY)
            episode += 1
            logger.info('Episode: {}, Reward: {}, e_greed: {}'.format(episode, train_reward, agent.e_greed))

        # 评估
        eval_reward = evaluate(agent, env, render=True)
        logger.info('Episode: {}, Evaluate reward:{}'.format(episode, eval_reward))


if __name__ == '__main__':
    main()
