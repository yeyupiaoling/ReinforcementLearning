import argparse
import os
import parl
import retro_util
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
SKILL_FRAME = 4  # 每次执行多少帧
RESIZE_SHAPE = (1, 112, 112)  # 训练缩放的大小，减少模型计算，原大小（224,240）


# 训练模型
def run_train_episode(env, agent, rpm, render=False):
    # 获取最后一帧图像
    obs = env.reset()[None, -1, :, :]
    total_reward = 0
    steps = 0
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
        next_obs, reward, isOver, info = env.step_sac(action)
        # 获取最后一帧图像
        next_obs = next_obs

        # 记录数据
        rpm.append(obs, action, reward, next_obs, isOver)

        # 训练模型
        if rpm.size() > WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if isOver:
            if render:
                env.render(close=True)
            break
    return total_reward, steps


# 评估模型
def run_evaluate_episode(env, agent, render=False):
    # 获取最后一帧图像
    obs = env.reset()[None, -1, :, :]
    total_reward = 0
    while True:
        if render:
            # 显示视频图像
            env.render()
        # 预测动作
        action = agent.predict(obs)
        # 获取动作
        action = [0 if a < 0 else 1 for a in action]
        # 执行游戏
        next_obs, reward, isOver, info = env.step_sac(action)
        total_reward += reward
        # 获取最后一帧图像
        obs = next_obs

        if isOver:
            if render:
                env.render(close=True)
            break
    return total_reward


def main():
    # 获取游戏，skill_frame每个动作执行的次数，resize_shape图像预处理的大小，render_preprocess是否显示预处理后的图像
    env = retro_util.RetroEnv(game=args.env,
                              resize_shape=RESIZE_SHAPE,
                              skill_frame=SKILL_FRAME,
                              render_preprocess=args.show_play,
                              is_train=True)
    env.seed(1)

    # 游戏的图像形状
    # obs_dim = env.observation_space.shape
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

    # 加载预训练模型
    if os.path.exists(args.model_path):
        logger.info("加载预训练模型...")
        agent.restore(args.model_path)

    # 创建记录数据存储器
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, action_dim)

    total_steps = 0
    step_train = 0
    print("开始训练模型。。。")
    while total_steps < args.train_total_steps:
        # 训练
        train_reward, steps = run_train_episode(env, agent, rpm, render=args.show_play)
        logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward))
        summary.add_scalar('train/episode_reward', train_reward, total_steps)
        total_steps += steps

        # 评估
        if step_train % 100 == 0:
            evaluate_reward = run_evaluate_episode(env, agent, render=args.show_play)
            logger.info('Steps {}, Evaluate reward: {}'.format(total_steps, evaluate_reward))
            summary.add_scalar('eval/episode_reward', evaluate_reward, total_steps)
        step_train += 1

        # 保存模型
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        agent.save(args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        default='SuperMarioBros-Nes',
                        help='Nes environment name')
    parser.add_argument('--train_total_steps',
                        type=int,
                        default=int(1e9),
                        help='maximum training steps')
    parser.add_argument('--show_play',
                        type=bool,
                        default=True,
                        help='if show game play')
    parser.add_argument('--model_path',
                        type=str,
                        default='models',
                        help='save model path')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.2,
                        help='Temperature parameter α determines the relative importance of the \
        entropy term against the reward (default: 0.2)')
    args = parser.parse_args()
    main()
