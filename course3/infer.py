import parl

import retro_util
from agent import Agent
from model import ActorModel, CriticModel

ACTOR_LR = 1e-4  # actor模型的学习率
CRITIC_LR = 1e-4  # critic模型的学习速率
GAMMA = 0.99  # 奖励系数
TAU = 0.005  # 衰减参数
RESIZE_SHAPE = (1, 112, 112)  # 训练缩放的大小
SAVE_MODEL_PATH = "models/model.ckpt"  # 保存模型路径


def main():
    # 获取游戏，skill_frame每个动作执行的次数，resize_shape图像预处理的大小，render_preprocess是否显示预处理后的图像
    env = retro_util.RetroEnv(game='SuperMarioBros-Nes',
                              skill_frame=4,
                              resize_shape=RESIZE_SHAPE,
                              render_preprocess=False)

    # 游戏的图像形状
    obs_dim = env.observation_space.shape
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

    # 加载模型
    agent.restore(SAVE_MODEL_PATH)

    # 开始游戏
    obs = env.reset()
    total_reward = 0
    isOver = False
    # 游戏未结束执行一直执行游戏
    while not isOver:
        env.render()
        # 获取动作
        action = agent.predict(obs)
        action = [0 if a < 0 else 1 for a in action]
        print('执行动作：', action)
        obs, reward, isOver, info = env.step(action)
        total_reward += reward
    env.close()
    print("最终得分为：{:.2f}".format(total_reward))


if __name__ == '__main__':
    main()
