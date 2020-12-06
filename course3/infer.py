import cv2
import numpy as np
import parl
import retro

from agent import Agent
from model import Model

RESIZE_SHAPE = (1, 224, 224)  # 训练缩放的大小
SAVE_MODEL_PATH = "models/model.ckpt"  # 保存模型路径


# 图像预处理
def preprocess(observation):
    assert RESIZE_SHAPE[0] == 1 or RESIZE_SHAPE[0] == 3
    observation = cv2.resize(observation, (RESIZE_SHAPE[1], RESIZE_SHAPE[2]))
    if RESIZE_SHAPE[0] == 1:
        # 把图像转成灰度图
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # 显示处理过的图像
        cv2.imshow("preprocess", observation)
        cv2.waitKey(1)
        observation = np.expand_dims(observation, axis=0)
    else:
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        # 显示处理过的图像
        cv2.imshow("preprocess", observation)
        cv2.waitKey(1)
        observation = observation.transpose((2, 0, 1))
    observation = observation / 255.0
    return observation


def main():
    # 初始化游戏
    env = retro.make(game='SnowBrothers-Nes')

    # 游戏的动作维度
    action_dim = env.action_space.shape[0]

    # 创建模型
    model = Model(action_dim)
    algorithm = parl.algorithms.DDPG(model, gamma=0.99, tau=0.001, actor_lr=1e-4, critic_lr=1e-3)
    agent = Agent(algorithm=algorithm,
                  obs_dim=RESIZE_SHAPE,
                  action_dim=action_dim)

    agent.restore(SAVE_MODEL_PATH)

    # 开始游戏
    obs = env.reset()
    episode_reward = 0
    isOver = False
    while not isOver:
        obs = preprocess(obs)
        action = agent.predict(obs.astype('float32'))
        # 将动作固定在0和1
        action = [1 if a > 0 else 0 for a in action]

        obs, reward, isOver, info = env.step(action)
        episode_reward += reward
    print("最终得分为：{:.2f}".format(episode_reward))


if __name__ == '__main__':
    main()
