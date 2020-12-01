import cv2
import numpy as np
import parl
import flappy_bird.wrapped_flappy_bird as flappyBird
from agent import Agent
from model import Model

RESIZE_SHAPE = (1, 224, 224)  # 训练缩放的大小
SAVE_MODEL_PATH = "models/model.ckpt"  # 保存模型路径


def preprocess(observation):
    observation = cv2.resize(observation, (RESIZE_SHAPE[1], RESIZE_SHAPE[2]))
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    observation = np.expand_dims(observation, axis=0)
    observation = observation / 255.0
    return observation


def main():
    # 初始化游戏
    env = flappyBird.GameState()

    # 动作维度
    action_dim = env.action_dim

    # 创建模型
    model = Model(act_dim=action_dim)
    algorithm = parl.algorithms.DQN(model, act_dim=action_dim, gamma=0.99, lr=0.0005)
    agent = Agent(algorithm=algorithm,
                  obs_dim=RESIZE_SHAPE,
                  act_dim=action_dim)

    agent.restore(SAVE_MODEL_PATH)

    # 开始游戏
    obs = env.reset()
    episode_reward = 0
    isOver = False
    while not isOver:
        obs = preprocess(obs)
        action = agent.predict(obs)
        obs, reward, isOver = env.step(action)
        episode_reward += reward
    print("最终得分为：{:.2f}".format(episode_reward))


if __name__ == '__main__':
    main()
