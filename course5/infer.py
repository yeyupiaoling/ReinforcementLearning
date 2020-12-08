import numpy as np
import parl
import retro
from config import config

import retro_util
from agent import Agent
from model import Model

RESIZE_SHAPE = (1, 112, 112)  # 训练缩放的大小
SAVE_MODEL_PATH = "models"  # 保存模型路径


def main():
    # 获取游戏，skill_frame每个动作执行的次数，resize_shape图像预处理的大小，render_preprocess是否显示预处理后的图像
    env = retro_util.RetroEnv(game=config['env_name'],
                              use_restricted_actions=retro.Actions.DISCRETE,
                              skill_frame=4,
                              resize_shape=config['obs_shape'],
                              render_preprocess=True)
    action_dim = env.action_space.n

    # 创建模型
    model = Model(action_dim)
    algorithm = parl.algorithms.A3C(model, vf_loss_coeff=config['vf_loss_coeff'])
    agent = Agent(algorithm, config)

    # 加载模型
    agent.restore(config['model_path'])

    # 开始游戏
    obs = env.reset()
    total_reward = 0
    isOver = False
    # 游戏未结束执行一直执行游戏
    while not isOver:
        env.render()
        obs = np.expand_dims(obs, axis=0)
        action = agent.predict(obs)[0]
        obs, reward, isOver, info = env.step(action)
        total_reward += reward
    env.close()
    print("最终得分为：{:.2f}".format(total_reward))


if __name__ == '__main__':
    main()
