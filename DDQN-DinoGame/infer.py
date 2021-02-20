import time

import numpy as np
import paddle

from env import DinoGame
from model import Model

resize_shape = (1, 30, 90)  # 训练缩放的大小
save_model_path = "models/model.pdparams"  # 保存模型路径
FPS = 25  # 控制游戏截图帧数


def main():
    # 初始化游戏
    env = DinoGame()
    # 图像输入形状和动作维度
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 创建模型
    model = Model(obs_dim, action_dim)
    model.load_dict(paddle.load(save_model_path))
    model.eval()

    # 开始游戏
    obs = env.reset()
    episode_reward = 0
    done = False
    last_time = time.time()
    # 游戏未结束执行一直执行游戏
    while not done:
        obs = np.expand_dims(obs, axis=0)
        obs = paddle.to_tensor(obs, dtype='float32')
        action = model(obs)
        action = paddle.argmax(action).numpy()[0]
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        # 防止截图太快
        fps_now = 1 / (time.time() - last_time)
        if fps_now > FPS:
            time.sleep(1 / FPS - 1 / fps_now)
        last_time = time.time()
    print("最终得分为：{:.2f}".format(episode_reward))


if __name__ == '__main__':
    main()
