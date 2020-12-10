import time
import retro_util
import cv2
import numpy as np


def main():
    # 获取游戏，skill_frame每个动作执行的次数，resize_shape图像预处理的大小，render_preprocess是否显示预处理后的图像
    env = retro_util.RetroEnv(game='SuperMarioBros-Nes',
                              resize_shape=(1, 112, 112),
                              skill_frame=4,
                              render_preprocess=True)
    obs = env.reset()
    print(env.observation_space.shape)

    while True:
        # 游戏生成的随机动作，长度为9的list，值为0或1
        action = env.action_space.sample()
        # 执行游戏
        obs, reward, terminal, info = env.step(action)
        # 显示连续动作
        obses = obs[0]
        for i in range(1, obs.shape[0]):
            obses = np.hstack([obses, obs[i]])
        cv2.imshow('obes', obses)
        env.render()
        print("=" * 50)
        print("action:", action)
        print("obs shape:", obs.shape)
        print("reward:", reward)
        print("terminal:", terminal)
        print("info:", info)
        if terminal:
            obs = env.reset()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
