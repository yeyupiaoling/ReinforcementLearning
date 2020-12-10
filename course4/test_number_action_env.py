import retro
import retro_util
import cv2
import numpy as np


def main():
    # 获取游戏
    env = retro_util.RetroEnv(game='SuperMarioBros-Nes',
                              use_restricted_actions=retro.Actions.DISCRETE,
                              skill_frame=4,
                              resize_shape=(1, 112, 112),
                              render_preprocess=True)
    obs = env.reset()

    while True:
        # 游戏生成的随机动作，int类型数值
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


if __name__ == "__main__":
    main()
