import cv2
import numpy

from env import TRexGame


def main():
    # 获取游戏
    env = TRexGame()
    print(env.observation_space.shape)
    print(env.action_space.n)

    while True:
        # 游戏生成的随机动作，int类型数值
        action = env.action_space.sample()
        # 执行游戏
        obs, reward, terminal, info = env.step(action)
        obs = numpy.squeeze(obs)
        cv2.imshow('obs', obs)
        cv2.waitKey(1)
        print("=" * 50)
        print("action:", action)
        print("obs shape:", obs.shape)
        print("reward:", reward)
        print("terminal:", terminal)
        print("info:", info)


if __name__ == "__main__":
    main()
