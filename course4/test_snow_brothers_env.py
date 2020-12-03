import cv2
import retro


def change_obs_color(obs, src, target):
    for i in range(len(src)):
        index = (obs == src[i])
        obs[index] = target[i]
    return obs


def main():
    env = retro.make(game='SnowBrothers-Nes')
    obs = env.reset()
    print(obs.shape)
    w, h, c = obs.shape

    for i in range(100000):
        # 把图像转成灰度图
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = obs[25:h, 15:w]
        obs = change_obs_color(obs, [66, 88, 114, 186, 189, 250], [255, 255, 255, 255, 255, 0])
        # 显示处理过的图像
        cv2.imshow("preprocess", obs)
        cv2.waitKey(1)
        # 游戏生成的随机动作
        action = env.action_space.sample()
        obs, reward, terminal, info = env.step(action)
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
