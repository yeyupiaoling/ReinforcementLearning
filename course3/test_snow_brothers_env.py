import cv2
import retro


def change_obs_color(obs, src, target):
    for i in range(len(src)):
        index = (obs == src[i])
        obs[index] = target[i]
    return obs


def main():
    # game指定游戏，state指定开始状态，use_restricted_actions指定动作类型，players指定玩家数量，obs_type指定输出obs的类型
    env = retro.RetroEnv(game='SnowBrothers-Nes',
                         state=retro.State.DEFAULT,
                         use_restricted_actions=retro.Actions.DISCRETE,
                         players=1,
                         obs_type=retro.Observations.IMAGE)
    obs = env.reset()
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
