import gym


def main():
    # 初始化游戏
    env = gym.make("CartPole-v1")
    # 开始游戏
    obs = env.reset()

    # 游戏未结束执行一直执行游戏
    while True:
        # 游戏生成的随机动作，int类型数值
        action = env.action_space.sample()
        # 执行游戏
        obs, reward, done, info = env.step(action)
        env.render()
        print("=" * 50)
        print("action:", action)
        print("obs shape:", obs.shape)
        print("reward:", reward)
        print("terminal:", done)
        print("info:", info)
        if done:
            obs = env.reset()


if __name__ == "__main__":
    main()
