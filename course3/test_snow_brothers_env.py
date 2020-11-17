import retro
import random


def main():
    env = retro.make(game='SnowBrothers-Nes')
    # 所有动作的组合，第一个是射击，倒数第三个为向左移动，倒数第二个为向右移动，倒数第一个为向跳，其他都没用的。
    # 射击和跳的动作要0和1两个动作结合在一起才可以
    actions = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1],
               [1, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 1, 0],
               [1, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0],
               [1, 0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 1],
               [1, 0, 0, 0, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 0, 0, 1, 0, 1],
               [1, 0, 0, 0, 0, 0, 1, 0, 1]]

    obs = env.reset()
    for i in range(100000):
        # 游戏生成的随机动作
        # action = env.action_space.sample()
        # 自定义的随机动作
        action = random.choice(actions)
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
