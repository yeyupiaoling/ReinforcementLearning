import retro


def main():
    env = retro.make(game='SnowBrothers-Nes')

    obs = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminal, info = env.step(action)
        env.render()
        print("="*50)
        print("action:", action)
        print("obs shape:", obs.shape)
        print("reward:", reward)
        print("terminal:", terminal)
        print("info:", info)
        if terminal:
            obs = env.reset()


if __name__ == "__main__":
    main()
