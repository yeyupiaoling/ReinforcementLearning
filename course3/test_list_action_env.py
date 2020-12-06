import retro_util


def main():
    # 获取游戏
    env = retro_util.RetroEnv(game='SnowBrothers-Nes',
                              skill_frame=1,
                              resize_shape=(1, 112, 112),
                              render_preprocess=True)
    obs = env.reset()
    print(env.observation_space)

    while True:
        # 游戏生成的随机动作，长度为9的list，值为0或1
        action = env.action_space.sample()
        # 执行游戏
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
