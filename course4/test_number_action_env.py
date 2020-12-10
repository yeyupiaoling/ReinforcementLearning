import retro
import retro_util


def main():
    # 获取游戏
    env = retro_util.RetroEnv(game='SuperMarioBros-Nes',
                              use_restricted_actions=retro.Actions.DISCRETE,
                              resize_shape=(1, 112, 112),
                              render_preprocess=True)
    obs = env.reset()

    while True:
        # 游戏生成的随机动作，int类型数值
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
