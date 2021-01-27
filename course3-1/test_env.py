
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from env import create_train_env


def main():
    # 获取游戏
    env = create_train_env(world=1, stage=2, actions=COMPLEX_MOVEMENT)

    print(env.observation_space.shape)
    print(env.action_space.n)

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
