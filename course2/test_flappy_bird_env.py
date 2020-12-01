import flappy_bird.wrapped_flappy_bird as flappyBird
import random


def main():
    env = flappyBird.GameState()

    obs = env.reset()
    while True:
        action = random.randint(0, 1)
        obs, reward, isOver = env.step(action)
        print("="*50)
        print("action:", action)
        print("obs shape:", obs.shape)
        print("reward:", reward)
        print("terminal:", isOver)
        if isOver:
            obs = env.reset()


if __name__ == "__main__":
    main()
