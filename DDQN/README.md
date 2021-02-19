# DQN-FlappyBird
本项目是使用DDQN模型训练Chrome浏览器的霸王龙游戏，这个是一个入门级的强化学习进阶项目，通过使用DDQN控制霸王龙通过障碍。

# 项目结构
```shell script
DQN-FlappyBird/
├── dino                霸王龙游戏程序
├── env.py              游戏处理
├── game.py             启动游戏程序
├── infer.py            使用训练好的模型进行推理
├── model.py            模型结构
├── replay_memory.py    游戏数据的记录器
├── test_env.py         测试游戏环境
├── train.py            训练DQN模型
```

# 超级马里奥

1. 安装gym游戏环境，命令如下。
```shell
pip install pygame
```

2. 使用`test_env.py`可以测试游戏环境，游戏在执行每一步都会返回`obs, reward, done, info`这四个数据，启动obs是游戏图像，reward是游戏奖励的分数，done是当前游戏是否结束，info是游戏返回的信息。
```python
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
```


# 训练

1. 首先安装PaddlePaddle以及相关库。
```shell
pip install paddlepaddle-gpu==2.0.0 -i https://mirrors.aliyun.com/pypi/simple/
```

2. 执行训练，执行训练的过程中，也会进行评估模型，通过可以查看评估的输出结果，了解训练的情况，随着训练的进行，模型能够控制小木棍不会掉下来。
```shell
python train.py
```


## 预测
预测程序会使用训练时保存的模型进行预测，这个预测程序需要在界面环境下执行。
```shell
python infer.py
```