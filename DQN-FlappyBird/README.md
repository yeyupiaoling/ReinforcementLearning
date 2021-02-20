# DQN-FlappyBird
本项目是使用DQN模型训练飞翔的小鸟游戏，这个是一个入门级的强化学习进阶项目，通过使用DQN控制小鸟通过障碍。

# 项目结构
```shell script
DQN-FlappyBird/
├── flappy_bird/        游戏程序
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
import flappy_bird.wrapped_flappy_bird as flappyBird


def main():
    # 初始化游戏
    env = flappyBird.GameState()
    # 开始游戏
    obs = env.reset()

    # 游戏未结束执行一直执行游戏
    while True:
        # 游戏生成的随机动作，int类型数值
        action = env.action_space()
        # 执行游戏
        obs, reward, done, info = env.step(action, is_train=False)
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


# 模型下载

| 模型 | 下载地址 |
| :---: | :---: |
| DQN | [点击下载](https://resource.doiduoyi.com/#ake9ca1) |

