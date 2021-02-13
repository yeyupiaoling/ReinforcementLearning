# 项目结构

# retro游戏模拟器

1. 安装retro
```shell
pip install gym-retro -i https://mirrors.aliyun.com/pypi/simple/
```

2. 查看retro支持的游戏，执行这两行代码会输出retro支持的游戏。
```python
import retro
retro.data.list_games()
```

3. 导入游戏的ROM文件，我们安装了retro不是就能够直接使用的，因为retro不会提供游戏的ROM文件，所以需要我们自己去下载和导入。跟着笔者下载游戏的ROM文件并导入到Python环境中。[这里](https://archive.org/download/No-Intro-Collection_2016-01-03_Fixed) 提供了很多游戏的ROM，笔者下载的是[Nintendo - Nintendo Entertainment System.zip](https://archive.org/download/No-Intro-Collection_2016-01-03_Fixed/Nintendo%20-%20Nintendo%20Entertainment%20System.zip) ，导入命令如下：
```shell script
python -m retro.import /ROM的文件夹/
```
**特别注意：** 如果ROM带有`.bin`扩展名，需要将其重命名为具有该系统正确的扩展名。如Atari类型的，需要改成`.a26`，以便`python -m retro.import`成功！
```shell script
.md： 世嘉创世纪（Mega Drive）
.sfc：超级任天堂娱乐系统
.nes：任天堂娱乐系统
.a26：Atari 2600
.gb： 任天堂游戏男孩
.gba：任天堂Game Boy Advance
.gbc：任天堂游戏男孩Color
.gg： 世嘉游戏装备
.pce：NEC TurboGrafx-16
.sms：世嘉Master System
```

4. 测试游戏环境，通过执行`test_env.py`可以测试游戏的环境。
```python
import cv2
import numpy as np

from env import create_train_env


def main():
    # 获取游戏
    env = create_train_env(game="SuperMarioBros-Nes")
    print(env.observation_space.shape)
    print(env.action_space.n)

    obs = env.reset()

    while True:
        # 游戏生成的随机动作，int类型数值
        action = env.action_space.sample()
        # 执行游戏
        obs, reward, terminal, info = env.step(action)
        # 显示连续动作
        obs = np.squeeze(obs)
        obses = obs[0]
        for i in range(1, obs.shape[0]):
            obses = np.hstack([obses, obs[i]])
        cv2.imshow('obes', obses)
        cv2.waitKey(1)
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
```

# 训练

1. 首先安装PaddlePaddle以及相关库。
```shell
pip install paddlepaddle-gpu==2.0.0 visualdl -i https://mirrors.aliyun.com/pypi/simple/
```

2. 执行训练，通过game参数可以指定想要选择的游戏，前提是retro得支持这个游戏并且已经导入了ROM文件。
```shell
python train.py --game=SuperMarioBros-Nes
```

3. 验证通过的游戏如下，最新的验证通过的游戏请参考本项目的Github文档。
```shell script
SuperMarioBros-Nes
SnowBrothers-Nes
```

4. 自行训练其他游戏，理论情况下只需要更改游戏名称就可以正常训练，但是为了模型更好的拟合，最好是可以修改一下游戏可执行的动作，动作在`actions.py`。

## 预测
预测程序会使用训练时保存的最好得分模型进行预测，这个预测程序需要在界面环境下执行，如果要在终端下执行，需要注释`env.render()`这行代码。
```shell
python infer.py --game=SuperMarioBros-Nes
```

