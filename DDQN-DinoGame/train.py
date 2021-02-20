import os
import time

import cv2
import numpy as np
import paddle

from env import DinoGame
from model import Model
from replay_memory import ReplayMemory

# 定义训练的参数
batch_size = 64  # batch大小
num_episodes = 10000  # 训练次数
memory_size = 20000  # 内存记忆
learning_rate = 1e-3  # 学习率大小
e_greed = 0.1  # 探索初始概率
gamma = 0.99  # 奖励系数
e_greed_decrement = 1e-6  # 在训练过程中，降低探索的概率
update_num = 0  # 用于计算目标模型更新次数
save_model_path = "models/model.pdparams"  # 保存模型路径
resize_shape = (1, 30, 90)  # 训练缩放的大小
FPS = 25  # 控制游戏截图帧数

# 实例化一个游戏环境，参数为游戏名称
env = DinoGame(reshape=resize_shape)
# 图像输入形状和动作维度
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 创建策略模型和目标模型，目标模型不参与训练
policyQ = Model(obs_dim, action_dim)
targetQ = Model(obs_dim, action_dim)
targetQ.eval()

# 数据记录器
rpm = ReplayMemory(memory_size)
# 优化方法
optimizer = paddle.optimizer.Adam(parameters=policyQ.parameters(),
                                  learning_rate=learning_rate)


# 评估模型
def evaluate():
    total_reward = 0
    obs = env.reset()
    last_time = time.time()
    while True:
        obs = np.expand_dims(obs, axis=0)
        obs = paddle.to_tensor(obs, dtype='float32')
        action = targetQ(obs)
        action = paddle.argmax(action).numpy()[0]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward
        if done:
            break
        # 防止截图太快
        fps_now = 1 / (time.time() - last_time)
        if fps_now > FPS:
            time.sleep(1 / FPS - 1 / fps_now)
        last_time = time.time()
    return total_reward


# 训练模型
def train():
    global e_greed, update_num
    total_reward = 0
    # 重置游戏状态
    obs = env.reset()
    last_time = time.time()
    while True:
        # 使用贪心策略获取游戏动作的来源
        e_greed = max(0.01, e_greed - e_greed_decrement)
        if np.random.rand() < e_greed:
            # 随机生成动作
            action = env.action_space.sample()
        else:
            # 策略模型预测游戏动作
            obs1 = np.expand_dims(obs, axis=0)
            action = policyQ(paddle.to_tensor(obs1, dtype='float32'))
            action = paddle.argmax(action).numpy()[0]

        # 执行游戏
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        # 记录游戏数据
        rpm.append((obs, action, reward, next_obs, done))
        obs = next_obs
        # 游戏结束
        if done:
            break
        # 记录的数据打印batch_size就开始训练
        if len(rpm) > batch_size:
            # 获取训练数据
            batch_obs, batch_action, batch_reword, batch_next_obs, batch_done = rpm.sample(batch_size)
            # 计算损失函数
            action_value = policyQ(batch_obs)
            action_onehot = paddle.nn.functional.one_hot(batch_action, action_dim)
            pred_action_value = paddle.sum(action_value * action_onehot, axis=1)

            next_action_value = policyQ(batch_next_obs)
            greedy_action = paddle.argmax(next_action_value, axis=-1)
            greedy_action_onehot = paddle.nn.functional.one_hot(greedy_action, action_dim)
            next_pred_value = targetQ(batch_next_obs)
            max_v = paddle.sum(greedy_action_onehot * next_pred_value, axis=1)
            max_v.stop_gradient = True

            target = batch_reword + gamma * max_v * (1.0 - batch_done)
            cost = paddle.nn.functional.mse_loss(pred_action_value, target)
            # 梯度更新
            optimizer.clear_grad()
            cost.backward()
            optimizer.step()
            # 指定的训练次数更新一次目标模型的参数
            if update_num % 200 == 0:
                targetQ.load_dict(policyQ.state_dict())
            update_num += 1
        # 防止截图太快
        fps_now = 1 / (time.time() - last_time)
        if fps_now > FPS:
            time.sleep(1 / FPS - 1 / fps_now)
        last_time = time.time()
        # 显示图像
        img = cv2.putText(np.squeeze(obs), "%dFPS" % int(fps_now), (20, 10), cv2.FONT_ITALIC, 0.3, 255, 1)
        cv2.imshow('obs', img)
        cv2.waitKey(1)
    return total_reward


if __name__ == '__main__':
    episode = 0
    while episode < num_episodes:
        for t in range(50):
            train_reward = train()
            episode += 1
            print('Episode: {}, Reward: {:.2f}, e_greed: {:.2f}'.format(episode, train_reward, e_greed))

        eval_reward = evaluate()
        print('Episode:{}    test_reward:{}'.format(episode, eval_reward))
        # 保存模型
        if not os.path.exists(os.path.dirname(save_model_path)):
            os.makedirs(os.path.dirname(save_model_path))
        paddle.save(targetQ.state_dict(), save_model_path)
