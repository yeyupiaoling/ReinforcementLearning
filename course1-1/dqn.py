import gym
import numpy as np
import paddle
from model import Model
from replay_memory import ReplayMemory

# 定义训练的参数
batch_size = 64  # batch大小
num_episodes = 10000  # 训练次数
memory_size = 10000  # 内存记忆
learning_rate = 1e-3  # 学习率大小
gamma = 1.0  # 奖励系数
e_greed = 0.1  # 探索初始概率
e_greed_decrement = 1e-6  # 在训练过程中，降低探索的概率
update_num = 0  # 用于计算目标模型更新次数

# 实例化一个游戏环境，参数为游戏名称
env = gym.make("CartPole-v1")
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
    state = env.reset()
    while True:
        # 显示游戏图像
        env.render()
        state = paddle.to_tensor(state, dtype='float32')
        action = targetQ(state)
        action = paddle.argmax(action).numpy()[0]
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward

        if done:
            break
    return total_reward


# 训练模型
def train():
    global e_greed, update_num
    total_reward = 0
    # 重置游戏状态
    state = env.reset()

    while True:
        # 显示游戏图像
        env.render()
        # 使用贪心策略获取游戏动作的来源
        e_greed = max(0.01, e_greed - e_greed_decrement)
        if np.random.rand() < e_greed:
            # 随机生成动作
            action = env.action_space.sample()
        else:
            # 策略模型预测游戏动作
            action = policyQ(paddle.to_tensor(state, dtype='float32'))
            action = paddle.argmax(action).numpy()[0]

        # 执行游戏
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        # 记录游戏数据
        rpm.append((state, action, reward, next_state, done))
        state = next_state
        # 游戏结束
        if done:
            break
        # 记录的数据打印batch_size就开始训练
        if rpm.__len__() > batch_size:
            # 获取训练数据
            batch_state, batch_action, batch_reword, batch_next_state, batch_done = rpm.sample(batch_size)
            # 计算损失函数
            action_value = policyQ(batch_state)
            action_onehot = paddle.nn.functional.one_hot(batch_action, 2)
            pred_action_value = paddle.sum(action_value * action_onehot, axis=1)

            best_v = targetQ(batch_next_state)
            best_v = paddle.max(best_v, axis=1)

            best_v.stop_gradient = False
            target = batch_reword + gamma * best_v * (1.0 - batch_done)

            cost = paddle.nn.functional.mse_loss(pred_action_value, target)
            # 梯度更新
            optimizer.clear_grad()
            cost.backward()
            optimizer.step()
            # 指定的训练次数更新一次目标模型的参数
            if update_num % 200 == 0:
                targetQ.load_dict(policyQ.state_dict())
            update_num += 1
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
