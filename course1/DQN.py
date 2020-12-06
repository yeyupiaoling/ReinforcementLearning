import numpy as np
import paddle.fluid as fluid
import random
import gym
from collections import deque
from paddle.fluid.param_attr import ParamAttr


# 定义一个深度神经网络
def Model(ipt, variable_field):
    fc1 = fluid.layers.fc(input=ipt,
                          size=24,
                          act='relu',
                          param_attr=ParamAttr(name='{}_fc1'.format(variable_field)),
                          bias_attr=ParamAttr(name='{}_fc1_b'.format(variable_field)))
    fc2 = fluid.layers.fc(input=fc1,
                          size=24,
                          act='relu',
                          param_attr=ParamAttr(name='{}_fc2'.format(variable_field)),
                          bias_attr=ParamAttr(name='{}_fc2_b'.format(variable_field)))
    out = fluid.layers.fc(input=fc2,
                          size=2,
                          param_attr=ParamAttr(name='{}_fc3'.format(variable_field)),
                          bias_attr=ParamAttr(name='{}_fc3_b'.format(variable_field)))
    return out


def dqn():
    # 定义输入数据
    state_data = fluid.data(name='state', shape=[None, 4], dtype='float32')
    action_data = fluid.data(name='action', shape=[None, 1], dtype='int64')
    reward_data = fluid.data(name='reward', shape=[None], dtype='float32')
    next_state_data = fluid.data(name='next_state', shape=[None, 4], dtype='float32')
    done_data = fluid.data(name='done', shape=[None], dtype='float32')

    # 获取策略网络
    policyQ = Model(state_data, 'policy')

    # 克隆预测程序
    predict_program = fluid.default_main_program().clone()

    action_onehot = fluid.layers.one_hot(action_data, 2)
    action_value = fluid.layers.elementwise_mul(action_onehot, policyQ)
    pred_action_value = fluid.layers.reduce_sum(action_value, dim=1)

    # 获取目标网络
    targetQ = Model(next_state_data, 'target')
    best_v = fluid.layers.reduce_max(targetQ, dim=1)
    # 停止梯度更新
    best_v.stop_gradient = True
    gamma = 1.0
    target = reward_data + gamma * best_v * (1.0 - done_data)

    # 定义损失函数
    cost = fluid.layers.square_error_cost(pred_action_value, target)
    avg_cost = fluid.layers.reduce_mean(cost)
    return predict_program, policyQ, avg_cost


# 定义更新参数程序
def _build_sync_target_network():
    # 获取所有的参数
    vars = list(fluid.default_main_program().list_vars())
    # 把两个网络的参数分别过滤出来
    policy_vars = list(filter(lambda x: 'GRAD' not in x.name and 'policy' in x.name, vars))
    target_vars = list(filter(lambda x: 'GRAD' not in x.name and 'target' in x.name, vars))
    policy_vars.sort(key=lambda x: x.name)
    target_vars.sort(key=lambda x: x.name)

    # 从主程序中克隆一个程序用于更新参数
    sync_program = fluid.default_main_program().clone()
    with fluid.program_guard(sync_program):
        sync_ops = []
        for i, var in enumerate(policy_vars):
            sync_op = fluid.layers.assign(policy_vars[i], target_vars[i])
            sync_ops.append(sync_op)
    # 完成更新参数
    sync_program = sync_program._prune(sync_ops)
    return sync_program


# 获取DQN程序
predict_program, policyQ, avg_cost = dqn()

# 获取更新参数程序
_sync_program = _build_sync_target_network()

# 定义优化方法
optimizer = fluid.optimizer.Adam(learning_rate=1e-3, epsilon=1e-3)
opt = optimizer.minimize(avg_cost)

# 创建执行器并进行初始化
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 定义训练的参数
batch_size = 64  # batch大小
num_episodes = 10000  # 训练次数
e_greed = 0.1  # 探索初始概率
e_greed_decrement = 1e-6  # 在训练过程中，降低探索的概率
update_num = 0


def run_train(env, replay_buffer):
    global update_num, e_greed
    total_reward = 0
    # 重置游戏状态
    state = env.reset()
    while True:
        # 显示游戏界面
        env.render()
        state = np.expand_dims(state, axis=0)
        # 定义贪心探索策略
        e_greed = max(0.01, e_greed - e_greed_decrement)
        if np.random.rand() < e_greed:
            # 以 e_greed 的概率选择随机下一步动作
            action = env.action_space.sample()
        else:
            # 使用模型预测作为结果下一步动作
            action = exe.run(predict_program,
                             feed={'state': state.astype('float32')},
                             fetch_list=[policyQ])[0]
            action = np.squeeze(action, axis=0)
            action = np.argmax(action)

        # 让游戏执行动作，获得执行完 动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        # 记录游戏输出的结果，作为之后训练的数据
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state

        # 如果游戏结束，就重新玩游戏
        if done:
            break

        # 如果收集的数据大于Batch的大小，就开始训练
        if len(replay_buffer) >= batch_size:
            batch_state, batch_action, batch_reward, batch_next_state, \
            batch_done = [np.array(a, np.float32) for a in
                          zip(*random.sample(replay_buffer, batch_size))]

            # 调整数据维度
            batch_action = np.expand_dims(batch_action, axis=-1)
            batch_state = np.squeeze(batch_state, axis=1)

            # 执行训练
            exe.run(program=fluid.default_main_program(),
                    feed={'state': batch_state,
                          'action': batch_action.astype('int64'),
                          'reward': batch_reward,
                          'next_state': batch_next_state,
                          'done': batch_done})

            # 更新参数
            if update_num % 200 == 0:
                exe.run(program=_sync_program)
            update_num += 1

    return total_reward


# 评估模型
def evaluate(env):
    total_reward = 0
    # 重置游戏状态
    state = env.reset()
    while True:
        state = np.expand_dims(state, axis=0)
        # 使用模型预测作为结果下一步动作
        action = exe.run(predict_program,
                         feed={'state': state.astype('float32')},
                         fetch_list=[policyQ])[0]
        action = np.squeeze(action, axis=0)
        action = np.argmax(action)
        next_state, reward, done, info = env.step(action)
        state = next_state

        total_reward += reward
        if done:
            break
    return total_reward


# 实例化一个游戏环境，参数为游戏名称
env = gym.make("CartPole-v1")
replay_buffer = deque(maxlen=10000)

# 开始训练
episode = 0
while episode < num_episodes:
    for t in range(50):
        train_reward = run_train(env, replay_buffer)
        episode += 1
        print('Episode: {}, Reward: {:.2f}, e_greed: {:.2f}'.format(episode, train_reward, e_greed))

    # 评估
    eval_reward = evaluate(env)
    print('episode:{}    test_reward:{}'.format(episode, eval_reward))
env.close()
