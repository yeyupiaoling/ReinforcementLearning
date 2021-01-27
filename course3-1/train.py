import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import paddle
from env import MultipleEnvironments
from model import Model
import multiprocessing as _mp
from utils import eval, print_arguments
from paddle.distribution import Categorical
import paddle.nn.functional as F
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world",            type=int,   default=1)
    parser.add_argument("--stage",            type=int,   default=1)
    parser.add_argument("--action_type",      type=str,   default="simple")
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--gamma',            type=float, default=0.9,  help='discount factor for rewards')
    parser.add_argument('--tau',              type=float, default=1.0,  help='parameter for GAE')
    parser.add_argument('--beta',             type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon',          type=float, default=0.2,  help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size',       type=int,   default=16)
    parser.add_argument('--num_epochs',       type=int,   default=10)
    parser.add_argument("--num_local_steps",  type=int,   default=512)
    parser.add_argument("--num_global_steps", type=int,   default=5e6)
    parser.add_argument("--num_processes",    type=int,   default=8)
    parser.add_argument("--save_interval",    type=int,   default=50,  help="Number of steps between savings")
    parser.add_argument("--max_actions",      type=int,   default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--saved_path",       type=str,   default="models")
    args = parser.parse_args()
    return args


# 训练模型
def train(args):
    # 使用 GPU训练
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu:0")
    # 创建多进程的游戏环境
    envs = MultipleEnvironments(args.world, args.stage, args.action_type, args.num_processes)
    # 固定初始化状态
    paddle.seed(123)
    # 创建模型
    model = Model(envs.num_states, envs.num_actions)
    # 创建保存模型的文件夹
    if not os.path.isdir(args.saved_path):
        os.makedirs(args.saved_path)
    paddle.save(model.state_dict(), "{}/model_{}_{}.pdparams".format(args.saved_path, args.world, args.stage))
    # 为游戏评估单独开一个进程
    mp = _mp.get_context("spawn")
    process = mp.Process(target=eval, args=(args, envs.num_states, envs.num_actions))
    process.start()
    # 创建优化方法
    clip_grad = paddle.nn.ClipGradByNorm(clip_norm=0.5)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip_grad)
    # 刚开始给每个进程的游戏执行初始化
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    # 获取游戏初始的界面
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = paddle.to_tensor(np.concatenate(curr_states, 0), dtype='float32')
    curr_episode = 0
    while True:
        curr_episode += 1
        old_log_policies, actions, values, states, rewards, dones = [], [], [], [], [], []
        for _ in range(args.num_local_steps):
            states.append(curr_states)
            # 执行预测
            logits, value = model(curr_states)
            # 计算每个动作的概率值
            policy = F.softmax(logits, axis=1)
            # 根据每个标签的概率随机生成符合概率的标签
            old_m = Categorical(policy)
            action = old_m.sample([1])
            action = paddle.t(action)
            # 记录预测数据
            actions.append(action)
            values.append(value.squeeze())
            #
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            # 向各个进程游戏发送动作
            [agent_conn.send(("step", int(act[0]))) for agent_conn, act in zip(envs.agent_conns, action)]
            # 将多进程的游戏数据打包
            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            # 进行数据转换
            state = paddle.to_tensor(np.concatenate(state, 0), dtype='float32')
            # 转换为tensor数据
            reward = paddle.to_tensor(reward, dtype='float32')
            done = paddle.to_tensor(done, dtype='float32')
            # 记录预测数据
            rewards.append(reward)
            dones.append(done)
            curr_states = state
        # 根据上面最后的图像预测
        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        old_log_policies = paddle.concat(old_log_policies).detach().squeeze()
        actions = paddle.concat(actions).squeeze()
        values = paddle.concat(values).detach().squeeze()
        states = paddle.concat(states).squeeze()

        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = paddle.to_tensor(gae * args.gamma * args.tau, dtype='float32')
            gae = gae + reward + args.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = paddle.concat(R).detach()
        advantages = R - values
        for i in range(args.num_epochs):
            indice = paddle.randperm(args.num_local_steps * args.num_processes)
            for j in range(args.batch_size):
                batch_indices = indice[
                                int(j * (args.num_local_steps * args.num_processes / args.batch_size)): int((j + 1) * (
                                        args.num_local_steps * args.num_processes / args.batch_size))]
                # 根据拿到的图像执行预测
                logits, value = model(paddle.gather(states, batch_indices))
                # 计算每个动作的概率值
                new_policy = F.softmax(logits, axis=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(paddle.gather(actions, batch_indices))
                # 计算actor损失
                ratio = paddle.exp(new_log_policy - paddle.gather(old_log_policies, batch_indices))
                advantages = paddle.gather(advantages, batch_indices)
                actor_loss = paddle.clip(ratio, 1.0 - args.epsilon, 1.0 + args.epsilon) * advantages
                actor_loss = paddle.to_tensor(ratio * advantages + actor_loss, dtype='float32')
                actor_loss = -paddle.mean(paddle.min(actor_loss, axis=0))

                # ratio = paddle.exp(new_log_policy - paddle.gather(old_log_policies, batch_indices))
                # advantages = paddle.gather(advantages, batch_indices)
                # actor_loss = paddle.to_tensor(list((ratio * advantages).numpy().astype("float32") +
                #                     (paddle.clip(ratio, 1.0 - args.epsilon, 1.0 + args.epsilon) * advantages).numpy().astype("float32")), dtype="float32")
                # actor_loss = -paddle.mean(paddle.min(actor_loss, axis=0))

                # 计算critic损失
                critic_loss = F.smooth_l1_loss(paddle.gather(R, batch_indices), value.squeeze())
                entropy_loss = paddle.mean(new_m.entropy())
                # 计算全部损失
                total_loss = actor_loss + critic_loss - args.beta * entropy_loss
                # 计算梯度
                optimizer.clear_grad()
                total_loss.backward()
                # for parameter in model.parameters():
                #     paddle.nn.clip_by_norm(parameter, 0.5)
                optimizer.step()
            paddle.save(model.state_dict(), "{}/model_{}_{}.pdparams".format(args.saved_path, args.world, args.stage))
        print("Episode: {}. Total loss: {:.4f}".format(curr_episode, total_loss.numpy()[0]))


if __name__ == "__main__":
    args = get_args()
    print_arguments(args)
    train(args)
