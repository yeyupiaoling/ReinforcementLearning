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
    parser.add_argument("--game",             type=str,   default="SuperMarioBros-Nes", help='游戏名称')
    parser.add_argument('--lr',               type=float, default=1e-4, help='模型的学习率')
    parser.add_argument('--gamma',            type=float, default=0.9,  help='奖励折扣率')
    parser.add_argument('--tau',              type=float, default=1.0,  help='GAE参数')
    parser.add_argument('--beta',             type=float, default=0.01, help='熵权')
    parser.add_argument('--epsilon',          type=float, default=0.2,  help='剪切替代目标参数')
    parser.add_argument('--batch_size',       type=int,   default=16,   help='训练数据的批量大小')
    parser.add_argument('--num_epochs',       type=int,   default=10,   help='每次采样训练多少轮')
    parser.add_argument("--num_local_steps",  type=int,   default=512,  help='每次采样的次数')
    parser.add_argument("--num_processes",    type=int,   default=16,   help='使用多少条线程启动游戏')
    parser.add_argument("--saved_path",       type=str,   default="models", help='保存模型的路径')
    parser.add_argument("--show_play",        type=bool,  default=False, help='是否显示评估游戏的界面，终端无法使用')
    args = parser.parse_args()
    return args


# 训练模型
def train(args):
    # 使用 GPU训练
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu:0")
    # 创建多进程的游戏环境
    envs = MultipleEnvironments(args.game, args.num_processes)
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
            policy = F.softmax(logits)
            # 根据每个标签的概率随机生成符合概率的标签
            old_m = Categorical(policy)
            action = old_m.sample([1]).squeeze()
            # 记录预测数据
            actions.append(action)
            values.append(value.squeeze())
            # 计算类别的概率的对数
            old_log_policy = old_m.log_prob(paddle.unsqueeze(action, axis=1))
            old_log_policy = paddle.squeeze(old_log_policy)
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
        values = paddle.concat(values).squeeze().detach()
        states = paddle.concat(states).squeeze()

        gae = 0.0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * args.gamma * args.tau
            gae = gae + reward + args.gamma * next_value.detach() * (1.0 - done) - value.detach()
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
                new_policy = F.softmax(logits)
                # 计算类别的概率的对数
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(paddle.unsqueeze(paddle.gather(actions, batch_indices), axis=1))
                new_log_policy = paddle.squeeze(new_log_policy)
                # 计算actor损失
                ratio = paddle.exp(new_log_policy - paddle.gather(old_log_policies, batch_indices))
                advantage = paddle.gather(advantages, batch_indices)
                actor_loss = paddle.clip(ratio, 1.0 - args.epsilon, 1.0 + args.epsilon) * advantage
                actor_loss = paddle.concat([paddle.unsqueeze(ratio * advantage, axis=0), paddle.unsqueeze(actor_loss, axis=0)])
                actor_loss = -paddle.mean(paddle.min(actor_loss, axis=0))
                # 计算critic损失
                critic_loss = F.smooth_l1_loss(paddle.gather(R, batch_indices), value.squeeze())
                entropy_loss = paddle.mean(new_m.entropy())
                # 计算全部损失
                total_loss = actor_loss + critic_loss - args.beta * entropy_loss
                # 计算梯度
                optimizer.clear_grad()
                total_loss.backward()
                optimizer.step()
            paddle.save(model.state_dict(), "{}/model_{}.pth".format(args.saved_path, args.game))
        print("Episode: {}. Total loss: {:.4f}".format(curr_episode, total_loss.numpy()[0]))


if __name__ == "__main__":
    args = get_args()
    print_arguments(args)
    train(args)
